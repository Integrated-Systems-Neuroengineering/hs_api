"""
DVS ANN-to-SNN Conversion Pipeline (Custom LIF version)
Converts trained DVS models to neuromorphic hardware-ready SNNs with 16-bit
quantization. Step 5 replaces IFNodes with Custom LIF nodes aligned to hardware.
"""

import torch
from spikingjelly.activation_based import ann2snn
from evaluate_dvs_snn import evaluate_dvs_snn, evaluate_dvs_snn_custom_lif, compare_ann_vs_snn
from ann_to_snn_dvs_utils import (
    create_dvs_environment_and_dataloader,
    graphmodule_to_sequential,
    fuse_and_remove_voltage_scalers,
    apply_16bit_quantization,
    save_dvs_models,
    print_conversion_summary,
)

import os
import sys
# Add paths - need to go up to project root to find hs_api
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # Go up 3 levels to reach project root
sys.path.insert(0, os.path.join(project_root, 'hs_api'))
sys.path.insert(0, os.path.join(project_root, 'fxpmath'))
sys.path.insert(0, os.path.join(project_root, 'hs_bridge'))
sys.path.insert(0, os.path.join(project_root, 'connectome_utils'))


def convert_ifnodes_to_custom_lif(module: torch.nn.Module,
                                  tau: float = 63.0,
                                  decay_input: bool = False,
                                  force_hard_reset: bool = True) -> torch.nn.Module:
    """
    Recursively replace IFNode layers with Custom_LIFNode layers aligned to hardware
    behavior.

    - Preserves threshold/reset/surrogate/detach_reset when available
    - Sets LIF parameters: tau and decay_input (default tau=63.0, decay_input=False)

    Args:
        module: Model to transform
        tau: Membrane time constant matching FPGA leak (e.g., 63.0)
        decay_input: Whether inputs decay in the LIF dynamics (False to match converter)

    Returns:
        The transformed module with IFNodes replaced by Custom_LIFNode.
    """
    from spikingjelly.clock_driven.neuron import IFNode as ClockDrivenIFNode
    from spikingjelly.activation_based.neuron import IFNode as ActivationBasedIFNode
    from spikingjelly.activation_based import surrogate as sj_surrogate
    from hs_api.Krish_custom_neurons import Custom_LIFNode

    for name, child in module.named_children():
        replaced = False
        if isinstance(child, (ActivationBasedIFNode, ClockDrivenIFNode)):
            v_threshold = float(getattr(child, 'v_threshold', 1.0))
            v_reset = 0.0 if force_hard_reset else getattr(child, 'v_reset', 0.0)
            surrogate_fn = getattr(child, 'surrogate_function', sj_surrogate.Sigmoid())
            detach_reset = bool(getattr(child, 'detach_reset', False))

            new_neuron = Custom_LIFNode(
                tau=float(tau),
                decay_input=bool(decay_input),
                v_threshold=v_threshold,
                v_reset=v_reset,
                surrogate_function=surrogate_fn,
                detach_reset=detach_reset,
            )
            setattr(module, name, new_neuron)
            replaced = True
            print(
                f"Converted {name}: IFNode -> Custom_LIFNode("
                f"tau={tau}, decay_input={decay_input}, v_th={v_threshold})"
            )

        if not replaced:
            convert_ifnodes_to_custom_lif(child, tau=tau, decay_input=decay_input, force_hard_reset=force_hard_reset)

    return module


def apply_leak_gain_compensation(seq: torch.nn.Sequential, gain: float) -> None:
    """
    Multiply weights (and biases) feeding into each Custom LIF by a compensation gain.
    Offsets amplitude reduction from LIF leak over a finite time window.
    Finds nearest preceding Conv2d/Linear for each LIF (skipping Identity/Flatten).
    """
    from hs_api.Krish_custom_neurons import Custom_LIFNode
    import torch.nn as nn

    layers = list(seq.children())
    for i, layer in enumerate(layers):
        if isinstance(layer, Custom_LIFNode):
            j = i - 1
            # Skip non-synaptic utility layers
            while j >= 0 and isinstance(layers[j], (nn.Identity, nn.Flatten)):
                j -= 1
            if j >= 0 and isinstance(layers[j], (nn.Conv2d, nn.Linear)):
                with torch.no_grad():
                    layers[j].weight.mul_(gain)
                    if layers[j].bias is not None:
                        layers[j].bias.mul_(gain)
                print(f"Applied leak compensation x{gain:.3f} to layer {j} feeding LIF at {i}")


def apply_per_lif_gain_compensation(seq: torch.nn.Sequential, time_steps: int, tau: float, flush_steps: int) -> None:
    """
    Apply per-LIF compensation gains based on effective time window seen by each layer
    when using flush steps. For N LIF layers and flush_steps >= N, the earliest LIF
    effectively integrates over time_steps + (N-1) steps, while the last one sees
    ~time_steps steps. Compute gain_i = W_i / sum_{k=0}^{W_i-1} a^k, a=1-1/tau.
    """
    from hs_api.Krish_custom_neurons import Custom_LIFNode
    import torch.nn as nn

    layers = list(seq.children())
    lif_indices = [idx for idx, l in enumerate(layers) if isinstance(l, Custom_LIFNode)]
    if not lif_indices:
        return

    a = 1.0 - 1.0 / float(tau)
    num_lif = len(lif_indices)
    # Effective extra steps diminish with layer depth; cap by flush_steps
    max_extra = min(flush_steps, num_lif) - 1 if flush_steps > 0 else 0

    for lif_order, lif_i in enumerate(lif_indices):
        extra = max(0, max_extra - lif_order)
        window = time_steps + extra
        leak_sum = sum(a ** k for k in range(window)) if window > 0 else 1.0
        gain = (window / leak_sum) if leak_sum > 0 else 1.0

        # Find nearest preceding synaptic layer
        j = lif_i - 1
        while j >= 0 and isinstance(layers[j], (nn.Identity, nn.Flatten)):
            j -= 1
        if j >= 0 and isinstance(layers[j], (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                layers[j].weight.mul_(gain)
                if layers[j].bias is not None:
                    layers[j].bias.mul_(gain)
            print(f"Applied per-LIF compensation x{gain:.3f} (window={window}) to layer {j} feeding LIF at {lif_i}")


def _lif_pairs_by_index(ref_seq: torch.nn.Sequential):
    from spikingjelly.activation_based.neuron import IFNode as ActivationBasedIFNode
    ref_layers = list(ref_seq.children())
    ref_if_idx = [i for i, l in enumerate(ref_layers) if isinstance(l, ActivationBasedIFNode)]
    return ref_if_idx


def _preceding_synaptic_index(layers, i):
    import torch.nn as nn
    j = i - 1
    while j >= 0 and isinstance(layers[j], (nn.Identity, nn.Flatten)):
        j -= 1
    return j if j >= 0 and isinstance(layers[j], (nn.Conv2d, nn.Linear)) else None


def calibrate_lif_scales(ref_if_seq: torch.nn.Sequential,
                         lif_seq: torch.nn.Sequential,
                         obs_tensor: torch.Tensor,
                         device: torch.device,
                         n_calib: int = 256,
                         time_steps: int = 20,
                         flush_steps: int = 4) -> None:
    """
    Calibrate per-LIF layer scales by matching IF (reference) and LIF (target) layer energies
    over a calibration set. For each LIF, compute s = sqrt(E_if / E_lif) and multiply the
    preceding Conv/Linear weights and bias by s.
    """
    from hs_api.Krish_custom_neurons import Custom_LIFNode
    from spikingjelly.activation_based import functional as afunc

    ref_layers = list(ref_if_seq.children())
    lif_layers = list(lif_seq.children())
    ref_if_idx = _lif_pairs_by_index(ref_if_seq)
    lif_idx = [i for i, l in enumerate(lif_layers) if isinstance(l, Custom_LIFNode)]
    if len(ref_if_idx) != len(lif_idx) or len(lif_idx) == 0:
        print("Calibration skipped: IF/LIF layer count mismatch or none found.")
        return

    # Accumulators of energies per layer
    E_if = [0.0 for _ in ref_if_idx]
    E_lif = [0.0 for _ in lif_idx]

    # Hooks to accumulate sum of squares at each forward
    def make_hook(acc_list, idx):
        def hook(_m, _inp, out):
            try:
                val = out.detach()
            except Exception:
                val = out
            acc_list[idx] += float((val.float() ** 2).sum().item())
        return hook

    ref_hooks = []
    lif_hooks = []
    for k, i in enumerate(ref_if_idx):
        h = ref_layers[i].register_forward_hook(make_hook(E_if, k))
        ref_hooks.append(h)
    for k, i in enumerate(lif_idx):
        h = lif_layers[i].register_forward_hook(make_hook(E_lif, k))
        lif_hooks.append(h)

    lif_seq.eval(); ref_if_seq.eval()
    # Use up to n_calib samples
    num = min(n_calib, obs_tensor.shape[0])
    with torch.no_grad():
        for n in range(num):
            obs = obs_tensor[n:n+1].to(device)
            # Reset both nets
            afunc.reset_net(lif_seq);
            afunc.reset_net(ref_if_seq)
            # Rate coding steps
            for _ in range(time_steps):
                _ = lif_seq(obs)
                _ = ref_if_seq(obs)
            if flush_steps and flush_steps > 0:
                blank = torch.zeros_like(obs)
                for _ in range(flush_steps):
                    _ = lif_seq(blank)
                    _ = ref_if_seq(blank)

    # Remove hooks
    for h in ref_hooks: h.remove()
    for h in lif_hooks: h.remove()

    # Compute and apply per-layer scales
    import torch.nn as nn
    eps = 1e-12
    for idx_pair, e_if, e_lif in zip(lif_idx, E_if, E_lif):
        if e_lif <= eps:
            continue
        s = (e_if / e_lif) ** 0.5
        pj = _preceding_synaptic_index(lif_layers, idx_pair)
        if pj is not None:
            with torch.no_grad():
                lif_layers[pj].weight.mul_(s)
                if lif_layers[pj].bias is not None:
                    lif_layers[pj].bias.mul_(s)
            print(f"Calibrated LIF at {idx_pair} via layer {pj}: scale x{s:.3f}")


def main():
    """Main DVS ANN-to-SNN conversion pipeline"""
    
    # Configuration
    # dvs_model_path = "../ann_training/checkpoints/dvs_63.pth"
    dvs_model_path = "../ann_training/trained_dvs_ann_weights.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("DVS ANN-TO-SNN CONVERSION PIPELINE")
    print("="*60)
    print(f"Device: {device}")
    print(f"DVS checkpoint: {dvs_model_path}")
    
    # =================== Step 1: Setup environment, load model, and create calibration data ===================
    print("\n" + "="*60)
    print("STEP 1: ENVIRONMENT SETUP AND MODEL LOADING")
    print("="*60)
    
    env, ann_model, model_architecture, loader, obs_tensor = create_dvs_environment_and_dataloader(
        dvs_model_path, device, num_observations=1000
    )
    
    # ================== Step 1.5: Evaluate Original ANN Performance ==========================================
    print("\n" + "="*60)
    print("STEP 1.5: EVALUATING ORIGINAL ANN PERFORMANCE")
    print("="*60)
    
    print("Evaluating original DVS ANN model...")
    print("Expected performance: ~20.4 average return")
    
    # Evaluate ANN with a simple forward pass (no rate coding)
    def evaluate_ann_simple(model, env, device, episodes=10):
        """Simple ANN evaluation without rate coding"""
        total_rewards = []
        
        for episode in range(episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < 5000:  # Max steps per episode
                # Convert observation to tensor
                if len(obs.shape) == 3 and obs.shape[0] in [2, 3]:  # Support both 2 and 3 channel DVS
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    print(f"Processing obs shape: {obs.shape} (channels: {obs.shape[0] if len(obs.shape) == 3 else 'N/A'})")
                    print("Warning: Unexpected observation format, skipping episode")
                    break
                
                with torch.no_grad():
                    q_values = model(obs_tensor)
                    action = q_values.argmax().item()
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            print(f"ANN Episode {episode + 1}: reward = {episode_reward}, steps = {steps}")
        
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"ANN Average reward over {episodes} episodes: {avg_reward:.2f}")
        return avg_reward
    
    # Test action selection method differences
    print(f"\n=== DEBUGGING ACTION SELECTION METHODS ===")
    test_obs, _ = env.reset()
    print(f"Test observation shape: {test_obs.shape}")
    print(f"Test observation range: [{test_obs.min():.3f}, {test_obs.max():.3f}]")
    
    # Method 1: Our current method
    obs_tensor_1 = torch.tensor(test_obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values_1 = ann_model(obs_tensor_1)
        action_1 = q_values_1.argmax().item()
    
    # Method 2: Training method (from DVS trainer)
    obs_tensor_2 = torch.FloatTensor(test_obs).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values_2 = ann_model(obs_tensor_2)
        action_2 = q_values_2.argmax(dim=1).item()
    
    print(f"Method 1 (current): Q-values = {q_values_1[0].cpu().numpy()}, action = {action_1}")
    print(f"Method 2 (training): Q-values = {q_values_2[0].cpu().numpy()}, action = {action_2}")
    print(f"Q-values match: {torch.allclose(q_values_1, q_values_2, atol=1e-6)}")
    print(f"Actions match: {action_1 == action_2}")
    
    # Check if model is in correct mode
    print(f"Model training mode: {ann_model.training}")
    ann_model.eval()
    print(f"After .eval(): {ann_model.training}")

    # ann_avg_reward = evaluate_ann_simple(ann_model, env, device, episodes=1)
    
    # if ann_avg_reward < -15:
    #     print(f"WARNING: ANN performance is poor ({ann_avg_reward:.2f})! Expected ~20.4")
    #     print("This suggests an issue with model loading or environment setup")
    #     print("\nPossible causes:")
    #     print("1. Wrong checkpoint file or loading issue")
    #     print("2. Environment setup differs from training")
    #     print("3. Model architecture mismatch") 
    #     print("4. DVS thresholds or frame_skip mismatch")
    #     print("\nContinuing conversion but SNN performance will likely be poor too...")
        
    #     # Ask user to verify
    #     print(f"\nDEBUG CHECKLIST:")
    #     print(f"- Does checkpoint score match? Check debug output above")
    #     print(f"- Are model weights reasonable? Check weight stats above") 
    #     print(f"- Is environment setup identical to training?")
    #     print(f"- frame_skip=1, change_threshold=10, static_threshold=100")
    #     print(f"- Architecture: {model_architecture}")
    
    # ================== Step 2: Convert ANN to SNN using SpikingJelly =========================================
    print("\n" + "="*60)
    print("STEP 2: SPIKINGJELLY ANN-TO-SNN CONVERSION")
    print("="*60)
    
    print("Converting DVS ANN to SNN using SpikingJelly ann2snn...")
    converter = ann2snn.Converter(dataloader=loader, mode='max')
    graph_snn = converter(ann_model).to(device)
    print("DVS Graph SNN created")
    print("Evaluating Graph SNN")
    print("Graph snn: ", graph_snn)

    # Use proper DVS SNN evaluation
    # evaluate_dvs_snn(graph_snn, ann_model, env, device, episodes=1, time_steps=18)

    # =================== Step 3: Convert GraphModule to Sequential =========================================
    print("\n" + "="*60)
    print("STEP 3: GRAPH-TO-SEQUENTIAL CONVERSION")
    print("="*60)
    
    flat_dvs_snn = graphmodule_to_sequential(graph_snn)
    print("DVS Sequential SNN structure:")
    print(flat_dvs_snn)
    print("Evaluating Sequential SNN")
    # evaluate_dvs_snn(flat_dvs_snn, ann_model, env, device, episodes=1, time_steps=18)

    # =================== Step 4: Fuse VoltageScalers for HiAER Spike compatibility ==========================
    print("\n" + "="*60)
    print("STEP 4: VOLTAGE SCALER FUSION")
    print("="*60)
    
    fused_dvs_snn = fuse_and_remove_voltage_scalers(flat_dvs_snn)
    print("DVS SNN fused (VoltageScalers removed):")
    print(fused_dvs_snn)
    print("Evaluating fused DVS SNN")
    # evaluate_dvs_snn(fused_dvs_snn, ann_model, env, device, episodes=1, time_steps=18)

    # =================== Step 5: Convert IFNodes to Custom LIF nodes =======================================
    print("\n" + "=" * 60)
    print("STEP 5: CONVERT IFNODES TO CUSTOM LIF")
    print("=" * 60)
    # Use hardware-aligned Custom LIF settings: tau matches FPGA leak, decay_input=False
    # Adjust tau/decay_input here if your trained model used different values.
    custom_lif_snn = convert_ifnodes_to_custom_lif(
        fused_dvs_snn,
        tau=63.0,
        decay_input=False,
        force_hard_reset=True,
    )
    print(custom_lif_snn)

    # Compute and apply leak compensation gains to layers feeding each LIF
    time_steps = 20
    tau_val = 63.0
    flush = 4
    # Global uniform compensation (baseline)
    a = 1.0 - 1.0 / float(tau_val)
    leak_sum = sum(a ** k for k in range(time_steps))
    comp_gain = (time_steps / leak_sum) if leak_sum > 0 else 1.0
    apply_leak_gain_compensation(custom_lif_snn, comp_gain)
    # Per-LIF refinement using effective window with flush
    apply_per_lif_gain_compensation(custom_lif_snn, time_steps=time_steps, tau=tau_val, flush_steps=flush)
    # Per-layer calibration to align IF and LIF layer energies over calibration set
    try:
        calibrate_lif_scales(
            ref_if_seq=fused_dvs_snn,
            lif_seq=custom_lif_snn,
            obs_tensor=obs_tensor,
            device=device,
            n_calib=256,
            time_steps=time_steps,
            flush_steps=flush,
        )
    except Exception as e:
        print(f"Calibration skipped due to error: {e}")

    # Use LIF-aware evaluator to ensure fair behavior
    evaluate_dvs_snn_custom_lif(
        custom_lif_snn,
        env,
        device,
        episodes=1,
        time_steps=time_steps,
        flush_steps=flush,
        assert_hard_reset=True,
        tau=tau_val,
        decay_input=False,
        log_stats=True,
    )



    # Step 7: Save all model variants
    print("\n" + "=" * 60)
    print("STEP 7: SAVING MODELS")
    print("=" * 60)

    save_dvs_models(fused_dvs_snn, custom_lif_snn, "pre_custom_lif_dvs_snn_84.pth", "custom_lif_snn_84.pth")

    # Step 8: Final ANN vs SNN comparison with optimized parameters (optional)
    # comparison_results = compare_ann_vs_snn(ann_model, custom_lif_snn, env, device, episodes=3)

    # Step 9: Print conversion summary
    print_conversion_summary(model_architecture, custom_lif_snn)

    env.close()
    print("DVS environment closed. Conversion complete!")


if __name__ == "__main__":
    main()
