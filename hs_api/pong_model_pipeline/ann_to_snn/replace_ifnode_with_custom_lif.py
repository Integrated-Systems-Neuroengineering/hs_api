"""
Replace SpikingJelly IFNode with Custom_LIFNode for Hardware Compatibility

This script converts the Pong SNN model from using standard SpikingJelly IFNodes
to using Krish's Custom_LIFNode which matches the HiAER Spike hardware behavior.

Key differences in Custom_LIFNode:
- Operation order: spike -> reset -> decay -> input (matches hardware)
- Uses '>' instead of '>=' for threshold comparison
- LIF dynamics with configurable tau and leak
"""

import torch
import torch.nn as nn
import sys
import os
import argparse

from evaluate_dvs_snn import evaluate_dvs_snn

# Add paths to import Custom_LIFNode and DVS environment
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))  # Add pong_stuff to path

# Import SNNCalibrator for threshold optimization
from ann_to_snn_hard_reset_calibration import SNNCalibrator, create_dvs_environment_and_dataloader
sys.path.insert(0, os.path.join(current_dir, '..', '..', '..', 'hs_api'))
sys.path.insert(0, os.path.join(current_dir, '..', '..', '..', 'fxpmath'))

# Import Custom_LIFNode directly from the module file to avoid hs_api.__init__ imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "custom_neurons",
    os.path.join(current_dir, '..', '..', '..', 'hs_api', 'hs_api', 'custom_neurons.py')
)
custom_neurons = importlib.util.module_from_spec(spec)
sys.modules['custom_neurons'] = custom_neurons
spec.loader.exec_module(custom_neurons)
Custom_LIFNode = custom_neurons.Custom_LIFNode
from spikingjelly.activation_based import surrogate, neuron

# Import DVS environment
from hs_api.pong_model_pipeline.DVSWrapper import make_dvs_pong_env

def replace_ifnodes_with_custom_lif(model, tau=63.0, decay_input=False):
    """
    Replace all IFNode instances with Custom_LIFNode in a Sequential model

    Args:
        model: torch.nn.Sequential model containing IFNodes
        tau: Membrane time constant for LIF neurons (default 63.0 to match DVS Gesture)
        decay_input: Whether input participates in decay (default False)

    Returns:
        New Sequential model with Custom_LIFNode replacing IFNode
    """
    new_modules = []
    replaced_count = 0

    print("=" * 60)
    print("REPLACING IFNode WITH Custom_LIFNode")
    print("=" * 60)
    print(f"Parameters: tau={tau}, decay_input={decay_input}")
    print()

    for idx, (name, module) in enumerate(model.named_children()):
        if isinstance(module, neuron.IFNode):
            # Extract parameters from IFNode
            v_threshold = float(module.v_threshold) if hasattr(module, 'v_threshold') else 1.0
            v_reset = float(module.v_reset) if hasattr(module, 'v_reset') and module.v_reset is not None else 0.0
            detach_reset = bool(module.detach_reset) if hasattr(module, 'detach_reset') else True

            # Get surrogate function (preserve alpha if it exists)
            if hasattr(module, 'surrogate_function'):
                surrogate_fn = module.surrogate_function
            else:
                surrogate_fn = surrogate.Sigmoid(alpha=4.0)

            print(f"Layer {idx} ({name}): IFNode -> Custom_LIFNode")
            print(f"  Original: v_threshold={v_threshold}, v_reset={v_reset}, detach_reset={detach_reset}")
            print(f"  Surrogate: {surrogate_fn}")

            # Create Custom_LIFNode with matching parameters
            custom_lif = Custom_LIFNode(
                tau=tau,
                decay_input=decay_input,
                v_threshold=v_threshold,
                v_reset=v_reset,
                surrogate_function=surrogate_fn,
                detach_reset=detach_reset,
                step_mode='s',
                backend='torch'
            )

            new_modules.append(custom_lif)
            replaced_count += 1
            print(f"  Replaced with: Custom_LIFNode(tau={tau}, decay_input={decay_input})")
            print()
        else:
            # Keep non-IFNode layers as-is
            new_modules.append(module)

    print(f"Total IFNodes replaced: {replaced_count}")
    print("=" * 60)
    print()

    # Create new Sequential model
    new_model = nn.Sequential(*new_modules)
    return new_model


def extract_and_save_thresholds(model, model_name="Model", output_file="custom_lif_thresholds.txt"):
    """
    Extract thresholds from Custom_LIFNode layers and save for HiAER Spike hardware calibration

    Args:
        model: PyTorch model with Custom_LIFNode layers
        model_name: Name for the threshold file
        output_file: Path to save threshold information
    """
    print("\n" + "=" * 60)
    print("EXTRACTING THRESHOLDS FOR HIAER SPIKE CALIBRATION")
    print("=" * 60)

    thresholds = {}
    layer_count = 0

    # Extract thresholds from each Custom_LIFNode
    for name, module in model.named_modules():
        if 'Custom_LIF' in str(type(module)):
            threshold = float(module.v_threshold) if hasattr(module, 'v_threshold') else 1.0
            tau = float(module.tau) if hasattr(module, 'tau') else 63.0
            thresholds[name] = {
                'threshold': threshold,
                'tau': tau,
                'layer_index': layer_count
            }
            layer_count += 1
            print(f"Layer {name}: threshold={threshold:.6f}, tau={tau:.1f}")

    # Save to file
    with open(output_file, 'w') as f:
        f.write(f"# Threshold Calibration for {model_name}\n")
        f.write(f"# Generated from Custom_LIFNode layers\n")
        f.write(f"# Total layers: {layer_count}\n\n")

        # Write Python code format for easy copy-paste
        f.write("# Copy these values to your HiAER Spike converter:\n\n")

        layer_names = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2', 'fc3', 'fc4']
        for i, (name, info) in enumerate(thresholds.items()):
            layer_name = layer_names[i] if i < len(layer_names) else f'layer{i}'
            f.write(f"threshold_{layer_name} = {int(info['threshold'])}  # Layer: {name}, tau={info['tau']:.1f}\n")

        f.write(f"\nleak_lif = {tau:.1f}  # LIF leak/tau parameter\n")
        f.write(f"perturbation = 0  # Noise perturbation\n\n")

        # Write LIF_neuron initialization code
        f.write("# LIF neuron initialization:\n")
        for i, (name, info) in enumerate(thresholds.items()):
            layer_name = layer_names[i] if i < len(layer_names) else f'layer{i}'
            f.write(f"LIF_{layer_name} = LIF_neuron(threshold_{layer_name}, perturbation, leak_lif)\n")

    print(f"\nThreshold calibration saved to: {output_file}")
    print("=" * 60)

    return thresholds


def calculate_adaptive_percentile(tau):
    """
    Calculate optimal calibration percentile based on tau value.
    Lower tau (more leak) -> lower percentile (lower thresholds) to compensate.

    Args:
        tau: Membrane time constant

    Returns:
        int: Target percentile (78-95)
    """
    if tau >= 5000:
        return 95  # Minimal leak, use standard percentile
    elif tau >= 1000:
        return 92
    elif tau >= 500:
        return 88
    elif tau >= 200:
        return 84
    elif tau >= 100:
        return 80
    else:  # tau <= 100 (including default 63)
        return 78  # Maximum compensation for leak


def run_tau_schedule(args, model, env):
    """
    Run progressive tau adaptation with calibration at each stage.

    Args:
        args: Command-line arguments
        model: Original model to convert
        env: Pong environment

    Returns:
        best_model: Model with best performance
        stage_results: List of results from each stage
    """
    import random
    import torch.utils.data as data_utils

    # Parse tau schedule
    tau_values = [float(t.strip()) for t in args.tau_schedule.split(',')]
    print(f"\n{'='*60}")
    print(f"PROGRESSIVE TAU ADAPTATION SCHEDULE")
    print(f"{'='*60}")
    print(f"Stages: {' -> '.join([str(t) for t in tau_values])}")
    print(f"Strategy: Test each tau, calibrate thresholds, keep best model")
    print()

    best_model = None
    best_performance = -float('inf')
    best_tau = None
    stage_results = []

    for stage_idx, tau in enumerate(tau_values):
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx + 1}/{len(tau_values)}: tau={tau}")
        print(f"{'='*60}")
        print(f"Leak rate: {100/tau:.4f}% per timestep")
        print()

        # Convert model with this tau
        converted_model = replace_ifnodes_with_custom_lif(
            model,
            tau=tau,
            decay_input=False
        )

        # Evaluate before calibration
        print(f"\nEvaluating with tau={tau} (before calibration):")
        pre_calib_results = evaluate_dvs_snn(
            converted_model, None, env=env, device='cpu',
            episodes=args.episodes, time_steps=args.time_steps
        )
        pre_calib_score = pre_calib_results['average_reward']
        print(f"Pre-calibration score: {pre_calib_score:.2f}")

        target_percentile = None

        # Calibrate if requested
        if args.calibrate_thresholds:
            print(f"\nCalibrating thresholds for tau={tau}...")

            # Determine percentile
            if args.adaptive_percentile:
                target_percentile = calculate_adaptive_percentile(tau)
                print(f"Using adaptive percentile: {target_percentile}")
            else:
                target_percentile = args.percentile
                print(f"Using fixed percentile: {target_percentile}")

            # Create dataloader
            print(f"Collecting {args.calibration_samples} observations...")
            calibration_data = []
            obs, _ = env.reset()
            calibration_data.append(obs)

            for _ in range(args.calibration_samples - 1):
                action = random.randint(0, 5)
                obs, _, terminated, truncated, _ = env.step(action)
                calibration_data.append(obs)
                if terminated or truncated:
                    obs, _ = env.reset()

            calibration_dataset = data_utils.TensorDataset(
                torch.stack([torch.FloatTensor(obs) for obs in calibration_data]),
                torch.zeros(len(calibration_data))
            )
            calibration_loader = data_utils.DataLoader(
                calibration_dataset,
                batch_size=32,
                shuffle=True,
                drop_last=True
            )

            # Run calibration
            device = torch.device('cpu')
            calibrator = SNNCalibrator(
                snn_model=converted_model,
                ann_model=None,
                dataloader=calibration_loader,
                device=device,
                time_steps=args.time_steps
            )
            calibrator.target_percentile = target_percentile

            calibration_results = calibrator.calibrate(
                env=env,
                max_iterations=args.calibration_iterations
            )

            post_calib_score = calibration_results['final_performance']
            print(f"Post-calibration score: {post_calib_score:.2f}")
        else:
            post_calib_score = pre_calib_score

        # Record results
        stage_results.append({
            'tau': tau,
            'leak_percent': 100/tau,
            'pre_calibration_score': pre_calib_score,
            'post_calibration_score': post_calib_score,
            'percentile': target_percentile
        })

        # Check if this is the best model so far
        if post_calib_score > best_performance:
            best_performance = post_calib_score
            best_model = converted_model
            best_tau = tau
            print(f"\n*** NEW BEST MODEL: tau={tau}, score={post_calib_score:.2f} ***")

    # Summary
    print(f"\n{'='*60}")
    print(f"TAU SCHEDULE COMPLETE")
    print(f"{'='*60}")
    print("\nResults by stage:")
    print(f"{'Tau':>8} {'Leak%':>8} {'Pre-Calib':>12} {'Post-Calib':>12} {'Percentile':>12}")
    print("-" * 60)
    for result in stage_results:
        percentile_str = str(result['percentile']) if result['percentile'] else 'N/A'
        print(f"{result['tau']:>8.0f} {result['leak_percent']:>7.4f}% "
              f"{result['pre_calibration_score']:>11.2f} "
              f"{result['post_calibration_score']:>11.2f} "
              f"{percentile_str:>12}")

    print(f"\nBest configuration:")
    print(f"  Tau: {best_tau}")
    print(f"  Score: {best_performance:.2f}")
    print(f"{'='*60}")

    return best_model, stage_results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Replace IFNode with Custom_LIFNode in Pong SNN')
    parser.add_argument('--input', type=str, default='dvs_84_no_bias_snn.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='dvs_84_no_bias_snn_custom_lif.pth', help='Output model path')
    parser.add_argument('--tau', type=float, default=63.0, help='Membrane time constant for LIF neurons (higher = less leak)')
    parser.add_argument('--tau-schedule', type=str, default=None, help='Comma-separated tau values for progressive adaptation (e.g., "1000,500,200,63")')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models on Pong (takes time)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--time-steps', type=int, default=18, help='Time steps for SNN rate coding')
    parser.add_argument('--calibrate', action='store_true', help='Extract and save thresholds for HiAER Spike')
    parser.add_argument('--calibrate-thresholds', action='store_true', help='Run membrane potential-based threshold calibration')
    parser.add_argument('--calibration-iterations', type=int, default=15, help='Number of calibration iterations')
    parser.add_argument('--calibration-samples', type=int, default=200, help='Number of samples for calibration')
    parser.add_argument('--adaptive-percentile', action='store_true', help='Automatically adjust calibration percentile based on tau')
    parser.add_argument('--percentile', type=int, default=95, help='Target percentile for threshold calibration (if not adaptive)')
    args = parser.parse_args()

    print("=" * 60)
    print("PONG SNN: IFNode to Custom_LIFNode Conversion")
    print("=" * 60)
    print()

    # Paths
    input_model_path = args.input
    output_model_path = args.output

    # Check if input file exists
    if not os.path.exists(input_model_path):
        print(f"ERROR: Input model not found: {input_model_path}")
        return

    print(f"Loading model from: {input_model_path}")
    model = torch.load(input_model_path, map_location='cpu', weights_only=False)
    print(f"Model type: {type(model)}")
    print()

    # Print original model structure
    print("ORIGINAL MODEL STRUCTURE:")
    print("-" * 60)
    print(model)
    print()

    # Verify it's a Sequential model
    if not isinstance(model, nn.Sequential):
        print(f"ERROR: Expected nn.Sequential, got {type(model)}")
        return

    # Create DVS Pong environment if needed
    env = None
    if args.evaluate or args.tau_schedule:
        print("Creating DVS Pong environment...")
        config = {
            'env': {
                'game': 'PongNoFrameskip-v4',
                'noop_max': 30,
                'frame_skip': 4,
                'episodic_life': True,
                'clip_rewards': True,
                'grayscale': True
            },
            'dvs': {
                'change_threshold': 10,
                'visualization': False
            }
        }
        env = make_dvs_pong_env(config)
        print()

        # Evaluate original model if not using tau-schedule
        if args.evaluate and not args.tau_schedule:
            print("Evaluating original SNN model on DVS Pong:")
            evaluate_dvs_snn(model, None, env=env, device='cpu',
                            episodes=args.episodes, time_steps=args.time_steps)
    else:
        print("Skipping evaluation (use --evaluate flag to enable)")
        print()

    # Check if tau-schedule mode
    if args.tau_schedule:
        # Run progressive tau adaptation
        best_model, schedule_results = run_tau_schedule(args, model, env)

        # Save best model
        output_path = args.output.replace('.pth', '_tau_schedule_best.pth')
        print(f"\nSaving best model to: {output_path}")
        torch.save(best_model, output_path)

        # Save results summary
        results_path = output_path.replace('.pth', '_results.json')
        import json
        with open(results_path, 'w') as f:
            json.dump(schedule_results, f, indent=2)
        print(f"Results saved to: {results_path}")

        # Extract thresholds from best model if requested
        if args.calibrate:
            extract_and_save_thresholds(
                best_model,
                model_name="Best Tau Schedule Model",
                output_file="tau_schedule_best_thresholds.txt"
            )

        # Close environment
        if env is not None:
            env.close()

        print("\n" + "=" * 60)
        print("TAU SCHEDULE MODE: COMPLETE")
        print("=" * 60)
        return  # Exit after schedule completes

    # Replace IFNodes with Custom_LIFNode
    # tau controls leak: higher = less leak (10000 ~ IFNode, 63 = hardware target)
    print(f"Converting with tau={args.tau} (leak = {100/args.tau:.3f}% per timestep)")
    converted_model = replace_ifnodes_with_custom_lif(
        model,
        tau=args.tau,
        decay_input=False
    )

    # Print converted model structure
    print("CONVERTED MODEL STRUCTURE:")
    print("-" * 60)
    print(converted_model)
    print()

    # Verify conversion
    print("VERIFICATION:")
    print("-" * 60)
    ifnode_count = sum(1 for m in model.modules() if isinstance(m, neuron.IFNode))
    custom_lif_count = sum(1 for m in converted_model.modules() if isinstance(m, Custom_LIFNode))

    print(f"Original model IFNodes: {ifnode_count}")
    print(f"Converted model Custom_LIFNodes: {custom_lif_count}")

    if ifnode_count == custom_lif_count and custom_lif_count > 0:
        print("SUCCESS: All IFNodes replaced with Custom_LIFNode")
    else:
        print("WARNING: Mismatch in neuron counts!")
    print()

    # Save converted model (before calibration)
    print(f"Saving converted model to: {output_model_path}")
    torch.save(converted_model, output_model_path)
    print()

    # Evaluate converted model if flag is set
    if args.evaluate and env is not None:
        print("Evaluating converted SNN model (before calibration):")
        evaluate_dvs_snn(converted_model, None, env=env, device='cpu',
                        episodes=args.episodes, time_steps=args.time_steps)
        print()

    # Membrane potential-based threshold calibration
    if args.calibrate_thresholds:
        print("\n" + "=" * 60)
        print("MEMBRANE POTENTIAL-BASED THRESHOLD CALIBRATION")
        print("=" * 60)
        print()

        # Need to create environment and dataloader if not already created
        if env is None:
            print("Creating DVS Pong environment for calibration...")
            config = {
                'env': {
                    'game': 'PongNoFrameskip-v4',
                    'noop_max': 30,
                    'frame_skip': 4,
                    'episodic_life': True,
                    'clip_rewards': True,
                    'grayscale': True
                },
                'dvs': {
                    'change_threshold': 10,
                    'visualization': False
                }
            }
            env = make_dvs_pong_env(config)

        # Create dataloader by collecting observations
        print(f"Collecting {args.calibration_samples} observations for calibration...")
        calibration_data = []
        obs, _ = env.reset()
        calibration_data.append(obs)

        import random
        for _ in range(args.calibration_samples - 1):
            action = random.randint(0, 5)  # Random actions
            obs, _, terminated, truncated, _ = env.step(action)
            calibration_data.append(obs)
            if terminated or truncated:
                obs, _ = env.reset()

        # Create simple dataloader
        import torch.utils.data as data_utils
        calibration_dataset = data_utils.TensorDataset(
            torch.stack([torch.FloatTensor(obs) for obs in calibration_data]),
            torch.zeros(len(calibration_data))  # Dummy labels
        )
        calibration_loader = data_utils.DataLoader(
            calibration_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True  # Prevent batch size mismatch in calibrator
        )
        print(f"Created dataloader with {len(calibration_data)} samples")
        print()

        # Initialize calibrator
        device = torch.device('cpu')
        calibrator = SNNCalibrator(
            snn_model=converted_model,
            ann_model=None,  # Not needed for this workflow
            dataloader=calibration_loader,
            device=device,
            time_steps=args.time_steps
        )

        # Determine target percentile based on tau and user preference
        if args.adaptive_percentile:
            target_percentile = calculate_adaptive_percentile(args.tau)
            print(f"Using adaptive percentile: {target_percentile} (based on tau={args.tau})")
            print(f"  Rationale: Lower tau (more leak) requires lower percentile to compensate")
        else:
            target_percentile = args.percentile
            print(f"Using fixed percentile: {target_percentile}")

        # Set calibrator's target percentile
        calibrator.target_percentile = target_percentile
        print()

        # Run calibration
        calibration_results = calibrator.calibrate(
            env=env,
            max_iterations=args.calibration_iterations
        )

        # Save calibrated model
        calibrated_output_path = output_model_path.replace('.pth', '_calibrated.pth')
        print(f"\nSaving calibrated model to: {calibrated_output_path}")
        torch.save(converted_model, calibrated_output_path)

        # Extract calibrated thresholds
        print("\nExtracting calibrated thresholds...")
        extract_and_save_thresholds(
            converted_model,
            model_name="Calibrated Custom_LIF Pong SNN",
            output_file="custom_lif_thresholds_calibrated.txt"
        )

        # Print calibration summary
        print("\n" + "=" * 60)
        print("CALIBRATION SUMMARY")
        print("=" * 60)
        print(f"Initial performance: {calibration_results['initial_performance']:.2f}")
        print(f"Final performance: {calibration_results['final_performance']:.2f}")
        print(f"Improvement: {calibration_results['final_performance'] - calibration_results['initial_performance']:.2f}")
        print("=" * 60)
        print()

    # Close environment if it was opened
    if env is not None and args.evaluate:
        env.close()
        print()

    # Extract and save thresholds for HiAER Spike calibration
    if args.calibrate:
        extract_and_save_thresholds(
            converted_model,
            model_name="Custom_LIF Pong SNN",
            output_file="custom_lif_thresholds.txt"
        )

    # Final summary
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Input:  {input_model_path}")
    print(f"Output: {output_model_path}")
    print(f"Neurons converted: {custom_lif_count}")
    print()
    print("The converted model now uses Custom_LIFNode which:")
    print("  - Matches HiAER Spike hardware operation order")
    print("  - Uses LIF dynamics with tau=63.0")
    print("  - Uses '>' threshold comparison (not '>=')")
    print("  - Should have better HiAER Spike performance")
    print()
    if args.calibrate:
        print("Threshold calibration file: custom_lif_thresholds.txt")
        print("  - Copy threshold values to your HiAER Spike converter")
        print("  - Use these with the quantized model for accurate hardware simulation")
    print()
    print("Usage examples:")
    print()
    print("  1. AUTOMATIC: Progressive tau adaptation (RECOMMENDED):")
    print("    python replace_ifnode_with_custom_lif.py \\")
    print("      --input dvs_84_no_bias_snn.pth \\")
    print("      --tau-schedule 1000,500,200,63 \\")
    print("      --calibrate-thresholds --adaptive-percentile \\")
    print("      --calibration-iterations 5 --episodes 2")
    print("    This will:")
    print("      - Test each tau value in sequence")
    print("      - Calibrate at each stage with adaptive percentile")
    print("      - Save the best-performing model")
    print("      - Generate results JSON file")
    print()
    print("  2. MANUAL: Test with minimal leak (prove leak is the issue):")
    print("    python replace_ifnode_with_custom_lif.py --tau 10000 --evaluate --episodes 5")
    print("    Expected: Score ~+20 (like original IFNode)")
    print()
    print("  3. MANUAL: Single tau with calibration:")
    print("    python replace_ifnode_with_custom_lif.py --tau 63 --calibrate-thresholds --adaptive-percentile --episodes 3")
    print()
    print("  4. Manual percentile control:")
    print("    python replace_ifnode_with_custom_lif.py --tau 63 --calibrate-thresholds --percentile 80")
    print()
    print("  5. Quick conversion (no evaluation):")
    print("    python replace_ifnode_with_custom_lif.py --input dvs_84_no_bias_snn.pth")
    print()
    print("Tau guidelines:")
    print("  - tau=10000: Minimal leak (0.01%/step) - approximates IFNode")
    print("  - tau=1000: Very mild leak (0.1%/step)")
    print("  - tau=500: Mild leak (0.2%/step)")
    print("  - tau=200: Moderate leak (0.5%/step)")
    print("  - tau=63: Full leak (1.6%/step) - hardware target")
    print("=" * 60)


if __name__ == "__main__":
    main()
