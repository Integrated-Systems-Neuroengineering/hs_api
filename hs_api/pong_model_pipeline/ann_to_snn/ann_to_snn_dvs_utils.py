"""
Utility functions for DVS ANN-to-SNN conversion pipeline
Contains all method definitions for environment setup, model loading, and conversion
"""

import torch
import torch.nn as nn
import gymnasium as gym
import ale_py
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.fx import GraphModule
import sys
import os

# Add ann_training to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def setup_dvs_environment():
    """
    Create DVS Pong evaluation environment matching DVS training setup
    
    Returns:
        env: Configured DVS environment
        config: DVS configuration used
    """
    print("Setting up DVS environment...")
    
    # Register ALE environments
    gym.register_envs(ale_py)
    
    from hs_api.pong_model_pipeline.DVSWrapper import make_dvs_pong_env
    
    # DVS Environment configuration (matching DVS trainer)
    dvs_config = {
        'env': {
            'game': 'PongNoFrameskip-v4',
            'noop_max': 30,
            'frame_skip': 4,  # FIXED: Match training environment (was 1)
            'episodic_life': True,
            'clip_rewards': True,
            'grayscale': True
        },
        'dvs': {
            'change_threshold': 10,
            'static_threshold': 100,
            'visualization': False,
            'vis_interval': 1000
        }
    }
    
    env = make_dvs_pong_env(dvs_config)
    print(f"DVS Environment created with observation space: {env.observation_space}")
    
    return env, dvs_config

def load_dvs_model(checkpoint_path, device):
    """
    Load DVS model from checkpoint with automatic architecture detection
    
    Args:
        checkpoint_path: Path to DVS checkpoint file
        device: torch device (cuda/cpu)
        
    Returns:
        model: Loaded DVS model
        architecture: Model architecture string
    """
    # Import relative to final_push directory
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ann_training.dvs_84_no_bias_model import NoBias84
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DVS checkpoint not found: {checkpoint_path}")
    
    print(f"Loading DVS model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model info from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        architecture = config['model'].get('architecture', 'silu_nature_cnn_2ch')
        input_channels = config['model'].get('input_channels', 2)  # DVS No Static has 2 channels (OFF, ON)
        n_actions = config['model'].get('n_actions', 6)
    else:
        # Default values if config not in checkpoint
        print("Warning: No config found in checkpoint, using defaults")
        architecture = 'silu_nature_cnn_2ch'
        input_channels = 2  # DVS No Static: OFF events, ON events only
        n_actions = 6
    
    print(f"Model architecture: {architecture}")
    print(f"Input channels: {input_channels}")
    print(f"Actions: {n_actions}")
    
    # Create original model to load weights, then create SJ-compatible version
    if architecture == 'silu_nature_cnn_2ch':
        original_model = NoBias84(input_channels, n_actions)
    else:
        # Default to NoBias84 for this specific use case
        print(f"Warning: Unknown architecture {architecture}, defaulting to NoBias84")
        original_model = NoBias84(input_channels, n_actions)

    # Load model state into original model
    if 'model_state_dict' in checkpoint:
        original_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'q_network' in checkpoint:
        original_model.load_state_dict(checkpoint['q_network'])
    else:
        # Checkpoint might be just the state dict
        original_model.load_state_dict(checkpoint)
    
    # Create SpikingJelly-compatible ReLU version and copy weights
    print("\n=== Creating SpikingJelly-compatible ReLU model ===")
    model = NoBias84(input_channels, n_actions)
    model = copy_weights_from_silu_model(original_model, model)
    
    # Verify checkpoint loaded correctly by checking first layer weights again
    first_param_after_load = next(model.parameters())
    print(f"After checkpoint loading - First layer weight stats: mean={first_param_after_load.mean().item():.6f}, std={first_param_after_load.std().item():.6f}")
    
    model = model.to(device)
    model.eval()
    
    # Add action_space attribute for compatibility with evaluate_snn
    class MockActionSpace:
        def __init__(self, n):
            self.n = n
    model.action_space = MockActionSpace(n_actions)
    
    print("DVS model loaded successfully!")
    
    # Print checkpoint info
    if 'episode' in checkpoint:
        print(f"Checkpoint episode: {checkpoint['episode']}")
    if 'global_step' in checkpoint:
        print(f"Checkpoint step: {checkpoint['global_step']:,}")
    if 'score' in checkpoint:
        print(f"Checkpoint score: {checkpoint['score']:.2f}")
    
    # Final weight verification after device move
    first_param = next(model.parameters())
    print(f"FINAL weight stats: mean={first_param.mean().item():.6f}, std={first_param.std().item():.6f}")
    
    # Check if weights are all zeros (common loading issue)
    if first_param.std().item() < 1e-6:
        print("WARNING: Model weights appear to be all zeros or constants!")
        print("This suggests a checkpoint loading issue")
    elif first_param.std().item() > 1.0:
        print("WARNING: Model weights have very high variance!")
        print("This suggests initialization rather than trained weights")
    
    # Additional debugging: check if checkpoint contains what we expect
    print(f"\nCheckpoint contents: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        state_keys = list(checkpoint['model_state_dict'].keys())
        print(f"Model state dict keys: {state_keys[:5]}...")  # First 5 keys
    elif 'q_network' in checkpoint:
        state_keys = list(checkpoint['q_network'].keys())  
        print(f"Q-network state dict keys: {state_keys[:5]}...")
    else:
        print("WARNING: Unusual checkpoint format - may not be loading correctly")
        
    # CRITICAL DEBUG: Test the model with a random input to see if it gives reasonable outputs
    print(f"\n=== TESTING MODEL WITH SAMPLE INPUT ===")
    test_input = torch.randn(1, input_channels, 84, 84).to(device) * 0.5 + 0.5  # DVS-like input in [0,1]
    model.eval()
    with torch.no_grad():
        test_output = model(test_input)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test input range: [{test_input.min().item():.3f}, {test_input.max().item():.3f}]")
        print(f"Test output shape: {test_output.shape}")
        print(f"Test Q-values: {test_output[0].cpu().numpy()}")
        print(f"Test argmax action: {test_output.argmax().item()}")
        print(f"Q-value range: [{test_output.min().item():.3f}, {test_output.max().item():.3f}]")
        
        # Check if all Q-values are the same (dead model)
        if torch.std(test_output) < 1e-6:
            print("ERROR: All Q-values are identical! Model appears dead.")
        elif torch.max(torch.abs(test_output)) > 1000:
            print("WARNING: Q-values are extremely large, possible training instability")
        else:
            print("Model test looks normal")
    
    return model, architecture

def collect_dvs_calibration_data(env, model, device, num_observations=1000, min_activity_threshold=0.01):
    """
    Collect DVS observations for SNN calibration with activity validation

    Args:
        env: DVS environment
        model: DVS model for action selection
        device: torch device
        num_observations: Number of observations to collect
        min_activity_threshold: Minimum fraction of pixels that should have DVS events

    Returns:
        observations: List of DVS observations
        obs_tensor: Tensor of calibration data
        loader: DataLoader for calibration
    """
    print(f"Collecting {num_observations} DVS observations for calibration...")
    print(f"Activity threshold: {min_activity_threshold*100:.1f}% of pixels must have DVS events")

    # Warm-up phase: play some steps to generate meaningful DVS activity
    print("Warm-up phase: generating initial DVS activity...")
    obs, info = env.reset()
    warmup_steps = 100

    for warmup_step in range(warmup_steps):
        if len(obs.shape) == 3 and obs.shape[0] in [2, 3]:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(obs_tensor)
                action = q_values.argmax().item()
        else:
            action = env.action_space.sample()  # Random action if obs format unexpected

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    print(f"Warm-up complete. Starting calibration data collection...")

    observations = []
    active_observations = 0
    total_activity = 0.0
    attempts = 0
    max_attempts = num_observations * 3  # Allow up to 3x attempts to get good data

    # print(f"DVS observation shape: {obs.shape}")
    # print(f"Initial DVS observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    while len(observations) < num_observations and attempts < max_attempts:
        attempts += 1

        # Calculate DVS activity for this observation
        if obs.shape[0] >= 2:  # At least OFF and ON channels
            dvs_activity = (obs[0].sum() + obs[1].sum()) / (obs.shape[1] * obs.shape[2])  # Activity per pixel
            total_pixels = obs.shape[1] * obs.shape[2]
            active_pixels = (obs[0] > 0).sum() + (obs[1] > 0).sum()
            activity_fraction = active_pixels / (2 * total_pixels)  # Fraction of pixels with events
        else:
            dvs_activity = 0.0
            activity_fraction = 0.0

        # Debug first few observations
        # if len(observations) < 3:
            # print(f"DVS observation {len(observations)} shape: {obs.shape}, dtype: {obs.dtype}")
            # print(f"  Activity fraction: {activity_fraction:.4f} (threshold: {min_activity_threshold:.4f})")
            # if obs.shape[0] == 2:  # 2-channel DVS (OFF, ON)
            #     print(f"  Channel sums: OFF={obs[0].sum():.0f}, ON={obs[1].sum():.0f}")
            #     print(f"  Channel ranges: OFF=[{obs[0].min():.3f},{obs[0].max():.3f}], ON=[{obs[1].min():.3f},{obs[1].max():.3f}]")
            # elif obs.shape[0] == 3:  # 3-channel DVS (OFF, ON, Static)
            #     print(f"  Channel sums: OFF={obs[0].sum():.0f}, ON={obs[1].sum():.0f}, Static={obs[2].sum():.0f}")
            #     print(f"  Channel ranges: OFF=[{obs[0].min():.3f},{obs[0].max():.3f}], ON=[{obs[1].min():.3f},{obs[1].max():.3f}], Static=[{obs[2].min():.3f},{obs[2].max():.3f}]")

        # Only include observations with sufficient DVS activity
        if activity_fraction >= min_activity_threshold:
            observations.append(obs.copy())
            active_observations += 1
            total_activity += dvs_activity

        # Convert DVS obs to tensor for model inference
        if len(obs.shape) == 3 and obs.shape[0] in [2, 3]:  # Support both 2 and 3 channel DVS
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            print(f"Unexpected DVS obs shape: {obs.shape}")
            break

        with torch.no_grad():
            q_values = model(obs_tensor)
            action = q_values.argmax().item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            obs, info = env.reset()

    if len(observations) < num_observations:
        print(f"WARNING: Only collected {len(observations)} active observations out of {num_observations} requested")
        print(f"Attempted {attempts} total steps. Consider lowering min_activity_threshold or playing longer")

    print(f"Collected {len(observations)} DVS observations with sufficient activity")
    print(f"Average DVS activity: {total_activity/max(len(observations), 1):.6f} events per pixel")
    print(f"Active observations: {active_observations}/{attempts} ({100*active_observations/max(attempts, 1):.1f}%)")

    if len(observations) == 0:
        raise ValueError("No active DVS observations collected! Check environment setup or lower activity threshold")

    # Validate collected data quality
    obs_array = np.stack(observations)
    print(f"DVS observations array shape: {obs_array.shape}")

    # Calculate statistics for validation
    channel_means = [obs_array[:, i].mean() for i in range(obs_array.shape[1])]
    channel_stds = [obs_array[:, i].std() for i in range(obs_array.shape[1])]
    total_events = obs_array.sum()

    print(f"Data quality validation:")
    for i, (mean, std) in enumerate(zip(channel_means, channel_stds)):
        print(f"  Channel {i}: mean={mean:.6f}, std={std:.6f}")
    print(f"  Total DVS events: {total_events:.0f}")
    print(f"  Events per observation: {total_events/len(observations):.1f}")

    if total_events < 100:  # Very low activity threshold
        print("WARNING: Very low DVS activity detected. Calibration may be less effective.")

    obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
    dummy_labels = torch.zeros(len(obs_tensor), dtype=torch.long)
    obs_dataset = TensorDataset(obs_tensor, dummy_labels)
    loader = DataLoader(obs_dataset, batch_size=32, shuffle=False)
    print("DVS calibration dataloader created")

    return observations, obs_tensor, loader

def validate_calibration_data_quality(obs_tensor, min_events_per_obs=10, min_variance=1e-6):
    """
    Validate quality of collected calibration data

    Args:
        obs_tensor: Tensor of calibration observations
        min_events_per_obs: Minimum DVS events per observation
        min_variance: Minimum variance to ensure data isn't constant

    Returns:
        dict: Validation results and statistics
    """
    print("Validating calibration data quality...")

    results = {
        'valid': True,
        'warnings': [],
        'statistics': {}
    }

    # Basic shape validation
    if len(obs_tensor.shape) != 4:
        results['valid'] = False
        results['warnings'].append(f"Invalid tensor shape: {obs_tensor.shape}, expected 4D")
        return results

    n_samples, n_channels, height, width = obs_tensor.shape
    total_pixels_per_obs = n_channels * height * width

    # Calculate statistics
    total_events = obs_tensor.sum().item()
    events_per_obs = total_events / n_samples
    mean_activity = obs_tensor.mean().item()
    variance = obs_tensor.var().item()

    # Channel-wise statistics
    channel_stats = []
    for ch in range(n_channels):
        ch_data = obs_tensor[:, ch]
        ch_stats = {
            'mean': ch_data.mean().item(),
            'std': ch_data.std().item(),
            'events': ch_data.sum().item(),
            'active_pixels': (ch_data > 0).sum().item()
        }
        channel_stats.append(ch_stats)

    results['statistics'] = {
        'total_samples': n_samples,
        'total_events': total_events,
        'events_per_obs': events_per_obs,
        'mean_activity': mean_activity,
        'variance': variance,
        'channel_stats': channel_stats
    }

    # Validation checks
    if events_per_obs < min_events_per_obs:
        results['warnings'].append(f"Low activity: {events_per_obs:.1f} events/obs < {min_events_per_obs} threshold")

    if variance < min_variance:
        results['warnings'].append(f"Low variance: {variance:.2e} < {min_variance:.2e} threshold")
        results['valid'] = False

    if total_events == 0:
        results['warnings'].append("No DVS events detected in calibration data!")
        results['valid'] = False

    # Check if all channels are active
    inactive_channels = []
    for ch, stats in enumerate(channel_stats):
        if stats['events'] == 0:
            inactive_channels.append(ch)

    if inactive_channels:
        results['warnings'].append(f"Inactive channels detected: {inactive_channels}")

    # Quality assessment
    if events_per_obs > min_events_per_obs * 10:
        quality = "excellent"
    elif events_per_obs > min_events_per_obs * 3:
        quality = "good"
    elif events_per_obs > min_events_per_obs:
        quality = "acceptable"
    else:
        quality = "poor"

    results['quality'] = quality

    # Print summary
    print(f"Calibration data validation results:")
    print(f"  Quality: {quality}")
    print(f"  Samples: {n_samples}")
    print(f"  Events per observation: {events_per_obs:.1f}")
    print(f"  Mean activity: {mean_activity:.6f}")
    print(f"  Data variance: {variance:.6f}")

    for ch, stats in enumerate(channel_stats):
        print(f"  Channel {ch}: {stats['events']:.0f} events, {stats['active_pixels']} active pixels")

    if results['warnings']:
        print(f"  Warnings: {len(results['warnings'])}")
        for warning in results['warnings']:
            print(f"    - {warning}")

    return results

def create_dvs_environment_and_dataloader(checkpoint_path, device, num_observations=1000):
    """
    Complete environment and dataloader creation for DVS conversion
    
    Args:
        checkpoint_path: Path to DVS checkpoint
        device: torch device
        num_observations: Number of calibration observations
        
    Returns:
        env: DVS environment
        model: Loaded DVS model
        architecture: Model architecture
        loader: Calibration dataloader
        obs_tensor: Calibration data tensor
    """
    # Setup environment
    env, dvs_config = setup_dvs_environment()
    
    # Load model
    model, architecture = load_dvs_model(checkpoint_path, device)
    
    # Collect calibration data with activity validation
    observations, obs_tensor, loader = collect_dvs_calibration_data(
        env, model, device, num_observations, min_activity_threshold=0.005  # 0.5% of pixels must have events
    )

    # Validate calibration data quality
    validation_results = validate_calibration_data_quality(obs_tensor)

    if not validation_results['valid']:
        raise ValueError(f"Calibration data validation failed: {validation_results['warnings']}")

    if validation_results['quality'] == 'poor':
        print("WARNING: Calibration data quality is poor. Consider collecting more data or adjusting thresholds.")

    return env, model, architecture, loader, obs_tensor

def graphmodule_to_sequential(graph_module):
    """
    Convert a torch.fx.GraphModule into a flat nn.Sequential
    
    Args:
        graph_module: SpikingJelly GraphModule SNN
        
    Returns:
        sequential_model: Flat Sequential model
    """
    print("Converting DVS GraphModule to Sequential...")
    
    class InputNormalization(nn.Module):
        """Module to handle input normalization if needed"""
        def forward(self, x):
            return x  # DVS input is already normalized (0.0-1.0)
    
    layers = []
    modules_dict = dict(graph_module.named_modules())
    
    # Process nodes in execution order
    for node in graph_module.graph.nodes:
        if node.op == 'call_module':
            # Get the actual submodule object
            submod = modules_dict[node.target]
            
            # If it's a nested sequential, flatten it
            if isinstance(submod, nn.Sequential):
                layers.extend(list(submod.children()))
            else:
                layers.append(submod)
        elif node.op == 'call_function':
            # Handle functions like torch.flatten
            if node.target == torch.flatten:
                layers.append(nn.Flatten())
        elif node.op == 'call_method':
            # Handle method calls like tensor.view()
            if node.target == 'view':
                layers.append(nn.Flatten(start_dim=1))  # Flatten from dim 1 onwards
    
    # Remove duplicates while preserving order
    seen = set()
    unique_layers = []
    for layer in layers:
        if id(layer) not in seen:
            seen.add(id(layer))
            unique_layers.append(layer)
    
    sequential_model = nn.Sequential(*unique_layers)
    print("DVS Sequential SNN created")
    
    return sequential_model

def validate_graph_to_sequential_conversion(graph_model, sequential_model, test_input, device, tolerance=1e-5):
    """
    Validate that Graph-to-Sequential conversion preserves functionality

    Args:
        graph_model: Original GraphModule SNN
        sequential_model: Converted Sequential SNN
        test_input: Test input tensor
        device: torch device
        tolerance: Numerical tolerance for comparison

    Returns:
        dict: Validation results
    """
    print("Validating Graph-to-Sequential conversion...")

    try:
        # Reset both models to ensure clean state
        from spikingjelly.activation_based import functional
        functional.reset_net(graph_model)
        functional.reset_net(sequential_model)

        # Run both models
        with torch.no_grad():
            graph_output = graph_model(test_input)
            sequential_output = sequential_model(test_input)

        # Compare outputs
        max_diff = torch.max(torch.abs(graph_output - sequential_output)).item()
        mean_diff = torch.mean(torch.abs(graph_output - sequential_output)).item()
        functionally_equivalent = max_diff < tolerance

        results = {
            'functionally_equivalent': functionally_equivalent,
            'max_difference': max_diff,
            'mean_difference': mean_diff,
            'tolerance': tolerance
        }

        print(f"  Output comparison:")
        print(f"    Max difference: {max_diff:.8f}")
        print(f"    Mean difference: {mean_diff:.8f}")
        print(f"    Functionally equivalent (tol={tolerance:.0e}): {functionally_equivalent}")

        return results

    except Exception as e:
        print(f"  Validation failed with error: {e}")
        return {
            'functionally_equivalent': False,
            'error': str(e),
            'max_difference': float('inf'),
            'mean_difference': float('inf')
        }

def fuse_and_remove_voltage_scalers(qnet):
    """
    Properly fuse VoltageScaler information into weights/biases, then remove them
    
    Args:
        qnet: Sequential SNN model with VoltageScalers
        
    Returns:
        fused_model: Model with VoltageScalers fused and removed
    """
    import copy
    import math
    
    from spikingjelly.activation_based.neuron import IFNode
    try:
        from spikingjelly.activation_based.ann2snn.modules import VoltageScaler as SJVoltageScaler
    except ImportError:  # pragma: no cover - spikingjelly always available in conversion env
        SJVoltageScaler = None
    
    def _is_voltage_scaler(module):
        if SJVoltageScaler is not None and isinstance(module, SJVoltageScaler):
            return True
        return module.__class__.__name__ == "VoltageScaler" and hasattr(module, "scale")
    
    weight_layer_types = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )
    
    layers = [copy.deepcopy(mod) for mod in qnet.children()]
    fused_layers = []
    pending_scale = 1.0
    
    with torch.no_grad():
        for layer in layers:
            if _is_voltage_scaler(layer):
                scale = getattr(layer, "scale", None)
                if scale is None and hasattr(layer, "_buffers"):
                    scale = layer._buffers.get("scale")
                if scale is None:
                    continue
                scale_value = scale.item() if hasattr(scale, "item") else float(scale)
                pending_scale *= scale_value
                continue
    
            if isinstance(layer, IFNode):
                if not math.isclose(pending_scale, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                    layer.v_threshold = float(layer.v_threshold) / pending_scale
                    pending_scale = 1.0
                fused_layers.append(layer)
                continue
    
            if isinstance(layer, weight_layer_types):
                if not math.isclose(pending_scale, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                    layer.weight.data.mul_(pending_scale)
                    pending_scale = 1.0
                fused_layers.append(layer)
                continue
    
            fused_layers.append(layer)
    
        if not math.isclose(pending_scale, 1.0, rel_tol=1e-9, abs_tol=1e-12) and fused_layers:
            for layer in reversed(fused_layers):
                if isinstance(layer, weight_layer_types):
                    layer.weight.data.mul_(pending_scale)
                    if getattr(layer, "bias", None) is not None:
                        layer.bias.data.mul_(pending_scale)
                    pending_scale = 1.0
                    break
                if isinstance(layer, IFNode):
                    layer.v_threshold = float(layer.v_threshold) / pending_scale
                    pending_scale = 1.0
                    break
            if not math.isclose(pending_scale, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                raise RuntimeError("Unconsumed VoltageScaler scaling factor could not be fused into the network.")
    
    return nn.Sequential(*fused_layers)

def validate_mathematically_accurate_fusion(original_model, fused_model, test_input, device, time_steps=3, tolerance=1e-3):
    """
    Validate that VoltageScaler fusion preserves mathematical accuracy

    Args:
        original_model: Model before VoltageScaler fusion
        fused_model: Model after VoltageScaler fusion
        test_input: Test input tensor
        device: torch device
        time_steps: Number of time steps to test
        tolerance: Numerical tolerance for comparison

    Returns:
        dict: Validation results with detailed analysis
    """
    print("Validating mathematically accurate VoltageScaler fusion...")

    try:
        from spikingjelly.activation_based import functional

        max_differences = []
        overall_functionally_equivalent = True

        for t in range(time_steps):
            print(f"  Testing functional equivalence across time steps...")

            # Reset both models
            functional.reset_net(original_model)
            functional.reset_net(fused_model)

            # Run for t time steps
            for step in range(t + 1):
                with torch.no_grad():
                    original_output = original_model(test_input)
                    fused_output = fused_model(test_input)

            # Compare final outputs
            max_diff = torch.max(torch.abs(original_output - fused_output)).item()
            max_differences.append(max_diff)

            step_equivalent = max_diff < tolerance
            if not step_equivalent:
                overall_functionally_equivalent = False

        max_difference_overall = max(max_differences)

        # Analyze IFNode threshold changes (for informational purposes)
        threshold_changes = []
        for name, module in fused_model.named_modules():
            if hasattr(module, 'v_threshold'):
                threshold_changes.append(module.v_threshold)

        avg_threshold_change = sum(threshold_changes) / len(threshold_changes) if threshold_changes else 1.0

        # Determine fusion quality
        if max_difference_overall < tolerance / 10:
            fusion_quality = "excellent"
        elif max_difference_overall < tolerance:
            fusion_quality = "good"
        elif max_difference_overall < tolerance * 10:
            fusion_quality = "acceptable"
        else:
            fusion_quality = "poor"

        results = {
            'overall_assessment': {
                'functionally_equivalent': overall_functionally_equivalent,
                'max_difference': max_difference_overall,
                'fusion_quality': fusion_quality,
                'average_ifnode_threshold_change_ratio': avg_threshold_change
            },
            'time_step_analysis': {
                'max_differences_per_step': max_differences,
                'tolerance': tolerance
            }
        }

        print(f"  Analyzing IFNode threshold changes...")
        print(f"    Max difference across time steps: {max_difference_overall:.8f}")
        print(f"    Functionally equivalent (tol={tolerance:.3f}): {overall_functionally_equivalent}")
        print(f"    Fusion quality: {fusion_quality}")
        print(f"    Average IFNode threshold change ratio: {avg_threshold_change:.6f}")

        return results

    except Exception as e:
        print(f"  Fusion validation failed with error: {e}")
        return {
            'overall_assessment': {
                'functionally_equivalent': False,
                'max_difference': float('inf'),
                'fusion_quality': 'failed',
                'error': str(e)
            }
        }

def convert_to_activation_based_ifnodes(module):
    """
    Convert clock-driven IFNodes to activation-based IFNodes for HiAER compatibility
    
    Args:
        module: SNN module with clock-driven IFNodes
        
    Returns:
        module: Module with activation-based IFNodes
    """
    print("Converting to activation-based IFNodes...")
    
    from spikingjelly.clock_driven.neuron import IFNode as ClockDrivenIFNode
    from spikingjelly.activation_based.neuron import IFNode as ActivationBasedIFNode
    
    for name, child in module.named_children():
        if isinstance(child, ClockDrivenIFNode):
            new_neuron = ActivationBasedIFNode(
                v_threshold=child.v_threshold,
                v_reset=child.v_reset,
                surrogate_function=child.surrogate_function,
                detach_reset=child.detach_reset
            )
            setattr(module, name, new_neuron)
            print(f"Converted {name}: clock_driven.IFNode -> activation_based.IFNode")
    
    return module

def apply_16bit_quantization(model, save_path="dvs_quantized_snn_16bit.pth"):
    """
    Apply 16-bit quantization to the DVS SNN model
    
    Args:
        model: SNN model to quantize
        save_path: Path to save quantized model
        
    Returns:
        quantized_data: Quantization results and statistics (includes 'model' key)
    """
    print("\n" + "="*60)
    print("APPLYING 16-BIT QUANTIZATION TO DVS SNN")
    print("="*60)
    
    try:
        from quantize_snn import quantize_snn_model_16bit
        
        print("Quantizing DVS SNN model to 16-bit integers...")
        quantized_data = quantize_snn_model_16bit(
            model,
            symmetric=True,
            per_channel=False,
            save_path=save_path
        )
        
        print("DVS 16-bit quantization completed successfully!")
        print(f"Compression ratio: {quantized_data['quantization_info']['global_stats']['compression_ratio']:.2f}x")
        print(f"Original size: {quantized_data['quantization_info']['global_stats']['original_size_mb']:.2f} MB")
        print(f"Quantized size: {quantized_data['quantization_info']['global_stats']['quantized_size_mb']:.2f} MB")
        
        return quantized_data
        
    except ImportError:
        print("Quantization skipped: quantize_snn.py not available")
        return None
    except Exception as e:
        print(f"DVS quantization failed: {e}")
        return None

def save_dvs_models(hiaer_snn, final_snn, first_path, second_path):
    """
    Save all DVS SNN model variants
    
    Args:
        hiaer_snn: HiAER Spike ready SNN model
        final_snn: SNN model without output IFNode
    """
    # Save the main models
    # torch.save(hiaer_snn, first_path)
    torch.save(hiaer_snn, first_path)
    print(f"Final DVS SNN model saved to: {first_path}")

    # torch.save(final_snn, "dvs_snn_no_output_ifnode.pth")
    torch.save(final_snn, second_path)
    print(f"DVS SNN model (no output IFNode) saved to: {second_path}")

def print_conversion_summary(model_architecture, quantized_data=None):
    """
    Print summary of DVS ANN-to-SNN conversion
    
    Args:
        model_architecture: Architecture string
        quantized_data: Quantization results (optional)
    """
    print("\n" + "="*60)
    print("DVS ANN TO SNN CONVERSION COMPLETE")
    print("="*60)
    print(f"DVS model architecture: {model_architecture}")
    print(f"Input channels: 3 (DVS: OFF, ON, Static)")
    print(f"Environment: DVS-encoded Pong with frame_skip=1")
    print("Output: HiAER Spike ready SNN with 16-bit quantization")
    print("Files saved:")
    print("  - dvs_smaller_hiaer_spike_ready_snn.pth (main SNN model)")
    print("  - dvs_smaller_snn_no_output_ifnode.pth (without output IFNode)")
    if quantized_data:
        print("  - dvs_quantized_snn_16bit.pth (16-bit quantized)")
    print("="*60)

def convert_ifnodes_to_custom_lif(module: nn.Module, tau: float = 84.0, decay_input: bool = False) -> nn.Module:
    """
    Recursively replace IFNode layers with Custom_LIFNode layers aligned to hardware behavior.

    - Preserves threshold/reset/surrogate/detach_reset when available
    - Sets LIF parameters: tau and decay_input (default tau=84.0, decay_input=False)

    Args:
        module: Model to transform
        tau: Membrane time constant matching FPGA leak (e.g., 84.0)
        decay_input: Whether inputs decay in the LIF dynamics (False to match converter)

    Returns:
        The transformed module with IFNodes replaced by Custom_LIFNode.
    """
    from spikingjelly.clock_driven.neuron import IFNode as ClockDrivenIFNode
    from spikingjelly.activation_based.neuron import IFNode as ActivationBasedIFNode
    from spikingjelly.activation_based import surrogate as sj_surrogate
    try:
        from hs_api.custom_neurons import Custom_LIFNode
    except Exception:
        # Fallback to activation_based LIF if custom is not available
        from spikingjelly.activation_based.neuron import LIFNode as Custom_LIFNode  # type: ignore

    for name, child in module.named_children():
        replaced = False
        if isinstance(child, (ActivationBasedIFNode, ClockDrivenIFNode)):
            v_threshold = float(getattr(child, 'v_threshold', 1.0))
            v_reset = getattr(child, 'v_reset', 0.0)
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
            print(f"Converted {name}: IFNode -> Custom_LIFNode(tau={tau}, decay_input={decay_input}, v_th={v_threshold})")

        if not replaced:
            convert_ifnodes_to_custom_lif(child, tau=tau, decay_input=decay_input)

    return module
