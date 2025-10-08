"""
DVS ANN-to-SNN Conversion Pipeline
Converts trained DVS models to neuromorphic hardware-ready SNNs with 16-bit quantization
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import ann2snn
from evaluate_dvs_snn import evaluate_dvs_snn, compare_ann_vs_snn
from ann_to_snn_dvs_utils import (
    create_dvs_environment_and_dataloader,
    graphmodule_to_sequential,
    fuse_and_remove_voltage_scalers,
    convert_to_activation_based_ifnodes,
    apply_16bit_quantization,
    save_dvs_models,
    print_conversion_summary
)

def main():
    """Main DVS ANN-to-SNN conversion pipeline"""
    
    # Configuration
    # dvs_model_path = "../ann_training/checkpoints/dvs_63_no_bias.pth"
    # dvs_model_path = "../ann_training/trained_dvs_ann_weights.pth"
    dvs_model_path = "../ann_training/dvs_84_no_bias_best.pth"
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

    ann_avg_reward = evaluate_ann_simple(ann_model, env, device, episodes=1)

    if ann_avg_reward < -15:
        print(f"WARNING: ANN performance is poor ({ann_avg_reward:.2f})! Expected ~20.4")
        print("This suggests an issue with model loading or environment setup")
        print("\nPossible causes:")
        print("1. Wrong checkpoint file or loading issue")
        print("2. Environment setup differs from training")
        print("3. Model architecture mismatch") 
        print("4. DVS thresholds or frame_skip mismatch")
        print("\nContinuing conversion but SNN performance will likely be poor too...")
    
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
    evaluate_dvs_snn(graph_snn, ann_model, env, device, episodes=1, time_steps=18)

    # =================== Step 3: Convert GraphModule to Sequential =========================================
    print("\n" + "="*60)
    print("STEP 3: GRAPH-TO-SEQUENTIAL CONVERSION")
    print("="*60)
    
    flat_dvs_snn = graphmodule_to_sequential(graph_snn)
    print("DVS Sequential SNN structure:")
    print(flat_dvs_snn)
    print("Evaluating Sequential SNN")
    evaluate_dvs_snn(flat_dvs_snn, ann_model, env, device, episodes=1, time_steps=18)

    # =================== Step 4: Fuse VoltageScalers for HiAER Spike compatibility ==========================
    print("\n" + "="*60)
    print("STEP 4: VOLTAGE SCALER FUSION")
    print("="*60)
    
    fused_dvs_snn = fuse_and_remove_voltage_scalers(flat_dvs_snn)
    print("DVS SNN fused (VoltageScalers removed):")
    print(fused_dvs_snn)
    print("Evaluating fused DVS SNN")
    evaluate_dvs_snn(fused_dvs_snn, ann_model, env, device, episodes=1, time_steps=18)
    
    # =================== Step 5: Convert existing IFNodes to activation-based ===============================
    print("\n" + "="*60)
    print("STEP 5: CONVERT IFNODES TO ACTIVATION-BASED")
    print("="*60)
    pre_quantized_snn = convert_to_activation_based_ifnodes(fused_dvs_snn)
    print(pre_quantized_snn)
    print("Evaluate activation-based IFNode SNN")
    evaluate_dvs_snn(pre_quantized_snn, ann_model, env, device, episodes=1, time_steps=18)

    # =================== Step 6: Apply 16-bit quantization ================================================
    print("\n" + "="*60)
    print("STEP 6: 16-BIT QUANTIZATION")
    print("="*60)
    
    quantized_info = apply_16bit_quantization(
        pre_quantized_snn,
        save_path="dvs_84_no_bias_snn_16bit.pth"
    )
    
    # Handle quantization results
    if quantized_info and 'model' in quantized_info:
        quantized_snn = quantized_info['model']
        print("Evaluating 16-bit quantized SNN")
        # try:
        #     evaluate_dvs_snn(quantized_snn, ann_model, env, device, episodes=1, time_steps=18)
        # except Exception as e:
        #     print(f"Quantized SNN evaluation failed: {e}")
        #     quantized_snn = pre_quantized_snn  # Fallback to pre-quantized
    else:
        print("Quantization failed or unavailable, using pre-quantized SNN")
        quantized_snn = pre_quantized_snn

    # Step 7: Save all model variants
    print("\n" + "="*60)
    print("STEP 7: SAVING MODELS")
    print("="*60)

    save_dvs_models(pre_quantized_snn, quantized_snn, "dvs_84_no_bias_snn.pth", "quantized_dvs_84_no_bias_snn.pth")

    # Step 8: Final ANN vs SNN comparison with optimized parameters
    print("\n" + "="*60)
    print("STEP 8: FINAL ANN VS SNN COMPARISON")
    print("="*60)
    
    comparison_results = compare_ann_vs_snn(ann_model, pre_quantized_snn, env, device, episodes=3)
    
    # Step 9: Print conversion summary
    print_conversion_summary(model_architecture, quantized_info)
    
    # Cleanup
    env.close()
    print("DVS environment closed. Conversion complete!")

if __name__ == "__main__":
    main()