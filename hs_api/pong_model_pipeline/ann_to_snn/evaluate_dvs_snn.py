#!/usr/bin/env python3
"""
DVS-specific SNN evaluation with proper environment handling
"""

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import functional as functional
from spikingjelly.clock_driven import functional as sf_func

def evaluate_dvs_snn(snn, ann_model, env, device, episodes=5, time_steps=18):
    """
    Evaluate DVS SNN model with proper rate coding and environment handling
    
    Args:
        snn: The spiking neural network model
        ann_model: Original ANN model (for compatibility)
        env: DVS environment
        device: torch device
        episodes: Number of episodes to evaluate
        time_steps: SNN time steps for rate coding
        
    Returns:
        dict: Evaluation results
    """
    print(f"=====EVALUATING DVS SNN ({time_steps} time steps)======")
    
    snn.eval()
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        print(f"DVS SNN Episode {episode + 1}:")
        
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        # Debug first episode
        if episode == 0:
            print(f"  Initial obs shape: {obs.shape}")
            print(f"  Initial obs range: [{obs.min():.3f}, {obs.max():.3f}]")
            if obs.shape[0] == 2:  # 2-channel DVS (OFF, ON)
                print(f"  Channel sums: OFF={obs[0].sum():.0f}, ON={obs[1].sum():.0f}")
            elif obs.shape[0] == 3:  # 3-channel DVS (OFF, ON, Static)
                print(f"  Channel sums: OFF={obs[0].sum():.0f}, ON={obs[1].sum():.0f}, Static={obs[2].sum():.0f}")
        
        while steps < 5000:  # Max steps per episode
            # Convert DVS observation to tensor
            # DVS obs is already in [0,1] range - no normalization needed
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Reset SNN state for each frame
            try:
                functional.reset_net(snn)  # activation_based
            except:
                sf_func.reset_net(snn)     # clock_driven fallback
            
            # Rate coding: accumulate SNN outputs over multiple time steps
            output_sum = torch.zeros(1, 6, device=device)  # 6 actions for Pong
            
            with torch.no_grad():
                for t in range(time_steps):
                    # Forward pass through SNN
                    snn_output = snn(obs_tensor)
                    output_sum += snn_output
            
            # Compute rate-coded Q-values (average over time steps)
            q_values = output_sum / time_steps
            action = q_values.argmax(dim=1).item()
            
            # Debug first few actions
            if episode == 0 and steps < 3:
                print(f"  Step {steps}: Q-values={q_values[0].cpu().numpy()}, action={action}")
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"  DVS SNN Episode {episode + 1} result: {episode_reward:.1f} reward, {steps} steps")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\nDVS SNN Evaluation Results:")
    print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Average length: {avg_length:.1f}")
    print(f"  Individual rewards: {episode_rewards}")
    
    return {
        'average_reward': avg_reward,
        'std_reward': std_reward,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

def compare_ann_vs_snn(ann_model, snn_model, env, device, episodes=3):
    """
    Direct comparison between ANN and SNN performance on same environment
    """
    print("="*60)
    print("DVS ANN vs SNN COMPARISON")
    print("="*60)
    
    # Evaluate ANN
    print("Evaluating ANN...")
    ann_rewards = []
    
    ann_model.eval()
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 5000:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = ann_model(obs_tensor)
                action = q_values.argmax().item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        ann_rewards.append(episode_reward)
        print(f"  ANN Episode {episode + 1}: {episode_reward:.1f} reward, {steps} steps")
    
    ann_avg = np.mean(ann_rewards)
    
    # Evaluate SNN with different time steps
    time_steps_list = [8, 12, 16, 20, 25]
    
    print(f"\nEvaluating SNN with different time steps...")
    
    best_snn_avg = -float('inf')
    best_time_steps = 0
    
    for time_steps in time_steps_list:
        print(f"\n--- Testing {time_steps} time steps ---")
        snn_results = evaluate_dvs_snn(snn_model, ann_model, env, device, 
                                       episodes=episodes, time_steps=time_steps)
        snn_avg = snn_results['average_reward']
        
        if snn_avg > best_snn_avg:
            best_snn_avg = snn_avg
            best_time_steps = time_steps
        
        print(f"SNN ({time_steps} steps): {snn_avg:.2f} avg reward")
    
    print(f"\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"ANN average reward: {ann_avg:.2f}")
    print(f"Best SNN average reward: {best_snn_avg:.2f} (at {best_time_steps} time steps)")
    print(f"Performance retention: {(best_snn_avg/ann_avg)*100:.1f}%")
    
    if best_snn_avg < ann_avg * 0.8:  # Less than 80% performance
        print("WARNING: Significant SNN performance drop detected!")
        print("Possible causes:")
        print("1. Rate coding time steps too low/high")
        print("2. Input preprocessing mismatch")
        print("3. SNN conversion issues")
        print("4. Environment state differences")
    
    return {
        'ann_avg': ann_avg,
        'ann_rewards': ann_rewards,
        'best_snn_avg': best_snn_avg,
        'best_time_steps': best_time_steps
    }


def evaluate_dvs_snn_custom_lif(
    snn,
    env,
    device,
    episodes=5,
    time_steps=20,
    flush_steps=0,
    assert_hard_reset=True,
    tau=63.0,
    decay_input=False,
    log_stats=False,
):
    """
    Evaluate a DVS SNN composed of Custom_LIFNode layers with hardware-aligned settings.

    - Resets membrane state before each environment step
    - Runs time_steps rate-coding steps per obs, then optional flush_steps of blanks
    - Optionally asserts LIF nodes are configured for hard reset and expected tau/decay

    Args:
        snn: SNN model (Sequential recommended)
        env: DVS environment
        device: torch device
        episodes: number of episodes
        time_steps: steps per observation
        flush_steps: extra blank steps after inputs to flush activity
        assert_hard_reset: if True, check Custom_LIFNode.v_reset == 0.0
        tau: expected tau for checks/logging
        decay_input: expected decay_input flag for checks/logging
        log_stats: if True, prints per-episode debug

    Returns:
        dict: evaluation statistics
    """
    print(f"=====EVALUATING DVS SNN (Custom LIF, T={time_steps}, flush={flush_steps})=====")

    # Attempt to import Custom_LIFNode for checks
    CustomLIF = None
    try:
        from hs_api.Krish_custom_neurons import Custom_LIFNode as CustomLIF
    except Exception:
        pass

    if CustomLIF is not None and assert_hard_reset:
        # Check LIF configuration
        for m in snn.modules():
            if isinstance(m, CustomLIF):
                if m.v_reset is None or float(m.v_reset) != 0.0:
                    print("WARNING: Custom_LIFNode v_reset is not 0.0 (hard reset). Current:", m.v_reset)
                if abs(float(getattr(m, 'tau', tau)) - tau) > 1e-3:
                    print("WARNING: Custom_LIFNode tau != expected:", getattr(m, 'tau', None))
                if bool(getattr(m, 'decay_input', decay_input)) != bool(decay_input):
                    print("WARNING: Custom_LIFNode decay_input mismatch:", getattr(m, 'decay_input', None))

    # Log geometric leak compensation info
    a = 1.0 - 1.0 / float(tau)
    leak_sum = sum(a**k for k in range(int(time_steps)))
    comp = (time_steps / leak_sum) if leak_sum > 0 else 1.0
    print(f"Leak geom sum={leak_sum:.4f}, suggested gain compensation≈{comp:.3f}")

    snn.eval()
    episode_rewards, episode_lengths = [], []
    
    for ep in range(episodes):
        obs, info = env.reset()
        ep_rew, steps = 0.0, 0

        if log_stats:
            print(f"Episode {ep+1}: obs shape={obs.shape}, range=[{obs.min():.3f},{obs.max():.3f}]")

        while steps < 5000:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Reset membrane state for this step
            try:
                functional.reset_net(snn)
            except Exception:
                sf_func.reset_net(snn)

            output_sum = torch.zeros(1, 6, device=device)
            with torch.no_grad():
                for t in range(time_steps):
                    q = snn(obs_tensor)
                    output_sum += q

                # Optional flush to propagate through depth
                if flush_steps and flush_steps > 0:
                    blank = torch.zeros_like(obs_tensor)
                    for _ in range(flush_steps):
                        _ = snn(blank)

            q_values = output_sum / time_steps
            action = q_values.argmax(dim=1).item()

            if log_stats and steps < 3:
                print(f"  Step {steps}: q={q_values[0].detach().cpu().numpy()}, a={action}")

            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            steps += 1
            if terminated or truncated:
                break

        episode_rewards.append(ep_rew)
        episode_lengths.append(steps)
        if log_stats:
            print(f"  Episode {ep+1} result: reward={ep_rew:.1f}, steps={steps}")

    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

    print("\nCustom LIF DVS SNN Evaluation:")
    print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Average length: {avg_length:.1f}")
    print(f"  Rewards: {episode_rewards}")

    return {
        'average_reward': avg_reward,
        'std_reward': std_reward,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'leak_compensation': comp,
    }
