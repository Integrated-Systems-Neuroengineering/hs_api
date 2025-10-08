#!/usr/bin/env python3
"""
DVS-encoded (No Static) Step-based Nature CNN DQN trainer
Adaptation of DVS_step_based_trainer.py to use DVS encoding without static channel
Uses 2-channel binary encoding: OFF events, ON events
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
import argparse
import os
import time
from datetime import datetime
import logging
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register ALE environments
import gymnasium as gym
try:
    import ale_py
    gym.register_envs(ale_py)
    print("ALE environments registered successfully")
except ImportError:
    print("Warning: ale_py not found, ALE environments may not be available")

from relu_nature_cnn_2ch_sj_compatible import ReLUNatureCNN2ChSJCompatible
from utils import ReplayBuffer, save_checkpoint
from hs_api.pong_model_pipeline.DVSWrapper import make_dvs_pong_env

class DVSNoStaticTrainer:
    """DVS-encoded (No Static) Step-based Nature CNN DQN trainer"""
    
    def __init__(self, config_path, device='cuda', training_mode='hybrid'):
        """
        Initialize DVS trainer without static channel
        
        Args:
            config_path: Path to configuration file
            device: Device to use ('cuda' or 'cpu')
            training_mode: Training mode ('fast', 'diagnostic', 'hybrid')
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.training_mode = training_mode
        
        # DVS No Static uses 2 input channels: [OFF_events, ON_events]
        input_channels = 2  # Fixed for DVS encoding without static
        n_actions = self.config['model'].get('n_actions', 6)
        architecture = self.config['model'].get('architecture', 'silu_nature_cnn_2ch')
        
        print(f"DVS No Static Trainer: Using {input_channels} input channels (OFF and ON events only)")
        
        # Initialize model - choose architecture
        if architecture == 'nature_cnn':
            self.q_network = NatureCNNDQN(input_channels, n_actions).to(self.device)
            self.target_network = NatureCNNDQN(input_channels, n_actions).to(self.device)
        elif architecture == 'leaky_nature_cnn':
            negative_slope = self.config['model'].get('leaky_relu_slope', 0.01)
            self.q_network = LeakyNatureCNNDQN(input_channels, n_actions, negative_slope).to(self.device)
            self.target_network = LeakyNatureCNNDQN(input_channels, n_actions, negative_slope).to(self.device)
        elif architecture == 'silu_nature_cnn':
            self.q_network = SiLUNatureCNNDQN(input_channels, n_actions).to(self.device)
            self.target_network = SiLUNatureCNNDQN(input_channels, n_actions).to(self.device)
        elif architecture == 'silu_nature_cnn_2ch':
            self.q_network = SiLUNatureCNN2Ch(input_channels, n_actions).to(self.device)
            self.target_network = SiLUNatureCNN2Ch(input_channels, n_actions).to(self.device)
        elif architecture == 'relu_nature_cnn_2ch_sj_compatible':
            self.q_network = ReLUNatureCNN2ChSJCompatible(input_channels, n_actions).to(self.device)
            self.target_network = ReLUNatureCNN2ChSJCompatible(input_channels, n_actions).to(self.device)
        elif architecture == 'binary_nature_cnn':
            self.q_network = BinaryNatureCNNDQN(input_channels, n_actions).to(self.device)
            self.target_network = BinaryNatureCNNDQN(input_channels, n_actions).to(self.device)
        elif architecture == 'calibrated_binary_nature_cnn':
            learnable_thresholds = self.config['model'].get('learnable_thresholds', True)
            self.q_network = CalibratedBinaryNatureCNNDQN(input_channels, n_actions, learnable_thresholds).to(self.device)
            self.target_network = CalibratedBinaryNatureCNNDQN(input_channels, n_actions, learnable_thresholds).to(self.device)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Store architecture for diagnostics
        self.architecture = architecture
        
        # Optimizer
        optimizer_type = self.config['training'].get('optimizer', 'adam').lower()
        lr = self.config['training']['lr']
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            momentum = self.config['training'].get('momentum', 0.9)
            self.optimizer = optim.SGD(self.q_network.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Step-based training parameters (CleanRL-style)
        self.total_steps = self.config['training'].get('total_steps', 5000000)
        self.learning_starts = self.config['training'].get('learning_starts', 40000)
        self.train_frequency = self.config['training'].get('train_frequency', 4)
        self.target_update_frequency = self.config['training'].get('target_update_frequency', 1000)
        
        # Diagnostic intervals based on training mode
        if training_mode == 'fast':
            self.log_interval = self.config['training'].get('log_interval', 10000)
            self.diagnostic_interval = self.total_steps + 1  # Never run diagnostics
        elif training_mode == 'diagnostic':
            self.log_interval = self.config['training'].get('log_interval', 1000)
            self.diagnostic_interval = self.config['training'].get('diagnostic_interval', 5000)
        else:  # hybrid
            self.log_interval = self.config['training'].get('log_interval', 5000)
            self.diagnostic_interval = self.config['training'].get('diagnostic_interval', 50000)
        
        # Replay buffer
        buffer_size = self.config['training'].get('buffer_size', 500000)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # DVS Environment with configuration (No Static channel)
        dvs_config = {
            'env': self.config.get('env', {}),
            'dvs': self.config.get('dvs', {
                'change_threshold': 10,
                'visualization': False,
                'vis_interval': 1000
            })
        }
        
        self.env = make_dvs_pong_env(dvs_config)
        print(f"DVS No Static Environment created with observation space: {self.env.observation_space}")
        print(f"Total axons: {np.prod(self.env.observation_space.shape)} (33% reduction from 3-channel)")
        
        # Epsilon schedule (linear)
        self.start_epsilon = self.config['training'].get('epsilon_start', 1.0)
        self.end_epsilon = self.config['training'].get('epsilon_end', 0.01)
        self.exploration_fraction = self.config['training'].get('exploration_fraction', 0.1)
        
        # Logging
        self.setup_logging()
        
        # Diagnostics (only if not in fast mode)
        if training_mode != 'fast':
            self.dead_relu_threshold = 0.01
            self.activation_stats = {}
        
        # Metrics tracking
        self.metrics = {
            'steps': 0,
            'episodes': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'q_values': [],
            'epsilons': [],
            'gradient_norms': []
        }
        
        print(f"\nDVS No Static Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Architecture: {architecture}")
        print(f"  Input channels: {input_channels} (OFF, ON)")
        print(f"  Actions: {n_actions}")
        print(f"  Training mode: {training_mode}")
        print(f"  Total steps: {self.total_steps:,}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config.get('save', {}).get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'dvs_no_static_training_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DVS No Static training started with config: {self.config}")
    
    def get_epsilon(self, step):
        """Get current epsilon for exploration"""
        exploration_steps = int(self.exploration_fraction * self.total_steps)
        if step < exploration_steps:
            epsilon = self.start_epsilon - (self.start_epsilon - self.end_epsilon) * (step / exploration_steps)
        else:
            epsilon = self.end_epsilon
        return epsilon
    
    def select_action(self, state, epsilon):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self, batch_size=32):
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor([float(d) for d in dones]).to(self.device)
        
        # Calculate current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calculate target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.config['training']['gamma'] * next_q_values * (1 - dones)
        
        # Calculate loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Calculate gradient norm for diagnostics
        total_norm = 0
        for p in self.q_network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.optimizer.step()
        
        return loss.item(), current_q_values.mean().item(), total_norm
    
    def run_diagnostics(self, step):
        """Run comprehensive diagnostics on the model"""
        if self.training_mode == 'fast':
            return
        
        self.logger.info(f"\n=== DVS No Static Diagnostics at step {step} ===")
        
        # Sample a batch for analysis
        if len(self.replay_buffer) < 32:
            return
        
        states, _, _, _, _ = self.replay_buffer.sample(32)
        states = torch.FloatTensor(states).to(self.device)
        
        # Analyze DVS encoding statistics
        off_events = states[:, 0, :, :]  # Channel 0: OFF events
        on_events = states[:, 1, :, :]   # Channel 1: ON events
        
        self.logger.info(f"DVS Encoding Statistics:")
        self.logger.info(f"  OFF events: mean={off_events.mean():.3f}, active={100*(off_events>0).float().mean():.1f}%")
        self.logger.info(f"  ON events: mean={on_events.mean():.3f}, active={100*(on_events>0).float().mean():.1f}%")
        
        # Model-specific diagnostics
        if 'relu' in self.architecture.lower() or 'leaky' in self.architecture.lower():
            self.diagnose_relu_death(states)
        elif 'silu' in self.architecture.lower():
            self.diagnose_silu_health(states)
        elif 'binary' in self.architecture.lower():
            self.diagnose_binary_activations(states)
    
    def diagnose_relu_death(self, sample_batch):
        """Diagnose ReLU death in the network"""
        self.q_network.eval()
        
        with torch.no_grad():
            x = sample_batch
            
            # Conv layers
            x = self.q_network.conv1(x)
            pre_relu1 = x.clone()
            x = F.relu(x) if 'leaky' not in self.architecture else F.leaky_relu(x, 0.01)
            dead_conv1 = (x == 0).float().mean().item()
            
            x = self.q_network.conv2(x)
            x = F.relu(x) if 'leaky' not in self.architecture else F.leaky_relu(x, 0.01)
            dead_conv2 = (x == 0).float().mean().item()
            
            x = self.q_network.conv3(x)
            x = F.relu(x) if 'leaky' not in self.architecture else F.leaky_relu(x, 0.01)
            dead_conv3 = (x == 0).float().mean().item()
            
            # FC layers
            x = x.view(x.size(0), -1)
            x = self.q_network.fc1(x)
            x = F.relu(x) if 'leaky' not in self.architecture else F.leaky_relu(x, 0.01)
            dead_fc1 = (x == 0).float().mean().item()
        
        self.logger.info(f"ReLU Death Analysis:")
        self.logger.info(f"  Conv1: {dead_conv1*100:.1f}% dead")
        self.logger.info(f"  Conv2: {dead_conv2*100:.1f}% dead")
        self.logger.info(f"  Conv3: {dead_conv3*100:.1f}% dead")
        self.logger.info(f"  FC1: {dead_fc1*100:.1f}% dead")
        
        self.q_network.train()
    
    def diagnose_silu_health(self, sample_batch):
        """Diagnose SiLU activation health"""
        self.q_network.eval()
        
        with torch.no_grad():
            x = sample_batch
            
            # Conv layers with SiLU
            x = F.silu(self.q_network.conv1(x))
            silu1_mean = x.mean().item()
            silu1_std = x.std().item()
            
            x = F.silu(self.q_network.conv2(x))
            silu2_mean = x.mean().item()
            silu2_std = x.std().item()
            
            x = F.silu(self.q_network.conv3(x))
            silu3_mean = x.mean().item()
            silu3_std = x.std().item()
            
            # FC layer
            x = x.view(x.size(0), -1)
            x = F.silu(self.q_network.fc1(x))
            fc1_mean = x.mean().item()
            fc1_std = x.std().item()
        
        self.logger.info(f"SiLU Activation Health:")
        self.logger.info(f"  Conv1: mean={silu1_mean:.3f}, std={silu1_std:.3f}")
        self.logger.info(f"  Conv2: mean={silu2_mean:.3f}, std={silu2_std:.3f}")
        self.logger.info(f"  Conv3: mean={silu3_mean:.3f}, std={silu3_std:.3f}")
        self.logger.info(f"  FC1: mean={fc1_mean:.3f}, std={fc1_std:.3f}")
        
        self.q_network.train()
    
    def diagnose_binary_activations(self, sample_batch):
        """Diagnose binary activation patterns"""
        self.logger.info("Binary activation diagnostics not yet implemented for 2-channel DVS")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting DVS No Static training...")
        
        # Initialize environment
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_start_time = time.time()
        
        for step in range(self.total_steps):
            # Select action
            epsilon = self.get_epsilon(step)
            action = self.select_action(obs, epsilon)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.push(obs, action, reward, next_obs, done)
            
            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Training
            if step >= self.learning_starts and step % self.train_frequency == 0:
                batch_size = self.config['training'].get('batch_size', 32)
                result = self.train_step(batch_size)
                
                if result is not None:
                    loss, q_value, grad_norm = result
                    self.metrics['losses'].append(loss)
                    self.metrics['q_values'].append(q_value)
                    self.metrics['gradient_norms'].append(grad_norm)
            
            # Update target network
            if step % self.target_update_frequency == 0 and step > 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                
            # Episode end
            if done:
                self.metrics['episodes'] += 1
                self.metrics['episode_rewards'].append(episode_reward)
                self.metrics['episode_lengths'].append(episode_length)
                
                # Log episode stats
                if self.metrics['episodes'] % 10 == 0:
                    recent_rewards = self.metrics['episode_rewards'][-100:]
                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    
                    self.logger.info(
                        f"Episode {self.metrics['episodes']}, "
                        f"Step {step}, "
                        f"Reward: {episode_reward:.1f}, "
                        f"Avg(100): {avg_reward:.1f}, "
                        f"Epsilon: {epsilon:.3f}"
                    )
                
                # Reset episode
                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_start_time = time.time()
            
            # Logging
            if step % self.log_interval == 0 and step > 0:
                self.log_metrics(step)
            
            # Diagnostics
            if step % self.diagnostic_interval == 0 and step > 0:
                self.run_diagnostics(step)
            
            # Save checkpoint
            if step % self.config['save'].get('checkpoint_interval', 100000) == 0 and step > 0:
                self.save_checkpoint(step)
            
            self.metrics['steps'] = step
        
        # Final save
        self.save_checkpoint(self.total_steps)
        self.logger.info("DVS No Static training completed!")
    
    def log_metrics(self, step):
        """Log training metrics"""
        recent_rewards = self.metrics['episode_rewards'][-100:]
        recent_lengths = self.metrics['episode_lengths'][-100:]
        recent_losses = self.metrics['losses'][-1000:] if self.metrics['losses'] else [0]
        recent_q_values = self.metrics['q_values'][-1000:] if self.metrics['q_values'] else [0]
        recent_grad_norms = self.metrics['gradient_norms'][-1000:] if self.metrics['gradient_norms'] else [0]
        
        self.logger.info(f"\n=== DVS No Static Training Metrics at Step {step} ===")
        self.logger.info(f"Episodes: {self.metrics['episodes']}")
        self.logger.info(f"Avg Reward (100 ep): {np.mean(recent_rewards) if recent_rewards else 0:.2f}")
        self.logger.info(f"Avg Length (100 ep): {np.mean(recent_lengths) if recent_lengths else 0:.1f}")
        self.logger.info(f"Avg Loss: {np.mean(recent_losses):.4f}")
        self.logger.info(f"Avg Q-Value: {np.mean(recent_q_values):.2f}")
        self.logger.info(f"Avg Gradient Norm: {np.mean(recent_grad_norms):.4f}")
        self.logger.info(f"Buffer Size: {len(self.replay_buffer)}")
        self.logger.info(f"Epsilon: {self.get_epsilon(step):.3f}")
    
    def save_checkpoint(self, step):
        """Save training checkpoint"""
        checkpoint_dir = os.path.join(self.config['save'].get('log_dir', 'logs'), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'dvs_no_static_step_{step}.pth')
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save as best model if it has highest average reward
        recent_rewards = self.metrics['episode_rewards'][-100:]
        if recent_rewards:
            avg_reward = np.mean(recent_rewards)
            best_path = self.config['save'].get('best_model_path', 'checkpoints/dvs_no_static_best.pth')
            
            # Check if this is the best model so far
            if not hasattr(self, 'best_avg_reward') or avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save(checkpoint, best_path)
                self.logger.info(f"New best model saved with avg reward: {avg_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description='DVS No Static DQN Training')
    parser.add_argument('--config', type=str, default='configs/step_based_pong_no_static.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--mode', type=str, default='hybrid',
                        choices=['fast', 'diagnostic', 'hybrid'],
                        help='Training mode')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = DVSNoStaticTrainer(args.config, args.device, args.mode)
    trainer.train()


if __name__ == "__main__":
    main()