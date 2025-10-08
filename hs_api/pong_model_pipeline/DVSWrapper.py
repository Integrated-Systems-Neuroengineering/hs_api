"""
DVS (Dynamic Vision Sensor) Wrapper for Pong Environment

This wrapper converts grayscale Pong observations into DVS-style temporal encoding
with 2 binary channels: OFF events, ON events.

Axon efficiency: 2×84×84 = 14,112 axons
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

class DVSPlusPongWrapper(gym.ObservationWrapper):
    """
    DVS-style temporal encoding wrapper for Pong environment.

    Converts grayscale observations into 2-channel binary encoding:
    - Channel 0: OFF events (brightness decreases)
    - Channel 1: ON events (brightness increases)  
    """
    
    def __init__(self, env, change_threshold=10):
        """
        Args:
            env: Base environment (should output 84x84 grayscale)
            change_threshold: Minimum pixel change to trigger change event
        """
        super().__init__(env)
        self.change_threshold = change_threshold
        self.prev_frame = None
        
        # Get original observation space for validation
        orig_shape = env.observation_space.shape
        if len(orig_shape) != 2:
            raise ValueError(f"Expected 2D grayscale input, got shape {orig_shape}")
        
        height, width = orig_shape
        
        # 3-channel output: [OFF_events, ON_events]
        self.observation_space = Box(
            low=0, high=1,
            shape=(2, height, width),
            dtype=np.float32
        )
        
        print(f"DVS Wrapper initialized:")
        print(f"  Input shape: {orig_shape}")
        print(f"  Output shape: {self.observation_space.shape}")
        print(f"  Change threshold: {change_threshold}")
        print(f"  Total axons: {np.prod(self.observation_space.shape)}")
    
    def observation(self, obs):
        """Convert grayscale observation to DVS 2-channel binary encoding"""

        if self.prev_frame is not None:
            # Calculate pixel differences
            diff = obs.astype(np.int16) - self.prev_frame.astype(np.int16)
            
            # Channel 0: OFF events (brightness decreases)
            off_events = (-diff > self.change_threshold).astype(np.float32)
            
            # Channel 1: ON events (brightness increases)
            on_events = (diff > self.change_threshold).astype(np.float32)
            
        else:
            # First frame: no change events
            height, width = obs.shape
            off_events = np.zeros((height, width), dtype=np.float32)
            on_events = np.zeros((height, width), dtype=np.float32)
        
        # Update frame history
        self.prev_frame = obs.copy()

        # Stack channels: shape becomes (2, height, width)
        dvs_obs = np.stack([off_events, on_events], axis=0)
        
        return dvs_obs
    
    def reset(self, **kwargs):
        """Reset environment and clear frame history"""
        self.prev_frame = None
        return super().reset(**kwargs)


class DVSVisualizationWrapper(gym.Wrapper):
    """
    Wrapper to visualize DVS channels for debugging.
    Saves sample frames periodically to understand encoding behavior.
    """
    
    def __init__(self, env, save_interval=100, save_dir="dvs_samples"):
        super().__init__(env)
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.step_count = 0
        
        # Create save directory
        import os
        os.makedirs(save_dir, exist_ok=True)
        print(f"DVS visualization will save samples to: {save_dir}")
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Save visualization samples
        if self.step_count % self.save_interval == 0:
            self._save_dvs_visualization(obs, self.step_count)
        
        self.step_count += 1
        return obs, reward, terminated, truncated, info
    
    def _save_dvs_visualization(self, obs, step):
        """Save visualization of DVS channels"""
        import matplotlib.pyplot as plt
        
        if len(obs.shape) != 3 or obs.shape[0] != 3:
            return  # Only visualize DVS observations
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Channel 0: OFF events (red)
        axes[0].imshow(obs[0], cmap='Reds', vmin=0, vmax=1)
        axes[0].set_title(f'OFF Events\n({obs[0].sum():.0f} pixels)')
        axes[0].axis('off')
        
        # Channel 1: ON events (green)  
        axes[1].imshow(obs[1], cmap='Greens', vmin=0, vmax=1)
        axes[1].set_title(f'ON Events\n({obs[1].sum():.0f} pixels)')
        axes[1].axis('off')
        
        # Combined view
        combined = np.stack([obs[0], obs[1], obs[2]], axis=2)
        axes[3].imshow(combined)
        axes[3].set_title('Combined\n(R=OFF, G=ON)')
        axes[3].axis('off')
        
        plt.suptitle(f'DVS Encoding - Step {step}')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/dvs_step_{step:06d}.png', dpi=100)
        plt.close()


def make_dvs_pong_env(config=None):
    """
    Create DVS-encoded Pong environment with standard CleanRL wrappers.
    
    Wrapper order:
    Raw RGB → AtariPreprocessing(grayscale) → ClipReward → DVSWrapper
    """
    if config is None:
        # Default configuration
        config = {
            'env': {
                'game': 'PongNoFrameskip-v4',
                'noop_max': 30,
                'frame_skip': 4,  # Standard frameskip for effective training
                'episodic_life': True,
                'clip_rewards': True,
                'grayscale': True
            },
            'dvs': {
                'change_threshold': 10,
                'visualization': False,
                'vis_interval': 500
            }
        }
    
    # Import wrappers
    from gymnasium.wrappers import ClipReward
    from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
    
    # Register ALE environments
    try:
        import ale_py
        gym.register_envs(ale_py)
        print("ALE environments registered for DVS")
    except ImportError:
        print("Warning: ale_py not found, ALE environments may not be available")
    
    # 1. Base environment  
    env_names = [
        config['env'].get('game', 'PongNoFrameskip-v4'),
        'PongNoFrameskip-v4',
        'PongNoFrameskip-v4', 
        'Pong-v5'
    ]
    
    env = None
    for env_name in env_names:
        try:
            env = gym.make(env_name, render_mode="rgb_array")
            print(f"Created environment: {env_name}")
            break
        except gym.error.Error:
            continue
    
    if env is None:
        raise RuntimeError(f"Could not create Pong environment. Tried: {env_names}")
    
    # 2. Standard Atari preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=config['env'].get('noop_max', 30),
        frame_skip=config['env'].get('frame_skip', 4),  # Standard frameskip=4 for effective training
        screen_size=84,  # Keep 84x84 for high resolution
        terminal_on_life_loss=config['env'].get('episodic_life', True),
        grayscale_obs=config['env'].get('grayscale', True),
        grayscale_newaxis=False,
        scale_obs=False  # We'll handle scaling in DVS wrapper
    )
    print("Applied AtariPreprocessing with grayscale conversion")
    
    # 3. Reward clipping
    if config['env'].get('clip_rewards', True):
        env = ClipReward(env, min_reward=-1, max_reward=1)
        print("Applied ClipReward wrapper")
    
    # 4. DVS conversion (main wrapper)
    dvs_config = config.get('dvs', {})
    env = DVSPlusPongWrapper(
        env,
        change_threshold=dvs_config.get('change_threshold', 10)
    )
    print("Applied DVS encoding wrapper")
    
    # 5. Optional visualization wrapper
    if dvs_config.get('visualization', False):
        env = DVSVisualizationWrapper(
            env,
            save_interval=dvs_config.get('vis_interval', 500),
            save_dir="dvs_visualization"
        )
        print("Applied DVS visualization wrapper")
    
    print(f"Final observation space: {env.observation_space}")
    return env


if __name__ == "__main__":
    """Test the DVS wrapper"""
    print("Testing DVS Pong Environment...")
    
    # Test configuration with visualization
    test_config = {
        'env': {
            'game': 'PongNoFrameskip-v4',
            'noop_max': 5,
            'frame_skip': 1,
            'episodic_life': False,  # Disable for testing
            'clip_rewards': True,
            'grayscale': True
        },
        'dvs': {
            'change_threshold': 10,
            'visualization': True,
            'vis_interval': 50  # Save every 50 steps for testing
        }
    }
    
    env = make_dvs_pong_env(test_config)
    
    # Test reset
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    print(f"Initial channel sums: OFF={obs[0].sum()}, ON={obs[1].sum()}")
    
    # Test several steps
    total_steps = 200
    for i in range(total_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 50 == 0:
            print(f"Step {i}: OFF={obs[0].sum():.0f}, ON={obs[1].sum():.0f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            obs, info = env.reset()
    
    env.close()
    print("DVS wrapper test completed successfully!")