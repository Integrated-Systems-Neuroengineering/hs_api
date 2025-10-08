"""Utilities for converting IFNodes to Custom_LIFNode, evaluating with flush steps, and optional calibration."""

from __future__ import annotations

import argparse
import os
import sys
import importlib.util
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spikingjelly.activation_based import functional
from spikingjelly.activation_based import neuron

# Paths for local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
for rel in ("hs_api", "fxpmath"):
    path = os.path.join(ROOT, rel)
    if path not in sys.path:
        sys.path.insert(0, path)

custom_neuron_path = os.path.join(ROOT, "hs_api", "hs_api", "custom_neurons.py")
if not os.path.isfile(custom_neuron_path):
    raise FileNotFoundError(f"Custom neuron definition not found: {custom_neuron_path}")
_spec = importlib.util.spec_from_file_location("hs_custom_neurons", custom_neuron_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load Custom_LIFNode from {custom_neuron_path}")
_custom_neurons = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_custom_neurons)
Custom_LIFNode = _custom_neurons.Custom_LIFNode

sys.path.insert(0, os.path.join(ROOT, "final_push", "pong_stuff"))

from hs_api.pong_model_pipeline.DVSWrapper import make_dvs_pong_env  # pylint: disable=wrong-import-position

from ann_to_snn_hard_reset_calibration import SNNCalibrator  # pylint: disable=wrong-import-position


def replace_ifnodes_with_custom_lif(
    model: nn.Sequential,
    tau: float = 63.0,
    decay_input: bool = False,
) -> Tuple[nn.Sequential, int]:
    """Clone a sequential SNN, swapping IFNodes for Custom_LIFNode."""
    converted = []
    replacements = 0

    for module in model:
        if isinstance(module, neuron.IFNode):
            surrogate_fn = getattr(module, "surrogate_function", None)
            lif = Custom_LIFNode(
                tau=tau,
                decay_input=decay_input,
                v_threshold=float(getattr(module, "v_threshold", 1.0)),
                v_reset=float(getattr(module, "v_reset", 0.0)) if module.v_reset is not None else 0.0,
                surrogate_function=surrogate_fn,
                detach_reset=bool(getattr(module, "detach_reset", True)),
                step_mode="s",
                backend="torch",
            )
            converted.append(lif)
            replacements += 1
        else:
            converted.append(module)

    return nn.Sequential(*converted), replacements


def evaluate_dvs_snn_custom_lif_with_flush(
    snn: nn.Module,
    env,
    device: torch.device,
    episodes: int = 5,
    time_steps: int = 18,
    flush_steps: int = 1,
) -> dict:
    """Evaluate a Custom_LIFNode SNN with optional post-input flush steps."""
    print(f"===== Custom LIF evaluation (T={time_steps}, flush={flush_steps}) =====")
    snn.to(device)
    snn.eval()
    rewards, lengths = [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        steps = 0

        while steps < 5000:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            functional.reset_net(snn)

            outputs = []
            with torch.no_grad():
                for _ in range(time_steps):
                    q = snn(obs_tensor)
                    outputs.append(q)

                if flush_steps > 0:
                    blank = torch.zeros_like(obs_tensor)
                    for _ in range(flush_steps):
                        q = snn(blank)
                        outputs.append(q)

            q_values = torch.stack(outputs).mean(dim=0)
            action = int(q_values.argmax(dim=1))

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

        rewards.append(ep_reward)
        lengths.append(steps)
        print(f"Episode {ep + 1}: reward={ep_reward:.1f}, steps={steps}")

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    std_reward = float(np.std(rewards)) if rewards else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0

    print("\nSummary:")
    print(f"  Reward avg: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Steps avg: {avg_length:.1f}")
    print(f"  All rewards: {rewards}")

    return {
        "average_reward": avg_reward,
        "std_reward": std_reward,
        "episode_rewards": rewards,
        "episode_lengths": lengths,
    }


def build_random_calibration_loader(
    env,
    samples: int,
    batch_size: int = 32,
) -> DataLoader:
    """Collect random observations from the environment and return a calibration DataLoader."""
    observations = []
    obs, _ = env.reset()
    observations.append(torch.as_tensor(obs, dtype=torch.float32))

    for _ in range(max(0, samples - 1)):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(torch.as_tensor(obs, dtype=torch.float32))
        if terminated or truncated:
            obs, _ = env.reset()

    data = torch.stack(observations)
    labels = torch.zeros(len(observations))  # Dummy labels
    effective_batch = max(1, min(batch_size, len(observations)))
    loader = DataLoader(
        TensorDataset(data, labels),
        batch_size=effective_batch,
        shuffle=True,
        drop_last=len(observations) > effective_batch,
    )
    print(f"Collected {len(observations)} calibration frames (batch={effective_batch})")
    return loader


def run_calibration(
    snn: nn.Module,
    env,
    device: torch.device,
    time_steps: int,
    samples: int,
    iterations: int,
) -> Optional[dict]:
    """Execute percentile-based calibration using SNNCalibrator."""
    if samples <= 0 or iterations <= 0:
        print("Calibration skipped: non-positive samples or iterations")
        return None

    loader = build_random_calibration_loader(env, samples=samples)
    calibrator = SNNCalibrator(
        snn_model=snn,
        ann_model=None,
        dataloader=loader,
        device=device,
        time_steps=time_steps,
    )

    print("\n===== Starting Custom LIF calibration =====")
    results = calibrator.calibrate(env=env, max_iterations=iterations)
    print("Calibration complete: ", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert IFNodes to Custom_LIF, evaluate, and calibrate with flush steps.")
    parser.add_argument("--model", default="dvs_84.pth", help="Path to IFNode-based SNN .pth file")
    parser.add_argument("--save", help="Optional path to save converted model")
    parser.add_argument("--tau", type=float, default=63.0, help="Custom_LIF tau constant")
    parser.add_argument("--decay-input", action="store_true", help="Use decay_input=True")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=18)
    parser.add_argument("--flush-steps", type=int, default=1)
    parser.add_argument("--calibrate", action="store_true", help="Run percentile calibration after evaluation")
    parser.add_argument("--calibration-samples", type=int, default=512, help="Number of observations for calibration data")
    parser.add_argument("--calibration-iterations", type=int, default=5, help="Max calibration iterations")
    parser.add_argument("--save-calibrated", help="Path to save calibrated model (after calibration)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model, map_location=device)
    if not isinstance(model, nn.Sequential):
        raise TypeError("Expected a torch.nn.Sequential model")

    converted, count = replace_ifnodes_with_custom_lif(
        model,
        tau=args.tau,
        decay_input=args.decay_input,
    )
    print(f"Replaced {count} IFNode layers with Custom_LIFNode")

    if args.save:
        torch.save(converted, args.save)
        print(f"Saved converted model to {args.save}")

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
    calibration_results = None
    try:
        evaluate_dvs_snn_custom_lif(
            converted,
            env,
            device=device,
            episodes=args.episodes,
            time_steps=args.time_steps,
            flush_steps=args.flush_steps,
        )

        if args.calibrate:
            env.reset()
            calibration_results = run_calibration(
                converted,
                env,
                device=device,
                time_steps=args.time_steps,
                samples=args.calibration_samples,
                iterations=args.calibration_iterations,
            )
            if calibration_results and args.save_calibrated:
                torch.save(converted, args.save_calibrated)
                print(f"Saved calibrated model to {args.save_calibrated}")
    finally:
        env.close()

    if calibration_results:
        print("\nCalibration summary:")
        for key, value in calibration_results.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
