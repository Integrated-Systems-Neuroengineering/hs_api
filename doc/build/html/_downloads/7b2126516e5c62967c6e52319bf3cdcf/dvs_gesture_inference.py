"""
DVS128 Gesture Inference on CRI Hardware
=========================================

This example demonstrates running inference on the DVS128 Gesture dataset
using a pre-converted spiking convolutional network (stride-2, 100 channels,
3-layer conv) deployed on CRI neuromorphic hardware.

The network was trained with SpikingJelly and converted to the CRI format
using ``hs_api``. Input frames are resized to 63×63 and binarized before
being fed to the hardware timestep-by-timestep.
"""

# %%
# Importing the necessary libraries
# ----------------------------------
# We need PyTorch and SpikingJelly for data loading and preprocessing,
# ``hs_api`` for the CRI network interface, and ``hs_bridge`` to reset
# membrane potentials between samples on the FPGA.

import os
import pickle
import urllib.request
import torch
import hs_bridge
from hs_api.api import CRI_network
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from spikingjelly.datasets import pad_sequence_collate
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %%
# Configuration
# -------------
# Set the path to the DVS128 Gesture dataset and select the compute device.
# The model configuration is downloaded from Dropbox if not already cached.

data_dir = "/home/ckdeng/myprojects/DVS_Gesture"
model_config_path = "DVS_model_config.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG_URL = (
    "https://www.dropbox.com/scl/fi/dz7lfg4hs2mjh3vw0jec8/"
    "DVS_model_config.pkl?rlkey=5scc386le9356arxxy3hkrm6e&st=ysecv0s3&dl=1"
)

if not os.path.exists(model_config_path):
    print("Downloading model configuration...")
    urllib.request.urlretrieve(MODEL_CONFIG_URL, model_config_path)
    print(f"  Saved to {model_config_path}")
else:
    print(f"  {model_config_path} already cached.")

# %%
# Defining the preprocessing transform
# --------------------------------------
# DVS128 Gesture frames are 128×128. The network expects 63×63 binary inputs,
# so we resize each frame with bilinear interpolation and then binarize
# (any non-zero value becomes 1).


class DVSResizeAndBinarize:
    """Resize and binarize DVS event frames along the temporal dimension."""

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        frames, label = data if isinstance(data, tuple) else (data, None)
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        T, C, H, W = frames.shape

        resized = torch.zeros(
            (T, C, self.size[0], self.size[1]),
            dtype=frames.dtype,
            device=frames.device,
        )
        for t in range(T):
            frame = frames[t]  # [C, H, W]
            resized_frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
            ).squeeze(0)
            resized[t] = (resized_frame > 0).float()
        return (resized, label) if label is not None else resized


resize_transform = DVSResizeAndBinarize(size=(63, 63))

# %%
# Loading the dataset
# --------------------
# We use the SpikingJelly ``DVS128Gesture`` dataset loader, splitting each
# recording into 10 frames of equal spike-count (``split_by="number"``).
# A custom collate function handles variable-length sequences.

test_set = DVS128Gesture(
    root=data_dir,
    frames_number=10,
    split_by="number",
    train=False,
    data_type="frame",
    duration=1600000,
    transform=resize_transform,
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    collate_fn=pad_sequence_collate,
)

print(f"Test samples: {len(test_set)}")

# %%
# Loading the model configuration
# --------------------------------
# The CRI network topology (axons, connections, output neuron IDs) was
# serialised to disk during the ``hs_api`` conversion step and is loaded here.

print("Loading model configuration...")
with open(model_config_path, "rb") as f:
    model_config = pickle.load(f)

axons = model_config["axons"]
connections = model_config["connections"]
outputs = model_config["outputs"]

# %%
# Creating the CRI network
# ------------------------
# A ``CRI_network`` object is initialised with the loaded topology and
# targets the physical CRI hardware (``target="CRI"``).

print("Creating CRI network...")
network = CRI_network(
    axons=axons,
    connections=connections,
    outputs=outputs,
    target="CRI",
)

# %%
# Running inference
# -----------------
# For each test sample we:
#
# 1. Clear the FPGA membrane potentials.
# 2. Feed each temporal frame as a list of active axon IDs.
# 3. Accumulate output spikes across all frames plus 6 drain timesteps.
# 4. Classify by the output neuron with the highest spike rate.
#
# Hardware performance counters (clock cycles, HBM accesses) are recorded
# for each sample.

print("Running inference on test set...")

data = []  # stores (clock_cycles, hbm_accesses) per sample
correct = 0
total = 0
loss_fn = nn.CrossEntropyLoss()
test_loss = 0

for img, label in test_set:
    # Reset membrane potentials before each new sample
    hs_bridge.FPGA_Execution.fpga_controller.clear(
        len(connections), False, 0
    )  # num_neurons, simDump, coreOverride

    img = img.to(device)  # [T, C, H, W]
    spike_counts = torch.zeros(len(outputs))

    # Feed each frame as a sparse spike list
    for t in range(img.shape[0]):
        frame = img[t, :, :, :]  # [C, H, W]
        flat = frame.unsqueeze(0).flatten(start_dim=1).to(torch.int16)
        inputs = [f"A{i}" for i, v in enumerate(flat[0]) if v.item() > 0]

        hardwareSpikes, _, _ = network.step(inputs)
        for spike in hardwareSpikes:
            if spike in outputs:
                spike_counts[spike] += 1

    # Drain remaining spikes with 6 empty timesteps
    for _ in range(6):
        hardwareSpikes, clock_cycles, hbm_accesses = network.step([])
        for spike in hardwareSpikes:
            if spike in outputs:
                spike_counts[spike] += 1

    data.append((clock_cycles, hbm_accesses))

    # Classify by average spike rate
    spike_counts = spike_counts / img.size(0)
    predicted = torch.argmax(spike_counts).item()

    total += 1
    if predicted == label:
        correct += 1

    # Cross-entropy loss against one-hot target
    label_onehot = F.one_hot(torch.tensor(label), num_classes=11).float()
    loss = loss_fn(spike_counts, label_onehot)
    test_loss += loss.item()

    print(
        f"[{total}] Predicted: {predicted}, Label: {label} | "
        f"Running accuracy: {100 * correct / total:.2f}%"
    )

# %%
# Results
# -------
# Print the final accuracy and average loss over the test set.

accuracy = 100 * correct / total
avg_loss = test_loss / total
print(f"Test accuracy: {accuracy:.2f}%")
print(f"Test loss:     {avg_loss:.4f}")
