"""
CIFAR-10 Inference with Bit-Sliced Inputs on HiAER-Spike Hardware
==================================================================

This example demonstrates running inference on the CIFAR-10 test set using a
fully-connected spiking neural network (16-100-100) deployed on HiAER-Spike
neuromorphic hardware.

Each 3-channel RGB image is converted to a 15-channel binary representation
by extracting the 5 most significant bit planes from each colour channel
(bit-slicing). The resulting binary tensor is fed to the hardware
timestep-by-timestep, and the output neuron with the highest spike rate
determines the predicted class.
"""
# sphinx_gallery_thumbnail_path = '_static/cifar10_bitslice_thumb.png'

# %%
# Importing the necessary libraries
# ----------------------------------
# We need PyTorch and torchvision for data loading and preprocessing,
# ``hs_api`` for the CRI network interface, and ``hs_bridge`` to reset
# membrane potentials between samples on the FPGA.

import io
import pickle
import urllib.request
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import hs_bridge
from hs_api.api import CRI_network

# %%
# Configuration
# -------------
# Select the compute device and set inference hyperparameters.
# ``T`` is the number of active input timesteps; ``extra_timesteps`` are
# additional empty steps used to drain spikes still propagating through
# the network after the last input frame.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 30
extra_timesteps = 5
MODEL_CONFIG_URL = 'https://www.dropbox.com/scl/fi/dqurfqfye1omyyozj0ps6/CIFAR10_model_config.pkl?rlkey=7lhl1ksk4n4j5evpyocwrh2y9&st=zcphe718&dl=1'

# %%
# Defining the bit-slicing utilities
# ------------------------------------
# CIFAR-10 images are 8-bit per channel. Rather than thresholding to a single
# binary value, we extract the 5 most significant bit planes from each of the
# R, G, and B channels, producing a 15-channel binary tensor per image.
# This preserves more colour information while keeping inputs binary.


class Cutout(object):
    """Randomly cut out square regions from an image tensor (data augmentation).

    Args:
        n_holes (int): Number of square holes to cut out.
        length (int): Side length of each square hole.
    """

    def __init__(self, n_holes: int, length: int):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        elif not isinstance(img, torch.Tensor):
            raise TypeError(
                f"Input must be a PIL Image or torch.Tensor, got {type(img)}"
            )

        C, H, W = img.shape
        mask = torch.ones((H, W), dtype=torch.bool, device=img.device)

        for _ in range(self.n_holes):
            y = torch.randint(H, (1,), device=img.device).item()
            x = torch.randint(W, (1,), device=img.device).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(H, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(W, x + self.length // 2)
            mask[y1:y2, x1:x2] = False

        return img * mask.unsqueeze(0).expand_as(img)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_holes={self.n_holes}, length={self.length})"


def extract_bit_planes(
    tensor: torch.Tensor,
    num_bits: int = 8,
    msb_first: bool = True,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Extract individual bit planes from a uint8 tensor.

    Args:
        tensor: uint8 input tensor of any shape.
        num_bits: Number of bit planes to extract (1–8).
        msb_first: If ``True``, extract from the most significant bit downward.
        target_dtype: dtype of the returned tensor.

    Returns:
        Tensor of shape ``(num_bits, *tensor.shape)`` containing 0/1 values.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if tensor.dtype != torch.uint8:
        raise ValueError(f"Input dtype must be torch.uint8, got {tensor.dtype}")
    if not (1 <= num_bits <= 8):
        raise ValueError(f"num_bits must be between 1 and 8, got {num_bits}")

    tensor = tensor.contiguous()
    dev = tensor.device

    if msb_first:
        shifts = torch.arange(7, 7 - num_bits, -1, device=dev, dtype=torch.uint8)
    else:
        shifts = torch.arange(0, num_bits, device=dev, dtype=torch.uint8)

    shift_shape = (num_bits,) + (1,) * tensor.ndim
    bit_planes = (tensor >> shifts.view(shift_shape)) & 1
    return bit_planes.to(target_dtype)


def cifar10_to_15channel_binary(image_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a uint8 RGB image to a 15-channel binary tensor via bit-slicing.

    Takes the 5 most significant bit planes from each of the R, G, and B
    channels, yielding a ``(15, H, W)`` float32 tensor.

    Args:
        image_tensor: uint8 tensor of shape ``(3, H, W)``.

    Returns:
        float32 tensor of shape ``(15, H, W)`` with values in {0, 1}.
    """
    if image_tensor.dtype != torch.uint8:
        raise TypeError(f"Input must be uint8, got {image_tensor.dtype}")

    r, g, b = torch.unbind(image_tensor, dim=0)
    r_bits = extract_bit_planes(r, num_bits=5, msb_first=True)
    g_bits = extract_bit_planes(g, num_bits=5, msb_first=True)
    b_bits = extract_bit_planes(b, num_bits=5, msb_first=True)
    return torch.cat([r_bits, g_bits, b_bits], dim=0)


class Binarize(object):
    """Threshold a float tensor to binary values.

    Args:
        threshold (float): Values above this become 1, others become 0.
    """

    def __init__(self, threshold=0.5):
        self.th = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.th).float()


# %%
# Defining the preprocessing transform
# --------------------------------------
# Images are resized to 32×32 (already the CIFAR-10 native size), converted
# to uint8 tensors with ``PILToTensor``, bit-sliced into 15 channels, and
# finally binarized with a 0.5 threshold.

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.PILToTensor(),
    cifar10_to_15channel_binary,
    Binarize(threshold=0.5),
])

# %%
# Loading the dataset
# --------------------
# We use the torchvision CIFAR-10 test split. After the transform each sample
# is a ``(15, 32, 32)`` binary tensor.

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transform,
    download=True,
)

C, H, W = test_dataset[0][0].shape
print(f"Input shape: {(C, H, W)}")
print(f"Test samples: {len(test_dataset)}")

# %%
# Loading the model configuration
# --------------------------------
# The CRI network topology (axons, connections, output neuron IDs) was
# serialised to disk during the ``hs_api`` conversion step and is loaded here.

print("Loading model configuration...")
with urllib.request.urlopen(MODEL_CONFIG_URL) as response:
    model_config = pickle.load(io.BytesIO(response.read()))

axons = model_config['axons']
connections = model_config['connections']
outputs = model_config['outputs']

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
# For each test image we:
#
# 1. Clear the FPGA membrane potentials.
# 2. Flatten the 15-channel binary image and feed active pixel indices as axon
#    IDs for each of the ``T`` timesteps.
# 3. Accumulate output spikes across all timesteps plus ``extra_timesteps``
#    drain steps.
# 4. Classify by the output neuron with the highest average spike rate.

print("Running inference on test set...")

correct = 0
total = 0

for img, label in test_dataset:
    # Reset membrane potentials before each new sample
    hs_bridge.FPGA_Execution.fpga_controller.clear(
        len(connections), False, 0
    )  # num_neurons, simDump, coreOverride

    img = img.to(device)                     # [C, H, W]
    flat = img.unsqueeze(0).flatten(start_dim=1)  # [1, C*H*W]
    spike_counts = torch.zeros(len(outputs))

    # Build spike list once — same binary image is replayed each timestep
    inputs = [f"A{i}" for i, v in enumerate(flat[0]) if v.item() > 0]

    for _ in range(T):
        hardwareSpikes, _, _ = network.step(inputs)
        for spike in hardwareSpikes:
            if spike in outputs:
                spike_counts[spike] += 1
            else:
                print(f"Warning: unexpected output spike {spike}")

    # Drain remaining spikes with empty timesteps
    for _ in range(extra_timesteps):
        hardwareSpikes, _, _ = network.step([])
        for spike in hardwareSpikes:
            if spike in outputs:
                spike_counts[spike] += 1

    spike_counts = spike_counts / T  # convert to average spike rate
    predicted = torch.argmax(spike_counts).item()

    total += 1
    if predicted == label:
        correct += 1

    print(
        f"[{total}] Predicted: {predicted}, Label: {label} | "
        f"Running accuracy: {100 * correct / total:.2f}%"
    )

# %%
# Results
# -------
# Print the final classification accuracy over the full 10 000-image test set.

accuracy = 100 * correct / total
print(f"Test accuracy: {accuracy:.2f}%")
