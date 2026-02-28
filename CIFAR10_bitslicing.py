import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from PIL import Image

class Cutout(object):
    """
    Cutout data augmentation applied to a tensor.

    This transform randomly cuts out one or more square regions (holes) from an image tensor.
    The pixel values within the holes are set to zero (multiplied by a boolean mask).

    Args:
        n_holes (int): Number of square holes to cut out.
        length (int): Side length of each square hole.
    """

    def __init__(self, n_holes: int, length: int):
        """
        Initializes the Cutout transform with specified parameters.
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: any) -> torch.Tensor:
        """
        Applies the Cutout transform to the input image.

        Args:
            img (PIL.Image.Image or torch.Tensor): The image to be augmented.

        Returns:
            torch.Tensor: The augmented image with holes cut out.
        
        Raises:
            TypeError: If the input image is neither a PIL Image nor a torch.Tensor.
        """
        # Convert PIL image to Tensor if necessary
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        # Ensure the input is a tensor if it wasn't a PIL image
        elif not isinstance(img, torch.Tensor):
            raise TypeError(f"Input image must be a PIL Image or a torch.Tensor, but got {type(img)}")

        # Get the dimensions of the image tensor (C, H, W)
        C, H, W = img.shape

        # Create a boolean mask of the same height and width, initialized to True (keep all pixels)
        mask = torch.ones((H, W), dtype=torch.bool, device=img.device)

        # Iterate through the number of holes to be added
        for n in range(self.n_holes):
            # Randomly select the center coordinates (y, x) for the hole
            y = torch.randint(H, (1,), device=img.device).item()
            x = torch.randint(W, (1,), device=img.device).item()

            # Calculate the coordinates for the top-left and bottom-right corners of the square
            # Ensure coordinates stay within image boundaries
            y1 = max(0, y - self.length // 2)
            y2 = min(H, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(W, x + self.length // 2)

            # Set the mask area within these bounds to False (cut out these pixels)
            mask[y1:y2, x1:x2] = False

        # Expand the 2D mask to match the 3D shape of the image tensor (C, H, W)
        # This makes the mask applicable across all color channels simultaneously
        mask = mask.unsqueeze(0).expand_as(img)

        # Apply the mask to the image tensor (False areas become 0) and return the result
        return img * mask

    def __repr__(self):
        """
        Provides a string representation of the transform configuration.
        """
        return f"{self.__class__.__name__}(n_holes={self.n_holes}, length={self.length})"


# Function to extract bit planes
def extract_bit_planes(
    tensor: torch.Tensor,
    num_bits: int = 8,
    msb_first: bool = True,
    target_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected input to be a torch.Tensor, but got {type(tensor)}")
    if tensor.dtype != torch.uint8:
        # The input CIFAR-10 data will be uint8 after ToTensor()
        raise ValueError(f"Input tensor dtype must be torch.uint8, but got {tensor.dtype}")
    
    # Ensure tensor is contiguous for performance
    tensor = tensor.contiguous()
    
    if not (1 <= num_bits <= 8):
        raise ValueError(f"num_bits must be between 1 and 8 (inclusive), got {num_bits}")
    
    device = tensor.device
    if msb_first:
        # Shifts from 7 down to 0 for full 8 bits
        shifts = torch.arange(7, 7 - num_bits, -1, device=device, dtype=torch.uint8)
    else:
        # Shifts from 0 up to 7
        shifts = torch.arange(0, num_bits, 1, device=device, dtype=torch.uint8)
    
    # Reshape shifts to enable broadcasting across the input tensor dimensions (B, C, H, W)
    # The new shape will be (num_bits, 1, 1, 1, ...) based on input tensor's ndim
    shift_shape = (num_bits,) + (1,) * tensor.ndim
    shift_tensor = shifts.view(shift_shape)
    
    # Apply bit shift and mask
    bit_planes = (tensor >> shift_tensor) & 1
    
    return bit_planes.to(target_dtype)

# 1. Define a custom transform to convert images to 15-channel binary (R 5-bit, G 5-bit, B 5-bit)
def cifar10_to_15channel_binary(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a uint8 image tensor of shape (C=3, H=32, W=32) into a 
    float32 tensor of shape (C=15, H=32, W=32) where each channel 
    represents one bit plane from R (5), G (5), and B (5) channels.
    """
    # Ensure input is uint8 as expected by extract_bit_planes
    if image_tensor.dtype != torch.uint8:
        raise TypeError(f"Input image tensor must be uint8, got {image_tensor.dtype}")

    # Split the 3-channel image into individual R, G, B tensors
    r, g, b = torch.unbind(image_tensor, dim=0)

    # Extract the 5 most significant bit planes for each channel
    # The output of extract_bit_planes will have shape (5, H, W)
    r_bits = extract_bit_planes(r, num_bits=5, msb_first=True)
    g_bits = extract_bit_planes(g, num_bits=5, msb_first=True)
    b_bits = extract_bit_planes(b, num_bits=5, msb_first=True)
    
    # Concatenate the bit planes along the channel dimension (dim=0)
    # The final shape will be (5 + 5 + 5 = 15, H, W)
    combined_bits = torch.cat([r_bits, g_bits, b_bits], dim=0)
    
    return combined_bits

# 2. Define the transformations for the CIFAR-10 dataset
# `ToTensor()` converts the PIL image into a uint8 torch.Tensor (0-255 range)
# with shape (C, H, W), which is the required input for our custom transform.
transform = transforms.Compose([
    transforms.PILToTensor(),
    cifar10_to_15channel_binary
])

# 3. Load the CIFAR-10 dataset using the custom transform
# Make sure to set `download=True` the first time you run this script
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# 4. Create a DataLoader to easily batch and load the data
BATCH_SIZE = 64
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2)