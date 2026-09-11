"""
LeNet-5 Stride 2 on MNIST
=========================

This example demonstrates how to evaluate a quantized LeNet-5 model (configured with a stride of 2) on the MNIST dataset using the HiAER-Spike API.

.. note::
    **NSG Submission:** Zip this script together with ``LeNet5_stride2_config.pkl`` and ``MNIST_test_dataset.pkl`` (both flat, in the same folder) before uploading to NSG.
"""

# %%
# Import Dependencies
# ------------------
# First, we import the required libraries and the HiAER-Spike network interface.
import pickle
import torch
from hs_api.api import CRI_network

MODEL_CONFIG_PATH = "LeNet5_stride2_config.pkl"
TEST_BATCH_PATH = "MNIST_test_dataset.pkl"

# %%
# Helper Function for Readout
# ---------------------------
# This function determines the maximum membrane potential across output neurons to predict the classification label.
def max_membrane_potential(outputs: list):
    max_val = float('-inf') # start lower than any real value
    max_label = None
    membrane_potentials_dict = dict(outputs) # convert to dict for easy look up
    for key in membrane_potentials_dict:       # iterate through all output neurons
        if membrane_potentials_dict[key] > max_val:
            max_val = membrane_potentials_dict[key]
            max_label = key
    
    return max_val, max_label  # return output neuron with greatest membrane potential

# %%
# Main Execution Loop
# -------------------
# Here we load the model configuration and test dataset, initialize the hardware network, and run inference across time steps.
def main():
    with open(MODEL_CONFIG_PATH, "rb") as f:
        model_config = pickle.load(f)
    with open(TEST_BATCH_PATH, "rb") as f:
        test_batch = pickle.load(f)

    axons = model_config["axons"]
    connections = model_config["connections"]
    outputs = model_config["outputs"]

    # Create network
    network = CRI_network(
        axons=axons,
        connections=connections,
        outputs=outputs,
        target="CRI"
    )

    # Run testing
    correct = 0
    total = 0

    predicted_MP = []
    for img, label in zip(test_batch['images'], test_batch['labels']):
        input = img.reshape(img.size(0), -1) # flatten input to [1, 36]
        input = input.to(torch.int16)        # change input from FP32 to INT16
    
        # Create input list
        inputs = []
        for i, elem in enumerate(input[0, :]):
            if elem.item() == 1:
                inputs.append(f"A{i}")
        
        # Running for 7 timesteps
        _ = network.step(inputs) # 1st and 2nd time step through Conv1
        _ = network.step([])     
        _ = network.step([])     # 3rd and 4th time step through Conv2
        _ = network.step([])
        _ = network.step([])     # 5th time step through fc1
        results = network.read_membrane(outputs)
    
        # Compare predicted with ground truth
        maxMP, predicted = max_membrane_potential(results)
        predicted_MP.append(maxMP)
        
        total += 1
        if predicted == label:
            correct += 1
    
        running_accuracy = 100 * correct / total
        print(f"Running accuracy : {running_accuracy:.2f} %")
    
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
                
if __name__ == "__main__":
    main()