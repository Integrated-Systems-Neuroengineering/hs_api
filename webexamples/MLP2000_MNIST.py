"""
Multi-Layer Perceptron (2000 hidden units) on MNIST
====================================================
This example demonstrates running inference on the MNIST test set using a
fully-connected spiking neural network (784-2000-10) deployed on HiAER-Spike
neuromorphic hardware.

The larger hidden layer (2000 units) provides higher capacity compared to the
128-unit variant, potentially improving accuracy at the cost of increased
hardware resource usage. Predictions are made by selecting the output neuron
with the highest membrane potential.
"""

# %%
# Importing the necessary libraries
# ----------------------------------
import pickle
import torch
from hs_api.api import CRI_network

MODEL_CONFIG_PATH = "MLP2000_config.pkl"
TEST_BATCH_PATH = "MNIST_test_dataset.pkl"

# %%
# Helper function for classification
# ------------------------------------
# Find the output neuron with maximum membrane potential to determine prediction
def max_membrane_potential(outputs: list):
    max_val = float('-inf')
    max_label = None
    membrane_potentials_dict = dict(outputs)
    for key in membrane_potentials_dict:
        if membrane_potentials_dict[key] > max_val:
            max_val = membrane_potentials_dict[key]
            max_label = key
    return max_val, max_label

# %%
# Main inference loop
# -------------------
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

    for img, label in zip(test_batch['images'], test_batch['labels']):
        input_flat = img.reshape(img.size(0), -1)
        input_flat = input_flat.to(torch.int16)
    
        # Create input spike list from binary image
        inputs = []
        for i, elem in enumerate(input_flat[0, :]):
            if elem.item() == 1:
                inputs.append(f"A{i}")
        
        # Run inference through the network
        _ = network.step(inputs)
        _ = network.step([])
        results = network.read_membrane(outputs)
    
        # Get prediction
        max_mp, predicted = max_membrane_potential(results)
        
        total += 1
        if predicted == label:
            correct += 1
    
        running_accuracy = 100 * correct / total
        print(f"Running accuracy: {running_accuracy:.2f}%")
    
    final_accuracy = 100 * correct / total
    print(f'Final accuracy on 10000 test images: {final_accuracy:.2f}%')

if __name__ == "__main__":
    main()
