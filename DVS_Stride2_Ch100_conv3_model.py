# test_dvs_inference_standalone.py
import pickle
import torch
import hs_bridge
from hs_api.api import CRI_network
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader, Subset
from spikingjelly.datasets import pad_sequence_collate
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

data_dir = "/home/ckdeng/myprojects/DVS_Gesture"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_config():
    """Load saved model configuration"""
    with open('/home/ckdeng/GitHub_repo/hs_api/tests/fixtures/DVS_model_config.pkl', 'rb') as f:
        return pickle.load(f)

#Loading the dataset and preprocessing
# resize transform that iterates over the temporal dimension, binarizes
class DVSResizeAndBinarize:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        frames, label = data if isinstance(data, tuple) else (data, None)
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        T, C, H, W = frames.shape

        resized = torch.zeros((T, C, self.size[0], self.size[1]), dtype=frames.dtype, device=frames.device)
        for t in range(T):
            frame = frames[t]  # [C, H, W]
            resized_frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False
            ).squeeze(0)
            binarized_frame = (resized_frame > 0).float()
            resized[t] = binarized_frame
        return (resized, label) if label is not None else resized
    
# Use our simple resize transform for all datasets
resize_transform = DVSResizeAndBinarize(size=(63, 63))  # resize from 128x128 to input_res


test_set = DVS128Gesture(
    root=data_dir, 
    frames_number=10, 
    split_by="number", 
    train=False, 
    data_type="frame", 
    duration=1600000,
    transform=resize_transform
)

# Create DataLoader for test set
test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    collate_fn=pad_sequence_collate,
)

print(f"Test samples: {len(test_set)}")

print("Loading model configuration...")
model_config = load_model_config()

axons = model_config['axons']
connections = model_config['connections']
outputs = model_config['outputs']

print("Creating CRI network...")
# Create network
network = CRI_network(
    axons=axons,
    connections=connections,
    outputs=outputs,
    target="CRI")

print("Running inference on test set...")

# Run inference on test set
#used to save clock cycles and hbm accesses
data = []


#test model
correct = 0
total = 0
images = 0 
loss_fn = nn.CrossEntropyLoss()
test_loss = 0
last_label = -1
for img, label in test_set:
    #reset membrane potnetials before each image
    hs_bridge.FPGA_Execution.fpga_controller.clear(
                    len(connections), False, 0
                )  ##Num_neurons, simDump, coreOverride

    initial = network.read_membrane(outputs)
    
    img = img.to(device) #shape [T, C, H, W]
    spike_counts = torch.zeros(len(outputs))  #to count spikes over all frames
    for t in range(img.shape[0]):  #iterate through all frames
        frame = img[t,:,:,:] #shape [C, H, W]

        #convert from [C, H, W] to [1, C*H*W]
        input = frame.unsqueeze(0)  #add dimension
        input = input.flatten(start_dim=1)
        input = input.to(torch.int16)        #change input from FP32 to INT16
        print(f"Input shape: {input.shape}")
        #create input list
        inputs = []
        #print all unique input values
        #print(f"Unique input values: {input.unique()}")
        for i, elem in enumerate(input[0, :]):
            if elem.item() > 0: #changed from == 1 for MNIST
                inputs.append(f"A{i}")
        
        results = network.read_membrane(outputs)
        print(f"Membrane potentials: {results}")

        hardwareSpikes, _, _ = network.step(inputs)
        print(f"Output spikes: {hardwareSpikes}")

        for spike in hardwareSpikes:
            if spike in outputs:
                spike_counts[spike] += 1
            else:
                print(f"Error: invalid output spike {spike}")

    #add 6 extra timesteps after lastinput frame to allow it to propogate through network
    for i in range(6):
        inputs = []  #no input spikes
        hardwareSpikes, clock_cycles, hbm_accesses = network.step(inputs)
        print(f"Output spikes: {hardwareSpikes}")

        for spike in hardwareSpikes:
            if spike in outputs:
                spike_counts[spike] += 1
            else:
                print(f"Error: invalid output spike {spike}")
    
    
    #record clock cycles and hbm accesses
    data.append((clock_cycles, hbm_accesses))

    spike_counts = spike_counts / img.size(0)  #average spike counts(spike rate)
    print(f"Spike counts: {spike_counts}")

    predicted = torch.argmax(spike_counts).item()
    print(f"Predicted: {predicted}, Label: {label}")

    if label != last_label:
        print(f"Label changed from {last_label} to {label}")
        last_label = label
    
    total += 1
    if predicted == label:
        correct += 1
    
    running_accuracy = 100 * correct / total
    print(f"Running accuracy : {running_accuracy:.2f} %")


    #calculate loss
    label_onehot = F.one_hot(torch.tensor(label), num_classes=11).float()
    loss = loss_fn(spike_counts, label_onehot)
    print(f"Loss: {loss}")
    test_loss += loss.item()


accuracy = 100 * correct / total
loss = test_loss / total
print(f'Accuracy of the network on test images: {accuracy:.2f} %')
print(f'Test Loss: {loss:.4f}')
