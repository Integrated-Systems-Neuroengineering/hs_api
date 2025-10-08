''''
Replacing SNN IFNodes with Custom_LIFNodes and evaluating with flush steps and tau=2^63 and finetuning on a dataset of ANN actions on observations.
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
import torch.nn.functional as F
import numpy as np

from evaluate_dvs_snn import evaluate_dvs_snn

# Add paths to import Custom_LIFNode and DVS environment
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))  # Add pong_stuff to path

# Import SNNCalibrator for threshold optimization
from ann_to_snn_dvs_utils import create_dvs_environment_and_dataloader
sys.path.insert(0, os.path.join(current_dir, '..', '..', '..', 'hs_api'))
sys.path.insert(0, os.path.join(current_dir, '..', '..', '..', 'fxpmath'))

# Import Custom_LIFNode directly from the module file to avoid hs_api.__init__ imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "custom_neurons",
    os.path.join(current_dir, '..', '..', '..', 'hs_api', 'hs_api', 'custom_neurons.py')
)
custom_neurons = importlib.util.module_from_spec(spec)
sys.modules['custom_neurons'] = custom_neurons
spec.loader.exec_module(custom_neurons)
Custom_LIFNode = custom_neurons.Custom_LIFNode
from spikingjelly.activation_based import surrogate, neuron, functional

# Import DVS environment
from hs_api.pong_model_pipeline.DVSWrapper import make_dvs_pong_env

from replace_ifnode_with_custom_lif import replace_ifnodes_with_custom_lif
from convert_to_custom_lif_with_flush import evaluate_dvs_snn_custom_lif_with_flush
os.path.join(current_dir, '..', 'ann_to_snn')

from evaluate_dvs_snn import evaluate_dvs_snn

#################################### PART 1: DATASET CREATION ####################################

# load ANN

print("\n" + "="*60)
print("STEP 1: ENVIRONMENT SETUP AND MODEL LOADING")
print("="*60)

dvs_model_path = "../ann_training/dvs_84_no_bias_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env, ann_model, model_architecture, loader, obs_tensor = create_dvs_environment_and_dataloader(
    dvs_model_path, device, num_observations=10000
)
print("ann_model:", ann_model)

# collect ANN actions on observations over 1,000,000 steps

print("\n" + "="*60)
print("STEP 2: COLLECTING ANN ACTIONS ON OBSERVATIONS")
print("="*60)

num_observations = 1000000
dataset_filename = f'finetuning_dataset_{num_observations}.npz'

# Check if dataset already exists
if os.path.exists(dataset_filename):
    print(f"Loading existing dataset from {dataset_filename}")
    data = np.load(dataset_filename)
    all_obs = data['observations']
    all_actions = data['actions']
    print(f"Loaded {len(all_actions)} samples from dataset")
else:
    print(f"Dataset not found. Collecting {num_observations} samples...")
    ann_model.eval()
    all_obs = np.zeros((num_observations, 2, 84, 84), dtype=np.float32)
    all_actions = np.zeros(num_observations, dtype=np.int32)

    obs, info = env.reset()
    all_obs[0] = obs

    for step in range(num_observations):
        if step % 10000 == 0:
            print(f"Step {step}/{num_observations}")
        obs_input = torch.from_numpy(obs).unsqueeze(0).to(device)
        action = ann_model(obs_input).argmax(dim=1).item()
        all_actions[step] = action

        obs, reward, terminated, truncated, info = env.step(action)
        if step < num_observations - 1:
            all_obs[step+1] = obs
        if terminated or truncated:
            obs, info = env.reset()

    # Save dataset
    print(f"Saving dataset to {dataset_filename}")
    np.savez_compressed(dataset_filename, observations=all_obs, actions=all_actions)
    print(f"Dataset saved successfully")
        
#################################### PART 2: REPLACE IFNODES WITH CUSTOM_LIF ####################################

print("\n" + "="*60)
print("STEP 3: REPLACING IFNODES WITH CUSTOM_LIFNODES")
print("="*60)

# load IFNode-based SNN

snn_model_path = "dvs_84.pth"
snn_model = torch.load(snn_model_path, map_location=device, weights_only=False)
print(snn_model)

# evaluate original IFNode-based SNN
# evaluate_dvs_snn(snn_model, ann_model=ann_model, env=env, device=device, episodes=5, time_steps=18)

custom_snn = replace_ifnodes_with_custom_lif(snn_model, tau=float(2**63), decay_input=False)

print(f"Converted model:\n{custom_snn}")

# evaluate Custom_LIFNode-based SNN with flush steps
evaluate_dvs_snn_custom_lif_with_flush(custom_snn, env=env, device=device, time_steps=18, flush_steps=1, episodes=5)

# ensure bias is off
for module in custom_snn.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if module.bias is not None:
            print("Setting bias to None for module:", module)
            module.bias = None

#################################### PART 3: SUPERVISED FINETUNING ####################################

print("\n" + "="*60)
print("STEP 4: SUPERVISED FINETUNING OF CUSTOM_LIFNODES")
print("="*60)

# finetune Custom_LIFNode-based SNN on collected dataset of ANN actions on observations
custom_snn.train()

# make sure surrogate gradients are enabled
for module in custom_snn.modules():
    if isinstance(module, neuron.BaseNode):
        module.surrogate_function = surrogate.ATan()

optimizer = torch.optim.Adam(custom_snn.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)
batch_size = 128
num_epochs = 20

dataset_size = len(all_actions)

# Train/validation split (80/20)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
all_indices = np.arange(dataset_size)
np.random.shuffle(all_indices)
train_indices = all_indices[:train_size]
val_indices = all_indices[train_size:]

num_train_batches = (train_size + batch_size - 1) // batch_size
num_val_batches = (val_size + batch_size - 1) // batch_size
print(f"Dataset size: {dataset_size}")
print(f"Train: {train_size}, Val: {val_size}")
print(f"Batches per epoch - Train: {num_train_batches}, Val: {num_val_batches}")

# Early stopping setup
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):

    # Shuffle training dataset each epoch
    np.random.shuffle(train_indices)
    epoch_loss = 0.0

    for batch_idx in range(num_train_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, train_size)
        batch_indices = train_indices[start_idx:end_idx]

        # Get batch data
        batch_obs = torch.from_numpy(all_obs[batch_indices]).to(device)
        batch_actions = torch.from_numpy(all_actions[batch_indices]).long().to(device)

        # Reset SNN state
        functional.reset_net(custom_snn)

        # Forward pass through SNN with time steps
        snn_outputs = []
        for t in range(1):  # time_steps=1
            output = custom_snn(batch_obs)
            snn_outputs.append(output)

        # sum outputs over time steps
        snn_qvalues = torch.stack(snn_outputs).sum(dim=0)

        # Calculate Cross-Entropy loss between SNN outputs and ANN action targets
        loss = F.cross_entropy(snn_qvalues, batch_actions)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(custom_snn.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{num_train_batches}, Loss: {loss.item():.4f}")

    avg_train_loss = epoch_loss / num_train_batches
    print(f"\nEpoch {epoch+1}/{num_epochs} Training completed. Average Train Loss: {avg_train_loss:.4f}")

    # Validation loop
    custom_snn.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_idx in range(num_val_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, val_size)
            batch_indices = val_indices[start_idx:end_idx]

            batch_obs = torch.from_numpy(all_obs[batch_indices]).to(device)
            batch_actions = torch.from_numpy(all_actions[batch_indices]).long().to(device)

            functional.reset_net(custom_snn)

            snn_outputs = []
            for t in range(1):
                output = custom_snn(batch_obs)
                snn_outputs.append(output)

            snn_qvalues = torch.stack(snn_outputs).sum(dim=0)
            loss = F.cross_entropy(snn_qvalues, batch_actions)
            val_loss += loss.item()

            # Calculate accuracy
            predicted = snn_qvalues.argmax(dim=1)
            val_correct += (predicted == batch_actions).sum().item()
            val_total += len(batch_actions)

    avg_val_loss = val_loss / num_val_batches
    val_accuracy = 100.0 * val_correct / val_total
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Learning rate scheduler step
    scheduler.step(avg_val_loss)

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model
        torch.save(custom_snn, 'dvs_84_no_bias_snn_finetuned_best.pth')
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience counter: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Evaluate with environment after each epoch
    print("\nEnvironment evaluation:")
    evaluate_dvs_snn_custom_lif_with_flush(custom_snn, env, device, time_steps=18, flush_steps=1, episodes=5)
    custom_snn.train()

print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)
print(f"Best validation loss: {best_val_loss:.4f}")
print("Best model saved as: dvs_84_no_bias_snn_finetuned_best.pth")