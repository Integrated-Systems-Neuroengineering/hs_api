import hs_bridge
from hs_api.api import CRI_network
from hs_api.neuron_models2 import IF_neuron
from quantizer import Quantize_Network
from spikingjelly.activation_based import functional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader, Subset
from spikingjelly.datasets import pad_sequence_collate
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

# ─── CHANGE HERE FOR DIFFERENT MODELS ─────────────────────────────────────────
# Swap model_file and PATH to test a different architecture.
# If the model class takes explicit spiking_neuron kwargs, pass them in _make_model().
# If it has the neuron baked in (like DVS_Stride2_Ch1_model), just call Model(num_classes).
#
# Ch6_Ch16 conv2: import DVS_Stride2_Ch6_Ch16_conv2_model; input_res = 90
# Ch10 conv2:     import DVS_Stride2_Ch10_conv2_model;     input_res = 63
# Ch50 conv3:     import DVS_Stride2_Ch50_conv3_model;     input_res = 63
# Ch100 conv3:    import DVS_Stride2_Ch100_conv3_model;    input_res = 63

import DVS_Stride2_Ch1_model as model_file

input_res     = model_file.input_res   # 63
data_dir      = "/home/k7arora/DVS128Gesture"
PATH          = "/home/k7arora/hs_api/examples/CRI_Mapping/chris_code/CIFAR10_SNN_Finetuning/Final_SNN_Model/DVS_weights_channels1"
frames_number = 10
batch_size    = 64

def _make_model():
    return model_file.Model(model_file.num_classes)
# ──────────────────────────────────────────────────────────────────────────────

threshold = 32767
N = IF_neuron(threshold)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_sj_accuracy(net, test_loader, device, num_drain_steps):
    net.eval()
    loss_fn = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for img, label, _ in test_loader:
            functional.reset_net(net)
            img = img.to(device)
            img = img.transpose(0, 1)  # (T, B, C, H, W)
            label = label.to(device)
            T = img.shape[0]
            out_fr = torch.zeros(label.shape[0], 11, device=device)
            for t in range(T):
                encoded_img = img[t]
                out_fr += net(encoded_img)
            blank_input = torch.zeros_like(encoded_img)
            for _ in range(num_drain_steps):
                out_fr += net(blank_input)
            out_fr = out_fr / T
            total_loss += loss_fn(out_fr, label).item() * label.numel()
            correct += (out_fr.argmax(1) == label).sum().item()
            total += label.numel()
    accuracy = correct / total
    avg_loss = total_loss / total
    print(f"accuracy: {accuracy:.4f}")
    return accuracy, avg_loss


def _conv_output_res(res, kernel, stride, padding=0):
    return int((res + 2 * padding - (kernel - 1) - 1) / stride + 1)


def _get_layers(model):
    conv_layers   = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return conv_layers, linear_layers


# Derive architecture from the model at import time for diagnostics
_arch             = _make_model()
conv_layers, linear_layers = _get_layers(_arch)
kernel_size_list  = [l.kernel_size[0] for l in conv_layers]
stride_list       = [l.stride[0]      for l in conv_layers]
padding_list      = [l.padding[0] if isinstance(l.padding, tuple) else l.padding for l in conv_layers]

conv_output_res_list = []
_res = input_res
for _i in range(len(conv_layers)):
    _res = _conv_output_res(_res, kernel_size_list[_i], stride_list[_i], padding_list[_i])
    conv_output_res_list.append(_res)

num_outputs     = [l.out_features for l in linear_layers][-1]
num_layers      = len(conv_layers) + len(linear_layers)   # pipeline drain steps
extra_timesteps = num_layers

print(f"kernel sizes:            {kernel_size_list}")
print(f"strides:                 {stride_list}")
print(f"paddings:                {padding_list}")
print(f"input resolution:        {input_res}")
print(f"conv output resolutions: {conv_output_res_list}")
print(f"num conv layers:         {len(conv_layers)}")
print(f"num linear layers:       {len(linear_layers)}")
print(f"num outputs:             {num_outputs}")
print(f"extra timesteps:         {extra_timesteps}")
print(f"weights path:            {PATH}")


class DVSResizeAndBinarize:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        frames, label = data if isinstance(data, tuple) else (data, None)
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        T, C, H, W = frames.shape
        resized = torch.zeros((T, C, self.size[0], self.size[1]),
                              dtype=frames.dtype, device=frames.device)
        for t in range(T):
            resized_frame = F.interpolate(
                frames[t].unsqueeze(0), size=self.size,
                mode="bilinear", align_corners=False
            ).squeeze(0)
            resized[t] = (resized_frame > 0).float()
        return (resized, label) if label is not None else resized


def main():
    resize_transform = DVSResizeAndBinarize(size=(input_res, input_res))

    # ── Data loading ───────────────────────────────────────────────────────────
    full_train_set = DVS128Gesture(
        root=data_dir, frames_number=frames_number, split_by="number",
        train=True, data_type="frame", duration=1600000,
        transform=resize_transform,
    )
    full_train_size = len(full_train_set)
    val_size        = int(0.15 * full_train_size)
    train_size      = full_train_size - val_size

    torch.manual_seed(1)
    indices       = torch.randperm(full_train_size)
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    train_set_aug = DVS128Gesture(
        root=data_dir, frames_number=frames_number, split_by="number",
        train=True, data_type="frame", duration=1600000,
        transform=resize_transform,
    )
    train_set = Subset(train_set_aug, train_indices)
    val_set   = Subset(full_train_set, val_indices)

    test_set = DVS128Gesture(
        root=data_dir, frames_number=frames_number, split_by="number",
        train=False, data_type="frame", duration=1600000,
        transform=resize_transform,
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        drop_last=True, pin_memory=True, collate_fn=pad_sequence_collate,
    )

    T, C, H, W = full_train_set[0][0].shape
    print(f"input shape: {(T, C, H, W)}")
    print(f"training samples:   {len(train_set)} ({len(train_set)/full_train_size*100:.1f}%)")
    print(f"validation samples: {len(val_set)} ({len(val_set)/full_train_size*100:.1f}%)")
    print(f"test samples:       {len(test_set)}")

    # ── Load model and test original accuracy ──────────────────────────────────
    model = _make_model().to(device)
    checkpoint = torch.load(PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["net"])
    model.eval()

    print("\n" + "="*50)
    print("TESTING ORIGINAL MODEL ACCURACY")
    print("="*50)
    original_accuracy, original_loss = _eval_sj_accuracy(model, test_loader, device, num_layers)

    # ── Quantize ───────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("QUANTIZING MODEL")
    print("="*50)
    qn = Quantize_Network(w_alpha=1, dynamic_alpha=False)
    print(f"w_alpha: {qn.w_alpha}  dynamic_alpha: {qn.dynamic_alpha}  "
          f"w_bits: {qn.w_bits}  w_delta: {qn.w_delta}")

    net_quan = qn.quantize(model)
    net_quan.eval()

    print("\n" + "="*50)
    print("TESTING QUANTIZED MODEL ACCURACY")
    print("="*50)
    quantized_accuracy, quantized_loss = _eval_sj_accuracy(net_quan, test_loader, device, num_layers)

    accuracy_drop = original_accuracy - quantized_accuracy
    print(f"\n{'='*50}")
    print(f"QUANTIZATION RESULTS:")
    print(f"Original Accuracy:   {original_accuracy:.4f}")
    print(f"Quantized Accuracy:  {quantized_accuracy:.4f}")
    print(f"Accuracy Drop:       {accuracy_drop:.4f} ({accuracy_drop/max(original_accuracy,1e-8)*100:.2f}%)")
    print(f"Original Loss:       {original_loss:.4f}")
    print(f"Quantized Loss:      {quantized_loss:.4f}")
    print(f"Loss increase:       {quantized_loss - original_loss:.4f}")
    print(f"{'='*50}")

    # Cast quantized weights to int16 (matching the specific converters)
    int16_sd     = net_quan.state_dict()
    weight_names = [k for k in int16_sd if k.endswith(".weight")]
    for name in weight_names:
        int16_sd[name] = int16_sd[name].to(torch.int16)
    print(f"int16_sd keys: {list(int16_sd.keys())}")

    conv_weight_names   = weight_names[:len(conv_layers)]
    linear_weight_names = weight_names[len(conv_layers):]

    for name in conv_weight_names:
        print(f"conv weight ({name}): {int16_sd[name].shape}")
    for name in linear_weight_names:
        print(f"FC   weight ({name}): {int16_sd[name].shape}")

    # ── Build axon / neuron / output dicts ─────────────────────────────────────
    axons       = {}
    connections = {}
    outputs     = []

    # One axon per pixel for both DVS polarity channels
    for i in range(C * input_res * input_res):
        axons[f"A{i}"] = []

    # Conv1: axons -> C1 neurons
    axon_map    = torch.arange(C * input_res**2, dtype=torch.float32).reshape(1, C, input_res, input_res)
    patch_tensor = F.unfold(axon_map, kernel_size=kernel_size_list[0],
                            stride=stride_list[0], padding=padding_list[0])
    patch_rows  = patch_tensor.transpose(1, 2).squeeze(0).to(torch.int16)

    print(f"conv1 weight shape: {int16_sd[conv_weight_names[0]].shape}")
    for feature_map, kernel in enumerate(int16_sd[conv_weight_names[0]]):
        flat_kernel = kernel.flatten()
        for index, row in enumerate(patch_rows):
            neuron_name = f"C1.{feature_map}.{index}"
            connections[neuron_name] = ([], N)
            for i, elem in enumerate(row):
                axons[f"A{int(elem.item())}"].append((neuron_name, flat_kernel[i].item()))

    # Conv2..N: Ck -> C(k+1) neurons
    for conv_idx in range(1, len(conv_layers)):
        prev_res     = conv_output_res_list[conv_idx - 1]
        c_map        = torch.arange(prev_res**2, dtype=torch.float32).reshape(1, 1, prev_res, prev_res)
        patch_tensor = F.unfold(c_map, kernel_size=kernel_size_list[conv_idx],
                                stride=stride_list[conv_idx], padding=padding_list[conv_idx])
        patch_rows   = patch_tensor.transpose(1, 2).squeeze(0).to(torch.int16)

        print(f"weight shape for conv{conv_idx+1}: {int16_sd[conv_weight_names[conv_idx]].shape}")
        for output_idx, output_channel in enumerate(int16_sd[conv_weight_names[conv_idx]]):
            for feature_map, kernel in enumerate(output_channel):
                flat_kernel = kernel.flatten()
                for j, row in enumerate(patch_rows):
                    neuron_name = f"C{conv_idx+1}.{output_idx}.{j}"
                    connections[neuron_name] = ([], N)
                    for i, elem in enumerate(row):
                        src_index = int(elem.item())
                        connections[f"C{conv_idx}.{feature_map}.{src_index}"][0].append(
                            (neuron_name, flat_kernel[i].item())
                        )

    # Last conv -> FC1
    feature_map    = 0
    last_res2      = conv_output_res_list[-1] ** 2
    print(f"fc1 shape: {int16_sd[linear_weight_names[0]].shape}")
    for col in range(int16_sd[linear_weight_names[0]].shape[1]):
        if col % last_res2 == 0 and col != 0:
            feature_map += 1
        src = f"C{len(conv_layers)}.{feature_map}.{col % last_res2}"
        for i, elem in enumerate(int16_sd[linear_weight_names[0]][:, col]):
            connections[src][0].append((f"FC1.{i}", elem.item()))

    # FC1 -> FC2 -> ... -> outputs
    for layer_idx in range(1, len(linear_weight_names)):
        print(f"fc{layer_idx+1} shape: {int16_sd[linear_weight_names[layer_idx]].shape}")
        is_last = (layer_idx == len(linear_weight_names) - 1)
        for col in range(int16_sd[linear_weight_names[layer_idx]].shape[1]):
            all_connections = []
            for i, elem in enumerate(int16_sd[linear_weight_names[layer_idx]][:, col]):
                dst = i if is_last else f"FC{layer_idx+1}.{i}"
                all_connections.append((dst, elem.item()))
            connections[f"FC{layer_idx}.{col}"] = (all_connections, N)

    # Output neurons
    for x in range(num_outputs):
        connections[x] = ([], N)
        outputs.append(x)

    # ── Save dicts for comparison with specific converters ─────────────────────
    model_name = type(model).__name__
    dict_dir   = os.path.join(os.path.dirname(__file__), "dvsdictionaries", model_name)
    os.makedirs(dict_dir, exist_ok=True)
    dict_path  = os.path.join(dict_dir, "conversion_dicts.pkl")
    with open(dict_path, "wb") as f:
        pickle.dump({"axons": axons, "connections": connections, "outputs": outputs}, f)
    print(f"Saved conversion dictionaries: {dict_path}")

    # ── Network stats ──────────────────────────────────────────────────────────
    num_synapses       = 0
    max_neuron_fanout  = 0
    max_axon_fanout    = 0
    for key in connections:
        n = len(connections[key][0])
        num_synapses += n
        if n > max_neuron_fanout:
            max_neuron_fanout = n
    for key in axons:
        n = len(axons[key])
        num_synapses += n
        if n > max_axon_fanout:
            max_axon_fanout = n

    print(f"Number of neurons:    {len(connections)}")
    print(f"Number of axons:      {len(axons)}")
    print(f"Number of synapses:   {num_synapses}")
    print(f"Max fan-out (neuron): {max_neuron_fanout}")
    print(f"Max fan-out (axon):   {max_axon_fanout}")
    print(f"Outputs:              {outputs}")

    # ── Create network ─────────────────────────────────────────────────────────
    network = CRI_network(axons=axons, connections=connections, outputs=outputs, target="CRI")
    print("Network Loaded onto HiAER Spike")

    # ── Hardware inference ─────────────────────────────────────────────────────
    data      = []
    correct   = 0
    total     = 0
    loss_fn   = nn.CrossEntropyLoss()
    test_loss = 0

    for img, label in test_set:
        hs_bridge.FPGA_Execution.fpga_controller.clear(len(connections), False, 0)

        network.read_membrane(outputs)

        img          = img.to(device)          # [T, C, H, W]
        spike_counts = torch.zeros(len(outputs))

        for t in range(img.shape[0]):
            flat = img[t].unsqueeze(0).flatten(start_dim=1).to(torch.int16)
            inputs = [f"A{i}" for i, v in enumerate(flat[0]) if v.item() > 0]

            network.read_membrane(outputs)
            hardware_spikes, _, _ = network.step(inputs)
            for spike in hardware_spikes:
                if spike in outputs:
                    spike_counts[spike] += 1
                else:
                    print(f"Error: invalid output spike {spike}")

        clock_cycles, hbm_accesses = 0, 0
        for _ in range(extra_timesteps):
            hardware_spikes, clock_cycles, hbm_accesses = network.step([])
            for spike in hardware_spikes:
                if spike in outputs:
                    spike_counts[spike] += 1
                else:
                    print(f"Error: invalid output spike {spike}")

        data.append((clock_cycles, hbm_accesses))

        spike_counts /= img.size(0)
        predicted  = torch.argmax(spike_counts).item()
        label_int  = int(label.item()) if torch.is_tensor(label) else int(label)
        print(f"Predicted: {predicted}, Label: {label_int}")

        total += 1
        if predicted == label_int:
            correct += 1
        print(f"Running accuracy: {100 * correct / total:.2f}%")

        label_onehot = F.one_hot(torch.tensor(label_int), num_classes=num_outputs).float()
        test_loss += loss_fn(spike_counts, label_onehot).item()

    snn_accuracy = 100 * correct / total
    print(f"Accuracy of the network on the test images: {snn_accuracy:.2f}%")
    print(f"Test Loss: {test_loss / total:.4f}")

    # ── Save results ───────────────────────────────────────────────────────────
    parent_directory = os.path.dirname(PATH)
    output_txt = os.path.join(parent_directory, "accuracies_dvs_converted.txt")
    mode = "a" if os.path.exists(output_txt) else "w"
    with open(output_txt, mode) as f:
        f.write(f"Original SNN Accuracy:   {original_accuracy:.2f}%\n")
        f.write(f"Quantized SNN Accuracy:  {quantized_accuracy:.2f}%\n")
        f.write(f"FPGA Converted Accuracy: {snn_accuracy:.2f}%\n")

    np.save(os.path.join(parent_directory, "clock_cycles_dvs_converted.npy"), np.asarray(data))


if __name__ == "__main__":
    main()
