import os
import hs_bridge
import pytest
import pickle
from hs_api.api import CRI_network
from hs_bridge.partitioning_code.hierarchical_partition import partition as netPart
_CORE_ID = int(os.environ.get("HIAER_CORE_ID", 0))
import torch

class TestDVSInferenceMulticore:
    """Test DVS model inference on hardware with METIS 16-core partitioning"""
    
    @pytest.fixture
    def model_config(self):
        with open('./fixtures/DVS_model_small_config_shift=-17.pkl', 'rb') as f:
            return pickle.load(f)
    
    @pytest.fixture
    def test_batch(self):
        with open('./fixtures/DVS_test_batch.pkl', 'rb') as f:
            return pickle.load(f)
    
    def test_dvs_accuracy_multicore(self, model_config, test_batch):
        axons = model_config['axons']
        connections = model_config['connections']
        outputs = model_config['outputs']

        for neuron_key in connections:
            neuron_obj = connections[neuron_key][1]
            neuron_obj.shift = -17
            neuron_obj.legacy_noise_en = 1

        # Build connectome in simpleSim to get partition format
        sim_net = CRI_network(axons=axons, connections=connections, outputs=outputs, target="simpleSim")
        data = sim_net.connectome.get_part_format()
        membership = netPart(data=data, n_clusters=16)
        # Convert globalIdx keys to user keys
        merged = sim_net.connectome.get_merged_neurons()
        membership_by_key = {merged[idx].get_user_key(): core for idx, core in membership.items()}
        print(f"METIS partition: {len(set(membership.values()))} cores, distribution: {sorted([(k, list(membership.values()).count(k)) for k in set(membership.values())])}")

        # Create hardware network with membership
        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="CRI",
            membership=membership_by_key,
            n_cores=16
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        correct = 0
        total = len(test_batch['images'])
        for img, label in zip(test_batch['images'], test_batch['labels']):
            for core in range(16):
                hs_bridge.FPGA_Execution.fpga_controller.clear(len(connections), False, core)

            img = img.to(device)
            spike_counts = torch.zeros(len(outputs))
            for t in range(img.shape[0]):
                frame = img[t,:,:,:]
                input = frame.unsqueeze(0)
                input = input.flatten(start_dim=1)
                input = input.to(torch.int16)

                inputs = []
                for i, elem in enumerate(input[0, :]):
                    if elem.item() > 0:
                        inputs.append(f"A{i}")

                results = network.read_membrane(outputs)
                print(f"Membrane potentials: {results}")

                hardwareSpikes, _, _ = network.step(inputs)
                print(f"Output spikes: {hardwareSpikes}")

                for spike in hardwareSpikes:
                    if spike in outputs:
                        spike_counts[outputs.index(spike)] += 1

            for i in range(4):
                inputs = []
                hardwareSpikes, _, _ = network.step(inputs)
                for spike in hardwareSpikes:
                    if spike in outputs:
                        spike_counts[outputs.index(spike)] += 1

            predicted = spike_counts.argmax().item()
            print(f"Spike counts: {spike_counts}")
            print(f"Predicted: {predicted}, Label: {label}")
            if predicted == label:
                correct += 1

        accuracy = correct / total * 100
        print(f"Final accuracy: {accuracy:.2f}%")
        assert accuracy >= 44.44, f"Accuracy {accuracy:.2f}% below threshold"