"""
MP-based accuracy test for the small DVS model, for use on cores where the
spike-readback path is not working (e.g. core 5 on the current multicore NoC
bitstream). Since flush_spikes doesn't report spikes correctly on non-zero
cores, we instead detect spikes from the membrane-potential trace: a neuron
that crosses threshold and then resets to (near) zero on the next read has
spiked. We count these MP-detected "spike events" per output neuron across
the trial and classify via argmax, exactly as the normal spike-counting
version does.

Usage:
    python3 test_DVS_small_mp_based.py [--core CORE_ID]
"""
import argparse
import pickle
import hs_bridge
from hs_api.api import CRI_network

MODEL_CONFIG_PATH = "fixtures/DVS_model_small_config_shift=-17.pkl"
TEST_BATCH_PATH = "fixtures/DVS_test_batch.pkl"
ACCURACY_THRESHOLD = 44

# A neuron is considered to have "reset" (i.e. spiked on the previous read)
# if its MP drops by at least this much between consecutive reads while the
# earlier reading was above this floor. These are heuristics -- tune if the
# detected spike counts look implausible relative to known-good spike-count
# runs on core 0.
RESET_DROP_THRESHOLD = 500
WAS_ACTIVE_FLOOR = 500


def detect_spike_events(mp_trace, neuron_key):
    """Given a list of mp values for one neuron across a trial, return the
    count of detected spike (reset) events."""
    count = 0
    for i in range(1, len(mp_trace)):
        prev_mp = mp_trace[i - 1]
        curr_mp = mp_trace[i]
        if prev_mp >= WAS_ACTIVE_FLOOR and (prev_mp - curr_mp) >= RESET_DROP_THRESHOLD:
            count += 1
    return count


def main(core_id):
    print(f"=== Running with core_id={core_id} ===")
    with open(MODEL_CONFIG_PATH, "rb") as f:
        model_config = pickle.load(f)
    with open(TEST_BATCH_PATH, "rb") as f:
        test_batch = pickle.load(f)

    axons = model_config["axons"]
    connections = model_config["connections"]
    outputs = model_config["outputs"]

    for key in connections:
        neuron_obj = connections[key][1]
        neuron_obj.legacy_noise_en = 1

    if core_id == 0:
        network = CRI_network(axons=axons, connections=connections, outputs=outputs, target="CRI")
    else:
        # Assign every neuron to core_id via the 3-element connection tuple format
        connections_multicore = {
            k: (v[0], v[1], core_id) for k, v in connections.items()
        }
        network = CRI_network(
            axons=axons, connections=connections_multicore, outputs=outputs,
            target="CRI", coreID=None,
        )
    print(f"axon cores: {set(a.get_core() for a in network.connectome.get_axons())}")
    print(f"neuron cores: {set(n.get_core() for n in network.connectome.get_neurons())}")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0
    total = len(test_batch["images"])
    for img, label in zip(test_batch["images"], test_batch["labels"]):
        hs_bridge.FPGA_Execution.fpga_controller.clear(len(connections), False, core_id)

        img = img.to(device)
        mp_traces = {out_key: [] for out_key in outputs}

        for t in range(img.shape[0]):
            frame = img[t, :, :, :]
            input_ = frame.unsqueeze(0).flatten(start_dim=1).to(torch.int16)
            inputs = [f"A{i}" for i, elem in enumerate(input_[0, :]) if elem.item() > 0]

            mps = network.read_membrane(outputs)
            for out_key, mp_val in mps:
                mp_traces[out_key].append(mp_val)

            network.step(inputs)

        # extra timesteps to let propagation finish
        for i in range(4):
            mps = network.read_membrane(outputs)
            for out_key, mp_val in mps:
                mp_traces[out_key].append(mp_val)
            network.step([])

        spike_events = {out_key: detect_spike_events(trace, out_key) for out_key, trace in mp_traces.items()}
        print(f"Detected spike events (label={label}): {spike_events}")

        predicted = max(spike_events, key=spike_events.get)
        print(f"Predicted: {predicted}, Label: {label}")

        if predicted == label:
            correct += 1

        running_accuracy = 100 * correct / total
        print(f"Running accuracy : {running_accuracy:.2f} %")

    accuracy = 100 * correct / total
    print(f"Final accuracy (MP-based detection): {accuracy:.2f} %")
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"PASS: accuracy {accuracy:.2f}% >= {ACCURACY_THRESHOLD}%")
    else:
        print(f"Below threshold: accuracy {accuracy:.2f}% < {ACCURACY_THRESHOLD}% (may need to tune RESET_DROP_THRESHOLD/WAS_ACTIVE_FLOOR)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=int, default=0, help="Core ID to run on (0 = single-core)")
    args = parser.parse_args()
    main(args.core)
