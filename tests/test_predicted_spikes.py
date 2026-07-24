"""
Self-checking 2-neuron spike test.

Network: 1 axon (A0) -> N0 -> N1, on a configurable core.
A0 fires at t=0 with weight 2000 (threshold 1000), so N0 is PREDICTED to spike
at t=0 and N1 is PREDICTED to spike at t=1 (one step after N0 drives it).

The test reads back hardware spikes at each timestep and compares against the
prediction. If a predicted spike is not observed, it prints a FAILURE message
and reports the mismatch. Also prints N0 and N1's hbmIdx for reference.

Usage:
    python3 test_predicted_spikes.py [--core CORE_ID]

Run with no --core (defaults to 0, single-core / L6m) or --core 5 (multicore).
"""
import argparse
import hs_bridge
from hs_api.api import CRI_network
from hs_api.neuron_models import IF_neuron

THRESH = 1000
WEIGHT = 2000


def main(core_id):
    axons = {"A0": [("N0", WEIGHT)]}
    if core_id == 0:
        # Plain 2-tuple connections for single-core / L6m (no manual core override)
        connections = {
            "N0": ([("N1", WEIGHT)], IF_neuron(THRESH, nu=0)),
            "N1": ([], IF_neuron(THRESH, nu=0)),
        }
        net = CRI_network(axons=axons, connections=connections, outputs=["N0", "N1"], target="CRI")
    else:
        # 3-tuple connections with manual core assignment for multicore
        connections = {
            "N0": ([("N1", WEIGHT)], IF_neuron(THRESH, nu=0), core_id),
            "N1": ([], IF_neuron(THRESH, nu=0), core_id),
        }
        net = CRI_network(axons=axons, connections=connections, outputs=["N0", "N1"], target="CRI", coreID=None)

    n0 = net.connectome.get_neuron_by_key("N0")
    n1 = net.connectome.get_neuron_by_key("N1")
    print(f"N0 hbmIdx: {n0.get_hbmIdx()}")
    print(f"N1 hbmIdx: {n1.get_hbmIdx()}")

    # Predicted spike timestep -> expected neuron key.
    # IMPORTANT: there is a consistent, reproducible one-execution-cycle lag
    # between when execute() is called and when the resulting spike is
    # observable via flush_spikes. This was confirmed on BOTH L6m (core 0) and
    # multicore (core 5) -- it is an inherent hardware/pipeline characteristic,
    # not a bug. It's invisible in tests that accumulate spike counts across
    # many timesteps (a uniform shift doesn't change the winning count), which
    # is why it wasn't noticed before. Predictions below are lag-adjusted.
    #
    # What IS core-5/multicore-specific: across repeated runs, core 5 often
    # shows NO spike at all (not just delayed) -- 5 of 6 observed runs had
    # zero spikes; only 1 run showed the expected lag-1 pattern. L6m (core 0)
    # showed the lag-1 pattern consistently on every run. This flakiness, not
    # the lag itself, is the real anomaly to report.
    predicted = {1: "N0", 2: "N1"}

    num_steps = 4
    failures = []
    for t in range(num_steps):
        inputs = ["A0"] if t == 0 else []
        result = net.step(inputs)
        # step() returns either a bare list (single-core path) or a
        # (spikeList, latency, hbmAcc) tuple (multicore path) -- normalize.
        spikes = result[0] if isinstance(result, tuple) else result

        expected = predicted.get(t)
        if expected is not None:
            if expected in spikes:
                print(f"t={t}: PASS - {expected} spiked as predicted (spikes={spikes})")
            else:
                msg = f"t={t}: FAILURE - expected {expected} to spike, but spikes={spikes}"
                print(msg)
                failures.append(msg)
        else:
            print(f"t={t}: spikes={spikes} (no prediction for this timestep)")

    print()
    if failures:
        print(f"TEST FAILED: {len(failures)} of {len(predicted)} predicted spikes were not observed.")
        for f in failures:
            print(f"  {f}")
    else:
        print("TEST PASSED: all predicted spikes were observed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=int, default=0, help="Core ID to run the network on (0 = single-core/L6m)")
    args = parser.parse_args()
    main(args.core)