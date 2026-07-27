"""
Minimal cross-core test.

Network: 1 axon (A0) -> N0 (on SOURCE_CORE) -> N1 (on DEST_CORE).
Since N0 and N1 are on DIFFERENT cores, the N0->N1 synapse becomes an
INTER_CORE synapse (opcode 001) routed through a relay axon on DEST_CORE,
rather than a LOCAL synapse.

A0 fires at t=0 with weight 2000 (threshold 1000). Based on the confirmed
one-execution-cycle lag (see test_predicted_spikes.py), N0 is expected to
spike/reset around t=1, and (if cross-core routing works) N1 should show
its MP accumulate on DEST_CORE some time after that.

This test is exploratory: it prints N0's MP (read on SOURCE_CORE) and N1's MP
(read on DEST_CORE) at every timestep so you can see exactly what's
happening, rather than asserting pass/fail up front.

Usage:
    python3 test_cross_core.py [--source-core N] [--dest-core M]
"""
import argparse
import hs_bridge
import hs_bridge.FPGA_Execution.fpga_controller as fc
from hs_api.api import CRI_network
from hs_api.neuron_models import IF_neuron

THRESH = 1000
WEIGHT = 2000


def main(source_core, dest_core):
    axons = {"A0": [("N0", WEIGHT)]}
    connections = {
        "N0": ([("N1", WEIGHT)], IF_neuron(THRESH, nu=0), source_core),
        "N1": ([], IF_neuron(THRESH, nu=0), dest_core),
    }
    outputs = ["N0", "N1"]

    net = CRI_network(axons=axons, connections=connections, outputs=outputs, target="CRI", coreID=None)

    n0 = net.connectome.get_neuron_by_key("N0")
    n1 = net.connectome.get_neuron_by_key("N1")
    print(f"N0: core={n0.get_core()}, coreTypeIdx={n0.get_coreTypeIdx()}, hbmIdx={n0.get_hbmIdx()}")
    print(f"N1: core={n1.get_core()}, coreTypeIdx={n1.get_coreTypeIdx()}, hbmIdx={n1.get_hbmIdx()}")
    print(f"axon cores: {set(a.get_core() for a in net.connectome.get_axons())}")
    print(f"neuron cores: {set(nn.get_core() for nn in net.connectome.get_neurons())}")

    # Confirm whether a relay axon actually got created
    all_axon_keys = [a.get_user_key() for a in net.connectome.get_axons()]
    relay_axons = [k for k in all_axon_keys if "RAx" in k]
    print(f"relay axons found: {relay_axons}")

    for step in range(5):
        inputs = ["A0"] if step == 0 else []
        spikes = net.step(inputs)
        n0_mp = fc.readSelect([n0.get_hbmIdx()], coreID=source_core)
        n1_mp = fc.readSelect([n1.get_hbmIdx()], coreID=dest_core)
        print(
            f"step {step}: inputs={inputs} "
            f"N0(core{source_core})_MP={n0_mp[0][3] if n0_mp else 'N/A'} "
            f"N1(core{dest_core})_MP={n1_mp[0][3] if n1_mp else 'N/A'} "
            f"spikes={spikes[0] if isinstance(spikes, tuple) else spikes}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-core", type=int, default=5)
    parser.add_argument("--dest-core", type=int, default=6)
    args = parser.parse_args()
    main(args.source_core, args.dest_core)
