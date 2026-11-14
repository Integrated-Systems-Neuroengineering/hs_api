"""test_delay_overflow_hw.py -- 1,100 axons each drive N1.0 through a 64-bit synapse of weight 1 and delay 5, all spiking at
step 0: 1,100 entries arrive in one delay slot, which holds 1,023 (fix 25).  N1.0 (threshold far above) must receive
nothing at steps 0-4 and exactly 1,023 at step 5; the status register must show bit 18 (delay-slot overflow) and not
bit 19.  Needs the compiler slot fix (empty 64-bit slots carry no delay), else empty slots also take delayed entries."""
import os
from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron
import hs_api.single_core_conductance_STDP as p2
from rec29_common import read_status
CORE = int(os.environ.get("HIAER_CORE_ID", 1)); N, DELAY, STEPS = 1100, 5, 8
def test_delay_overflow():
    model = ANN_neuron(4 * N, shift=-17)
    axons = {f"A{i}": [("N1.0", 1)] for i in range(N)}
    net = CRI_network(axons=axons, connections={"N1.0": ([], model)}, outputs=["N1.0"], target="CRI", coreID=CORE,
                      syn_64bit=True, syn_delay=DELAY)
    p2.send_cmd13_psc_params(coreID=CORE, delta_mode=1, syn_64bit_en=1)
    mps = []
    try:
        for s in range(STEPS):
            net.step([f"A{i}" for i in range(N)] if s == 0 else [])
            mps.append(dict(net.read_membrane(["N1.0"]))["N1.0"])
    finally:
        es, fw = read_status(CORE)
        p2.send_cmd13_psc_params(coreID=CORE)
    print("N1.0 per step: %s; error_status %s; feature word %s" % (mps, None if es is None else hex(es), None if fw is None else hex(fw)))
    assert mps[:DELAY] == [0] * DELAY, "input arrived before the %d-step delay: %s" % (DELAY, mps)
    assert mps[DELAY] == 1023, "N1.0 at step %d: %s, expected 1023 (a full slot)" % (DELAY, mps[DELAY])
    assert es is not None and (es >> 18) & 1 == 1 and (es >> 19) & 1 == 0, "status: %s (expected bit 18 set, bit 19 clear)" % (None if es is None else hex(es))
