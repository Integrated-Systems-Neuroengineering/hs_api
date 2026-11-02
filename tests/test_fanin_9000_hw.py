"""test_fanin_9000_hw.py -- 9,000 axons (two input microphases: 8,192 + 808, the second input-only) each drive N1.0 with
weight 1; N1.0's threshold is far above 9,000, so it never fires.  After step 0 its membrane must read exactly 9,000 --
every pointer of the input-only microphase delivered (fix 26 waits on the input processor there) -- and the status
register (CMD 10) must show none of bits 16-19 (no watchdog, bypass, delay-slot overflow or Phase-2 fallback)."""
import os
from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron
from rec29_common import read_status
CORE = int(os.environ.get("HIAER_CORE_ID", 1)); N = 9000
def test_fanin_9000():
    model = ANN_neuron(4 * N, shift=-17)
    axons = {f"A{i}": [("N1.0", 1)] for i in range(N)}
    net = CRI_network(axons=axons, connections={"N1.0": ([], model)}, outputs=["N1.0"], target="CRI", coreID=CORE)
    net.step([f"A{i}" for i in range(N)])
    mp = dict(net.read_membrane(["N1.0"]))["N1.0"]
    es, fw = read_status(CORE)
    print("N1.0 after step 0: %s; error_status %s; feature word %s" % (mp, None if es is None else hex(es), None if fw is None else hex(fw)))
    assert mp == N, "N1.0 membrane %s, expected %d" % (mp, N)
    assert es is not None and (es >> 16) & 0xF == 0, "status bits 16-19 set: %s" % (None if es is None else hex(es))
