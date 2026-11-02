"""test_syn64_isolated_hw.py -- isolated firing (round 19): 32 neurons, each with its own driver axon (weight 1000,
threshold 300) and no other synapse.  Driver n fires only at step n, so neuron n must spike exactly at step n+1.
test_isolated_64: 64-bit synapses, plus one inert plastic synapse (an axon that never fires -> N1.31, weight 0) so the
plasticity controller is configured and its spike captures run.  test_isolated_32: the same network, 32-bit, control.  Not recording: each test asserts neuron n spiked exactly at n+1.
Writes hardware indices and the schedule to $ISO_OUT.  PLAST_RECORD=1: recording only (feature word overridden,
stdp_enable sent as 0 -- the replay copy sets it -- and no waiting for step-done packets: only the commands matter)."""
import os, json
from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron
from hs_api.single_core_conductance_STDP import send_cmd13_psc_params
import hs_bridge.FPGA_Execution.fpga_controller as fc
from hs_bridge.FPGA_Execution import plasticity as PL
CORE = int(os.environ.get("HIAER_CORE_ID", 1)); RECORD = os.environ.get("PLAST_RECORD") == "1"
N, STEPS = 32, 36
class Rule:
    def __init__(self, A2p, A3p, A2m, w_min, w_max, soft=True, A3m=0, A0=0, s2p=0, s3p=0, s2m=0, s3m=0, s0=0,
                 elig=False, eta=256, elig_reset=False, e_shift=0, lut_p=0, lut_m=1, lut_y=2, lut_x=3, lut_e=4):
        self.__dict__.update(locals()); del self.__dict__['self']
        self.inv_range = min(int(round(2 ** 32 / (w_max - w_min))), 2 ** 32 - 1)
def _run(syn64):
    if RECORD:                     # recording only: the commands are what is recorded, not the card's answers
        fake = lambda *a, **k: 0xF10A001F
        import importlib; hn = importlib.import_module("hs_bridge.network")
        fc.bitstream_features = fake; hn.bitstream_features = fake
        nowait = lambda *a, **k: ([], 0, 0)        # flush_spikes only reads; a card without these features never
        fc.flush_spikes = nowait; hn.flush_spikes = nowait   # sends the step-done packet, so do not wait for it
    model = ANN_neuron(300, shift=-17)
    connections = {f"N1.{i}": ([], model) for i in range(N)}
    axons = {f"D{i}": [(f"N1.{i}", 1000)] for i in range(N)}
    kw = {}
    if syn64:
        axons["X"] = [("N1.31", 0, False, 1)]                     # inert plastic synapse: X never fires
        kw = dict(syn_64bit=True, plasticity={'rules': {1: Rule(5120, 0, 5120, -250, 250)},
                                              'tables': {t: PL.trace_table(tau) for t, tau in enumerate((16.8, 33.7, 40.0, 714.0))}})
    net = CRI_network(axons=axons, connections=connections, outputs=[f"N1.{i}" for i in range(N)], target="CRI", coreID=CORE, **kw)
    send_cmd13_psc_params(None, delta_mode=1, syn_64bit_en=1 if syn64 else 0, stdp_enable=1 if (syn64 and not RECORD) else 0, coreID=CORE)
    hw = {k: net.connectome.connectomeDict[k].get_coreTypeIdx() for k in list(axons) + list(connections)}
    sched = [[f"D{s}"] if s < N else [] for s in range(STEPS)]
    fired = {}
    for s in range(STEPS):
        out = net.step(sched[s])
        if not RECORD:
            for k in (out[0] if isinstance(out, tuple) else out): fired.setdefault(k, []).append(s)
    send_cmd13_psc_params(None, delta_mode=1, syn_64bit_en=0, stdp_enable=0, coreID=CORE)
    out = os.environ.get("ISO_OUT")
    if out: json.dump({"syn64": syn64, "steps": STEPS, "n": N, "core": CORE, "hw": hw, "schedule": sched}, open(out, "w"), indent=1)
    if not RECORD:                  # on a bitstream with the fixes: neuron n spikes exactly once, at step n+1
        bad = {f"N1.{n}": fired.get(f"N1.{n}") for n in range(N) if fired.get(f"N1.{n}") != [n + 1]}
        extra = sorted(k for k in fired if not k.startswith("N1.") or int(k[3:]) >= N)
        assert not bad and not extra, "isolated firing (%s-bit) wrong: %s, unexpected %s" % (64 if syn64 else 32, dict(list(bad.items())[:8]), extra[:8])
def test_isolated_64(): _run(True)
def test_isolated_32(): _run(False)
