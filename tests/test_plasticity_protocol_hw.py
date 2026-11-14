"""test_plasticity_protocol_hw.py -- plasticity-on protocol (round 14 recording; on a plasticity bitstream the card's
spikes per step and read-back rows are saved too, for hw2tx.py + plast_check.py).
Pre spikes: inputs to pre axons at seeded random steps.  Post spikes: whatever the neurons actually do (each post has a
driver axon, weight 1000, threshold 300; the checker takes post and neuron-pre spike times from the replay's reports).
Plastic synapses under five rules (soft triplet, anti-Hebbian, Vogels-like with the pre-spike constant, eligibility
keep / reset), rewards after steps 25, 50, 70; neuromod_level 160.  At the end every plastic HBM row is read back over
CMD 2.  The protocol (hardware indices, rules, tables, schedule, rewards, layout, rows read) goes to $PLAST_OUT.
PLAST_RECORD=1: RECORDING ONLY on a bitstream without plasticity -- the feature word is overridden so the recording
contains the plasticity load and opcode 18, and stdp_enable is sent as 0 (the replay copy of the recording sets it)."""
import os, json, random
import numpy as np
from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron
from hs_api.single_core_conductance_STDP import send_cmd13_psc_params
import hs_bridge.FPGA_Execution.fpga_controller as fc
from hs_bridge.FPGA_Execution import plasticity as PL
CORE = int(os.environ.get("HIAER_CORE_ID", 1))
RECORD = os.environ.get("PLAST_RECORD") == "1"
OUT = os.environ.get("PLAST_OUT", "plasticity_protocol.json")
STEPS, G = 80, 160
REWARDS = {25: 256, 50: -192, 70: 384}                     # after step s, Q8.8
class Rule:
    """Same fields and reciprocal as stdp_ref.Rule (v3)."""
    def __init__(self, A2p, A3p, A2m, w_min, w_max, soft=True, A3m=0, A0=0, s2p=0, s3p=0, s2m=0, s3m=0, s0=0,
                 elig=False, eta=256, elig_reset=False, e_shift=0, lut_p=0, lut_m=1, lut_y=2, lut_x=3, lut_e=4):
        self.__dict__.update(locals()); del self.__dict__['self']
        self.inv_range = min(int(round(2 ** 32 / (w_max - w_min))), 2 ** 32 - 1)
q = lambda u: int(round(u * 256))
RULES = {1: Rule(q(20), q(30), q(25), -250, 250, soft=True, A3m=q(8)),
         2: Rule(q(15), 0, q(15), -250, 250, soft=False, s2p=1, s2m=1),
         3: Rule(q(10), 0, q(10), -250, 250, soft=False, A0=q(3), s2m=1, s0=1, lut_p=5, lut_m=5),
         4: Rule(q(20), q(10), q(20), -250, 250, soft=True, elig=True, eta=256, e_shift=0, lut_e=4),
         5: Rule(q(25), 0, q(15), -250, 250, soft=False, elig=True, eta=128, elig_reset=True, e_shift=0, lut_e=4)}
TABLES_SPEC = {0: ("trace", 16.8, 0), 1: ("trace", 33.7, 0), 2: ("trace", 40.0, 0), 3: ("trace", 714.0, 0),
               4: ("decay", 100.0, 0), 5: ("trace", 20.0, 0)}
def tables():
    return {t: (PL.trace_table(tau) if k == "trace" else PL.decay_table(tau, sh)) for t, (k, tau, sh) in TABLES_SPEC.items()}
def read_row(row):
    import hs_bridge.wrapped_dmadump.dmadump as dmadump
    a = '0' + np.binary_repr(row, 23)
    cmd = [2, (CORE & 0x1F) << 3] + [0] * 27 + [int(a[:8], 2), int(a[8:16], 2), int(a[16:], 2)] + [0] * 32
    pkt = np.flip(np.array(cmd, dtype=np.uint64))
    dmadump.dma_dump_write(pkt, len(pkt), 1, 0, 0, 0, dmadump.DmaMethodNormal)
    for _ in range(2000):                                      # the reply is the next BB packet
        code, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
        bts = bytes(int(x) & 0xFF for x in list(data)[:64])
        if len(bts) == 64 and bts[63] == 0xBB and bts[62] == 0xBB:
            return int.from_bytes(bts[:32], "little")
    raise RuntimeError("no BB reply for row %d" % row)
def test_plasticity_protocol():
    if RECORD:                                                # recording only: see the module docstring
        fake = lambda *a, **k: 0xF10A001F
        fc.bitstream_features = fake
        import importlib
        hn = importlib.import_module("hs_bridge.network")    # the module: hs_bridge re-exports its class 'network'
        hn.bitstream_features = fake
    r = random.Random(14)
    model = ANN_neuron(300, shift=-17)
    NP = 24
    connections = {f"N1.{i}": ([], model) for i in range(NP)}
    axons = {f"D{i}": [(f"N1.{i}", 1000)] for i in range(NP)}
    syn = []                                                  # (pre key, post key, rule, w0)
    for k in range(12):
        for post, rule in ((k, 1 + k % 5), (16 + k % 8, 1 + (k + 2) % 5)):
            syn.append((f"P{k}", f"N1.{post}", rule, r.randint(-100, 100)))
    for i in range(4):
        syn.append((f"N1.{i}", f"N1.{20 + i}", 1 + i % 2, r.randint(-100, 100)))
    for pre, post, rule, w0 in syn:
        entry = (post, w0, False, rule)
        if pre.startswith("P"): axons.setdefault(pre, []).append(entry)
        else: connections[pre][0].append(entry)
    outputs = [f"N1.{i}" for i in range(NP)]
    net = CRI_network(axons=axons, connections=connections, outputs=outputs, target="CRI", coreID=CORE,
                      syn_64bit=True, plasticity={'rules': RULES, 'tables': tables()})
    send_cmd13_psc_params(None, delta_mode=1, syn_64bit_en=1, neuromod_level=G, stdp_enable=0 if RECORD else 1, coreID=CORE)
    hw = {k: net.connectome.connectomeDict[k].get_coreTypeIdx() for k in list(axons) + list(connections)}
    schedule, hw_spikes = [], []
    for s in range(STEPS):
        fire = [f"D{i}" for i in range(NP) if r.random() < 0.12] + [f"P{k}" for k in range(12) if r.random() < 0.2]
        schedule.append(fire)
        res = net.step(fire)
        hw_spikes.append(sorted(hw[k] for k in (res[0] if isinstance(res, tuple) else res)))
        if s in REWARDS: net.reward(REWARDS[s])
    bases = net.CRI.plasticity_bases[CORE]
    rows = list(range(bases['PLASTIC_START'], bases['PLASTIC_END']))
    hw_rows = ["%064x" % read_row(row) for row in rows]
    json.dump({"core": CORE, "steps": STEPS, "g": G, "rewards": REWARDS, "record_mode": RECORD,
               "rules": {s: {k: v for k, v in rl.__dict__.items()} for s, rl in RULES.items()},
               "tables": TABLES_SPEC, "hw": hw,
               "synapses": [{"pre": p, "post": q_, "rule": ru, "w0": w0} for p, q_, ru, w0 in syn],
               "schedule": schedule, "bases": bases, "rows_read": rows,
               "hw_spikes": hw_spikes, "hw_rows": hw_rows}, open(OUT, "w"), indent=1)
