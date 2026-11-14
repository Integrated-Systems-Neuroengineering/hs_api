"""
test_bio_features_hw.py -- hardware tests of the biological neuron model.

Each test runs a one-neuron network (axon A0 -> neuron N0, plus A1 in F04), configures the
core with CMD 13, steps it, reads N0's membrane potential after every step, and compares
every value with tests/hw_ref_model.py, which implements the neuron update of
internal_events_processor.v exactly.  A test passes only if every step matches.

    F00  control, legacy delta mode (also measures the read timing used by all others)
    F01  exponential PSC, excitatory (CUBA)
    F02  decay-constant effect
    F03  inhibitory PSC (I_in)
    F04  dual time constant, excitatory + inhibitory on one neuron
    F05  COBA conductance synapses
    F06  AdEx spike-frequency adaptation
    F07  neuromodulation, excitability bias +100
    F08  neuromodulation, excitability bias -100
    F09  refractory period
    F10  soft reset
    F11  64-bit synapse format (must reproduce F01)
    F12  per-synapse delay of 2 steps

Run from the hs_api checkout with the hardware environment active:
    HIAER_CORE_ID=1 python -m pytest tests/test_bio_features_hw.py -v
Optional: HIAER_BIO_TRACES=<file.json> writes the hardware and reference traces of every test.
See tests/HARDWARE_TESTS.md.
"""
import json
import os

import pytest

from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron
from hs_api import single_core_conductance_STDP as p2

try:
    from .hw_ref_model import simulate
except ImportError:  # run as a plain script from tests/
    from hw_ref_model import simulate

_CORE_ID = int(os.environ.get("HIAER_CORE_ID", 0))
_TRACES = os.environ.get("HIAER_BIO_TRACES")

BIG = 1_000_000
EVERY = lambda n: list(range(n))

# Every test uses a different threshold: a threshold change makes the core clear all
# neuron state (V and the synaptic currents) on the next step, so nothing carries over.
CONTROL = dict(id="F00", name="control_delta_mode", cmd13=dict(delta_mode=1),
               syn={"A0": 1000}, sched={"A0": [0]}, theta=BIG + 0, steps=8)

FEATURES = [
    dict(id="F01", name="exp_psc_excitatory", cmd13=dict(delta_mode=0, decay_ex=3354),
         syn={"A0": 1000}, sched={"A0": [0]}, theta=BIG + 1, steps=12),
    dict(id="F02", name="decay_constant", cmd13=dict(delta_mode=0, decay_ex=2048),
         syn={"A0": 1000}, sched={"A0": [0]}, theta=BIG + 2, steps=12),
    dict(id="F03", name="inhibitory_psc", cmd13=dict(delta_mode=0, decay_in=3354),
         syn={"A0": -600}, sched={"A0": [0]}, theta=BIG + 3, steps=12),
    dict(id="F04", name="dual_tau_exc_inh", cmd13=dict(delta_mode=0, decay_ex=3800, decay_in=1500),
         syn={"A0": 800, "A1": -400}, sched={"A0": [0], "A1": [0]}, theta=BIG + 4, steps=14),
    dict(id="F05", name="coba", cmd13=dict(delta_mode=0, decay_ex=3354, coba_mode=1, E_ex=1500, E_in=-1500),
         syn={"A0": 1000}, sched={"A0": [0]}, theta=BIG + 5, steps=12),
    dict(id="F06", name="adex_adaptation", cmd13=dict(delta_mode=0, decay_ex=2048, decay_w=128, delta_w=40),
         syn={"A0": 300}, sched={"A0": EVERY(30)}, theta=600, steps=30),
    dict(id="F07", name="neuromod_bias_plus100", cmd13=dict(delta_mode=0, decay_ex=2048, neuromod_excitability_bias=100),
         syn={"A0": 300}, sched={"A0": EVERY(30)}, theta=1000, steps=30),
    dict(id="F08", name="neuromod_bias_minus100", cmd13=dict(delta_mode=0, decay_ex=2048, neuromod_excitability_bias=-100),
         syn={"A0": 300}, sched={"A0": EVERY(30)}, theta=1001, steps=30),
    dict(id="F09", name="refractory_period", cmd13=dict(delta_mode=0, decay_ex=2048),
         syn={"A0": 300}, sched={"A0": EVERY(30)}, theta=602, steps=30, refractory_max=3),
    dict(id="F10", name="soft_reset", cmd13=dict(delta_mode=0, decay_ex=2048),
         syn={"A0": 300}, sched={"A0": EVERY(30)}, theta=603, steps=30, soft_reset=True),
    dict(id="F11", name="syn64_no_delay", cmd13=dict(delta_mode=0, decay_ex=3354, syn_64bit_en=1),
         syn={"A0": 1000}, sched={"A0": [0]}, theta=BIG + 11, steps=12, syn_64bit=True, syn_delay=0),
    dict(id="F12", name="syn64_delay_2", cmd13=dict(delta_mode=0, decay_ex=3354, syn_64bit_en=1),
         syn={"A0": 1000}, sched={"A0": [0]}, theta=BIG + 12, steps=14, syn_64bit=True, syn_delay=2),
]


def _to_s32(v):
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _model(f):
    ins = {}
    for ax, steps in f["sched"].items():
        for t in steps:
            ins.setdefault(t + f.get("syn_delay", 0), []).append(f["syn"][ax])
    p = dict(f["cmd13"])
    p.update(threshold=f["theta"], refractory_max=f.get("refractory_max", 0),
             soft_reset=f.get("soft_reset", False),
             neuromod_bias=f["cmd13"].get("neuromod_excitability_bias", 0))
    return simulate(f["steps"], ins, p)


def _hardware(f):
    axons = {ax: [("N0", w)] for ax, w in f["syn"].items()}
    neuron = LIF_neuron(threshold=f["theta"], shift=-17, leak=63,
                        refractory_max=f.get("refractory_max", 0),
                        soft_reset_en=f.get("soft_reset", False))
    net = CRI_network(axons=axons, connections={"N0": ([], neuron)}, outputs=["N0"],
                      target="CRI", coreID=_CORE_ID,
                      syn_64bit=f.get("syn_64bit", False), syn_delay=f.get("syn_delay", 0))
    if f.get("syn_64bit"):
        net.CRI.compiledNetwork.create_synapses(syn_64bit=True)  # write synapses again in 64-bit format
    p2.send_cmd13_psc_params(coreID=_CORE_ID, **f["cmd13"])
    hw, spikes = [], []
    try:
        for t in range(f["steps"]):
            res = net.step([ax for ax, steps in f["sched"].items() if t in steps])
            spikes.append("N0" in (res[0] if isinstance(res, tuple) else res))
            hw.append(_to_s32(net.read_membrane(["N0"])[0][1]))
    finally:
        p2.send_cmd13_psc_params(coreID=_CORE_ID)  # back to the reset defaults (delta_mode=1)
    return hw, spikes


def _record(f, hw, model, offset, status):
    print("\n%s %s core %d: %s\n  hw    = %s\n  model = %s" % (f["id"], f["name"], _CORE_ID, status, hw, model))
    if not _TRACES:
        return
    data = json.load(open(_TRACES)) if os.path.exists(_TRACES) else {"core": _CORE_ID, "tests": {}}
    data["tests"][f["id"] + "_" + f["name"]] = dict(status=status, read_offset=offset, hw=hw, model=model)
    json.dump(data, open(_TRACES, "w"), indent=1)


@pytest.fixture(scope="module")
def read_offset():
    """Run the legacy-mode control once and measure the read timing."""
    model, _ = _model(CONTROL)
    hw, _ = _hardware(CONTROL)
    target = model[-1]
    first_hw = next((t for t, v in enumerate(hw) if v == target), None)
    first_m = next((t for t, v in enumerate(model) if v == target), None)
    if first_hw is None:
        _record(CONTROL, hw, model, None, "FAIL")
        pytest.fail("control on core %d never reached V=%d: hw=%s. The remaining tests cannot be "
                    "judged on this core." % (_CORE_ID, target, hw))
    offset = first_hw - first_m
    _record(CONTROL, hw, model, offset, "PASS")
    return offset


def test_F00_control_delta_mode(read_offset):
    assert read_offset in (0, 1), "unexpected read timing offset %d" % read_offset


@pytest.mark.parametrize("f", FEATURES, ids=[f["id"] + "_" + f["name"] for f in FEATURES])
def test_feature(f, read_offset):
    model, _ = _model(f)
    hw, _ = _hardware(f)
    for t, v in enumerate(hw):
        m = t - read_offset
        expected = model[m] if 0 <= m < len(model) else 0
        if v != expected:
            _record(f, hw, model, read_offset, "FAIL")
            pytest.fail("%s on core %d: first difference at step %d: hardware %d, expected %d\n"
                        "  hw    = %s\n  model = %s" % (f["name"], _CORE_ID, t, v, expected, hw, model))
    _record(f, hw, model, read_offset, "PASS")
