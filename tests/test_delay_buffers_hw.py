"""
test_delay_buffers_hw.py -- per-axon delay (CMD 14) and pending delayed inputs across a network load.

    F21  per-axon delay: axon row 0 delayed 3 steps through CMD 14; must equal F01 shifted 3 steps
    F22  a per-axon-delayed input still pending when a new network is loaded must not reach it
    F23  a per-synapse-delayed (64-bit) synapse still pending at a network load must not reach it

CMD 14 (command_interpreter.v): one delay-table entry per packet, byte 63 = 14, byte 62 = core << 3,
bits [18:6] = 13-bit table address, bits [5:0] = delay (0-63).  The table address is the AXON ROW
(16 axons per row, one event per row; axon_delay_buffer.v), so all 16 axons of a row share a delay.

Same output format and reference model as test_bio_features_hw.py (F00 runs first and measures the
read timing).  Run: HIAER_CORE_ID=1 python -m pytest tests/test_delay_buffers_hw.py -v -s
"""
import os

import numpy as np
import pytest

import hs_bridge.wrapped_dmadump.dmadump as d
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron
from hs_api import single_core_conductance_STDP as p2

from .test_bio_features_hw import BIG, _CORE_ID, _model, _record, _to_s32, read_offset  # noqa: F401 (fixture)


def axon_delay(core, row, delay):
    """CMD 14: set the delay of axon row `row` (16 axons) to `delay` timesteps (0-63)."""
    if not (0 <= row < 8192 and 0 <= delay <= 63):
        raise ValueError("row %d / delay %d out of range" % (row, delay))
    b = np.zeros(64, dtype=np.uint64)
    b[63] = 14
    b[62] = (core & 0x1F) << 3
    v = (row << 6) | delay
    b[0], b[1], b[2] = v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF
    d.dma_dump_write(b, 64, 1, 0, 0, 0, d.DmaMethodNormal)


def _net(syn, theta, syn_64bit=False, syn_delay=0):
    neuron = LIF_neuron(threshold=theta, shift=-17, leak=63)
    net = CRI_network(axons={ax: [("N0", w)] for ax, w in syn.items()}, connections={"N0": ([], neuron)},
                      outputs=["N0"], target="CRI", coreID=_CORE_ID, syn_64bit=syn_64bit, syn_delay=syn_delay)
    if syn_64bit:
        net.CRI.compiledNetwork.create_synapses(syn_64bit=True)
    return net


def _v(net):
    return _to_s32(net.read_membrane(["N0"])[0][1])


def _check(f, hw, model, offset):
    for t, v in enumerate(hw):
        m = t - offset
        expected = model[m] if 0 <= m < len(model) else 0
        if v != expected:
            _record(f, hw, model, offset, "FAIL")
            pytest.fail("%s on core %d: first difference at step %d: hardware %d, expected %d\n  hw    = %s\n"
                        "  model = %s" % (f["name"], _CORE_ID, t, v, expected, hw, model))
    _record(f, hw, model, offset, "PASS")


@pytest.fixture(scope="module", autouse=True)
def _delay_table_clean():
    """A killed earlier run can leave axon row 0 delayed on the card (CMD 14 persists until the next
    flash).  Put it back to 0 before and after this module, whatever happens in between."""
    axon_delay(_CORE_ID, 0, 0)
    yield
    axon_delay(_CORE_ID, 0, 0)


F21 = dict(id="F21", name="axon_delay_3", cmd13=dict(delta_mode=0, decay_ex=3354),
           syn={"A0": 1000}, sched={"A0": [0]}, theta=BIG + 21, steps=15, syn_delay=3)   # model: input at t + 3


def test_F21_axon_delay(read_offset):
    f = F21
    model, _ = _model(f)
    net = _net(f["syn"], f["theta"])
    hw = []
    try:
        axon_delay(_CORE_ID, 0, 3)
        p2.send_cmd13_psc_params(coreID=_CORE_ID, delta_mode=0)          # clear leftover state (as the bio suite)
        for _ in range(8):
            net.step([])
        p2.send_cmd13_psc_params(coreID=_CORE_ID, **f["cmd13"])
        for t in range(f["steps"]):
            net.step(["A0"] if t == 0 else [])
            hw.append(_v(net))
    finally:
        axon_delay(_CORE_ID, 0, 0)
        p2.send_cmd13_psc_params(coreID=_CORE_ID)
    _check(f, hw, model, read_offset)


def _carryover(f, first, second, delay_cmd):
    """Run `first` with a delayed input at step 0 for 3 steps (the delivery is due at step 6), load
    `second` and run it 10 steps with no input: every value must stay 0."""
    model = [0] * 10
    hw = []
    try:
        net1 = _net(**first)
        axon_delay(_CORE_ID, 0, 6 if delay_cmd else 0)
        p2.send_cmd13_psc_params(coreID=_CORE_ID, delta_mode=0, decay_ex=3354,
                                 syn_64bit_en=1 if first.get("syn_64bit") else 0)
        for t in range(3):
            net1.step(["A0"] if t == 0 else [])
        if delay_cmd:
            axon_delay(_CORE_ID, 0, 0)
        net2 = _net(**second)
        p2.send_cmd13_psc_params(coreID=_CORE_ID, delta_mode=0, decay_ex=3354,
                                 syn_64bit_en=1 if second.get("syn_64bit") else 0)
        for t in range(10):
            net2.step([])
            hw.append(_v(net2))
    finally:
        axon_delay(_CORE_ID, 0, 0)
        p2.send_cmd13_psc_params(coreID=_CORE_ID)
    return hw, model


def test_F22_axon_delay_carryover(read_offset):
    f = dict(id="F22", name="axon_delay_carryover")
    hw, model = _carryover(f, dict(syn={"A0": 1000}, theta=BIG + 22), dict(syn={"A0": 1000}, theta=BIG + 122), True)
    _check(f, hw, model, 0)


def test_F23_synapse_delay_carryover(read_offset):
    f = dict(id="F23", name="synapse_delay_carryover")
    hw, model = _carryover(f, dict(syn={"A0": 1000}, theta=BIG + 23, syn_64bit=True, syn_delay=6),
                           dict(syn={"A0": 1000}, theta=BIG + 123, syn_64bit=True, syn_delay=0), False)
    _check(f, hw, model, 0)
