"""CRI membrane readout names the right neuron on every core.

run_step returns each core's membrane values one core after another, each
tagged with its core, in hardware (hbmIdx) order; hbmIdx restarts at 0 on
every core.  No hardware needed:

    python -m pytest tests/test_membrane_keys.py
"""
import random
import types

import pytest

from connectome_utils.connectome import connectome, neuron
from hs_api.api import CRI_network


def placed_network(n_neurons, n_cores, seed=0):
    """A connectome laid out the way CRI_network lays it out: cores assigned,
    models padded, per-core hbmIdx from get_class_ordered_list."""
    neuron.reset_count()
    rng = random.Random(seed)
    c = connectome()
    for i in range(n_neurons):
        c.addNeuron(neuron("n%d" % i, "neuron", neuronModel=rng.choice(["LIF_a", "LIF_b"])))
    for i in range(n_neurons):
        c.get_neuron_by_key("n%d" % i).set_core(rng.randrange(n_cores))
    c.pad_models()
    return c, c.get_class_ordered_list()


def readout(ordered):
    """What run_step hands back: per core, in hbmIdx order, (..., potential, ..., core)."""
    out = []
    for core in sorted({n.get_core() for _, n in ordered}):
        on_core = sorted((n for _, n in ordered if n.get_core() == core), key=lambda n: n.get_hbmIdx())
        for n in on_core:
            # read_membranes' fields, with the potential at [3]; the test potential
            # encodes the neuron so a wrong match shows up
            out.append((0, 0, 0, "v_" + n.get_user_key(), core))
    return out


@pytest.mark.parametrize("n_cores", [1, 2, 4, 16])
def test_every_value_names_its_own_neuron(n_cores):
    c, ordered = placed_network(300, n_cores, seed=n_cores)
    net = types.SimpleNamespace(connectome=c)
    keyed = CRI_network._membrane_keys(net, readout(ordered))
    assert len(keyed) == len(ordered)
    assert all(value == "v_" + key for key, value in keyed)
    assert {key for key, _ in keyed} == {key for key, _ in ordered}


def test_one_core_matches_the_old_numbering():
    """With one core the old code (position across all values -> hbmIdx) was
    right; the new one gives the same keys."""
    c, ordered = placed_network(200, 1)
    out = readout(ordered)
    old = [(c.get_neuron_by_hbmIdx(idx).get_user_key(), data[3]) for idx, data in enumerate(out)]
    assert CRI_network._membrane_keys(types.SimpleNamespace(connectome=c), out) == old


def test_a_value_past_the_last_neuron_is_an_error():
    c, ordered = placed_network(64, 2)
    out = readout(ordered) + [(0, 0, 0, "extra", 1)]
    with pytest.raises(RuntimeError, match="belongs to no neuron"):
        CRI_network._membrane_keys(types.SimpleNamespace(connectome=c), out)
