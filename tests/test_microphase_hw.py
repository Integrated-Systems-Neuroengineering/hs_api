"""
test_microphase_hw.py -- test_max_number_axons from test_bitstream_hardware_fast.py, unchanged in body,
at sizes that cross the microphase boundary (512 rows of 16 neurons = 8,192): 10,000 reaches row 624,
16,000 row 999, 16,384 the full 1,024 rows.  Uses the fast suite's HIAER_CORE_ID core selection.
"""
import pytest
from hs_api.neuron_models import ANN_neuron
from tests.test_bitstream_hardware_fast import TestBitStream as _FastBase

class TestMicrophase(_FastBase):
    @pytest.mark.parametrize("numberAxons", [10000, 16000, 16384])
    def test_microphase_axons(self, setup_dictionaries, numberAxons):
        network, inputs, outputs = setup_dictionaries(
            numberAxons=numberAxons,
            numberNeurons=numberAxons,
            weight=1,
            neuron_model=ANN_neuron(0, shift=-17),
            fully_connect_axons_to_neurons=False)
        currSpikes1 = network.step(inputs)
        mp1 = network.read_membrane(outputs)
        currSpikes2 = network.step([])
        mp2 = network.read_membrane(outputs)
        mp1_dict, mp2_dict = dict(mp1), dict(mp2)
        assert len(currSpikes1[0]) == 0
        assert len(currSpikes2[0]) != 0
        bad = [i for i in range(numberAxons) if mp1_dict[f"N1.{i}"] != 1 or mp2_dict[f"N1.{i}"] != 0]
        assert not bad, "%d neurons wrong; first: %s (row = i//16, group = i%%16)" % (
            len(bad), [(i, i // 16, i % 16, mp1_dict[f"N1.{i}"], mp2_dict[f"N1.{i}"]) for i in bad[:6]])
