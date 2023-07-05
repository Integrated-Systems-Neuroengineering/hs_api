import pytest

from hs_api.api import *

@pytest.mark.parametrize(
    ('input_pm','expected'),
    (
        (36,'bad perturbMag'),
        (-1, 'bad perturbMag'),
        (2., 'bad perturbMag'),
    ),
)
def test_perturbMag_parsing(input_pm,expected):
    config = {}
    config['neuron_type'] = "I&F"
    config['global_neuron_params'] = {}
    config['global_neuron_params']['v_thr'] = 9

    inputs = {}
    for i in range(100):
        inputs[i] = ['alpha']

    print(inputs)

    #Define an axons dictionary
    axons = {'alpha': [('a', 1.0),('b', 2.0),('c', 3.0),('d', 4.0),('e',5.0)]}

    #Define a connections dictionary
    connections = {
                'a': [('z',4)],
                'b': [('y',4)],
                'c': [('x',4)],
                'd': [('w',4)],
                'e': [('v',4)],
                'v': [],
                'w': [],
                'x': [],
                'y': [],
                'z': []}

    outputs = ['z']

    with pytest.raises(perturbMagError) as execinfo:
        CRI_network(axons,connections,config, outputs, target = 'simpleSim', simDump = False, coreID=0, perturbMag = input_pm)
    msg, = execinfo.value.args
    assert msg == expected
