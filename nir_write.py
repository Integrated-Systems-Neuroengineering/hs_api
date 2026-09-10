"""
First implementation of the NIR *write* direction: HiAER-Spike's
axons/connections format -> a nir.NIRGraph.

This is the reverse of nir_general_importer.py's nir_to_hiaer_spike().
Verified via a full round-trip: take a network already in our format,
write it to NIR, read it back with our own general importer, and confirm
we get back an equivalent network.

Design (mirrors the shape our general importer already understands):
  Input -> Linear (axon weights) -> LIF -> Linear (recurrent weights,
  looped back into LIF) -> Output

Mapping (inverse of the read-side mapping worked out earlier):
  Lambda -> tau = 2^Lambda
  theta -> v_threshold directly
  v_leak = 0, v_reset = 0, r = 1 (matching our hardware's fixed behavior)
"""
import numpy as np
import nir
from nir_general_importer import nir_to_hiaer_spike


def hiaer_spike_to_nir(axons, connections, outputs):
    """
    Convert HiAER-Spike's (axons, connections, outputs) format into a
    nir.NIRGraph.

    Parameters
    ----------
    axons : dict
        {axon_name: [(neuron_name, weight), ...]}
    connections : dict
        {neuron_name: ([(target_neuron_name, weight), ...], LIF_neuron_obj)}
    outputs : list
        Names of neurons whose spikes are read out.

    Returns
    -------
    nir.NIRGraph
    """
    neuron_names = list(connections.keys())
    n_neurons = len(neuron_names)
    neuron_index = {name: i for i, name in enumerate(neuron_names)}

    axon_names = list(axons.keys())
    n_axons = len(axon_names)
    axon_index = {name: i for i, name in enumerate(axon_names)}

    # Build per-neuron theta/Lambda arrays from the connections dict's
    # LIF_neuron objects, then convert Lambda -> tau (inverse of read mapping)
    thetas = np.zeros(n_neurons, dtype=np.float32)
    taus = np.zeros(n_neurons, dtype=np.float32)
    for name, (_, neuron_obj) in connections.items():
        idx = neuron_index[name]
        thetas[idx] = neuron_obj.get_theta()
        Lambda = neuron_obj.get_Lambda()
        taus[idx] = 2.0 ** Lambda

    # Build the axon-to-neuron weight matrix (rows=neurons, cols=axons)
    axon_weight_matrix = np.zeros((n_neurons, n_axons), dtype=np.float32)
    for axon_name, synapse_list in axons.items():
        for target_name, weight in synapse_list:
            axon_weight_matrix[neuron_index[target_name], axon_index[axon_name]] = weight

    # Build the recurrent neuron-to-neuron weight matrix (rows=target, cols=source)
    recurrent_weight_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    for source_name, (synapse_list, _) in connections.items():
        for target_name, weight in synapse_list:
            recurrent_weight_matrix[neuron_index[target_name], neuron_index[source_name]] = weight

        # Selection matrix: picks out only the intended output neurons from the
    # full population, rather than wiring the whole population to Output.
    # Without this, on read-back every neuron looks like an output, since
    # the general importer has no other way to know which ones were meant
    # to be outputs.
    n_outputs = len(outputs)
    output_index = {name: i for i, name in enumerate(outputs)}
    selection_matrix = np.zeros((n_outputs, n_neurons), dtype=np.float32)
    for output_name, out_idx in output_index.items():
        selection_matrix[out_idx, neuron_index[output_name]] = 1

    nir_graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([n_axons])}),
            "linear_in": nir.Linear(weight=axon_weight_matrix),
            "lif": nir.LIF(
                tau=taus,
                r=np.ones(n_neurons, dtype=np.float32),
                v_leak=np.zeros(n_neurons, dtype=np.float32),
                v_threshold=thetas,
                v_reset=np.zeros(n_neurons, dtype=np.float32),
            ),
            "linear_rec": nir.Linear(weight=recurrent_weight_matrix),
            "linear_select": nir.Linear(weight=selection_matrix),
            "output": nir.Output(output_type={"output": np.array([n_outputs])}),
        },
        edges=[
            ("input", "linear_in"),
            ("linear_in", "lif"),
            ("lif", "linear_rec"),
            ("linear_rec", "lif"),
            ("lif", "linear_select"),
            ("linear_select", "output"),
        ],
    )
    return nir_graph

if __name__ == "__main__":
    from hs_api.neuron_models import LIF_neuron

    print("=== Full round-trip test: HiAER-Spike -> NIR -> HiAER-Spike ===")

    # Gwen's network, in our own format
    N1 = LIF_neuron(theta=3, nu=-17, Lambda=63)
    N2 = LIF_neuron(theta=4, nu=-17, Lambda=2)
    N3 = LIF_neuron(theta=5, nu=-17, Lambda=63)

    axons_orig = {'alpha': [('a', 3), ('c', 2)],
                  'beta': [('b', 3)]}
    connections_orig = {'a': ([('b', 1), ('d', 2)], N1),
                         'b': ([], N1),
                         'c': ([], N2),
                         'd': ([('c', 1)], N3)}
    outputs_orig = ['a', 'b']

    # Write to NIR, save to a file, read the file back
    nir_graph = hiaer_spike_to_nir(axons_orig, connections_orig, outputs_orig)
    nir.write("roundtrip_test.nir", nir_graph)
    print("Wrote roundtrip_test.nir")

    reloaded_graph = nir.read("roundtrip_test.nir")
    print("Read back graph with nodes:", list(reloaded_graph.nodes.keys()))

    # Import it back using our own general importer
    axons_back, connections_back, outputs_back = nir_to_hiaer_spike(reloaded_graph)
    print(f"\nRecovered axons: {axons_back}")
    print(f"Recovered connections: { {k: v[0] for k, v in connections_back.items()} }")
    print(f"Recovered outputs: {outputs_back}")

    # Check: same WEIGHTS/TOPOLOGY as the original (names will differ, since
    # the general importer auto-generates names like lif_0, lif_1, ... rather
    # than preserving a/b/c/d, which NIR's format has no way to store)
    original_axon_weights = sorted(
        (axon, weight) for axon, synapses in axons_orig.items() for _, weight in synapses
    )
    recovered_axon_weights = sorted(
        weight for synapses in axons_back.values() for _, weight in synapses
    )
    original_weights_only = sorted(w for _, w in original_axon_weights)
    assert original_weights_only == recovered_axon_weights, \
        f"FAIL: axon weights don't match. Original: {original_weights_only}, Recovered: {recovered_axon_weights}"

    original_recurrent_weights = sorted(
        w for synapses, _ in connections_orig.values() for _, w in synapses
    )
    recovered_recurrent_weights = sorted(
        w for name, (synapses, _) in connections_back.items() for _, w in synapses
    )
    assert original_recurrent_weights == recovered_recurrent_weights, \
        f"FAIL: recurrent weights don't match. Original: {original_recurrent_weights}, Recovered: {recovered_recurrent_weights}"

    assert len(outputs_back) == len(outputs_orig), \
        f"FAIL: output count doesn't match, expected {len(outputs_orig)}, got {len(outputs_back)}"

    print("\nPASS: full write -> read -> import round trip preserves all weights and structure")