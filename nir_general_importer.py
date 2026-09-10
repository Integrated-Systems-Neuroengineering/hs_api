"""
General-purpose NIR -> HiAER-Spike importer.

Unlike the two earlier demos (which hand-coded a specific graph shape),
this walks an arbitrary NIR graph made of Input, Output, LIF, and Linear
nodes -- including recurrent connectivity (a LIF population connected back
to itself via a Linear node) -- and builds HiAER-Spike's axons/connections
format automatically.

Current limitations (first general version, to be extended):
- Only Input, Output, LIF, and Linear node types are supported. Other types
  (IF, CubaLIF, AdEx, Conv2d, etc.) raise NotImplementedError.
- Exactly one Input and one Output node are supported.
- A feedforward chain of LIF/Linear layers is supported, each LIF layer may
  additionally have its own recurrent Linear loop.
- All imported neurons use nu=-17 (no noise); NIR doesn't currently have a
  stochasticity field on LIF to translate from anyway.
- theta/Lambda are recovered via the tau=2^Lambda mapping worked out
  earlier; non-power-of-2 tau is rounded to the nearest Lambda (silently,
  for now -- a warning system is future work per Leif's request).
"""
import numpy as np
import nir
from hs_api.neuron_models import LIF_neuron


def nir_to_hiaer_spike(nir_graph):
    """
    Convert a NIR graph into HiAER-Spike's (axons, connections, outputs) format.

    Parameters
    ----------
    nir_graph : nir.NIRGraph

    Returns
    -------
    axons : dict
    connections : dict
    outputs : list
    """
    nodes = nir_graph.nodes
    edges = nir_graph.edges

    outgoing = {name: [] for name in nodes}
    incoming = {name: [] for name in nodes}
    for src, dst in edges:
        outgoing[src].append(dst)
        incoming[dst].append(src)

    # Locate Input/Output
    input_names = [n for n, obj in nodes.items() if isinstance(obj, nir.Input)]
    output_names = [n for n, obj in nodes.items() if isinstance(obj, nir.Output)]
    if len(input_names) != 1 or len(output_names) != 1:
        raise NotImplementedError(
            "Only graphs with exactly one Input and one Output node are currently supported"
        )
    input_name = input_names[0]
    output_name = output_names[0]

    # Check for unsupported node types
    supported_types = (nir.Input, nir.Output, nir.LIF, nir.Linear)
    for name, obj in nodes.items():
        if not isinstance(obj, supported_types):
            raise NotImplementedError(
                f"Node '{name}' has unsupported type {type(obj).__name__}; "
                f"only Input, Output, LIF, and Linear are currently supported"
            )

    # Classify Linear nodes as recurrent (loops back to a node that feeds it)
    # or feedforward (connects two different things)
    linear_names = [n for n, obj in nodes.items() if isinstance(obj, nir.Linear)]
    recurrent_linears = {}
    feedforward_linears = []
    for lin in linear_names:
        srcs = set(incoming[lin])
        dsts = set(outgoing[lin])
        loop_targets = srcs & dsts
        if loop_targets:
            recurrent_linears[lin] = list(loop_targets)[0]
        else:
            feedforward_linears.append(lin)

    # Expand each LIF node into individually named neurons
    lif_names = [n for n, obj in nodes.items() if isinstance(obj, nir.LIF)]
    neuron_names_by_lif = {}
    neuron_objs = {}
    for lif_name in lif_names:
        lif_node = nodes[lif_name]
        n = len(lif_node.tau)
        names = [f"{lif_name}_{i}" for i in range(n)]
        neuron_names_by_lif[lif_name] = names
        for i, name in enumerate(names):
            theta = int(round(lif_node.v_threshold[i]))
            Lambda = int(round(np.log2(lif_node.tau[i])))
            neuron_objs[name] = LIF_neuron(theta=theta, nu=-17, Lambda=Lambda)

    # Build axons from Input -> Linear -> LIF
    # Build axons from Input -> [Linear ->] LIF. Input may connect directly
    # to a LIF with no Linear in between (as in the single-neuron demo) --
    # treat that as an implicit identity connection (weight 1 per channel),
    # as long as the channel count matches the neuron count exactly.
    axons = {}
    num_input_channels = int(list(nodes[input_name].input_type.values())[0][0])
    input_axon_names = [f"axon_{i}" for i in range(num_input_channels)]
    for axon_name in input_axon_names:
        axons[axon_name] = []

    direct_lif_targets = [dst for dst in outgoing.get(input_name, []) if isinstance(nodes[dst], nir.LIF)]
    for target_lif in direct_lif_targets:
        target_neurons = neuron_names_by_lif[target_lif]
        if len(target_neurons) != num_input_channels:
            raise NotImplementedError(
                f"Input connects directly to LIF node '{target_lif}' with no Linear "
                f"node, but channel count ({num_input_channels}) doesn't match neuron "
                f"count ({len(target_neurons)}). An implicit identity connection only "
                f"makes sense when these match; a mismatched direct connection isn't "
                f"supported."
            )
        for axon_name, neuron_name in zip(input_axon_names, target_neurons):
            axons[axon_name].append((neuron_name, 1))

    for lin in feedforward_linears:
        if input_name in incoming[lin]:
            target_lif = outgoing[lin][0]
            weight_matrix = nodes[lin].weight
            target_neurons = neuron_names_by_lif[target_lif]
            for neuron_idx, neuron_name in enumerate(target_neurons):
                for axon_idx, axon_name in enumerate(input_axon_names):
                    w = weight_matrix[neuron_idx, axon_idx]
                    if w != 0:
                        axons[axon_name].append((neuron_name, int(w)))

    # Build connections: start every neuron with an empty outgoing list
    connections = {name: ([], obj) for name, obj in neuron_objs.items()}

    # Recurrent connections (within one LIF population)
    for lin, lif_name in recurrent_linears.items():
        weight_matrix = nodes[lin].weight
        neuron_list = neuron_names_by_lif[lif_name]
        for target_idx, target_name in enumerate(neuron_list):
            for source_idx, source_name in enumerate(neuron_list):
                w = weight_matrix[target_idx, source_idx]
                if w != 0:
                    connections[source_name][0].append((target_name, int(w)))

    # Feedforward connections between different LIF populations
    for lin in feedforward_linears:
        if input_name in incoming[lin]:
            continue  # already handled as an axon above
        source_lif = incoming[lin][0]
        target_lif = outgoing[lin][0] if outgoing[lin] else None
        if source_lif not in neuron_names_by_lif or target_lif not in neuron_names_by_lif:
            continue  # connects to something other than a LIF (e.g. straight to Output)
        weight_matrix = nodes[lin].weight
        source_neurons = neuron_names_by_lif[source_lif]
        target_neurons = neuron_names_by_lif[target_lif]
        for target_idx, target_name in enumerate(target_neurons):
            for source_idx, source_name in enumerate(source_neurons):
                w = weight_matrix[target_idx, source_idx]
                if w != 0:
                    connections[source_name][0].append((target_name, int(w)))

    # Outputs: whichever LIF population feeds directly into Output
        # Outputs: either a LIF population feeds directly into Output (whole
    # population is output), or a Linear node sits between them, selecting
    # a specific subset. HiAER-Spike can only report individual neurons'
    # spikes, not weighted combinations, so a selection Linear must have
    # exactly one nonzero (weight=1) entry per output row.
    outputs = []
    for lif_name in lif_names:
        if output_name in outgoing[lif_name]:
            outputs.extend(neuron_names_by_lif[lif_name])

    for lin in linear_names:
        if output_name in outgoing.get(lin, []):
            source_lif = incoming[lin][0]
            weight_matrix = nodes[lin].weight
            neuron_list = neuron_names_by_lif[source_lif]
            for row in weight_matrix:
                nonzero_indices = np.nonzero(row)[0]
                if len(nonzero_indices) != 1 or row[nonzero_indices[0]] != 1:
                    raise NotImplementedError(
                        "A Linear node selecting outputs must have exactly one "
                        "weight=1 entry per row (a direct neuron selection); "
                        "weighted combinations as outputs aren't representable "
                        "in HiAER-Spike, which can only report individual "
                        "neurons' spikes."
                    )
                outputs.append(neuron_list[nonzero_indices[0]])
    return axons, connections, outputs


if __name__ == "__main__":
    # ---- Self-test 1: reproduce the single-neuron demo ----
    print("=== Test 1: single neuron ===")
    theta, Lambda = 3, 6
    tau = 2.0 ** Lambda
    graph1 = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1])}),
            "lif": nir.LIF(
                tau=np.array([tau], dtype=np.float32),
                r=np.array([1.0], dtype=np.float32),
                v_leak=np.array([0.0], dtype=np.float32),
                v_threshold=np.array([theta], dtype=np.float32),
                v_reset=np.array([0.0], dtype=np.float32),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[("input", "lif"), ("lif", "output")],
    )
    axons1, connections1, outputs1 = nir_to_hiaer_spike(graph1)
    print(f"Axons: {axons1}")
    print(f"Connections: { {k: v[0] for k, v in connections1.items()} }")
    print(f"Outputs: {outputs1}")
    assert axons1 == {'axon_0': [('lif_0', 1)]}, f"FAIL: expected identity weight=1, got {axons1}"
    assert connections1 == {'lif_0': ([], connections1['lif_0'][1])}, f"FAIL: unexpected connections, got {connections1}"
    assert outputs1 == ['lif_0'], f"FAIL: outputs incorrect, got {outputs1}"
    print("PASS: direct Input->LIF connection correctly imported as identity (weight=1)")

    # ---- Self-test 2: reproduce Gwen's 4-neuron network demo ----
    print("\n=== Test 2: Gwen's 4-neuron recurrent network ===")
    thetas = np.array([3, 3, 4, 5], dtype=np.float32)
    lambdas = np.array([63, 63, 2, 63], dtype=np.float32)
    taus = 2.0 ** lambdas
    axon_weights = np.array([[3, 0], [0, 3], [2, 0], [0, 0]], dtype=np.float32)
    recurrent_weights = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [2, 0, 0, 0],
    ], dtype=np.float32)

    graph2 = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "linear_in": nir.Linear(weight=axon_weights),
            "lif": nir.LIF(
                tau=taus,
                r=np.ones(4, dtype=np.float32),
                v_leak=np.zeros(4, dtype=np.float32),
                v_threshold=thetas,
                v_reset=np.zeros(4, dtype=np.float32),
            ),
            "linear_rec": nir.Linear(weight=recurrent_weights),
            "output": nir.Output(output_type={"output": np.array([4])}),
        },
        edges=[
            ("input", "linear_in"),
            ("linear_in", "lif"),
            ("lif", "linear_rec"),
            ("linear_rec", "lif"),
            ("lif", "output"),
        ],
    )

    axons2, connections2, outputs2 = nir_to_hiaer_spike(graph2)
    print(f"Axons: {axons2}")
    print(f"Connections: { {k: v[0] for k, v in connections2.items()} }")
    print(f"Outputs: {outputs2}")

    assert axons2 == {'axon_0': [('lif_0', 3), ('lif_2', 2)], 'axon_1': [('lif_1', 3)]}, \
        f"FAIL: axons don't match expected topology, got {axons2}"
    expected_conn_topology = {'lif_0': [('lif_1', 1), ('lif_3', 2)], 'lif_1': [], 'lif_2': [], 'lif_3': [('lif_2', 1)]}
    actual_conn_topology = {k: v[0] for k, v in connections2.items()}
    assert actual_conn_topology == expected_conn_topology, \
        f"FAIL: connections don't match expected topology, got {actual_conn_topology}"
    assert set(outputs2) == {'lif_0', 'lif_1', 'lif_2', 'lif_3'}, f"FAIL: outputs incorrect, got {outputs2}"

    print("\nPASS: general importer produces exactly the expected topology on the recurrent network")

    # ---- Self-test 3: a genuine two-layer feedforward network ----
    # Input (2 channels) -> Linear -> LIF1 (2 neurons) -> Linear -> LIF2 (1 neuron) -> Output
    # This tests cross-population connectivity, distinct from Test 2's single
    # recurrent population -- closer to what a real multi-layer NIR-exported
    # model (e.g. from PyTorch) would actually look like.
    print("\n=== Test 3: two-layer feedforward network ===")
    layer1_thetas = np.array([2, 2], dtype=np.float32)
    layer1_lambdas = np.array([4, 4], dtype=np.float32)
    layer2_thetas = np.array([3], dtype=np.float32)
    layer2_lambdas = np.array([5], dtype=np.float32)

    w_in = np.array([[1, 0], [0, 1]], dtype=np.float32)  # 2 channels -> 2 neurons, identity
    w_out = np.array([[2, 3]], dtype=np.float32)  # 2 neurons -> 1 neuron

    graph3 = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "linear1": nir.Linear(weight=w_in),
            "lif1": nir.LIF(
                tau=2.0 ** layer1_lambdas,
                r=np.ones(2, dtype=np.float32),
                v_leak=np.zeros(2, dtype=np.float32),
                v_threshold=layer1_thetas,
                v_reset=np.zeros(2, dtype=np.float32),
            ),
            "linear2": nir.Linear(weight=w_out),
            "lif2": nir.LIF(
                tau=2.0 ** layer2_lambdas,
                r=np.ones(1, dtype=np.float32),
                v_leak=np.zeros(1, dtype=np.float32),
                v_threshold=layer2_thetas,
                v_reset=np.zeros(1, dtype=np.float32),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[
            ("input", "linear1"),
            ("linear1", "lif1"),
            ("lif1", "linear2"),
            ("linear2", "lif2"),
            ("lif2", "output"),
        ],
    )

    axons3, connections3, outputs3 = nir_to_hiaer_spike(graph3)
    print(f"Axons: {axons3}")
    print(f"Connections: { {k: v[0] for k, v in connections3.items()} }")
    print(f"Outputs: {outputs3}")

    assert axons3 == {'axon_0': [('lif1_0', 1)], 'axon_1': [('lif1_1', 1)]}, \
        f"FAIL: axons don't match expected topology, got {axons3}"
    expected_conn3 = {'lif1_0': [('lif2_0', 2)], 'lif1_1': [('lif2_0', 3)], 'lif2_0': []}
    actual_conn3 = {k: v[0] for k, v in connections3.items()}
    assert actual_conn3 == expected_conn3, f"FAIL: connections don't match expected, got {actual_conn3}"
    assert outputs3 == ['lif2_0'], f"FAIL: outputs incorrect, got {outputs3}"

    print("\nPASS: general importer correctly handles cross-population feedforward connectivity")