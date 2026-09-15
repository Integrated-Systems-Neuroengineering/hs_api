"""
General importer for arbitrary NIR graphs.

This walks an arbitrary NIR graph made of Input, Output, LIF, IF, and Linear
nodes -- including feedforward multi-layer connectivity.

Limitations:

- Only Input, Output, LIF, IF, and Linear node types are supported.
- A feedforward chain of LIF/IF/Linear layers is supported.

Returns:
  (axons, connections, outputs) tuple
"""

import numpy as np
import nir
from hs_api.neuron_models import LIF_neuron, IF_neuron


def import_nir_graph(nir_graph: nir.NIRGraph):
    """Import a NIR graph into HiAER-Spike format."""

    nodes = nir_graph.nodes
    edges = nir_graph.edges

    supported_types = (nir.Input, nir.Output, nir.LIF, nir.IF, nir.Linear)
    for name, node in nodes.items():
        if not isinstance(node, supported_types):
            raise NotImplementedError(f"only Input, Output, LIF, IF, and Linear are supported")

    outgoing = {}
    incoming = {}
    for (src, dst), edge in edges.items():
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(dst, []).append(src)

    axons = {}
    neuron_objs = {}
    connections = {}
    outputs = []

    input_name = [n for n, obj in nodes.items() if isinstance(obj, nir.Input)][0]
    output_nodes = [n for n, obj in nodes.items() if isinstance(obj, nir.Output)]

    lif_names = [n for n, obj in nodes.items() if isinstance(obj, nir.LIF)]
    neuron_names_by_lif = {}

    for lif_name in lif_names:
        lif_node = nodes[lif_name]
        n = len(lif_node.tau)
        names = [f"{lif_name}_{i}" for i in range(n)]
        neuron_names_by_lif[lif_name] = names
        for i, name in enumerate(names):
            theta = int(round(float(lif_node.v_threshold[i])))
            Lambda = int(round(np.log2(float(lif_node.tau[i]))))
            neuron_objs[name] = LIF_neuron(theta=theta, nu=-17, Lambda=Lambda)

    if_names = [n for n, obj in nodes.items() if isinstance(obj, nir.IF)]
    neuron_names_by_if = {}
    for if_name in if_names:
        if_node = nodes[if_name]
        n = len(if_node.tau)
        names = [f"{if_name}_{i}" for i in range(n)]
        neuron_names_by_if[if_name] = names
        for i, name in enumerate(names):
            theta = int(round(float(if_node.v_threshold[i])))
            Lambda = int(round(np.log2(float(if_node.tau[i]))))
            neuron_objs[name] = IF_neuron(theta=theta, nu=-17, Lambda=Lambda)

    linear_names = [n for n, obj in nodes.items() if isinstance(obj, nir.Linear)]
    for lin in linear_names:
        lin_node = nodes[lin]
        if input_name in incoming.get(lin, []):
            axon_name = f"axon_{lin}"
            axons[axon_name] = [
                (f"{lin}_{i}", float(lin_node.weight[i, j]))
                for i in range(lin_node.weight.shape[0])
                for j in range(lin_node.weight.shape[1])
                if lin_node.weight[i, j] != 0
            ]

    # Linear -> LIF connections
    for lin in linear_names:
        lin_node = nodes[lin]
        target_lif = next((dst for dst in outgoing.get(lin, []) if isinstance(nodes[dst], nir.LIF)), None)
        target_if = next((dst for dst in outgoing.get(lin, []) if isinstance(nodes[dst], nir.IF)), None)

        if target_lif:
            target_neurons = neuron_names_by_lif[target_lif]
            for i, target_name in enumerate(target_neurons):
                connections[target_name] = [
                    (target_name, float(lin_node.weight[i, j]))
                    for j in range(lin_node.weight.shape[1])
                    if lin_node.weight[i, j] != 0
                ]

        if target_if:
            target_neurons = neuron_names_by_if[target_if]
            for i, target_name in enumerate(target_neurons):
                connections[target_name] = [
                    (target_name, float(lin_node.weight[i, j]))
                    for j in range(lin_node.weight.shape[1])
                    if lin_node.weight[i, j] != 0
                ]

    # Output selection
    for output_node_name in output_nodes:
        source_lif = next((src for src in incoming.get(output_node_name, []) if src in lif_names), None)
        source_if = next((src for src in incoming.get(output_node_name, []) if src in if_names), None)

        if source_lif:
            source_neurons = neuron_names_by_lif[source_lif]
        elif source_if:
            source_neurons = neuron_names_by_if[source_if]
        else:
            raise ValueError(f"Output {output_node_name} has no source")

        outputs.extend(source_neurons)

    # Initialize all neurons in connections dict
    for neuron_name in neuron_objs:
        connections.setdefault(neuron_name, [])

    return axons, connections, outputs


if __name__ == "__main__":
    print("=== Test 1: single neuron ===")
    lif = nir.LIF(tau=np.array([10.0]), v_threshold=np.array([1.0]), r=np.array([1.0]), v_leak=np.array([0.0]))
    input_node = nir.Input(input_type=[1])
    output_node = nir.Output(output_type=[1])
    linear = nir.Linear(weight=np.array([[1.0]]))

    edges = {
        ("input", "linear"): None,
        ("linear", "lif"): None,
        ("lif", "output"): None,
    }
    nodes = {"input": input_node, "linear": linear, "lif": lif, "output": output_node}
    graph = nir.NIRGraph(nodes, edges)

    axons, connections, outputs = import_nir_graph(graph)
    print(f"Axons: {axons}")
    print(f"Connections: {connections}")
    print(f"Outputs: {outputs}")
    assert "lif_0" in connections
    print("PASS: single neuron test")

    print("\n=== Test 2: two-layer feedforward network ===")
    lif1 = nir.LIF(tau=np.array([10.0, 10.0]), v_threshold=np.array([1.0, 1.0]), r=np.array([1.0, 1.0]), v_leak=np.array([0.0, 0.0]))
    lif2 = nir.LIF(tau=np.array([10.0]), v_threshold=np.array([1.0]), r=np.array([1.0]), v_leak=np.array([0.0]))
    lin1 = nir.Linear(weight=np.array([[1.0], [1.0]]))  # 1 input -> 2 outputs
    lin2 = nir.Linear(weight=np.array([[2.0, 3.0]]))    # 2 inputs -> 1 output

    edges_2layer = {
        ("input", "linear1"): None,
        ("linear1", "lif1"): None,
        ("lif1", "linear2"): None,
        ("linear2", "lif2"): None,
        ("lif2", "output"): None,
    }
    nodes_2layer = {
        "input": input_node,
        "linear1": lin1,
        "lif1": lif1,
        "linear2": lin2,
        "lif2": lif2,
        "output": output_node,
    }
    graph_2layer = nir.NIRGraph(nodes_2layer, edges_2layer)

    axons_2layer, connections_2layer, outputs_2layer = import_nir_graph(graph_2layer)
    print(f"Axons: {axons_2layer}")
    print(f"Connections: {connections_2layer}")
    print(f"Outputs: {outputs_2layer}")
    assert "lif1_0" in connections_2layer
    assert "lif2_0" in connections_2layer
    print("PASS: two-layer feedforward network test")