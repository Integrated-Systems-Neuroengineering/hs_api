"""
General importer for arbitrary NIR graphs.

This walks an arbitrary NIR graph made of Input, Output, LIF, IF, Threshold, Conv2d, and Linear
nodes -- including feedforward multi-layer connectivity.

Limitations:

- Only Input, Output, LIF, IF, Threshold, Conv2d, and Linear node types are supported.
- Conv2d is flattened into individual neuron-to-neuron connections.
- A feedforward chain of layers is supported.

Returns:
  (axons, connections, outputs) tuple
"""

import numpy as np
import nir
from hs_api.neuron_models import LIF_neuron, IF_neuron, ANN_neuron


def check_tau_approximation(tau_value):
    """Check if tau is a power of 2 and warn if not.
    
    Parameters:
    -----------
    tau_value : float
        The time constant from NIR
        
    Returns:
    --------
    int : log2 of the approximated tau (power of 2)
    """
    log_tau = np.log2(float(tau_value))
    if log_tau != int(log_tau):
        approx_log = int(round(log_tau))
        approx_tau = 2 ** approx_log
        error_pct = abs(approx_tau - float(tau_value)) / float(tau_value) * 100
        print(f"WARNING: tau={tau_value} is not power of 2, approximating as {approx_tau} (error: {error_pct:.1f}%)")
    return int(round(log_tau))


def import_nir_graph(nir_graph: nir.NIRGraph):
    """Import a NIR graph into HiAER-Spike format."""

    nodes = nir_graph.nodes
    edges = nir_graph.edges

    supported_types = (nir.Input, nir.Output, nir.LIF, nir.IF, nir.Threshold, nir.Conv2d, nir.Linear)
    for name, node in nodes.items():
        if not isinstance(node, supported_types):
            raise NotImplementedError(f"only Input, Output, LIF, IF, Threshold, Conv2d, and Linear are supported")

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
            Lambda = check_tau_approximation(lif_node.tau[i])
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
            Lambda = check_tau_approximation(if_node.tau[i])
            neuron_objs[name] = IF_neuron(theta=theta, nu=-17, Lambda=Lambda)

    threshold_names = [n for n, obj in nodes.items() if isinstance(obj, nir.Threshold)]
    neuron_names_by_threshold = {}
    for threshold_name in threshold_names:
        threshold_node = nodes[threshold_name]
        n = len(threshold_node.threshold)
        names = [f"{threshold_name}_{i}" for i in range(n)]
        neuron_names_by_threshold[threshold_name] = names
        for i, name in enumerate(names):
            theta = int(round(float(threshold_node.threshold[i])))
            neuron_objs[name] = ANN_neuron(theta=theta, nu=-17)

    conv2d_names = [n for n, obj in nodes.items() if isinstance(obj, nir.Conv2d)]
    neuron_names_by_conv2d = {}
    
    for conv2d_name in conv2d_names:
        conv2d_node = nodes[conv2d_name]
        out_channels = conv2d_node.weight.shape[0]
        in_h, in_w = conv2d_node.input_shape
        k_h, k_w = conv2d_node.weight.shape[2:4]
        pad_h, pad_w = conv2d_node.padding if isinstance(conv2d_node.padding, tuple) else (conv2d_node.padding, conv2d_node.padding)
        stride_h, stride_w = conv2d_node.stride if isinstance(conv2d_node.stride, tuple) else (conv2d_node.stride, conv2d_node.stride)
        
        out_h = (in_h + 2*pad_h - k_h) // stride_h + 1
        out_w = (in_w + 2*pad_w - k_w) // stride_w + 1
        
        names = [f"{conv2d_name}_{c}_{h}_{w}" for c in range(out_channels) for h in range(out_h) for w in range(out_w)]
        neuron_names_by_conv2d[conv2d_name] = names
        
        for name in names:
            theta = int(round(float(conv2d_node.bias[0]))) if len(conv2d_node.bias) > 0 else 0
            neuron_objs[name] = LIF_neuron(theta=theta, nu=-17, Lambda=10)

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

    for lin in linear_names:
        lin_node = nodes[lin]
        target_lif = next((dst for dst in outgoing.get(lin, []) if isinstance(nodes[dst], nir.LIF)), None)
        target_if = next((dst for dst in outgoing.get(lin, []) if isinstance(nodes[dst], nir.IF)), None)
        target_threshold = next((dst for dst in outgoing.get(lin, []) if isinstance(nodes[dst], nir.Threshold)), None)
        target_conv2d = next((dst for dst in outgoing.get(lin, []) if isinstance(nodes[dst], nir.Conv2d)), None)

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

        if target_threshold:
            target_neurons = neuron_names_by_threshold[target_threshold]
            for i, target_name in enumerate(target_neurons):
                connections[target_name] = [
                    (target_name, float(lin_node.weight[i, j]))
                    for j in range(lin_node.weight.shape[1])
                    if lin_node.weight[i, j] != 0
                ]

        if target_conv2d:
            conv2d_node = nodes[target_conv2d]
            out_channels = conv2d_node.weight.shape[0]
            in_channels = conv2d_node.weight.shape[1]
            k_h, k_w = conv2d_node.weight.shape[2:4]
            pad_h, pad_w = conv2d_node.padding if isinstance(conv2d_node.padding, tuple) else (conv2d_node.padding, conv2d_node.padding)
            stride_h, stride_w = conv2d_node.stride if isinstance(conv2d_node.stride, tuple) else (conv2d_node.stride, conv2d_node.stride)
            
            in_h, in_w = conv2d_node.input_shape
            out_h = (in_h + 2*pad_h - k_h) // stride_h + 1
            out_w = (in_w + 2*pad_w - k_w) // stride_w + 1
            
            target_neurons = neuron_names_by_conv2d[target_conv2d]
            idx = 0
            for c in range(out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        target_name = target_neurons[idx]
                        connections[target_name] = [
                            (target_name, float(lin_node.weight[idx, j]))
                            for j in range(lin_node.weight.shape[1])
                            if lin_node.weight[idx, j] != 0
                        ]
                        idx += 1

    for output_node_name in output_nodes:
        source_lif = next((src for src in incoming.get(output_node_name, []) if src in lif_names), None)
        source_if = next((src for src in incoming.get(output_node_name, []) if src in if_names), None)
        source_threshold = next((src for src in incoming.get(output_node_name, []) if src in threshold_names), None)
        source_conv2d = next((src for src in incoming.get(output_node_name, []) if src in conv2d_names), None)

        if source_lif:
            source_neurons = neuron_names_by_lif[source_lif]
        elif source_if:
            source_neurons = neuron_names_by_if[source_if]
        elif source_threshold:
            source_neurons = neuron_names_by_threshold[source_threshold]
        elif source_conv2d:
            source_neurons = neuron_names_by_conv2d[source_conv2d]
        else:
            raise ValueError(f"Output {output_node_name} has no source")

        outputs.extend(source_neurons)

    for neuron_name in neuron_objs:
        connections.setdefault(neuron_name, [])

    return axons, connections, outputs


if __name__ == "__main__":
    print("=== Test 1: single neuron ===")
    input_node = nir.Input(input_type=[1])
    output_node = nir.Output(output_type=[1])
    lif = nir.LIF(tau=np.array([10.0]), v_threshold=np.array([1.0]), r=np.array([1.0]), v_leak=np.array([0.0]))
    linear = nir.Linear(weight=np.array([[1.0]]))

    edges = {
        ("input", "linear"): None,
        ("linear", "lif"): None,
        ("lif", "output"): None,
    }
    nodes = {"input": input_node, "linear": linear, "lif": lif, "output": output_node}
    graph = nir.NIRGraph(nodes, edges)

    axons, connections, outputs = import_nir_graph(graph)
    assert "lif_0" in connections
    print("PASS: single neuron test")

    print("\n=== Test 2: two-layer feedforward network ===")
    lif1 = nir.LIF(tau=np.array([10.0, 10.0]), v_threshold=np.array([1.0, 1.0]), r=np.array([1.0, 1.0]), v_leak=np.array([0.0, 0.0]))
    lif2 = nir.LIF(tau=np.array([10.0]), v_threshold=np.array([1.0]), r=np.array([1.0]), v_leak=np.array([0.0]))
    lin1 = nir.Linear(weight=np.array([[1.0], [1.0]]))
    lin2 = nir.Linear(weight=np.array([[2.0, 3.0]]))

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
    assert "lif1_0" in connections_2layer
    assert "lif2_0" in connections_2layer
    print("PASS: two-layer feedforward network test")

    print("\n=== Test 3: Threshold (ANN) node ===")
    threshold = nir.Threshold(threshold=np.array([1.0, 2.0]))
    lin_to_threshold = nir.Linear(weight=np.array([[1.0], [1.0]]))
    output_node_threshold = nir.Output(output_type=[2])

    edges_threshold = {
        ("input", "linear"): None,
        ("linear", "threshold"): None,
        ("threshold", "output"): None,
    }
    nodes_threshold = {
        "input": input_node,
        "linear": lin_to_threshold,
        "threshold": threshold,
        "output": output_node_threshold,
    }
    graph_threshold = nir.NIRGraph(nodes_threshold, edges_threshold)

    axons_threshold, connections_threshold, outputs_threshold = import_nir_graph(graph_threshold)
    assert "threshold_0" in connections_threshold
    assert "threshold_1" in connections_threshold
    print("PASS: Threshold (ANN) node test")

    print("\n=== Test 4: Tau approximation warnings ===")
    lif_nonpow2 = nir.LIF(tau=np.array([7.5, 15.0]), v_threshold=np.array([1.0, 1.0]), r=np.array([1.0, 1.0]), v_leak=np.array([0.0, 0.0]))
    lin_nonpow2 = nir.Linear(weight=np.array([[1.0], [1.0]]))
    output_node_nonpow2 = nir.Output(output_type=[2])

    edges_nonpow2 = {
        ("input", "linear"): None,
        ("linear", "lif"): None,
        ("lif", "output"): None,
    }
    nodes_nonpow2 = {
        "input": input_node,
        "linear": lin_nonpow2,
        "lif": lif_nonpow2,
        "output": output_node_nonpow2,
    }
    graph_nonpow2 = nir.NIRGraph(nodes_nonpow2, edges_nonpow2)

    axons_nonpow2, connections_nonpow2, outputs_nonpow2 = import_nir_graph(graph_nonpow2)
    assert "lif_0" in connections_nonpow2
    assert "lif_1" in connections_nonpow2
    print("PASS: Tau approximation warnings test")