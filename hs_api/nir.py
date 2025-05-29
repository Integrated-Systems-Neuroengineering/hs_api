#!/usr/bin/env python3

import nir
import connectome_utils

def parse_graph(graph: nir.NIRGraph):
    # Create a dictionary of nodes
    axons = {}
    connections = {}
    outputs = []
    for name, node in graph.nodes.items():
        # Match the node to your platform's primitive
        if isinstance(node, nir.Input):
            #nodes[name] = MyPlatformInput()
            axonKey = name
            connectome.addNeuron(neuron(axonKey, "axon", axonType="Uaxon"))
        elif isinstance(node, nir.LIF):
            #nodes[name] = MyPlatformLeakyIntegrator(node.tau, node.r, node.v_leak)
            neuronKey = name
            neuron_model = LIF(node.tau,node.r,node.v_leak,node.threshold)
            #for now assume the neuron is not an output neuron
            self.connectome.addNeuron(
                neuron(
                    neuronKey,
                    "neuron",
                    neuronModel=neuron_model,
                )
            )
        elif isinstance(node, nir.Output):
            #We don't exactly have the same type of fake outputs
            neuron = connectome.get_neuron_by_key(node)
            neuron.set_output(output)

        elif isinstance(node, nir.NIRGraph): # Recurse through subgraphs
            nodes[name] = parse_graph(node)
        else:
            raise NotImplementedError(f"Node {node} not supported.")

    # Connect the nodes
    for edge in graph.edges:
        # Connect the nodes
        nodes[edge[0]].connect(nodes[edge[1]])

    return nodes
