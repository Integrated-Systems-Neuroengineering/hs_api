"""
NIR <-> HiAER-Spike demo using Gwen's tiny 4-neuron network (simplified to
all-LIF, no-noise per Leif's request), including real neuron-to-neuron
recurrent connectivity, not just a single isolated neuron.

Neuron order fixed as [a, b, c, d] (indices 0-3) throughout.

Original network (from Gwen's website example, simplified):
  N1 = LIF(theta=3, nu=-17, Lambda=63)   -- used by a, b
  N2 = LIF(theta=4, nu=-17, Lambda=2)    -- used by c
  N3 = LIF(theta=5, nu=-17, Lambda=63)   -- used by d (was ANN_neuron originally)

  axons: alpha -> a (w=3), alpha -> c (w=2), beta -> b (w=3)
  synapses: a -> b (w=1), a -> d (w=2), d -> c (w=1)
  outputs: a, b

NIR graph shape: Input -> Linear(axon weights) -> LIF -> Linear(recurrent
neuron weights) -> LIF (looped back) -> Output. The recurrent edge (LIF ->
linear_rec -> LIF) is what represents neuron-to-neuron connectivity, since
NIR's LIF node itself has no internal connectivity, that lives in Linear
nodes and the edges between them.
"""
import numpy as np
import nir
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

# ---- Step 1: build the NIR graph ----

# theta values per neuron (order: a, b, c, d)
thetas = np.array([3, 3, 4, 5], dtype=np.float32)
# Lambda values per neuron (order: a, b, c, d)
lambdas = np.array([63, 63, 2, 63], dtype=np.float32)
taus = 2.0 ** lambdas  # our mapping: tau = 2^Lambda

axon_weights = np.array([
    [3, 0],  # a <- alpha, beta
    [0, 3],  # b <- alpha, beta
    [2, 0],  # c <- alpha, beta
    [0, 0],  # d <- alpha, beta
], dtype=np.float32)

recurrent_weights = np.array([
    [0, 0, 0, 0],  # a <- a, b, c, d
    [1, 0, 0, 0],  # b <- a, b, c, d
    [0, 0, 0, 1],  # c <- a, b, c, d
    [2, 0, 0, 0],  # d <- a, b, c, d
], dtype=np.float32)

nir_graph = nir.NIRGraph(
    nodes={
        "input": nir.Input(input_type={"input": np.array([2])}),  # alpha, beta
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
        ("linear_rec", "lif"),  # recurrent loop: neuron-to-neuron connectivity
        ("lif", "output"),
    ],
)

nir.write("gwen_network_nir_demo.nir", nir_graph)
print("Wrote gwen_network_nir_demo.nir")

imported_graph = nir.read("gwen_network_nir_demo.nir")
print("Read back graph with nodes:", list(imported_graph.nodes.keys()))

# ---- Step 2: minimal importer -- convert into our connections/axons format ----
neuron_names = ["a", "b", "c", "d"]

lif_node = imported_graph.nodes["lif"]
linear_in_node = imported_graph.nodes["linear_in"]
linear_rec_node = imported_graph.nodes["linear_rec"]

recovered_thetas = [int(round(v)) for v in lif_node.v_threshold]
recovered_lambdas = [int(round(np.log2(t))) for t in lif_node.tau]
print(f"Recovered thetas: {recovered_thetas}")
print(f"Recovered Lambdas: {recovered_lambdas}")

neuron_objs = {
    name: LIF_neuron(theta=recovered_thetas[i], nu=-17, Lambda=recovered_lambdas[i])
    for i, name in enumerate(neuron_names)
}

# Build axons dict from linear_in's weight matrix (rows=neurons, cols=axons)
axon_names = ["alpha", "beta"]
axons = {axon_name: [] for axon_name in axon_names}
for neuron_idx, neuron_name in enumerate(neuron_names):
    for axon_idx, axon_name in enumerate(axon_names):
        w = linear_in_node.weight[neuron_idx, axon_idx]
        if w != 0:
            axons[axon_name].append((neuron_name, int(w)))

# Build connections dict: outgoing synapses per neuron, from linear_rec's
# weight matrix (rows=target, cols=source -- so column = source's outgoing list)
connections = {}
for source_idx, source_name in enumerate(neuron_names):
    outgoing = []
    for target_idx, target_name in enumerate(neuron_names):
        w = linear_rec_node.weight[target_idx, source_idx]
        if w != 0:
            outgoing.append((target_name, int(w)))
    connections[source_name] = (outgoing, neuron_objs[source_name])

outputs = ["a", "b"]

print(f"\nImported axons: {axons}")
print(f"Imported connections: { {k: (v[0], 'neuron_obj') for k, v in connections.items()} }")

# ---- Step 3: build and run ----
network = CRI_network(axons=axons, connections=connections, outputs=outputs, target="simpleSim")

for step_num in range(1, 4):
    inputs = ["alpha", "beta"] if step_num == 1 else []
    spikes = network.step(inputs)
    mp = network.read_membrane(outputs)
    print(f"Step {step_num}: spikes={spikes}, MP={mp}")

print("\nPASS: NIR round-trip with Gwen's full 4-neuron recurrent network completed successfully")
