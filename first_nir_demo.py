"""
First demo of NIR <-> HiAER-Spike interop, per Leif's request.

Starts as simple as possible: a single LIF neuron, Input -> LIF -> Output,
written to a NIR file and read back, then imported into our CRI_network
format and run on the simulator.

Parameter mapping (NIR's continuous-time LIF -> our discrete-time LIF):
  Our hardware:  V_new = V_old * (1 - 1/2^Lambda) + input   (decays toward 0)
  NIR (discretized, dt=1): V_new = V_old * (1 - dt/tau) + ...

  Matching: dt/tau = 1/2^Lambda  =>  tau = 2^Lambda  (with dt=1)
  v_leak = 0   (our hardware has no resting-potential offset, always decays to 0)
  r = 1        (no separate input resistance scaling in our model)
  v_reset = 0  (hard reset, matching our default reset mode)
  v_threshold -> theta directly
"""
import numpy as np
import nir
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

# ---- Step 1: build a simple NIR graph (Input -> LIF -> Output) ----
theta = 3
Lambda = 6  # our hardware's leak parameter
tau = 2 ** Lambda  # matching mapping above

nir_graph = nir.NIRGraph(
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
    edges=[
        ("input", "lif"),
        ("lif", "output"),
    ],
)

# ---- Step 2: write to a NIR file, then read it back ----
nir.write("first_nir_demo.nir", nir_graph)
print("Wrote first_nir_demo.nir")

imported_graph = nir.read("first_nir_demo.nir")
print("Read back graph with nodes:", list(imported_graph.nodes.keys()))

# ---- Step 3: minimal importer -- convert the NIR graph into our format ----
# This is intentionally minimal: handles exactly the Input -> LIF -> Output
# shape, one neuron. A general importer would need to walk arbitrary graphs
# and expand Linear/population nodes into many neurons; that's future work.

lif_node = imported_graph.nodes["lif"]

# Recover our discrete Lambda from NIR's tau (inverse of the mapping above)
recovered_lambda = int(round(np.log2(lif_node.tau[0])))
recovered_theta = int(round(lif_node.v_threshold[0]))

print(f"Recovered from NIR: theta={recovered_theta}, Lambda={recovered_lambda}")

imported_neuron = LIF_neuron(theta=recovered_theta, nu=-17, Lambda=recovered_lambda)

axons = {"A0": [("N0", 2000)]}  # single input axon, arbitrary weight for demo
connections = {"N0": ([], imported_neuron)}
outputs = ["N0"]

network = CRI_network(axons=axons, connections=connections, outputs=outputs, target="simpleSim")

# ---- Step 4: run it ----
spikes = network.step(["A0"])
print(f"Spikes after step 1: {spikes}")
mp = network.read_membrane(outputs)
print(f"Membrane potential after step 1: {mp}")

spikes2 = network.step([])
print(f"Spikes after step 2: {spikes2}")
mp2 = network.read_membrane(outputs)
print(f"Membrane potential after step 2: {mp2}")

print("\nPASS: NIR round-trip (write -> read -> import -> run) completed successfully")