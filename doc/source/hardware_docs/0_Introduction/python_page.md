# Our hs_api Example

## The Example Network: A Code Walkthrough

We'll use this network throughout the documentation as our reference example. Let's walk through the code:

### Imports

```python
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron
```

The hs_api library provides two key abstractions: `CRI_network` represents the entire spiking neural network (both the connectivity structure and the execution engine), while `LIF_neuron` defines the computational model of individual neurons. CRI stands for "Configurable Research Interface," reflecting the hardware's reconfigurable nature.

### Defining the Neuron Model

```python
lif_model = LIF_neuron(threshold=2000, shift=0, leak=63)
```

This creates a Leaky Integrate-and-Fire neuron model with three critical parameters:

**`threshold=2000`**: The threshold voltage. Each neuron maintains a membrane potential (voltage) that accumulates inputs. When this potential reaches or exceeds 2000, the neuron fires (spikes) and resets to zero. Think of this like a bucket that fills with water, and when it reaches the 2000-unit mark, it tips over (spikes) and empties.

**`shift=0`**: Random noise perturbation magnitude. Setting this to 0 means the system is deterministic—given the same inputs, you'll always get the same outputs. This is crucial for understanding hardware behavior, as it eliminates stochasticity.

**`leak=63`**: The leak rate controls how membrane potential decays over time. The value 63 is special: it's `2^6 - 1`, which in the hardware's fixed-point arithmetic effectively means "no leak." The neuron becomes a pure integrator (Integrate-and-Fire rather than Leaky Integrate-and-Fire), accumulating all inputs without any decay. This simplifies our analysis.

### Defining Axons (Input Layer)

```python
axons = {
    'a0': [('h0', 1000), ('h1', 1000), ('h2', 1000), ('h3', 1000), ('h4', 1000)],
    'a1': [('h0', 1000), ('h1', 1000), ('h2', 1000), ('h3', 1000), ('h4', 1000)],
    'a2': [('h0', 1000), ('h1', 1000), ('h2', 1000), ('h3', 1000), ('h4', 1000)],
    'a3': [('h0', 1000), ('h1', 1000), ('h2', 1000), ('h3', 1000), ('h4', 1000)],
    'a4': [('h0', 1000), ('h1', 1000), ('h2', 1000), ('h3', 1000), ('h4', 1000)]
}
```

Axons are external inputs to the network, they represent stimuli from the outside world. This dictionary defines 5 axons (`a0` through `a4`), each connecting to all 5 hidden neurons (`h0` through `h4`).

The structure is: `axon_name: [(target_neuron, synaptic_weight), ...]`

For example, `'a0': [('h0', 1000), ('h1', 1000), ...]` means:
- When axon `a0` fires, it sends a signal to neuron `h0` with weight 1000
- It simultaneously sends signals to `h1`, `h2`, `h3`, and `h4`, all with weight 1000

The **synaptic weight** (1000) represents the strength of the connection. When an axon fires, each target neuron's membrane potential increases by the synaptic weight. Since our threshold is 2000, a neuron needs to receive 2 inputs of weight 1000 to spike.

This creates a **fully-connected** topology: every axon connects to every hidden neuron. Total connections: 5 axons × 5 neurons = 25 synapses.

### Defining Neuron Connections (Hidden → Output Layer)

```python
connections = {
    'h0': ([('o0', 1000), ('o1', 1000), ('o2', 1000), ('o3', 1000), ('o4', 1000)], lif_model),
    'h1': ([('o0', 1000), ('o1', 1000), ('o2', 1000), ('o3', 1000), ('o4', 1000)], lif_model),
    'h2': ([('o0', 1000), ('o1', 1000), ('o2', 1000), ('o3', 1000), ('o4', 1000)], lif_model),
    'h3': ([('o0', 1000), ('o1', 1000), ('o2', 1000), ('o3', 1000), ('o4', 1000)], lif_model),
    'h4': ([('o0', 1000), ('o1', 1000), ('o2', 1000), ('o3', 1000), ('o4', 1000)], lif_model)
}
```

The `connections` dictionary defines the internal network structure: which neurons connect to which other neurons. The format is slightly different from `axons`:

`neuron_name: ([(target_neuron, weight), ...], neuron_model)`

Each entry has two parts:
1. **A list of outgoing connections**: Where this neuron sends its spikes and with what weight
2. **The neuron model**: Which computational dynamics this neuron follows

For example, `'h0': ([('o0', 1000), ('o1', 1000), ...], lif_model)` means:
- Neuron `h0` uses the `lif_model` dynamics (the LIF model we defined earlier)
- When `h0` spikes, it sends signals to all 5 output neurons (`o0` through `o4`)
- Each signal has weight 1000

Again, this is fully connected: 5 hidden neurons × 5 output neurons = 25 synapses.

**Key distinction between axons and connections:**
- `axons`: External inputs with no internal state (no membrane potential)
- `connections`: Internal neurons with state (membrane potential), dynamics (LIF model), and computation

### Output Neurons

```python
outputs = ['o0', 'o1', 'o2', 'o3', 'o4']
```

This list specifies which neurons we want to monitor, i.e. which neurons' spikes should be recorded and returned to the host computer. In our network, we're only interested in the output layer (`o0` through `o4`). The hidden layer spikes (`h0`-`h4`) occur internally but aren't reported unless explicitly added to this list.

This is analogous to setting breakpoints in a debugger: you're telling the system "I care about these specific events, report them to me."

### Creating the Network

```python
network = CRI_network(
    axons=axons,
    connections=connections,
    outputs=outputs,
    target='simpleSim'
)
```

This instantiates the network object by combining all the definitions we've created. The `CRI_network` constructor takes:

**`axons`**: The input connectivity we defined (5 axons → 5 hidden neurons)

**`connections`**: The internal connectivity we defined (5 hidden → 5 output neurons)

**`outputs`**: Which neurons to monitor (the 5 output neurons)

**`target='simpleSim'`**: The execution backend. Here's where the hardware abstraction becomes visible:
- `'simpleSim'`: Software simulation (Python-based, runs on CPU)
- `'CRI'`: FPGA hardware (compiles network to hardware, runs on neuromorphic chip)

For this example, we use software simulation so you can run it immediately. Later chapters will show what happens when `target='CRI'` and the exact same network definition gets compiled into FPGA configuration data, loaded into hardware memory, and executed at microsecond timescales.

When you create a `CRI_network` object with `target='CRI'`, the constructor:
1. Compiles your network structure into binary hardware representations
2. Allocates memory addresses in HBM for synapses
3. Initializes FPGA configuration registers
4. Transfers all data to the neuromorphic hardware via PCIe

With `target='simpleSim'`, it instead creates a software model that mimics hardware behavior.

### Running the Simulation

```python
inputs = ['a0', 'a1', 'a2']

for timestep in range(10):
    spikes = network.step(inputs)
    print(f"Timestep {timestep:2d}: Spikes = {spikes}")
```

This is the execution loop. We run the network for 10 discrete timesteps, where each timestep represents one unit of simulated time.

**`inputs = ['a0', 'a1', 'a2']`**: Defines which axons fire at each timestep. We're providing constant input: axons a0, a1, and a2 fire every timestep, while a3 and a4 remain silent.

**`network.step(inputs)`**: Advances the network by one timestep:
1. Applies the specified axon spikes (a0, a1, a2 fire)
2. Propagates spikes through synaptic connections (each firing axon adds 1000 to connected neurons)
3. Updates all neuron membrane potentials according to their dynamics (LIF model)
4. Detects threshold crossings (neurons with V ≥ 2000 spike)
5. Returns a list of which output neurons spiked

**Return value**: The `spikes` variable contains a list of output neuron names that spiked during this timestep, for example: `['o0', 'o1', 'o2', 'o3', 'o4']`.

### Expected Behavior

Let's trace what happens in the first few timesteps:

**Timestep 0:**
- Axons a0, a1, a2 fire
- Each hidden neuron (h0-h4) receives 3 inputs × 1000 weight = 3000 total input
- Hidden neurons start at V=0, so V_new = 0 + 3000 = 3000
- Since 3000 > 2000 (threshold), all hidden neurons spike immediately
- When a hidden neuron spikes: V resets to 0, but it sends its spike to all output neurons
- Each output neuron receives 5 inputs × 1000 weight = 5000 total input
- Output neurons: V_new = 0 + 5000 = 5000, which exceeds threshold → all output neurons spike
- **Result: `['o0', 'o1', 'o2', 'o3', 'o4']`**

**Timestep 1:**
- Hidden neurons start at V=0 (they reset after spiking)
- Output neurons also start at V=0 (they reset after spiking)
- Same pattern repeats: 3 axons fire → hidden neurons receive 3000 → spike → outputs receive 5000 → spike
- **Result: `['o0', 'o1', 'o2', 'o3', 'o4']`**

This pattern continues for all 10 timesteps. The network reaches a steady state where all output neurons spike every timestep because the input is constant and strong enough (3 simultaneous axon spikes) to drive the hidden layer above threshold immediately.

**Why this example?** This simple, predictable network lets us trace exact hardware behavior in later chapters. We'll see exactly which memory addresses get written, which transistors switch, and which clock cycles execute which operations—all because the computation is deterministic and well-defined.

---

Now that we understand the network from a software perspective, the following sections will peel back the abstraction layers: how this Python code becomes binary data, how that data moves through physical communication channels, how hardware circuits perform the computation, and how the results flow back to your program.
