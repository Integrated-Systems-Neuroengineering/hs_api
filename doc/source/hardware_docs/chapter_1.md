# 1 Initializing the Network

## Introduction

When you call `network = CRI_network(target="CRI")` in hs_api, you're asking the system to take your high-level network definition (axons, neurons, synapses, weights) and transform it into a physical configuration on the FPGA hardware. This chapter explains exactly what happens during this initialization process.

We'll use our example network from the Introduction:
- **5 axons** (a0-a4) → **5 hidden neurons** (h0-h4) → **5 output neurons** (o0-o4)
- All synaptic weights = 1000
- Fully connected between layers

By the end of initialization, this entire network will be programmed into the FPGA's memory systems, ready to process spikes in microseconds.

```{toctree}
:hidden:
:maxdepth: 1

1_Initializing_the_Network/Chapter_1_1
1_Initializing_the_Network/Chapter_1_2
1_Initializing_the_Network/CRI_Stack_Diagram
```
