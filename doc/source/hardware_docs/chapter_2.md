# 2 The Network Comes to Life

## Introduction

In Chapter 1, we initialized our network—programming the connectivity structure into HBM, clearing neuron states in URAM, and configuring network parameters. The FPGA now holds a frozen snapshot of our network architecture, but nothing is happening yet. The neurons are silent, their membrane potentials at zero.

In this chapter, we bring the network to life. We'll trace what happens when you call `network.step(['a0', 'a1', 'a2'])` to send input spikes to the first three axons. We'll follow these spikes as they:
1. **Phase 0:** Get written to BRAM as input patterns
2. **Phase 1:** Trigger reads from HBM to fetch synaptic connections
3. **Phase 2:** Flow into neurons, causing membrane potentials to rise and neurons to spike
4. **Phase 3:** Get collected and sent back to the host

We'll use our example network from the Introduction:
- **5 axons** (a0-a4) → **5 hidden neurons** (h0-h4) → **5 output neurons** (o0-o4)
- All weights = 1000
- Threshold = 2000
- **Input:** Axons a0, a1, a2 fire every timestep

By the end of this chapter, you'll understand exactly what happens in hardware during a single timestep, from input arrival to spike output, at the clock-cycle level.
```{toctree}
:hidden:
:maxdepth: 1

2_The_Network_Comes_to_Life/Chapter_2_1
2_The_Network_Comes_to_Life/Chapter_2_2
2_The_Network_Comes_to_Life/Chapter_2_3
2_The_Network_Comes_to_Life/FPGA_Timestep_Diagram
```
