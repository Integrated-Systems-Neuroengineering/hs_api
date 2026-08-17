# 0 Introduction

This documentation teaches the low-level hardware implementation of hs_api to software experts and computational neuroscientists. hs_api provides a high-level Python interface for creating spiking neural networks, but the true hardware implementation is introduced here. This covers how your network definition translates into hardware-level operations on FPGA hardware, how data flows through PCIe transactions, how neurons are stored in memory and updated at nanosecond timescales. We'll build this understanding from the ground up, starting with a simple network example and tracing its journey from Python code through software layers, across physical communication buses, into programmable logic, and finally to individual memory cells storing neuron states.
```{toctree}
:hidden:
:maxdepth: 1

0_Introduction/python_page
0_Introduction/hardware_map
0_Introduction/hardware_component_definitions
0_Introduction/hs_bridge_role
```
