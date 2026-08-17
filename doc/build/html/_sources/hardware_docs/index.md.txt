# HiAER Spike Hardware Docs

Welcome 👋

This documentation teaches hs_api experts and computational neuroscientists how the hardware behind hs_api works. HiAER Spike is built on FPGA technology, which is programmable via hardware language (SystemVerilog) and communicates with the host system (CPU) via hs_bridge. Rather than walking through each file, this documentation teaches from the top down:

## [**Introduction**](introduction.md) shows a basic hs_api Python implementation for a tiny, 2 layer network.
We will learn everything in the context of this network so that each component is grounded in an example.
- The [Python file](0_Introduction/python_page.md) defines and compiles the network and runs 10 timesteps of inputs on the network
- Has a [page](0_Introduction/hardware_component_definitions.md) introducing all our hardware components, how they're connected, units of communication, capabilities
- Has a [map](0_Introduction/hardware_map.md) visualizing the hardware components and connections

## [**Chapter 1**](chapter_1.md) shows you how the network looks in hardware.
[**1.1**](1_Initializing_the_Network/Chapter_1_1.md) Explains the locations of everything conceptually, including a visualization  
[**1.2**](1_Initializing_the_Network/Chapter_1_2.md) Explains how the network is written into the FPGA from the host using hs_bridge and SystemVerilog code

## [**Chapter 2**](chapter_2.md) shows how the hardware processes a timestep of the network in multiple phases
[**2.1**](2_The_Network_Comes_to_Life/Chapter_2_1.md) Walks through the phases of a timestep *conceptually*, no code. Introduces FPGA/Verilog modules as black boxes.  
[**2.2**](2_The_Network_Comes_to_Life/Chapter_2_2.md) Walks through phases of a timestep in  SystemVerilog code; beginning to explain the FPGA/SystemVerilog modules.  
[**2.3**](2_The_Network_Comes_to_Life/Chapter_2_3.md) Introduces the concept and implementation of the hardware's state machine.

## [**Chapter 3**](chapter_3.md) further explains the hardware design
Each page of Chapter 3 is dedicated to a SystemVerilog file. The first half of the page is essentially a README for the file, followed by the SystemVerilog file itself (heavily commented)

## [**Chapter 4**](chapter_4.md) shows how to implement R-STDP on HiAER-Spike
R-STDP is a form of biologically inspired supervised weight update rule ([What is RSTDP?](4_Implementing_RSTDP/what_is_rstdp.md)). To implement it, we need to know ...
- how to update weights:
  - [**Reading and Writing Synapses**](4_Implementing_RSTDP/synapse_read_write.md) explains how the pre-existing hs_bridge code for read synapse weights and writing synapse weights works.
- how to implement coincidence STDP:
  - (not yet written)
- how to implement the eligibility trace:
  - (not yet written)


## [**Supplementary Information**](supplementary_information.md)
Includes:
- [**Appendix**](supplementary_information/appendix.md): a long list of key terms throughout the whole documentation, thoroughly explained for hardware beginners
- [**Packet Encoding Explained**](supplementary_information/packet_encoding.md): an explanation of how PCIe TLP packets work for Host-to-FPGA and FPGA-to-Host communication
- [**Addresses Explained**](supplementary_information/addresses_explained.md): an explanation of how addresses are encoded/
- **Bit Encoding** = explanation of how information is standardly encoded for hardware in hex code

```{toctree}
:hidden:
:maxdepth: 2
:caption: Hardware Guide Book

introduction
chapter_1
chapter_2
chapter_3
chapter_4
supplementary_information
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: PCIe and HBM Packet Definitions

pcie_hbm_packet_definitions
```
