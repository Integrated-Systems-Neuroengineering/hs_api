# 3 Verilog Files Review

## Overview

The Verilog File directory contains the FPGA hardware implementation for a neuromorphic computing system designed to simulate large-scale spiking neural networks. This implementation targets Xilinx XCVU37p FPGAs with High Bandwidth Memory (HBM) and is part of the CRI (Cognitive Research Infrastructure) neuromorphic computing cluster at San Diego Supercomputer Center.

### System Scale
- **40 FPGA boards** across 5 compute servers
- **32 cores per FPGA** (each core can process 128K neurons)
- **4M neurons per FPGA**, 160M neurons total system capacity
- **400+ GBps HBM bandwidth** per board

### Key Features
- Supports **16 neuron groups** per core (8192 neurons each)
- **Multi-clock domain design**: 225 MHz (aclk) and 450 MHz (aclk450)
- **Three-tier memory hierarchy**:
  - BRAM for axon/external event data
  - URAM for neuron state data
  - HBM for synaptic connectivity data
- **PCIe Gen3 x16 interface** for host communication via DMA
- **Two-phase execution model**: External events (Phase 1) → Internal updates (Phase 2)

---

## High-Level Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │         Host Computer (Python)              │
                    │    hs_bridge: Network definition & control  │
                    └──────────────────┬──────────────────────────┘
                                       │
                                  PCIe Gen3 x16
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │           FPGA (Xilinx XCVU37p)             │
                    │                                              │
                    │  ┌────────────────────────────────────────┐ │
                    │  │      pcie2fifos (AXI4 to FIFO)         │ │
                    │  │  512-bit data path @ 225 MHz           │ │
                    │  └──────┬─────────────────────┬───────────┘ │
                    │         │                     │              │
                    │    Input FIFO            Output FIFO         │
                    │         │                     │              │
                    │  ┌──────▼─────────────────────▼───────────┐ │
                    │  │      command_interpreter                │ │
                    │  │  Parses commands, routes to modules     │ │
                    │  └──┬─────────┬──────────┬─────────┬──────┘ │
                    │     │         │          │         │         │
                    │     │         │          │         │         │
        ┌───────────┼─────▼─────────┴──────────┴─────────┴─────┐  │
        │  BRAM     │  input_data_handler (BRAM Arbiter)        │  │
        │  2^15 x   │  - Arbitrates CI vs EEP access            │  │
        │  256-bit  │  - 3-cycle read latency                   │  │
        └───────────┼──────┬──────────────────────────┬─────────┘  │
                    │      │                          │             │
                    │  ┌───▼──────────────┐   ┌───────▼──────────┐ │
                    │  │ external_events_ │   │  spike_fifo_     │ │
                    │  │    processor     │   │   controller     │ │
                    │  │ - Reads axon/    │   │ - Collects spikes│ │
                    │  │   BRAM data      │   │   from 8 FIFOs   │ │
                    │  │ - Generates HBM  │   │ - Round-robin    │ │
                    │  │   read requests  │   │   arbitration    │ │
                    │  └───┬──────────────┘   └──────────────────┘ │
                    │      │                                        │
                    │  ┌───▼────────────────────────────────────┐  │
                    │  │       hbm_processor                     │  │
                    │  │  - Manages HBM access (AXI4)            │  │
                    │  │  - Reads synaptic pointers              │  │
                    │  │  - Prefetches synapse data              │  │
                    │  │  - Handles pointer chains               │  │
                    │  └───┬────────────────────────────────┬───┘  │
                    │      │                                │       │
        ┌───────────┼──────▼────────────────────┐   ┌──────▼─────┐ │
        │  HBM      │  pointer_fifo_controller  │   │ hbm_       │ │
        │  33-bit   │  - 16 pointer FIFOs       │   │ register_  │ │
        │  address  │  - Round-robin dispatch   │   │ slice      │ │
        │  256-bit  │  - Manages BRAM/URAM      │   │ (timing)   │ │
        │  data     │    spike flags            │   └────────────┘ │
        │  400+     └───┬───────────────────────┘                  │
        │  GBps                │                                    │
        └───────────┼──────────▼────────────────────────────────┐  │
                    │  internal_events_processor                │  │
                    │  - Processes 16 URAM banks                │  │
                    │  - Updates neuron states                  │  │
                    │  - Detects spikes                         │  │
                    │  - Read-modify-write hazard resolution    │  │
        ┌───────────┼──────────────────────────────────────────┐│  │
        │  URAM     │  16 banks x 4096 rows x 72 bits          ││  │
        │  12-bit   │  (2 neurons per 72-bit word)             ││  │
        │  address  │  @450 MHz for high throughput            ││  │
        │  72-bit   └──────────┬────────────────────────────────┘│  │
        │  data                │                                  │  │
        └──────────────────────┼──────────────────────────────────┘  │
                    │          │                                     │
                    │     Spike outputs → spike_fifo_controller      │
                    │          (17-bit neuron addresses)             │
                    └────────────────────────────────────────────────┘
```

---

## Directory Structure

### `CRI_proj/` - Single-Core Verilog Implementation

Original single-core implementation using Verilog. Contains the core processing modules:

| Module | Purpose | Key Features |
|--------|---------|--------------|
| **command_interpreter.v** | PCIe command interface | Parses commands, routes to processors |
| **pcie2fifos.v** | PCIe to FIFO bridge | AXI4 512-bit interface, handles DMA |
| **input_data_handler.v** | BRAM arbiter | Arbitrates between command interpreter and external events processor |
| **external_events_processor.v** | Axon event processing | Reads BRAM, generates HBM requests for synapses |
| **hbm_processor.v** | HBM memory controller | Manages synapse data access, pointer chaining |
| **pointer_fifo_controller.v** | Pointer distribution | Distributes pointers to 16 neuron groups |
| **internal_events_processor.v** | Neuron state updates | Processes 16 URAM banks, updates neurons, detects spikes |
| **spike_fifo_controller.v** | Spike collection | Collects spikes from 8 FIFOs, round-robin arbitration |

**Variants:**
- `external_events_processor_simple.v` - Simplified version with fixed pipeline depth
- `external_events_processor_v2.v` - Enhanced version with improved timing

### `N_cores/` - Multi-Core SystemVerilog Implementation

Multi-core implementation using SystemVerilog with improved modularity and timing closure:

| Module | Purpose | Key Features |
|--------|---------|--------------|
| **single_core.sv** | Top-level core integration | Instantiates all processors, 16 URAMs, FIFOs |
| **core_wrapper.sv** | Core wrapper with reset sync | Adds HBM register slice for timing closure |
| **types.sv** | Interface definitions | AXI4, AXILite, AXIStream, FIFO, RAM interfaces |
| **Xilinx_IP_wrappers.sv** | Xilinx IP macros | FIFO and URAM wrapper macros |
| **FIFO_AXI_Converters.sv** | Protocol converters | FIFO ↔ AXI Stream conversion modules |
| **reset_synchronizer.sv** | Reset synchronization | 2-FF synchronizer for clock domain crossing |

---

## Memory Organization

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  Memory Type  │  Size/Dimensions  │  Purpose               │
├───────────────┼───────────────────┼────────────────────────┤
│  BRAM         │  2^15 x 256-bit   │  Axon/External Events  │
│  (Block RAM)  │  = 1 MB           │  - Row addresses       │
│               │                   │  - Spike masks         │
├───────────────┼───────────────────┼────────────────────────┤
│  URAM         │  16 banks         │  Neuron States         │
│  (UltraRAM)   │  4096 x 72-bit    │  - Membrane potential  │
│               │  = 294 Kb/bank    │  - Threshold           │
│               │  @450 MHz         │  - Refractory state    │
│               │                   │  (2 neurons/word)      │
├───────────────┼───────────────────┼────────────────────────┤
│  HBM          │  33-bit address   │  Synaptic Connectivity │
│  (High        │  256-bit data     │  - Pointer chains      │
│   Bandwidth   │  400+ GBps        │  - Synapse weights     │
│   Memory)     │  8 GB total       │  - Target neuron IDs   │
└───────────────┴───────────────────┴────────────────────────┘
```

### Address Space Layout

**BRAM Address (15-bit):**
```
[14:0] Row address → 32,768 rows
       Each row: 256 bits = 16 x 16-bit masks (one per neuron group)
```

**URAM Address (12-bit per bank):**
```
[11:0] Row address → 4,096 rows per bank
       16 banks total
       72 bits per row = 2 neurons x 36 bits each
       Total: 8,192 neurons per bank x 16 banks = 131,072 neurons
```

**HBM Address (33-bit):**
```
[32:0] Byte address → 8 GB addressable space
       Stores synapse data in pointer-chain format:
       [31:0]  Next pointer (32-bit HBM address)
       [47:32] Synapse weight (16-bit)
       [63:48] Target neuron ID (16-bit)
```

---

## Data Flow and Execution Phases

### Phase 1: External Event Processing

```
1. Host → PCIe → pcie2fifos → Input FIFO
2. command_interpreter extracts axon spike events
3. external_events_processor:
   - Reads BRAM (axon data) via input_data_handler
   - Generates HBM read requests for synapse pointers
4. hbm_processor:
   - Fetches synapse pointer chains from HBM
   - Prefetches synapse data
5. pointer_fifo_controller:
   - Distributes pointers to 16 neuron groups
   - Sets spike flags in BRAM/URAM
```

### Phase 2: Internal Event Processing

```
1. internal_events_processor:
   - Reads 16 URAM banks (neuron states) @450 MHz
   - Applies synaptic inputs from pointer FIFOs
   - Updates membrane potentials
   - Detects threshold crossings (spikes)
   - Writes back updated states
2. Spike outputs → spike_fifo_controller
3. spike_fifo_controller:
   - Collects from 8 spike FIFOs (round-robin)
   - Sends spike events back to external_events_processor
   OR sends to host via Output FIFO
4. Output FIFO → pcie2fifos → PCIe → Host
```

---

## Clock Domains

The design uses two primary clock domains:

| Clock Domain | Frequency | Usage |
|--------------|-----------|-------|
| **aclk** | 225 MHz | PCIe, command interpreter, BRAM, most processing logic |
| **aclk450** | 450 MHz | URAM access for high-throughput neuron updates |

**Clock Domain Crossing:**
- Reset signals synchronized with `reset_synchronizer.sv` (2-FF synchronizer)
- Data crossing handled by async FIFOs with independent read/write clocks
- Critical path timing improved with `hbm_register_slice` in AXI4 HBM interface

---

## Module Hierarchy

```
single_core (top-level)
├── core_wrapper
│   ├── reset_synchronizer (aclk domain)
│   ├── reset_synchronizer (aclk450 domain)
│   ├── hbm_register_slice (AXI4 pipeline)
│   └── core
│       ├── pcie2fifos (PCIe ↔ FIFO bridge)
│       ├── command_interpreter
│       ├── input_data_handler (BRAM arbiter)
│       │   └── BRAM (2^15 x 256)
│       ├── external_events_processor
│       ├── hbm_processor
│       │   └── AXI4 Master (HBM interface)
│       ├── pointer_fifo_controller
│       │   └── 16 x pointer_FIFO (32-bit)
│       ├── internal_events_processor
│       │   └── 16 x URAM (4096 x 72-bit) @450MHz
│       └── spike_fifo_controller
│           └── 8 x spike_FIFO (17-bit)
```

---

## Key Terms and Definitions

| Term | Definition |
|------|------------|
| **Axon** | Presynaptic neuron output; external events stored in BRAM |
| **BRAM** | Block RAM - On-chip memory for axon/external event data |
| **HBM** | High Bandwidth Memory - Off-chip DRAM for synaptic connectivity |
| **URAM** | UltraRAM - High-density on-chip memory for neuron states |
| **Neuron Group** | 8,192 neurons sharing a URAM bank and processing pipeline |
| **Pointer Chain** | Linked-list structure in HBM storing synapses for each axon |
| **Spike** | Action potential - neuron firing event when threshold is crossed |
| **Synapse** | Connection between neurons with associated weight |
| **AXI4** | ARM Advanced eXtensible Interface - High-performance memory protocol |
| **DMA** | Direct Memory Access - Host-FPGA data transfer without CPU involvement |
| **FIFO** | First-In-First-Out buffer for asynchronous data transfer |
| **Round-Robin** | Fair scheduling algorithm cycling through multiple requesters |
| **Register Slice** | Pipeline stage for timing closure in high-speed interfaces |
| **FWFT** | First-Word Fall-Through - FIFO mode with zero-latency reads |

---

## Interface Specifications

### PCIe Interface (pcie2fifos.v)
- **Protocol:** AXI4 (512-bit data width)
- **Clock:** 225 MHz (aclk)
- **Bandwidth:** ~14 GB/s theoretical (512 bits × 225 MHz / 8)
- **Latency:** Command-to-response ~10-20 cycles typical

### HBM Interface (hbm_processor.v)
- **Protocol:** AXI4 (256-bit data width)
- **Address Width:** 33 bits (8 GB addressable)
- **Clock:** 225 MHz (aclk)
- **Bandwidth:** 400+ GB/s (multiple HBM channels)
- **Latency:** ~100-200 ns typical read latency

### BRAM Interface (input_data_handler.v)
- **Width:** 256 bits
- **Depth:** 32,768 rows (15-bit address)
- **Read Latency:** 3 cycles
- **Arbitration:** Command interpreter has priority over external events processor

### URAM Interface (internal_events_processor.v)
- **Width:** 72 bits (2 neurons × 36 bits)
- **Depth:** 4,096 rows per bank (12-bit address)
- **Banks:** 16 independent banks
- **Clock:** 450 MHz (aclk450)
- **Read Latency:** 1 cycle
- **Special Feature:** Read-modify-write hazard detection and resolution

---

## Cross-References

### Software Stack Integration
- **Python API:** `hs_bridge/` directory contains the host-side software
  - `network.py`: High-level network definition
  - `compile_network.py`: Converts network to HBM memory layout
  - `FPGA_Execution/fpga_controller.py`: Sends commands to command_interpreter
  - `wrapped_dmadump/`: DMA library for PCIe communication

### Hardware Configuration
- **System Information:** `CRI_stack_information` - Details on 40-board cluster
- **Configuration Files:** `hs_bridge/config.yaml` - FPGA/network parameters

### Documentation Flow
1. Start with this README for overall architecture
2. Read `CRI_proj/` or `N_cores/` module documentation for implementation details
3. Refer to interface specifications for integration
4. Cross-reference with Python code for command protocols

---

## Implementation Notes

### Design Evolution
The codebase shows evidence of scaling from 8 to 16 neuron groups:
- Commented code in `spike_fifo_controller.v` shows previous 16-FIFO support (now 8)
- `pointer_fifo_controller.v` has full 16-FIFO implementation
- Address widths expanded: 12-bit → 13-bit for row addresses

### Performance Optimization
- **Pipeline Depth:** External events processor uses 3-stage pipeline
- **Register Slicing:** HBM interface includes register slice for timing closure
- **Dual Clock:** URAM runs at 450 MHz (2x) for doubled throughput
- **Round-Robin:** Fair arbitration prevents starvation in multi-FIFO controllers

### Safety Features
- **Reset Synchronization:** Prevents metastability across clock domains
- **Hazard Resolution:** Read-modify-write conflicts detected in internal_events_processor
- **Flow Control:** FIFO full/empty signals prevent data loss
- **Default States:** All state machines include default cases returning to reset

---

## Getting Started

### Prerequisites
- Xilinx Vivado for FPGA synthesis (version compatible with XCVU37p)
- Python 3.x with hs_bridge package for host control
- PCIe Gen3 x16 connection to FPGA board
- Access to CRI cluster or compatible hardware setup

### Build Flow
1. Synthesize modules in `CRI_proj/` or `N_cores/`
2. Integrate with Xilinx HBM and PCIe IP cores
3. Place and route with timing constraints for 225/450 MHz
4. Generate bitstream and program FPGA
5. Use `hs_bridge` Python library to configure and run networks

### Testing
- Unit tests: Simulate individual modules with test vectors
- Integration tests: Full-core simulation with spike input/output
- Hardware validation: Use simple network patterns to verify connectivity

---

## Future Development

Potential areas for enhancement:
- **Scaling:** Support for 32+ neuron groups per core
- **Precision:** Configurable bit widths for weights and potentials
- **Learning:** Hardware support for on-chip STDP or other plasticity rules
- **Multi-Core:** Inter-core communication for distributed networks
- **Monitoring:** Built-in performance counters and debug interfaces

---

## Contact and Support

For questions about this hardware implementation:
- Refer to `hs_bridge` documentation for software interface
- Check individual module documentation for detailed logic descriptions
- See `CRI_stack_information` for system architecture and configuration

---

**Last Updated:** December 2025 (Generated Documentation)
**Hardware Version:** 16 neuron groups, 225/450 MHz dual-clock design
**Target Device:** Xilinx XCVU37p with HBM


```{toctree}
:hidden:
:maxdepth: 1

3_Verilog_Files_Review/command_interpreter
3_Verilog_Files_Review/external_events_processor
3_Verilog_Files_Review/hbm_processor
3_Verilog_Files_Review/input_data_handler
3_Verilog_Files_Review/internal_events_processor
3_Verilog_Files_Review/pcie2fifos
3_Verilog_Files_Review/pointer_fifo_controller
3_Verilog_Files_Review/spike_fifo_controller
```
