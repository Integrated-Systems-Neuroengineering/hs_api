# internal_events_processor.v

## Module Overview

### Purpose and Role in Stack

The **internal_events_processor** is the **core neuron computation engine**, responsible for updating neuron states and detecting spikes. This module:

- **Manages 16 URAM banks** storing neuron states (131,072 neurons total)
- **Applies synaptic inputs** from HBM processor via pointer FIFOs
- **Updates membrane potentials** using configurable neuron models
- **Detects threshold crossings** (spike generation)
- **Implements two-phase execution:**
  - **Phase 1**: Read neuron states while processing external events
  - **Phase 2**: Update neurons with synaptic inputs from HBM
- **Provides host access** for reading/writing individual neuron states

In the software/hardware stack:
```
External Events → HBM Processor → Pointer FIFO Controller
                        ↓                     ↓
                   Synapse Data       Neuron Addresses
                        ↓                     ↓
                  internal_events_processor
                        ↓
                  16 × URAM Banks (neuron states)
                        ↓
                  Spike Detection → Spike FIFOs
```

This module is the heart of the neuromorphic computation, where all neuron dynamics occur.

---

## Module Architecture

### High-Level Block Diagram

```
              internal_events_processor
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  ┌───────────────────────────────────────────────┐     │
    │  │   Command Interpreter Interface               │     │
CI  │  │   - Read/write individual neurons             │     │
FIFO├─►│   - ci2iep_dout[53:0]                         │     │
    │  │     [53]=R/W, [52:49]=group, [48:36]=row      │     │
    │  │     [35:0]=data                                │     │
    │  └────────────────────────┬──────────────────────┘     │
    │                           │                            │
    │  ┌────────────────────────▼──────────────────────┐     │
    │  │   Address Multiplexer                         │     │
    │  │   - Phase 1: Sequential scan (uram_raddr)     │     │
    │  │   - Phase 2: HBM addresses (exec_hbm_rdata)   │     │
    │  │   - CI access: SET_ROW from command           │     │
    │  └────────────────────────┬──────────────────────┘     │
    │                           │                            │
    │  ┌────────────────────────▼──────────────────────┐     │
    │  │   16 × URAM Interfaces (72-bit each)          │     │
URAM│  │   - Each URAM: 4096 × 72-bit                  │     │
    ├─►│   - 2 neurons per word (36-bit each)          │     │
    │  │   - Read latency: 1 cycle                     │     │
    │  │   - Addresses: [12:1] (bit [0] selects neuron)│     │
    │  └────────────────────────┬──────────────────────┘     │
    │                           │                            │
    │  ┌────────────────────────▼──────────────────────┐     │
    │  │   Read-Modify-Write Hazard Resolution         │     │
    │  │   - Detects consecutive writes to same addr   │     │
    │  │   - Uses uram_rmwdata registers               │     │
    │  │   - Compares: uram_waddr == uram_waddr_reg    │     │
    │  └────────────────────────┬──────────────────────┘     │
    │                           │                            │
    │  ┌────────────────────────▼──────────────────────┐     │
    │  │   HBM Data Processing                         │     │
HBM │  │   - 512-bit packets (16 × 32-bit)             │     │
Data├─►│   - [511,479,447...31]: Null flags (16 bits)  │     │
    │  │   - [495:480,...,015:000]: Weights (16×16-bit)│     │
    │  │   - [508:496,...,028:016]: Neuron IDs (16×13) │     │
    │  │   - Sign extension: 16-bit → 36-bit           │     │
    │  └────────────────────────┬──────────────────────┘     │
    │                           │                            │
    │  ┌────────────────────────▼──────────────────────┐     │
    │  │   Neuron Model Selection                      │     │
    │  │   exec_neuron_model[1:0]:                     │     │
    │  │   0: Memoryless (reset to 0)                  │     │
    │  │   1: Incremental (+group_id per timestep)     │     │
    │  │   2: Leaky I&F (decay: >> 3)                  │     │
    │  │   3: Non-leaky (no decay)                     │     │
    │  └────────────────────────┬──────────────────────┘     │
    │                           │                            │
    │  ┌────────────────────────▼──────────────────────┐     │
    │  │   Membrane Potential Update                   │     │
    │  │   - Phase 1: Apply decay/increment            │     │
    │  │   - Phase 2: Add synaptic weight              │     │
    │  │   - Check threshold crossing                  │     │
    │  │   - Reset if spiked                           │     │
    │  └────────────────────────┬──────────────────────┘     │
    │                           │                            │
    │  ┌────────────────────────▼──────────────────────┐     │
    │  │   Spike Detection                             │     │
    │  │   - exec_uram_spiked[15:0]                    │     │
    │  │   - 1 bit per neuron group                    │     │
    │  │   - Used by pointer_fifo_controller           │     │
    │  └───────────────────────────────────────────────┘     │
    │                                                         │
    │  ┌───────────────────────────────────────────────┐     │
    │  │   State Machine                               │     │
    │  │   IDLE → FILL_PIPE → WAIT_BRAM → PUSH_PTR →  │     │
    │  │   PHASE1_DONE → POP_PTR → PHASE2_DONE → IDLE │     │
    │  └───────────────────────────────────────────────┘     │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

---

## Interface Specification

### Clock and Reset

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `clk` | Input | 1 | 225 MHz system clock (note: NOT 450 MHz as initially planned) |
| `resetn` | Input | 1 | Active-low synchronous reset |

**Note:** Comments in code suggest URAMs may run at 450 MHz in some configurations, but this module operates at 225 MHz.

### Network Parameters

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `num_outputs` | Input | 17 | Number of output neurons (max 131,072) |
| `threshold` | Input | 36 (signed) | Spike threshold for membrane potential |
| `exec_neuron_model` | Input | 2 | Neuron model selection (0-3) |

**Neuron Models:**
```
2'd0 = Memoryless (resets to 0 after spike or each timestep)
2'd1 = Incremental (adds group_id each cycle, for testing)
2'd2 = Leaky Integrate-and-Fire (decay: V -= V>>3 each cycle)
2'd3 = Non-leaky (holds value, no decay)
```

### Execution Control

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `exec_run` | Input | 1 | Start new time-step execution |
| `exec_bram_phase1_done` | Input | 1 | External events processor finished Phase 1 |
| `exec_uram_phase1_ready` | Output (reg) | 1 | Pipeline filled, ready for synaptic data |
| `exec_uram_phase1_done` | Output (reg) | 1 | Phase 1 complete (all neurons scanned) |
| `exec_uram_phase2_done` | Output (reg) | 1 | Phase 2 complete (all updates applied) |
| `exec_hbm_rx_phase2_done` | Input | 1 | HBM processor finished sending data |

**Phase Sequence:**
```
1. exec_run pulse
2. Fill URAM pipeline (2 cycles)
3. Wait for exec_bram_phase1_done
4. assert exec_uram_phase1_ready
5. Process HBM data (Phase 1)
6. assert exec_uram_phase1_done
7. Process remaining HBM data (Phase 2)
8. assert exec_uram_phase2_done
9. Return to IDLE
```

### HBM Processor Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `exec_hbm_rdata` | Input | 512 | Synapse data packet (16 neuron groups) |
| `exec_hbm_rvalidready` | Input | 1 | HBM data valid (FIFO not empty) |
| `hbm2iep_rden` | Output (wire) | 1 | Pop HBM data FIFO |

**HBM Data Packet Format (`exec_hbm_rdata[511:0]`):**

Per neuron group (32 bits × 16 groups = 512 bits):
```
Group 0:  [511]    = Null flag (1=no data)
          [510:509]= Reserved (2 bits)
          [508:496]= Target neuron address (13 bits)
          [495:480]= Synaptic weight (16-bit signed)

Group 1:  [479:448] (same structure)
...
Group 15: [031:000] (same structure)
```

**Null Flag Behavior:**
- If `exec_hbm_rdata[group*32 + 31]` == 1: No synapse, weight forced to 0
- Otherwise: Use `exec_hbm_rdata[group*32 + 15:group*32]` as weight

### Spike Output

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `exec_uram_spiked` | Output (reg) | 16 | Spike flags (1 per neuron group) |

**Usage:**
- `exec_uram_spiked[i]` = 1 if neuron in group `i` crossed threshold
- Consumed by `pointer_fifo_controller`
- Valid during Phase 1 when `exec_uram_phase1_ready & !exec_uram_phase1_done`

### Command Interpreter Interface

**Input (CI to IEP):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `ci2iep_empty` | Input | 1 | Command FIFO empty flag |
| `ci2iep_dout` | Input | 54 | Command data |
| `ci2iep_rden` | Output (reg) | 1 | Command FIFO read enable |

**Command Format (`ci2iep_dout[53:0]`):**
```
[53]      = R/W (0=read, 1=write)
[52:49]   = Neuron group (4 bits, 0-15)
[48:36]   = Row address (13 bits)
[35:0]    = Data (36-bit signed membrane potential)
```

**Output (IEP to CI):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `iep2ci_full` | Input | 1 | Response FIFO full flag |
| `iep2ci_din` | Output (reg) | 53 | Response data |
| `iep2ci_wren` | Output (reg) | 1 | Response FIFO write enable |

**Response Format (`iep2ci_din[52:0]`):**
```
[52:49]   = Neuron group (echoed from request)
[48:36]   = Row address (echoed from request)
[35:0]    = Membrane potential (read data)
```

### URAM Interfaces (16 banks)

**Read Interface (per bank):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `uram_raddr_0` ... `uram_raddr_15` | Output (reg) | 12 | Read address (4096 rows) |
| `uram_rden_0` ... `uram_rden_15` | Output (reg) | 1 | Read enable |
| `uram_rdata_0` ... `uram_rdata_15` | Input | 72 | Read data (2 neurons) |

**Write Interface (per bank):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `uram_waddr_0` ... `uram_waddr_15` | Output (wire) | 12 | Write address |
| `uram_wdata_0` ... `uram_wdata_15` | Output (reg) | 72 | Write data (2 neurons) |
| `uram_wren_0` ... `uram_wren_15` | Output (wire) | 1 | Write enable |

**URAM Data Format (72-bit word):**
```
[71:36] = Upper neuron membrane potential (36-bit signed)
[35:0]  = Lower neuron membrane potential (36-bit signed)

Address mapping:
  Full neuron address: [16:0] (131,072 neurons)
  [16:13] = Neuron group (4 bits, 0-15)
  [12:1]  = URAM row address (12 bits, 0-4095)
  [0]     = Neuron select (0=lower [35:0], 1=upper [71:36])
```

### Debug Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `iep_curr_state` | Output (wire) | 4 | Current state machine state (for VIO) |
| `curr_uram_waddr` | Output (wire) | 13 | Current write address (for VIO) |

---

## Detailed Logic Description

### State Machine

**States:**
```verilog
STATE_RESET                 (4'd0)  - Reset state
STATE_IDLE                  (4'd1)  - Wait for commands
STATE_FILL_PIPE_PHASE1      (4'd2)  - Fill URAM read pipeline
STATE_WAIT_BRAM_PHASE1_DONE (4'd3)  - Wait for external events
STATE_PUSH_PTR_FIFO         (4'd4)  - Process HBM data (Phase 1)
STATE_PHASE1_DONE           (4'd5)  - Phase 1 complete
STATE_POP_PTR_FIFO          (4'd6)  - Process HBM data (Phase 2)
STATE_PHASE2_DONE           (4'd7)  - Phase 2 complete
STATE_READ_URAM_0           (4'd8)  - CI read (cycle 1)
STATE_READ_URAM_1           (4'd9)  - CI read (cycle 2, send response)
STATE_WRITE_URAM_0          (4'd11) - CI write (read for RMW)
STATE_WRITE_URAM            (4'd10) - CI write (perform write)
```

**State Transition Diagram (Execution Flow):**

```
        ┌──────────────┐
        │ STATE_RESET  │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
    ┌──▶│ STATE_IDLE   │◄───────────────────┐
    │   └──┬───────┬───┘                    │
    │      │       │                        │
    │ exec_run      │ !ci2iep_empty          │
    │      │       │  & phase2_done         │
    │      │       ├─ R/W==0 ──> READ_URAM_0 ──> READ_URAM_1 ─┘
    │      │       └─ R/W==1 ──> WRITE_URAM_0 ──> WRITE_URAM ─┘
    │      │
    │      ▼
    │ FILL_PIPE_PHASE1
    │  (2 cycles)
    │      │
    │      ▼
    │ WAIT_BRAM_PHASE1_DONE
    │  (wait for external events)
    │      │ exec_bram_phase1_done
    │      ▼
    │ PUSH_PTR_FIFO
    │  (process HBM data)
    │  (assert phase1_ready)
    │      │ uram_waddr[0] == LIMIT
    │      ▼
    │ PHASE1_DONE
    │  (assert phase1_done)
    │      │
    │      ▼
    │ POP_PTR_FIFO
    │  (continue HBM processing)
    │      │ exec_hbm_rx_phase2_done
    │      ▼
    │ PHASE2_DONE
    │  (assert phase2_done)
    │      │
    └──────┘
```

### Read-Modify-Write (RMW) Hazard Resolution

**Problem:** Writing to the same URAM address in consecutive cycles creates a read-before-write hazard.

**Solution:** `uram_rmwdata` registers bypass stale URAM data when addresses match.

**Logic:**
```verilog
// Detect hazard: same address, same group, consecutive writes
wire hazard = (uram_waddr[i] == uram_waddr_reg[i]) &&
              (SET_GROUP == SET_GROUP_reg) &&
              uram_wren[i] && uram_wren[i]_reg;

// Use bypassed data if hazard, otherwise use URAM read data
uram_rmwdata[i] = hazard ? uram_wdata_reg[i] : uram_rdata[i];
```

**Example Scenario:**
```
Cycle 0: Write neuron 100 (group 0, row 50, lower)
Cycle 1: Write neuron 101 (group 0, row 50, upper)
         - Same URAM row (50), different neuron within word
         - URAM read returns OLD data (write not yet complete)
         - RMW logic uses wdata_reg[0] from previous cycle instead
```

**Special Cases:**
- Different neuron groups: RMW disabled (writes to different URAMs)
- CI write states: RMW checks both address AND group match
- Normal execution: RMW checks only address (all groups access simultaneously)

### Neuron Update Logic

**Phase 1: Baseline Decay/Increment**

During `PUSH_PTR_FIFO` state (before synaptic inputs applied):

```verilog
// For each neuron group, increment based on neuron model
if (potential > threshold) {
    new_potential = 0;  // Reset after spike
} else {
    switch (exec_neuron_model) {
        case 0: new_potential = 0;              // Memoryless
        case 1: new_potential = potential + group_id; // Incremental
        case 2: new_potential = potential - (potential >>> 3); // Leaky (divide by 8)
        case 3: new_potential = potential;      // Non-leaky
    }
}
```

**Note:** Different neuron groups add different increments in model 1:
- Group 0: +1, Group 1: +2, ..., Group 15: +16

This creates distinguishable behavior for testing.

**Phase 2: Synaptic Input Addition**

During `POP_PTR_FIFO` state (when `exec_hbm_rvalidready`):

```verilog
// Extract weight from HBM packet (16-bit signed)
weight_16bit = exec_hbm_rdata[group*32 + 15 : group*32];

// Sign-extend to 36-bit
weight_36bit = (weight_16bit[15]) ? {20'hFFFFF, weight_16bit}
                                  : {20'h00000, weight_16bit};

// If null flag set, force weight to 0
if (exec_hbm_rdata[group*32 + 31])
    weight_36bit = 0;

// Add to membrane potential (no saturation implemented)
new_potential = current_potential + weight_36bit;
```

### Spike Detection

```verilog
// During Phase 1, check threshold crossings
for (group = 0; group < 16; group++) {
    // Determine which neuron in the word (based on LSB of address)
    if (~uram_raddr_full[group][0])
        neuron_potential = uram_rmwdata_upper[group];
    else
        neuron_potential = uram_rmwdata_lower[group];

    // Set spike flag
    exec_uram_spiked[group] = (neuron_potential > threshold);
}
```

**Output:** 16-bit vector indicating which neuron groups have spiking neurons.

**Usage:** `pointer_fifo_controller` uses this to:
1. Set spike flags in BRAM/URAM
2. Generate spike addresses for output FIFOs

### Address Multiplexing

**Three Address Sources:**

1. **Phase 1 Sequential Scan:**
   ```verilog
   uram_raddr_full[i] = uram_raddr[12:0];  // All groups same address
   uram_raddr[i] = uram_raddr_full[12:1];  // Drop LSB for URAM address
   ```

2. **Phase 2 HBM-Directed:**
   ```verilog
   // Extract neuron address from HBM packet (13 bits per group)
   uram_raddr_full[0] = exec_hbm_rdata[508:496];
   uram_raddr_full[1] = exec_hbm_rdata[476:464];
   ...
   uram_raddr_full[15] = exec_hbm_rdata[028:016];
   ```

3. **CI Access:**
   ```verilog
   uram_raddr_full[i] = SET_ROW[12:0];  // From ci2iep_dout[48:36]
   ```

**Write Address Pipeline:**
- Write address = registered read address (1-cycle delay)
- Allows read-modify-write pattern
- Hazard resolution ensures correct data

### URAM Read/Write Coordination

**Read Cycle:**
```
Cycle N:   Assert uram_rden[i], set uram_raddr[i]
Cycle N+1: uram_rdata[i] valid (1-cycle latency)
           Register to uram_waddr[i] via uram_raddr_full
Cycle N+2: Use uram_rdata[i] or uram_rmwdata[i] for computation
           Assert uram_wren[i], output uram_wdata[i]
```

**Write Cycle:**
```
Cycle N:   Compute new_potential
           Set uram_wdata[i] = {upper_neuron, lower_neuron}
Cycle N+1: URAM performs write
           Data becomes readable on next access
```

---

## Memory Map

### URAM Organization

**Per Bank:**
- **Depth:** 4096 rows (12-bit address)
- **Width:** 72 bits (2 × 36-bit neurons)
- **Total:** 294,912 bits per bank

**System Total:**
- **16 banks** × 8192 neurons = **131,072 neurons**
- **16 banks** × 294 Kb = **4.7 Mb total URAM**

**Address Breakdown:**

```
Neuron Address [16:0]:
  [16:13] = Neuron group (4 bits) → Selects URAM bank (0-15)
  [12:1]  = Row address (12 bits) → URAM row (0-4095)
  [0]     = Neuron select → Position in 72-bit word

URAM Bank Assignment:
  Group 0:  Neurons 0 - 8191
  Group 1:  Neurons 8192 - 16383
  ...
  Group 15: Neurons 122880 - 131071
```

**Example:**
- Neuron 10000 (decimal) = 0x2710 (hex) = 0b00010_0111000_10000
  - Group: 0b0010 = 2
  - Row: 0b011100010000 = 912
  - Position: 0 (lower 36 bits)
  - URAM: Bank 2, Row 912, Bits [35:0]

---

## Timing Diagrams

### Execution: Phase 1 (Sequential Scan)

```
Cycle:   0      1      2      3      4      5      ...    N
         │      │      │      │      │      │      │      │
exec_run ▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
         │      │      │      │      │      │      │      │
State    IDLE   │FILL  │FILL  │WAIT  │WAIT  │PUSH  │PUSH  │
         │      │_PIPE │_PIPE │_BRAM │_BRAM │_PTR  │_PTR  │
         │      │      │      │      │      │      │      │
uram_    ▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│
rden     │      │      │      │      │      │      │      │
         │      │      │      │      │      │      │      │
uram_    0      │0      │1      │1      │1      │1      │2
raddr    │      │      │      │      │      │      │      │
         │      │      │      │      │      │      │      │
exec_    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│
uram_    │      │      │      │      │      │      │      │
phase1   │      │      │      │      │      │      │      │
_ready   │      │      │      │      │      │      │      │
         │      │      │      │      │      │      │      │
exec_hbm ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▁▁▁▁│▔▔▁▁▁▁│
_rvalid  │      │      │      │      │      │      │      │
ready    │      │      │      │      │      │      │      │
```

**Notes:**
- Cycles 1-2: Fill pipeline with 2 reads
- Cycles 3-4: Wait for `exec_bram_phase1_done`
- Cycle 5+: Process HBM data, increment address on each `rvalidready`

### Execution: Phase 2 (HBM-Directed Updates)

```
Cycle:   N      N+1    N+2    N+3    N+4    N+5
         │      │      │      │      │      │
State    PUSH   │PHASE1│POP   │POP   │POP   │PHASE2
         _PTR   │_DONE │_PTR  │_PTR  │_PTR  │_DONE
         │      │      │      │      │      │
exec_hbm ▔▔▔▔▔▔▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁│▔▔▔▔▔▔▁▁
_rvalid  │      │      │      │      │      │
ready    │      │      │      │      │      │
         │      │      │      │      │      │
exec_hbm DATA1 │DATA1 │DATA2 │DATA2 │DATA3 │DATA3
_rdata   │      │      │      │      │      │
         │      │      │      │      │      │
uram_    ▔▔▔▔▔▔▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁│▔▔▔▔▔▔▁▁
rden     │      │      │      │      │      │
         │      │      │      │      │      │
uram_    ADDR1 │ADDR1 │ADDR2 │ADDR2 │ADDR3 │ADDR3
raddr_i  │(from │      │(from │      │(from │
         │ HBM) │      │ HBM) │      │ HBM) │
         │      │      │      │      │      │
uram_    ▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔
wren_i   │      │      │      │      │      │
         │      │      │      │      │      │
uram_    XXXX   │UPDATE│UPDATE│UPDATE│UPDATE│UPDATE
wdata_i  │      │(+wt1)│(+wt2)│(+wt2)│(+wt3)│(+wt3)
```

**Notes:**
- Each HBM packet triggers read of 16 neuron addresses (one per group)
- Read data valid after 1 cycle
- Write occurs 1 cycle after read
- Process continues until `exec_hbm_rx_phase2_done`

### CI Read Transaction

```
Cycle:   0      1      2      3      4
         │      │      │      │      │
State    IDLE   │READ  │READ  │IDLE  │
         │      │_0    │_1    │      │
         │      │      │      │      │
ci2iep   ▔▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁
_empty   │      │      │      │      │
         │      │      │      │      │
ci2iep   CMD(R) │CMD(R)│CMD(R)│XXXX  │
_dout    │      │      │      │      │
         │      │      │      │      │
uram_    ▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁
rden_i   │      │ (grp)│      │      │
         │      │      │      │      │
uram_    XXXX   │ADDR  │ADDR  │XXXX  │
raddr_i  │      │(SET  │      │      │
         │      │_ROW) │      │      │
         │      │      │      │      │
iep2ci   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁
_wren    │      │      │      │      │
         │      │      │      │      │
iep2ci   XXXX   │XXXX  │{GRP, │XXXX  │
_din     │      │      │ ROW, │      │
         │      │      │ DATA}│      │
```

**Notes:**
- 1-cycle URAM read latency
- Response includes group, row, and data
- Only reads selected neuron group URAM

### CI Write Transaction (with RMW)

```
Cycle:   0      1      2      3
         │      │      │      │
State    IDLE   │WRITE │WRITE │IDLE
         │      │_0    │      │
         │      │      │      │
ci2iep   ▔▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁
_empty   │      │      │      │
         │      │      │      │
uram_    ▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁
rden_i   │      │(read │      │
         │      │for   │      │
         │      │ RMW) │      │
         │      │      │      │
uram_    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁
wren_i   │      │      │(grp) │
         │      │      │      │
uram_    XXXX   │XXXX  │{NEW, │
wdata_i  │      │      │ OLD} │
         │      │      │(masked)
```

**Notes:**
- Write requires read first (masked write - preserve other neuron in word)
- LSB of address determines which 36-bit half to update
- Other half preserved via `uram_rmwdata`

---

## Key Terms and Definitions

| Term | Definition |
|------|------------|
| **URAM** | UltraRAM - High-density on-chip memory (Xilinx primitive) |
| **Neuron Group** | 8,192 neurons sharing a URAM bank |
| **Membrane Potential** | 36-bit signed value representing neuron state |
| **Threshold** | Potential value at which neuron fires (spikes) |
| **Spike** | Event when neuron crosses threshold |
| **Phase 1** | Initial scan of all neurons, baseline updates |
| **Phase 2** | Synaptic input application from HBM |
| **RMW** | Read-Modify-Write - Pattern to update part of memory word |
| **Hazard** | Conflict when consecutive operations access same address |
| **Sign Extension** | Expanding signed value to wider bit-width |
| **Leaky I&F** | Leaky Integrate-and-Fire neuron model |
| **FWFT** | First-Word Fall-Through FIFO mode |
| **Null Flag** | Bit indicating no synaptic data for this group |

---

## Performance Characteristics

### Throughput

**Phase 1 (Sequential Scan):**
- **Rate:** 1 row per cycle when HBM data available
- **Neurons per cycle:** 32 (16 groups × 2 neurons/row)
- **Total time:** ~4,096 cycles for full scan (131,072 neurons / 32)
- **Duration:** ~18.2 µs @ 225 MHz

**Phase 2 (Synaptic Updates):**
- **Rate:** Limited by HBM FIFO throughput
- **Typical:** 1-10 updates per neuron per timestep
- **Variable duration:** Depends on network connectivity

**CI Access:**
- **Read latency:** 2 cycles (1 read + 1 response)
- **Write latency:** 2 cycles (1 RMW read + 1 write)
- **Throughput:** ~112M reads/sec or writes/sec @ 225 MHz

### Latency

| Operation | Cycles | Time @ 225 MHz |
|-----------|--------|----------------|
| URAM Read | 1 | 4.4 ns |
| Neuron Update (Phase 1) | 1 | 4.4 ns |
| Neuron Update (Phase 2) | 2 | 8.9 ns (read + write) |
| Spike Detection | 0 | Combinational |
| CI Read | 2 | 8.9 ns |
| CI Write | 2 | 8.9 ns |

---

## Cross-References

### Related Modules

| Module | Relationship | Interface |
|--------|--------------|-----------|
| **command_interpreter.v** | Bidirectional | `ci2iep_*` and `iep2ci_*` FIFOs |
| **hbm_processor.v** | Upstream | `exec_hbm_rdata`, `exec_hbm_rvalidready` |
| **pointer_fifo_controller.v** | Upstream | Uses `exec_uram_spiked` flags |
| **external_events_processor.v** | Coordination | `exec_bram_phase1_done` signal |
| **URAM (Xilinx IP)** | Memory | 16 × simple dual-port URAM instances |

### Software Integration

**Python (hs_bridge):**
- `fpga_controller.read_neuron(address)` → Sends CI read command
- `fpga_controller.write_neuron(address, value)` → Sends CI write command
- `network.set_threshold(value)` → Configures spike threshold
- `network.set_neuron_model(model_id)` → Selects neuron dynamics

---

## Design Evolution

### Evidence of Scaling (8 → 16 Neuron Groups)

**Address Width Changes:**
```verilog
// OLD (8 groups):
wire [13:0] uram_raddr_full;  // 14 bits
wire [2:0] SET_GROUP;          // 3 bits

// NEW (16 groups):
wire [12:0] uram_raddr_full;  // 13 bits
wire [3:0] SET_GROUP;          // 4 bits
```

**Commented Code:**
```verilog
// Line 160: "Previously 14'd0 for 8 NGs"
// Line 509: "Previously exec_hbm_rdata_reg[253:240]" (8 groups, 256-bit HBM)
// Line 681: "Previously unregistered exec_hbm_rdata[511:0]"
```

**Neuron Model Increments:**
- Groups 0-7 originally: +1 to +8
- Groups 8-15 added: +9 to +16
- Allows differentiation of all 16 groups in model 1

---

## Common Issues and Debugging

### Problem: RMW Hazards Causing Incorrect Updates

**Symptoms:** Neurons in same URAM row overwrite each other

**Debug Steps:**
1. Check `uram_waddr[i] == uram_waddr_reg[i]` match detection
2. Verify `SET_GROUP == SET_GROUP_reg` for CI writes
3. Check `uram_wren[i]` and `uram_wren[i]_reg` both asserted
4. Monitor `uram_rmwdata[i]` vs `uram_rdata[i]`

**Common Cause:** Consecutive writes to different neurons in same row, same group

### Problem: Phase Transitions Never Occur

**Symptoms:** State machine stuck in WAIT or PUSH/POP states

**Debug Steps:**
1. Check `exec_bram_phase1_done` - should assert after external events
2. Check `exec_hbm_rvalidready` - should pulse when HBM data available
3. Check `uram_waddr[0] == URAM_ADDR_LIMIT` - ensure limit calculated correctly
4. Monitor `exec_hbm_rx_phase2_done` - should assert when HBM finished

**Common Cause:** Missing handshake signals from other modules

### Problem: Spikes Not Detected

**Symptoms:** `exec_uram_spiked` always zero

**Debug Steps:**
1. Check `threshold` parameter - may be set too high
2. Verify neuron potentials increasing (read via CI)
3. Check `exec_uram_phase1_ready & !exec_uram_phase1_done` timing
4. Monitor `uram_rmwdata_upper/lower[i]` values

**Common Cause:** Threshold misconfiguration or insufficient synaptic input

### VIO/ILA Probes (Recommended)

```verilog
(*mark_debug = "true"*) reg [3:0] curr_state;
(*mark_debug = "true"*) wire [12:0] uram_raddr;
(*mark_debug = "true"*) wire [12:0] uram_waddr[0];
(*mark_debug = "true"*) wire [15:0] exec_uram_spiked;
(*mark_debug = "true"*) wire exec_hbm_rvalidready;
(*mark_debug = "true"*) wire exec_uram_phase1_ready;
(*mark_debug = "true"*) wire [35:0] uram_rmwdata_upper_0;
(*mark_debug = "true"*) wire signed [35:0] threshold;
```

---

## Safety and Edge Cases

### Reset Behavior

On `resetn` deassertion:
- All state registers → 0
- Phase flags → idle (phase1_done=1, phase2_done=1)
- URAM addresses → 0
- Spike flags → 0

### Neuron Model Edge Cases

**Model 0 (Memoryless):**
- Always resets to 0, regardless of input
- Useful for testing spike detection without accumulation

**Model 2 (Leaky I&F):**
- Decay: `V - (V >>> 3)` = V × 7/8
- No saturation at zero (can go negative if synaptic inputs negative)
- Decay applied before synaptic inputs in same cycle

**Model 3 (Non-leaky):**
- No reset after spike (continues accumulating)
- Can overflow 36-bit signed range

### Overflow Protection

**None implemented!** Potentials can wrap around:
- Max positive: 2^35 - 1 = 17,179,869,183
- Overflow wraps to large negative value
- Software must manage weight scaling

---

## Future Enhancement Opportunities

1. **Saturation Arithmetic:** Clamp potentials to min/max instead of wrapping

2. **Additional Neuron Models:** Izhikevich, Hodgkin-Huxley approximations

3. **Configurable Decay Rate:** Instead of fixed `>>> 3`, use parameter

4. **Refractory Period:** Prevent immediate re-spiking after threshold crossing

5. **Performance Counters:** Track spike rates, hazard occurrences

6. **Multi-Precision:** Support 16-bit or 64-bit neuron states

7. **Hazard-Free Architecture:** Separate read and write banks to eliminate RMW

---

**Document Version:** 1.0
**Last Updated:** December 2025
**Module File:** `internal_events_processor.v`
**Module Location:** `CRI_proj/cri_fpga/code/new/hyddenn2/vivado/single_core.srcs/sources_1/new/`
**Purpose:** Core neuron computation engine
**Neuron Capacity:** 131,072 (16 groups × 8,192)
**URAM Usage:** 16 banks × 294 Kb = 4.7 Mb
**Clock Frequency:** 225 MHz
