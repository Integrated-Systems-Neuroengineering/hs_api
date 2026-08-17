# input_data_handler.v

## Module Overview

### Purpose and Role in Stack

The **input_data_handler** module acts as a **BRAM arbiter**, managing access to the shared Block RAM (BRAM) that stores axon/external event data. This module:

- **Arbitrates between two requesters:**
  - Command interpreter (CI) - for host read/write access
  - External events processor (EEP) - for runtime axon event processing
- **Enforces priority:** Command interpreter has higher priority than external events processor
- **Handles BRAM read latency:** Implements 3-cycle pipeline to account for BRAM read delay
- **Routes responses** back to appropriate requester with address passthrough

In the software/hardware stack:
```
Command Interpreter ──┐
                      ├──► input_data_handler ──► BRAM (2^15 x 256-bit)
External Events       │         (Arbiter)              │
Processor         ────┘                                │
                                                       │
                         ┌─────────────────────────────┘
                         │
                    Response Router
                         │
            ┌────────────┴─────────────┐
            ▼                          ▼
    Command Interpreter      External Events Processor
    (read response)          (read response)
```

This module is essential for **efficient BRAM utilization**, allowing both configuration/debug access (via CI) and high-speed runtime processing (via EEP) to share the same memory resource.

---

## Module Architecture

### High-Level Block Diagram

```
        input_data_handler
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │         ┌───────────────────────────────┐                  │
    │         │   Command Interpreter FIFO    │                  │
    │         │   (Input: Local Read)          │                  │
CI→FIFO  ────►│ ci2idp_dout[271:0]            │                  │
(local)       │  [271] = R/W command          │                  │
empty/rden    │  [270:256] = 15-bit address   │                  │
              │  [255:0] = 256-bit data        │                  │
              └───────────┬───────────────────┘                  │
                          │                                      │
    │                     │                                      │
    │         ┌───────────▼───────────────────┐                  │
    │         │   External Events Proc FIFO   │                  │
    │         │   (Input: Local Read)          │                  │
EEP→FIFO  ────►│ eep2idp_dout[14:0]           │                  │
(local)       │  15-bit address only          │                  │
empty/rden    └───────────┬───────────────────┘                  │
              │           │                                      │
              │           │                                      │
    │         │   ┌───────▼─────────────────────────────┐        │
    │         │   │   Priority Arbiter                  │        │
    │         │   │   - CI has priority over EEP        │        │
    │         │   │   - Selects address source          │        │
    │         │   │   - Generates BRAM control signals  │        │
    │         │   └───────┬─────────────────────────────┘        │
    │         │           │                                      │
    │         │           ▼                                      │
    │         │   ┌────────────────────────┐                    │
    │         │   │   BRAM Interface       │                    │
BRAM  ◄───────┼───┤ addr[14:0]             │                    │
Interface     │   │ din[255:0] (write data)│                    │
(2^15 x 256)  │   │ dout[255:0] (read data)│                    │
              │   │ wren (write enable)    │                    │
              │   └────────┬───────────────┘                    │
    │         │            │                                     │
    │         │            ▼                                     │
    │         │   ┌──────────────────────────────────┐          │
    │         │   │   3-Cycle Read Pipeline          │          │
    │         │   │   (Compensates for BRAM latency) │          │
    │         │   │                                  │          │
    │         │   │   IDLE → WAIT_0 → WAIT_1 →      │          │
    │         │   │         → WAIT_2 → output       │          │
    │         │   │                                  │          │
    │         │   └──────────┬───────────────────────┘          │
    │         │              │                                   │
    │         │              ▼                                   │
    │         │   ┌──────────────────────────────────┐          │
    │         │   │   Response Router                │          │
    │         │   │   - Directs read data to         │          │
    │         │   │     original requester           │          │
    │         │   │   - Includes address passthrough │          │
    │         │   └──────┬─────────┬─────────────────┘          │
    │         │          │         │                             │
    │         │          ▼         ▼                             │
    │         │   ┌──────────┐ ┌──────────┐                    │
    │         │   │ idp2ci   │ │ idp2eep  │                    │
CI←FIFO  ◄──────┤ FIFO     │ │ FIFO     │◄───────────EEP←FIFO │
(remote)        │ (Output: │ │ (Output: │                (remote)
full/wren       │  Remote) │ │  Remote) │                        │
data            └──────────┘ └──────────┘                        │
                │                                                 │
                └─────────────────────────────────────────────────┘
```

---

## Interface Specification

### Clock and Reset

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `clk` | Input | 1 | 225 MHz system clock |
| `resetn` | Input | 1 | Active-low synchronous reset |

### Command Interpreter Interface

**Input FIFO (Local - CI to IDP):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `ci2idp_empty` | Input | 1 | Input FIFO empty flag |
| `ci2idp_dout` | Input | 272 | Input FIFO data output |
| `ci2idp_rden` | Output (reg) | 1 | Input FIFO read enable |

**Data Format (`ci2idp_dout[271:0]`):**
```
[271]       = R/W command (0=read, 1=write)
[270:256]   = 15-bit BRAM address
[255:0]     = 256-bit write data
```

**Output FIFO (Remote - IDP to CI):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `idp2ci_full` | Input | 1 | Output FIFO full flag |
| `idp2ci_din` | Output | 271 | Output FIFO data input |
| `idp2ci_wren` | Output (reg) | 1 | Output FIFO write enable |

**Data Format (`idp2ci_din[270:0]`):**
```
[270:256]   = 15-bit BRAM address (echoed from request)
[255:0]     = 256-bit read data
```

### External Events Processor Interface

**Input FIFO (Local - EEP to IDP):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `eep2idp_empty` | Input | 1 | Input FIFO empty flag |
| `eep2idp_dout` | Input | 15 | Input FIFO data output (address only) |
| `eep2idp_rden` | Output (reg) | 1 | Input FIFO read enable |

**Data Format (`eep2idp_dout[14:0]`):**
```
[14:0] = 15-bit BRAM address (read request only)
```

**Output FIFO (Remote - IDP to EEP):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `idp2eep_full` | Input | 1 | Output FIFO full flag |
| `idp2eep_din` | Output | 271 | Output FIFO data input |
| `idp2eep_wren` | Output (reg) | 1 | Output FIFO write enable |

**Data Format (`idp2eep_din[270:0]`):**
```
[270:256]   = 15-bit BRAM address (echoed from request)
[255:0]     = 256-bit read data
```

### BRAM Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `bram_addr` | Output (reg) | 15 | BRAM address (0 to 32,767) |
| `bram_din` | Output | 256 | BRAM write data |
| `bram_wren` | Output (reg) | 1 | BRAM write enable |
| `bram_dout` | Input | 256 | BRAM read data (3-cycle latency) |

**BRAM Specifications:**
- **Depth:** 32,768 rows (2^15)
- **Width:** 256 bits per row
- **Total Size:** 1 MB (32,768 × 256 bits = 8,388,608 bits)
- **Read Latency:** 3 clock cycles
- **Write Latency:** 1 clock cycle (synchronous write)

---

## Detailed Logic Description

### Command Decoder

```verilog
localparam CMD_READ  = 1'b0;
localparam CMD_WRITE = 1'b1;

wire command = ci2idp_dout[271];  // Extract R/W bit
```

### State Machine

**States:**
```verilog
localparam [2:0] STATE_RESET                = 3'd0;
localparam [2:0] STATE_IDLE                 = 3'd1;
localparam [2:0] STATE_EEP_WAIT_BRAM_READ_0 = 3'd2;
localparam [2:0] STATE_EEP_WAIT_BRAM_READ_1 = 3'd3;
localparam [2:0] STATE_EEP_WAIT_BRAM_READ_2 = 3'd4;
localparam [2:0] STATE_CI_WAIT_BRAM_READ_0  = 3'd5;
localparam [2:0] STATE_CI_WAIT_BRAM_READ_1  = 3'd6;
localparam [2:0] STATE_CI_WAIT_BRAM_READ_2  = 3'd7;
```

**State Transition Diagram:**

```
                   ┌──────────────┐
                   │ STATE_RESET  │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
              ┌───▶│ STATE_IDLE   │◄────────────────┬─────────────────┐
              │    │ (Arbitrate)  │                 │                 │
              │    └──┬───────┬───┘                 │                 │
              │       │       │                     │                 │
              │  !eep │       │ !ci                 │                 │
              │  empty│       │ empty               │                 │
              │       │       │                     │                 │
              │       │       └─ CMD_READ           │                 │
              │       │              │              │                 │
              │       │              ▼              │                 │
              │       │       STATE_CI_WAIT_0       │                 │
              │       │              │              │                 │
              │       │              ▼              │                 │
              │       │       STATE_CI_WAIT_1       │                 │
              │       │              │              │                 │
              │       │              ▼              │                 │
              │       │       STATE_CI_WAIT_2       │                 │
              │       │              │              │                 │
              │       │              │!idp2ci_full  │                 │
              │       │              └──────────────┘                 │
              │       │                                               │
              │       │ CMD_WRITE                                     │
              │       └─(immediate pop)──────────────────────────────┘
              │       │
              │       ▼
              │    STATE_EEP_WAIT_0
              │       │
              │       ▼
              │    STATE_EEP_WAIT_1
              │       │
              │       ▼
              │    STATE_EEP_WAIT_2
              │       │
              │       │!idp2eep_full
              └───────┘
```

### Priority Arbitration Logic

**IDLE State Behavior:**
```verilog
STATE_IDLE: begin
    if (~eep2idp_empty) begin
        // EEP has pending request
        bram_addr  = eep2idp_dout;
        next_state = STATE_EEP_WAIT_BRAM_READ_0;

    end else if (~ci2idp_empty) begin
        // CI has pending request (higher priority)
        bram_addr = ci2idp_dout[270:256];  // Extract 15-bit address

        if (command==CMD_READ)
            next_state = STATE_CI_WAIT_BRAM_READ_0;
        else begin  // CMD_WRITE
            bram_wren   = 1'b1;
            ci2idp_rden = 1'b1;
            next_state  = STATE_IDLE;  // Write completes immediately
        end
    end
end
```

**Priority Rules:**
1. **CI Write:** Highest priority, completes in 1 cycle (no wait states)
2. **CI Read:** High priority, 3-cycle wait for BRAM latency
3. **EEP Read:** Lower priority, serviced only when CI FIFO empty
4. **No Starvation:** EEP will eventually be serviced due to finite CI request rate

### BRAM Read Pipeline (3-Cycle Latency)

**Cycle Breakdown:**

```
Cycle 0: Request arrives in IDLE state
         - bram_addr = address from FIFO
         - Transition to WAIT_0

Cycle 1: STATE_WAIT_0
         - BRAM internal pipeline stage 1
         - bram_addr held stable
         - Transition to WAIT_1

Cycle 2: STATE_WAIT_1
         - BRAM internal pipeline stage 2
         - bram_addr held stable
         - Transition to WAIT_2

Cycle 3: STATE_WAIT_2
         - bram_dout now valid
         - Wait for output FIFO not full
         - Write to output FIFO (wren pulse)
         - Pop input FIFO (rden pulse)
         - Transition to IDLE
```

**EEP Read Example:**
```verilog
STATE_EEP_WAIT_BRAM_READ_0: begin
    bram_addr  = eep2idp_dout;  // Hold address stable
    next_state = STATE_EEP_WAIT_BRAM_READ_1;
end

STATE_EEP_WAIT_BRAM_READ_1: begin
    bram_addr  = eep2idp_dout;
    next_state = STATE_EEP_WAIT_BRAM_READ_2;
end

STATE_EEP_WAIT_BRAM_READ_2: begin
    bram_addr = eep2idp_dout;
    if (~idp2eep_full) begin
        idp2eep_wren = 1'b1;  // Write read data to output FIFO
        eep2idp_rden = 1'b1;  // Pop request from input FIFO
        next_state = STATE_IDLE;
    end
    // else: stall until output FIFO has space
end
```

**CI Read:** Same pattern using `ci2idp_dout[270:256]` for address and `idp2ci` FIFOs.

### Output Data Routing

**Assignments:**
```verilog
assign idp2eep_din = {bram_addr, bram_dout};  // [270:256]=addr, [255:0]=data
assign idp2ci_din  = {bram_addr, bram_dout};
assign bram_din    = ci2idp_dout[255:0];      // Only CI can write
```

**Address Passthrough:**
- Read responses include the original address
- Allows requester to correlate response with request
- Critical for pipelined operation (though this module doesn't pipeline)

---

## Timing Diagrams

### CI Write Transaction

```
Cycle:     0      1      2
           │      │      │
State      IDLE   │IDLE  │
           │      │      │
ci2idp     ▁▁▁▁▁▁▁│▔▔▔▔▔▔│  (WR, Addr=0x1234, Data=0xABCD...)
_empty     │      │      │
           │      │      │
ci2idp     ▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁
_rden      │      │      │
           │      │      │
bram_addr  XXXX   │0x1234│
           │      │      │
bram_wren  ▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁
           │      │      │
bram_din   XXXX   │0xABCD│
           │      │...   │
```

**Notes:**
- Single-cycle write operation
- No wait states required
- Returns to IDLE immediately

### CI Read Transaction

```
Cycle:     0      1      2      3      4      5
           │      │      │      │      │      │
State      IDLE   │WAIT_0│WAIT_1│WAIT_2│IDLE  │
           │      │      │      │      │      │
ci2idp     ▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│  (RD, Addr=0x5678)
_empty     │      │      │      │      │      │
           │      │      │      │      │      │
ci2idp     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁
_rden      │      │      │      │      │      │
           │      │      │      │      │      │
bram_addr  XXXX   │0x5678│0x5678│0x5678│0x5678│
           │      │      │      │      │      │
bram_dout  XXXX   │XXXX  │XXXX  │XXXX  │DATA  │
           │      │      │      │      │      │
idp2ci     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁
_wren      │      │      │      │      │      │
           │      │      │      │      │      │
idp2ci_din XXXX   │XXXX  │XXXX  │XXXX  │{0x5678,
           │      │      │      │      │ DATA}
```

**Notes:**
- 3-cycle wait for BRAM read latency
- Address held stable during wait states
- Response includes address + data

### Priority Arbitration: EEP Deferred

```
Cycle:     0      1      2      3      4      5      6      7      8
           │      │      │      │      │      │      │      │      │
State      IDLE   │WAIT_0│WAIT_1│WAIT_2│IDLE  │WAIT_0│WAIT_1│WAIT_2│
           │      │      │      │      │      │      │      │      │
eep2idp    ▔▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│  (pending request)
_empty     │      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │      │
ci2idp     ▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  (higher priority)
_empty     │      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │      │
Serviced   -      │CI    │CI    │CI    │CI    │EEP   │EEP   │EEP   │EEP
           │      │      │      │      │      │      │      │      │
```

**Notes:**
- Cycle 0: Both FIFOs have requests, CI serviced first
- Cycles 1-4: CI read completes (3-cycle wait)
- Cycle 5: EEP request now serviced
- Demonstrates priority enforcement

---

## Cross-References

### Related Modules

| Module | Relationship | Interface |
|--------|--------------|-----------|
| **command_interpreter.v** | Upstream | Connects to `ci2idp_*` and `idp2ci_*` FIFOs |
| **external_events_processor.v** | Upstream | Connects to `eep2idp_*` and `idp2eep_*` FIFOs |
| **BRAM (Xilinx IP)** | Downstream | `bram_*` signals control Block RAM |

### BRAM Structure (Parent: pcie2fifos → command_interpreter)

**Data Stored in BRAM:**
- **Axon/External Event Data**
- Each row: 256 bits = 16 × 16-bit masks (one per neuron group)
- Row address: Axon ID / 16

**Example Row at Address 0x1000:**
```
Bits [255:240] = Mask for neuron group 15
Bits [239:224] = Mask for neuron group 14
...
Bits [31:16]   = Mask for neuron group 1
Bits [15:0]    = Mask for neuron group 0

Each 16-bit mask: One bit per neuron group indicating which received axon spike
```

---

## Key Terms and Definitions

| Term | Definition |
|------|------------|
| **Arbiter** | Logic that decides which requester gains access to shared resource |
| **Priority** | CI requests serviced before EEP when both pending |
| **Read Latency** | 3 clock cycles from address presentation to valid data |
| **Passthrough** | Address echoed back with read data for correlation |
| **Local FIFO** | FIFO in same clock domain as module (input side) |
| **Remote FIFO** | FIFO potentially in different clock domain (output side) |
| **CMD_READ** | Command bit value 0, triggers read transaction |
| **CMD_WRITE** | Command bit value 1, triggers write transaction |
| **BRAM** | Block RAM - On-chip synchronous memory primitive |
| **FIFO Backpressure** | Waiting for output FIFO not full before writing |

---

## Performance Characteristics

### Throughput

**Best Case (No Contention):**
- **CI Write:** 1 operation per clock cycle = 225 MHz = 225M writes/sec
- **CI Read:** 4 cycles per operation (1 IDLE + 3 WAIT) = 56.25M reads/sec
- **EEP Read:** 4 cycles per operation = 56.25M reads/sec (when CI idle)

**Worst Case (Contention):**
- **EEP Read (with CI active):** Indefinitely deferred until CI idle
- **CI Read (with output FIFO full):** Stalled in WAIT_2 state

**Realistic (Mixed Workload):**
- CI accesses: Infrequent (configuration, debug)
- EEP accesses: Burst during Phase 1 execution
- Typical: EEP dominates, achieving ~50M reads/sec effective rate

### Latency

| Operation | Latency (Cycles) | Latency (ns @ 225 MHz) | Notes |
|-----------|------------------|------------------------|-------|
| CI Write | 1 | 4.4 ns | Immediate, no wait |
| CI Read | 4 | 17.8 ns | 1 IDLE + 3 WAIT |
| EEP Read | 4 | 17.8 ns | When CI idle |
| EEP Read (deferred) | 4 + CI latency | Variable | Must wait for CI completion |

### Stall Conditions

**Input Side Stalls:**
- None - FIFOs assumed to handle backpressure

**Output Side Stalls:**
- **WAIT_2 State:** If output FIFO full, module holds until space available
- **Impact:** Backpressure propagates to input FIFO (requesters must wait)

---

## Design Considerations

### Why Priority to CI?

1. **Low Frequency:** CI accesses are rare (host-initiated)
2. **Latency Sensitive:** Host expects fast response for debug/config
3. **No Starvation:** EEP can afford to wait a few cycles
4. **Simplicity:** Avoids complex round-robin or fair arbitration

### Why 3-Cycle Wait?

- **BRAM Primitive:** Xilinx Block RAM has inherent 2-3 cycle read latency
- **Pipeline Registers:** Additional registering for timing closure
- **Fixed Latency:** Simplifies state machine design (no variable wait)

### Alternative Designs

**Round-Robin Arbitration:**
- Pros: Fair access, prevents EEP starvation
- Cons: More complex, CI latency increases

**Pipelined Operation:**
- Pros: Higher throughput (overlapped requests)
- Cons: Requires buffering, address tracking, out-of-order handling
- Not needed: Current design adequate for workload

---

## Common Issues and Debugging

### Problem: EEP Never Gets Access

**Symptoms:** EEP input FIFO fills up, no reads complete

**Debug Steps:**
1. Check `ci2idp_empty` - should toggle to 1 occasionally
2. Check state machine - should eventually reach `STATE_EEP_WAIT_0`
3. Verify CI not continuously sending requests

**Common Cause:** CI stuck in continuous read/write loop

### Problem: Read Data Incorrect

**Symptoms:** Returned data doesn't match expected values

**Debug Steps:**
1. Check `bram_addr` during WAIT states - should be stable
2. Verify `bram_dout` on cycle 3 (WAIT_2 state)
3. Confirm write operations completed before read
4. Check address calculation in requester module

**Common Cause:** Address mismatch or read-before-write hazard

### Problem: Module Stuck in WAIT_2

**Symptoms:** State machine doesn't return to IDLE

**Debug Steps:**
1. Check output FIFO full flag (`idp2ci_full` or `idp2eep_full`)
2. Verify downstream module consuming from output FIFO
3. Check for clock domain crossing issues (if FIFOs are async)

**Common Cause:** Output FIFO overflow or downstream stall

### VIO/ILA Probes (Recommended)

```verilog
(*mark_debug = "true"*) reg [2:0] curr_state;
(*mark_debug = "true"*) wire command = ci2idp_dout[271];
(*mark_debug = "true"*) wire [14:0] ci_addr = ci2idp_dout[270:256];
(*mark_debug = "true"*) wire [14:0] eep_addr = eep2idp_dout;
(*mark_debug = "true"*) wire ci_request = ~ci2idp_empty;
(*mark_debug = "true"*) wire eep_request = ~eep2idp_empty;
(*mark_debug = "true"*) wire [14:0] bram_addr;
(*mark_debug = "true"*) wire bram_wren;
```

---

## Safety and Edge Cases

### Reset Behavior

On `resetn` deassertion:
- State machine → `STATE_RESET` → `STATE_IDLE`
- All output signals → 0 (no spurious FIFO operations)
- BRAM address → `15'dX` (don't care)

### Simultaneous Requests

**Both FIFOs have data at IDLE state:**
- CI serviced first (priority)
- EEP serviced after CI completes

**Write During Read:**
- Write completes in 1 cycle
- Subsequent read sees updated value (BRAM write latency = 1 cycle)

### FIFO Full During WAIT_2

- Module stalls in WAIT_2 state
- `bram_addr` held stable (safe to stall)
- No timeout - waits indefinitely for FIFO space
- Assumes downstream will eventually consume

---

## Potential Enhancements

1. **Pipelined Reads:** Allow new request while waiting for previous read
   - Requires FIFO buffering and address tracking
   - Could double read throughput

2. **Write Acknowledgment:** Provide write confirmation to CI
   - Currently fire-and-forget
   - Useful for verification

3. **Round-Robin or Weighted Arbitration:** Fairer access to EEP
   - Prevent worst-case starvation scenarios
   - At cost of CI latency

4. **Variable BRAM Latency:** Support configurable wait cycles
   - Adapt to different BRAM configurations
   - Requires parameterization

5. **Performance Counters:** Track utilization and contention
   - CI access count
   - EEP access count
   - Stall cycles
   - Useful for profiling

6. **Error Detection:** Detect protocol violations
   - Write with read-pending
   - Address out of range
   - Currently no error reporting

---

**Document Version:** 1.0
**Last Updated:** December 2025
**Module File:** `input_data_handler.v`
**Module Location:** `CRI_proj/cri_fpga/code/new/hyddenn2/vivado/single_core.srcs/sources_1/new/`
**Purpose:** BRAM arbiter for shared axon/external event memory
**BRAM Size:** 1 MB (2^15 × 256-bit)
**Read Latency:** 3 cycles
