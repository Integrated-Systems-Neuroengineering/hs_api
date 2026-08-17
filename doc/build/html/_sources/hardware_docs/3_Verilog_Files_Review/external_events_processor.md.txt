# External Events Processor Module Family

## Overview

The **External Events Processor** family manages input spike events (axons) in the neuromorphic FPGA system. These modules maintain two Block RAMs in a double-buffering scheme: one for the "present" time step (currently being processed) and one for the "future" time step (accumulating new events). This architecture allows continuous operation without dropping input events.

Three variants exist:
1. **external_events_processor.v** - Base version with full pipeline hazard handling
2. **external_events_processor_simple.v** - Simplified single-core version with wider data paths
3. **external_events_processor_v2.v** - Enhanced version with debugging capabilities

### Role in the Software/Hardware Stack

```
Host Application (Python/C++)
         |
    [hs_bridge]
         |
  [PCIe Interface]
         |
  [Command Interpreter] -----> [External Events Processor] <---- External spike events
         |                              |
         |                     Present BRAM (8 or 16 axons/row)
         |                     Future BRAM (8 or 16 axons/row)
         |                              |
    [HBM Processor] <------ exec_bram_spiked (spike mask)
         |                              |
  [Internal Events] <-------- exec_bram_phase1_done
    Processor
```

**Function**:
- Receive input spike events from external sources or command interpreter
- Store events in double-buffered BRAMs (present/future)
- Synchronize event delivery with HBM read operations
- Clear processed events after reading
- Handle pipeline hazards for concurrent writes during multi-core operation

**Key Innovation**: Double-buffering allows new events to accumulate in the "future" BRAM while the "present" BRAM is being read and cleared, ensuring no event loss during processing.

---

## Variant Comparison

| Feature | Base Version | Simple Version | V2 Version |
|---------|-------------|----------------|------------|
| **File** | external_events_processor.v | external_events_processor_simple.v | external_events_processor_v2.v |
| **Axons per row** | 8 | 16 | 8 |
| **Address width** | 14 bits | 13 bits | 14 bits |
| **Data width** | 8 bits | 16 bits | 8 bits |
| **Target** | Multi-core | Single-core | Debug/verification |
| **Future pipeline** | 3-stage hazard handling | Direct write (no hazards) | Simplified RMW |
| **State machine** | 5 states | 4 states | 5 states + debug FSM |
| **Debug features** | None | None | CI interface, debug ports |
| **Complexity** | High | Low | High |

---

## Module Architecture (Base Version)

```
                                    ┌─────────────────────────────────┐
                                    │  External Events Processor      │
                                    │                                 │
   setArray_go ──────────┐          │  ┌──────────────────────────┐  │
   setArray_addr[13:0] ──┼──────────┼─>│  Future BRAM Control     │  │
   setArray_data[7:0] ───┘          │  │  - 3-stage pipeline      │  │
                                    │  │  - Hazard detection      │  │
   exec_run ────────────────────────┼─>│  - waddr/wdata/wren[2:0] │  │
                                    │  └────────┬─────────────────┘  │
                                    │           │                     │
                                    │           v                     │
                                    │  ┌──────────────────────────┐  │
                                    │  │   BRAM Multiplexer       │  │
                                    │  │   bram_select toggle     │  │
    ┌───────────────────────────────┼─>│   - BRAM0 ←→ Present    │  │
    │                               │  │   - BRAM1 ←→ Future      │  │
    │  ┌────────────────────────────┼─<│                          │  │
    │  │                            │  └───────┬──────────────────┘  │
    │  │                            │          │                     │
    │  │                            │          v                     │
    │  │                            │  ┌──────────────────────────┐  │
    │  │   exec_hbm_rvalidready ────┼─>│ Present BRAM Control     │  │
    │  │                            │  │ - State machine (5)      │  │
    │  │                            │  │ - Pipeline fill          │  │
    │  │                            │  │ - Read & clear           │  │
    │  │                            │  │ - raddr/waddr tracking   │  │
    │  │                            │  └────────┬─────────────────┘  │
    │  │                            │           │                     │
    │  └───────── exec_bram_spiked[7:0] <───────┘                    │
    │                               │                                 │
    └───────── exec_bram_phase1_done ────────────────────────────────┤
                                    │                                 │
                                    └─────────────────────────────────┘

    BRAM0 (18Kb)                    BRAM1 (18Kb)
    ┌────────────┐                  ┌────────────┐
    │ 16384 × 8b │                  │ 16384 × 8b │
    │            │                  │            │
    │ Toggles:   │                  │ Toggles:   │
    │ Present ←→ │                  │ Future ←→  │
    │ Future     │                  │ Present    │
    └────────────┘                  └────────────┘
```

### Data Flow (Two-Phase Operation)

**Phase 0: Setup (between time steps)**
```
1. exec_run pulse triggers:
   - bram_select toggles (swaps present ←→ future)
   - State machine resets to IDLE

2. Present BRAM now contains accumulated events from previous "future"
3. Future BRAM ready to accumulate new events
```

**Phase 1: Event Processing (during time step)**
```
STATE_FILL_PIPE (cycles 0-2):
   ├─> Read BRAM addresses 0, 1, 2
   └─> Fill 3-stage pipeline (no writes yet)

STATE_READ_INPUTS (cycles 3 to completion):
   ├─> Wait for exec_hbm_rvalidready
   ├─> Read next BRAM address (bramPresent_raddr++)
   ├─> Write 0 to lagging address (bramPresent_waddr++)
   ├─> Output exec_bram_spiked[7:0] to downstream
   └─> Loop until bramPresent_waddr == BRAM_ADDR_LIMIT

STATE_PHASE1_DONE:
   └─> Assert exec_bram_phase1_done
```

**Concurrent Future Writes** (throughout processing):
```
setArray_go pulse:
   ├─> Check for pipeline hazards (same address in stages 0, 1, 2)
   ├─> Merge with in-flight data if hazard detected
   ├─> Propagate through 3-stage pipeline
   └─> Write to Future BRAM after 3 cycles
```

---

## Interface Specification

### Base Version (external_events_processor.v)

#### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `PIPE_DEPTH` | 3 | BRAM read pipeline depth (matches BRAM latency) |

#### Clock and Reset
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | Input | 1 | System clock (225 MHz) |
| `resetn` | Input | 1 | Active-low asynchronous reset |

#### Configuration
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `num_inputs` | Input | 17 | Total number of input axons (max 131,072) |

#### External Event Input Interface
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `setArray_go` | Input | 1 | Write pulse for new axon event |
| `setArray_addr` | Input | 14 | BRAM row address (8 axons per row) |
| `setArray_data` | Input | 8 | Bit mask (1=spike, 0=no spike) |

#### Execution Control Interface
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `exec_run` | Input | 1 | Start new time step (toggles BRAMs) |
| `exec_bram_phase1_ready` | Output | 1 | Pipeline filled, ready for reads |
| `exec_hbm_rvalidready` | Input | 1 | HBM data valid & ready (advance BRAM) |
| `exec_bram_spiked` | Output | 8 | Current spike mask (8 axons) |
| `exec_bram_phase1_done` | Output | 1 | All inputs read, phase 1 complete |

#### BRAM0 Interface
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `bram0_waddr` | Output | 14 | Write address |
| `bram0_wdata` | Output | 8 | Write data |
| `bram0_wren` | Output | 1 | Write enable |
| `bram0_raddr` | Output | 14 | Read address |
| `bram0_rden` | Output | 1 | Read enable |
| `bram0_rdata` | Input | 8 | Read data (3-cycle latency) |

#### BRAM1 Interface
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `bram1_waddr` | Output | 14 | Write address |
| `bram1_wdata` | Output | 8 | Write data |
| `bram1_wren` | Output | 1 | Write enable |
| `bram1_raddr` | Output | 14 | Read address |
| `bram1_rden` | Output | 1 | Read enable |
| `bram1_rdata` | Input | 8 | Read data (3-cycle latency) |

### Simple Version (external_events_processor_simple.v)

Key differences from base version:
- **13-bit addresses**: `axonEvent_addr[12:0]`, `bram0/1_*addr[12:0]`
- **16-bit data**: `axonEvent_data[15:0]`, `bram0/1_*data[15:0]`, `exec_eep_spiked[15:0]`
- **16 axons per row**: `axon_addr_limit = num_inputs[16:4]` (not `[16:3]`)
- **Additional output**: `hbm2eep_rden` (HBM FIFO read enable)
- **Debug outputs**: `eep_curr_state[1:0]`, `curr_bram_waddr[12:0]`
- **Renamed ports**: `exec_eep_*` instead of `exec_bram_*`

### V2 Version (external_events_processor_v2.v)

Additional interfaces (beyond base version):

#### Command Interpreter Debug Interface
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `ci2eep_empty` | Input | 1 | Debug command FIFO empty flag |
| `ci2eep_dout` | Input | 14 | Debug read address from CI |
| `ci2eep_rden` | Output | 1 | Debug command FIFO read enable |
| `eep2ci_full` | Input | 1 | Debug response FIFO full flag |
| `eep2ci_din` | Output | 22 | Debug response data (addr + data) |
| `eep2ci_wren` | Output | 1 | Debug response FIFO write enable |

#### Debug BRAM Read Ports
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `bram0_raddr_dbg` | Output | 14 | Debug read address for BRAM0 |
| `bram0_rdata_dbg` | Input | 8 | Debug read data from BRAM0 |
| `bram1_raddr_dbg` | Output | 14 | Debug read address for BRAM1 |
| `bram1_rdata_dbg` | Input | 8 | Debug read data from BRAM1 |

---

## Detailed Logic Description

### Base Version State Machine

#### Present BRAM Control FSM

**States:**
```verilog
STATE_RESET        = 3'd0  // Reset addresses and flags
STATE_IDLE         = 3'd1  // Wait for exec_run
STATE_FILL_PIPE    = 3'd2  // Fill 3-stage BRAM read pipeline
STATE_READ_INPUTS  = 3'd3  // Read inputs, clear memory, sync with HBM
STATE_PHASE1_DONE  = 3'd4  // Signal completion
```

**State Transitions:**
```
    RESET
      |
      v
    IDLE <────────────────┐
      |                   │
      | exec_run          │
      v                   │
   FILL_PIPE              │
      |                   │
      | raddr >= 3        │
      v                   │
  READ_INPUTS             │
      |                   │
      | waddr == limit    │
      v                   │
  PHASE1_DONE ────────────┘
```

**State Behaviors:**

```verilog
STATE_RESET:
    bramPresent_addr_rst = 1'b1        // Reset raddr and waddr to 0
    next_state = STATE_IDLE

STATE_IDLE:
    if (exec_run)
        bramPresent_addr_rst = 1'b1    // Reset for new time step
        next_state = STATE_FILL_PIPE

STATE_FILL_PIPE:
    if (bramPresent_raddr < PIPE_DEPTH)
        bramPresent_rden = 1'b1        // Issue read
        bramPresent_addr_inc = 1'b1    // Increment raddr
    else
        next_state = STATE_READ_INPUTS // Pipeline full

STATE_READ_INPUTS:
    if (exec_hbm_rvalidready)          // HBM ready for next data
        bramPresent_rden = 1'b1        // Read next address
        bramPresent_addr_inc = 1'b1    // Increment both raddr and waddr
        if (bramPresent_waddr == BRAM_ADDR_LIMIT)
            next_state = STATE_PHASE1_DONE

STATE_PHASE1_DONE:
    next_state = STATE_IDLE            // Return to idle
```

#### Address Management (Present BRAM)

The module maintains two addresses with different roles:

**Read Address (raddr)** - Leading edge:
```verilog
// Advances PIPE_DEPTH cycles ahead of write address
// Points to data that will be available after pipeline latency
always @(posedge clk) begin
    if (~resetn | exec_run | bramPresent_addr_rst)
        bramPresent_raddr <= 14'd0;
    else if (bramPresent_addr_inc)
        bramPresent_raddr <= bramPresent_raddr + 1'b1;
end
```

**Write Address (waddr)** - Lagging edge:
```verilog
// Trails read address by PIPE_DEPTH cycles
// Points to data currently emerging from pipeline
always @(posedge clk) begin
    if (~resetn | exec_run | bramPresent_addr_rst)
        bramPresent_waddr <= 14'd0;
    else if (bramPresent_addr_inc && exec_bram_phase1_ready)
        bramPresent_waddr <= bramPresent_waddr + 1'b1;
end
```

**Address Relationship:**
```
Cycle 0-2 (FILL_PIPE):
   raddr: 0→1→2→3
   waddr: 0→0→0→0  (not advancing until exec_bram_phase1_ready)

Cycle 3+ (READ_INPUTS):
   raddr: 3→4→5→6→...
   waddr: 0→1→2→3→...  (maintaining 3-cycle lag)
```

**Why Two Addresses?**
- BRAM has 3-cycle read latency
- raddr issues read requests
- waddr writes zeros to addresses whose data has emerged from pipeline
- This implements "read first" behavior: read data, then clear it

#### Future BRAM Pipeline Hazard Handling

The base version implements a sophisticated 3-stage pipeline to handle concurrent writes to the same BRAM address during the PIPE_DEPTH filling phase.

**Problem**: If two `setArray_go` pulses target the same address within 3 cycles, data could be lost.

**Solution**: Three-stage pipeline with hazard detection and data merging.

```verilog
// Pipeline registers
reg [13:0] bramFuture_waddr [2:0];  // Stages 2→1→0
reg        bramFuture_wren  [2:0];
reg  [7:0] bramFuture_wdata [2:0];

// Stage assignments (stage 2 is newest, stage 0 is oldest)
always @(posedge clk) begin
    if (~resetn) begin
        // Initialize all stages
        bramFuture_wdata[2] <= 8'd0;
        bramFuture_wdata[1] <= 8'd0;
        bramFuture_wdata[0] <= 8'd0;
        // ... (similar for waddr, wren)
    end else if (setArray_go) begin
        // Check for hazards at each stage
        if (setArray_addr == bramFuture_waddr[2]) begin
            // Hazard in stage 2: merge immediately
            bramFuture_wdata[2] <= 8'd0;
            bramFuture_wdata[1] <= bramFuture_wdata[2] | setArray_data;
            bramFuture_wdata[0] <= bramFuture_wdata[1];
            bramFuture_wren[2]  <= 1'b0;
        end else if (setArray_addr == bramFuture_waddr[1]) begin
            // Hazard in stage 1: merge with stage 1 data
            bramFuture_wdata[2] <= 8'd0;
            bramFuture_wdata[1] <= bramFuture_wdata[2];
            bramFuture_wdata[0] <= bramFuture_wdata[1] | setArray_data;
            bramFuture_wren[2]  <= 1'b0;
        end else if (setArray_addr == bramFuture_waddr[0]) begin
            // Hazard in stage 0: data will merge at BRAM (commented out)
            // Current code doesn't merge (see lines 95-96, 103-104)
            bramFuture_wdata[2] <= 8'd0;
            bramFuture_wdata[1] <= bramFuture_wdata[2];
            bramFuture_wdata[0] <= bramFuture_wdata[1];
            bramFuture_wren[2]  <= 1'b0;
        end else begin
            // No hazard: normal pipeline operation
            bramFuture_wdata[2] <= setArray_data;
            bramFuture_wdata[1] <= bramFuture_wdata[2];
            bramFuture_wdata[0] <= bramFuture_wdata[1];
            bramFuture_wren[2]  <= 1'b1;
        end

        // Always propagate addresses and enables
        bramFuture_waddr[2] <= setArray_addr;
        bramFuture_waddr[1] <= bramFuture_waddr[2];
        bramFuture_waddr[0] <= bramFuture_waddr[1];
        bramFuture_wren[1]  <= bramFuture_wren[2];
        bramFuture_wren[0]  <= bramFuture_wren[1];
    end else begin
        // No new write: propagate with zeros
        bramFuture_wdata[2] <= 8'd0;
        bramFuture_wdata[1] <= bramFuture_wdata[2];
        bramFuture_wdata[0] <= bramFuture_wdata[1];
        // ... (propagate addresses/enables)
    end
end
```

**Hazard Example:**

```
Cycle | setArray_go | addr | data | Stage2    | Stage1    | Stage0    | Action
------|-------------|------|------|-----------|-----------|-----------|------------------
  0   |      1      | 100  | 0x01 | 100/0x01  |    -/-    |    -/-    | New write
  1   |      1      | 100  | 0x02 | 100/0x00  | 100/0x03  |    -/-    | Hazard! Merge 0x01|0x02=0x03
  2   |      1      | 200  | 0x04 | 200/0x04  | 100/0x00  | 100/0x03  | No hazard
  3   |      0      |  -   |  -   |    -/0x00 | 200/0x04  | 100/0x00  | Propagate
  4   |      0      |  -   |  -   |    -/0x00 |    -/0x00 | 200/0x04  | Write 100(0x03)
  5   |      0      |  -   |  -   |    -/0x00 |    -/0x00 |    -/0x00 | Write 200(0x04)
```

**Note**: Lines 95-96 and 103-104 show debugging modifications that bypass the final merge operation:
```verilog
// Original (with full hazard handling):
// assign bram0_wdata = ~bram_select ? bramPresent_wdata : bramFuture_wdata[0] | bramFuture_rdata | setArray_data_pipe;

// Debug version (simpler, may lose events):
assign bram0_wdata = ~bram_select ? bramPresent_wdata : bramFuture_wdata[0];
```

### Simple Version Logic

The simple version removes complex hazard handling for single-core operation:

**Key Simplifications:**

1. **Direct Future Write** (no pipeline):
```verilog
// No pipeline registers - direct assignment
assign bramFuture_waddr = axonEvent_addr_reg;
assign bramFuture_wdata = axonEvent_data_reg;
assign bramFuture_wren  = axonEvent_set_reg;
```

2. **No Future BRAM Read** (unless debugging):
```verilog
assign bramFuture_raddr = 13'd0;
assign bramFuture_rden  = 1'b0;  // Disabled to avoid pipeline issues
```

3. **Simplified State Machine** (4 states instead of 5):
```verilog
// Removed STATE_PHASE1_DONE, completion detected in STATE_READ_INPUTS
STATE_READ_INPUTS: begin
    if (exec_hbm_rvalidready) begin
        bramPresent_rden = 1'b1;
        bramPresent_wren = 1'b1;
        if (bramPresent_waddr == axon_addr_limit) begin
            phase1_done_set = 1'b1;
            next_state = STATE_IDLE;  // Direct transition
        end
    end
end
```

4. **16 Axons Per Row**:
```verilog
// Base version: 8 axons per row
// BRAM_ADDR_LIMIT = num_inputs[16:3]  // Divide by 8

// Simple version: 16 axons per row
// axon_addr_limit = num_inputs[16:4]  // Divide by 16
```

**Address Calculation Example:**
- `num_inputs = 17'd131072` (max neurons)
- Base: `BRAM_ADDR_LIMIT = 131072 >> 3 = 16384` rows
- Simple: `axon_addr_limit = 131072 >> 4 = 8192` rows

5. **Registered Input Events**:
```verilog
// Better place-and-route by registering inputs
always @(posedge clk) begin
    if (~resetn) begin
        axonEvent_set_reg  <= 1'b0;
        axonEvent_addr_reg <= 13'd0;
        axonEvent_data_reg <= 16'd0;
    end else begin
        axonEvent_set_reg  <= axonEvent_set;
        axonEvent_addr_reg <= axonEvent_addr;
        axonEvent_data_reg <= axonEvent_data;
    end
end
```

### V2 Version Enhancements

The V2 version adds debug capabilities while simplifying the future write logic:

#### Simplified Future Write (Read-Modify-Write)

Instead of complex pipeline hazard detection, V2 uses RMW:

```verilog
// Read the current value
assign bramFuture_raddr = setArray_addr[16:3];  // Note: only uses upper bits
assign bramFuture_rden  = ci2eep_rden | setArray_go | bramFuture_wren[2] | bramFuture_wren[1] | bramFuture_wren[0];
assign bramFuture_rdata = bram_select ? bram0_rdata : bram1_rdata;

// Merge with new data via OR operation
assign bramFuture_wdata = bramFuture_rdata | setArray_data;

// Propagate through 3-stage pipeline (addresses and enables only)
always @(posedge clk) begin
    if (~resetn) begin
        bramFuture_waddr[2] <= 14'd0;
        bramFuture_waddr[1] <= 14'd0;
        bramFuture_waddr[0] <= 14'd0;
        bramFuture_wren[2]  <= 1'b0;
        bramFuture_wren[1]  <= 1'b0;
        bramFuture_wren[0]  <= 1'b0;
    end else begin
        bramFuture_waddr[2] <= setArray_addr;
        bramFuture_waddr[1] <= bramFuture_waddr[2];
        bramFuture_waddr[0] <= bramFuture_waddr[1];
        bramFuture_wren[2]  <= setArray_go;
        bramFuture_wren[1]  <= bramFuture_wren[2];
        bramFuture_wren[0]  <= bramFuture_wren[1];
    end
end
```

**Why This Works:**
- Always read before write (RMW pattern)
- OR operation merges new spikes with existing ones
- Simpler than explicit hazard detection
- Relies on BRAM "read first" mode

#### Debug State Machine

V2 adds a separate FSM for debug access:

**Debug States:**
```verilog
DBG_STATE_RESET   = 3'd0  // Reset debug logic
DBG_STATE_IDLE    = 3'd1  // Wait for debug command
DBG_STATE_WAIT_0  = 3'd2  // Issue first read
DBG_STATE_WAIT_1  = 3'd3  // Wait cycle 2
DBG_STATE_WAIT_2  = 3'd4  // Wait cycle 3
DBG_STATE_WAIT_3  = 3'd5  // Wait cycle 4
DBG_STATE_DONE    = 3'd6  // Send response, pop command
```

**Debug Flow:**
```
1. Command Interpreter writes address to ci2eep FIFO
2. Debug FSM detects ~ci2eep_empty
3. Issue BRAM read (bram0/1_raddr_dbg = ci2eep_dout)
4. Wait 4 cycles for pipeline (WAIT_0→1→2→3)
5. Write response to eep2ci FIFO (address + data)
6. Pop command from ci2eep (ci2eep_rden = 1)
7. Return to IDLE
```

**Debug Interface Behavior:**
```verilog
always @(*) begin
    ci2eep_rden = 1'b0;
    eep2ci_wren = 1'b0;
    dbg_next_state = dbg_curr_state;

    case (dbg_curr_state)
        DBG_STATE_IDLE: begin
            if (~ci2eep_empty)
                dbg_next_state = DBG_STATE_WAIT_0;
        end
        DBG_STATE_WAIT_0: begin
            if (~eep2ci_full)
                eep2ci_wren = 1'b1;  // Write response
            dbg_next_state = DBG_STATE_WAIT_1;
        end
        // ... (similar for WAIT_1, WAIT_2, WAIT_3)
        DBG_STATE_DONE: begin
            ci2eep_rden = 1'b1;      // Pop command
            dbg_next_state = DBG_STATE_IDLE;
        end
    endcase
end

// Response format: {address, data}
assign eep2ci_din = bram_select ? {bram0_raddr_dbg, bram0_rdata_dbg}
                                : {bram1_raddr_dbg, bram1_rdata_dbg};

// Debug read addresses (always driven)
assign bram0_raddr_dbg = ci2eep_dout;
assign bram1_raddr_dbg = ci2eep_dout;
```

---

## Memory Map

### Base and V2 Versions (8 axons/row)

**BRAM Organization:**
- **Depth**: 16,384 rows (14-bit address)
- **Width**: 8 bits (1 bit per axon)
- **Total Capacity**: 131,072 axons
- **Dual BRAMs**: BRAM0 and BRAM1 (double buffering)

**Address Mapping:**
```
Axon ID Range     | BRAM Address | Bit Position
------------------|--------------|-------------
0 - 7             | 0x0000       | [0] to [7]
8 - 15            | 0x0001       | [0] to [7]
16 - 23           | 0x0002       | [0] to [7]
...               | ...          | ...
131064 - 131071   | 0x3FFF       | [0] to [7]
```

**Bit Encoding:**
- Bit = 1: Axon spiked
- Bit = 0: No spike

**Address Calculation:**
```verilog
bram_addr = axon_id[16:3];      // Upper 14 bits
bit_pos   = axon_id[2:0];       // Lower 3 bits
```

### Simple Version (16 axons/row)

**BRAM Organization:**
- **Depth**: 8,192 rows (13-bit address)
- **Width**: 16 bits (1 bit per axon)
- **Total Capacity**: 131,072 axons
- **Dual BRAMs**: BRAM0 and BRAM1

**Address Mapping:**
```
Axon ID Range     | BRAM Address | Bit Position
------------------|--------------|-------------
0 - 15            | 0x0000       | [0] to [15]
16 - 31           | 0x0001       | [0] to [15]
32 - 47           | 0x0002       | [0] to [15]
...               | ...          | ...
131056 - 131071   | 0x1FFF       | [0] to [15]
```

**Address Calculation:**
```verilog
bram_addr = axon_id[16:4];      // Upper 13 bits
bit_pos   = axon_id[3:0];       // Lower 4 bits
```

**Memory Utilization:**
- Base/V2: 16,384 × 8 = 128 Kb per BRAM → 256 Kb total
- Simple: 8,192 × 16 = 128 Kb per BRAM → 256 Kb total
- Same total capacity, different organization

---

## Timing Diagrams

### Time Step Transition (Double Buffer Swap)

```
         ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────
clk      ┘     └─────┘     └─────┘     └─────┘     └─────┘

exec_run ──────┐     ┌─────────────────────────────────────────
               └─────┘

         Time Step N-1    │ Time Step N
         ─────────────────┼─────────────────────────────────────
                          │
bram_select = 0           │ Toggle → 1
BRAM0 = Present           │ BRAM0 → Future
BRAM1 = Future            │ BRAM1 → Present
                          │
State:   READ_INPUTS      │ IDLE → FILL_PIPE → READ_INPUTS

bramPresent = BRAM0       │ bramPresent = BRAM1
bramFuture  = BRAM1       │ bramFuture  = BRAM0
```

### Present BRAM Read Sequence (Base & V2)

```
Cycle    0    1    2    3    4    5    6    7    8    9
         ─────┬────┬────┬────┬────┬────┬────┬────┬────┬────
State    FILL │FILL│FILL│READ│READ│READ│READ│READ│READ│...
         ────┬┴────┴────┴────┴────┴────┴────┴────┴────┴────

rden     ────┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───
         ────└────┘   └────┘   └────┘   └────┘   └────┘

raddr        0    1    2    3    4    5    6    7    8    9

wren     ───────────────────┐    ┌───┐    ┌───┐    ┌───┐
         ───────────────────└────┘   └────┘   └────┘   └────

waddr        0    0    0    0    1    2    3    4    5    6

                     ┌─────────┐
phase1_ready ────────┘         └──────────────────────────────
                          (asserted in READ_INPUTS state)

hbm_rvalidready ────────────┐    ┌───┐    ┌───┐    ┌───┐
                ────────────└────┘   └────┘   └────┘   └────

exec_bram_spiked    X    X    X    D0   D1   D2   D3   D4   D5
                    (pipeline latency = 3 cycles)

BRAM Data Flow:
  Cycle 0: Issue read addr 0  →  (3-cycle latency)  →  Cycle 3: Data 0 emerges
  Cycle 1: Issue read addr 1  →  (3-cycle latency)  →  Cycle 4: Data 1 emerges
  Cycle 2: Issue read addr 2  →  (3-cycle latency)  →  Cycle 5: Data 2 emerges
  Cycle 3: Issue read addr 3, Write 0 to addr 0
  Cycle 4: Issue read addr 4, Write 0 to addr 1
  ...
```

### Future BRAM Write with Hazard (Base Version)

```
Cycle        0    1    2    3    4    5    6    7
             ────┬────┬────┬────┬────┬────┬────┬────
setArray_go  ───┐    ┌───┐    ┌───────────────────
             ───└────┘   └────┘

setArray_addr    100  200  100  -    -    -    -    -
setArray_data    0x01 0x04 0x02 -    -    -    -    -

Pipeline Stage 2:
  waddr          100  200  100  -    -    -    -    -
  wdata          0x01 0x04 0x00 -    -    -    -    -
  wren           1    1    0    -    -    -    -    -

Pipeline Stage 1:
  waddr          -    100  200  100  -    -    -    -
  wdata          -    0x01 0x04 0x03 -    -    -    - ← Merged!
  wren           -    1    1    0    -    -    -    -

Pipeline Stage 0:
  waddr          -    -    100  200  100  -    -    -
  wdata          -    -    0x01 0x04 0x00 -    -    -
  wren           -    -    1    1    0    -    -    -

BRAM Write:
  Addr 100 ──────────────────────────────────────┐
    Data = 0x01 (from cycle 2) ───────────────────┘

  Addr 200 ─────────────────────────────────────────┐
    Data = 0x04 (from cycle 3) ───────────────────┘

  Addr 100 ────────────────────────────────────────────┐
    Data = 0x03 (merged 0x01|0x02 from cycle 4) ───────┘

Note: Cycle 2 write to addr 100 detected hazard with cycle 0, merged data in stage 1.
```

### Simple Version: No Pipeline (Direct Write)

```
Cycle        0    1    2    3    4    5
             ────┬────┬────┬────┬────┬────
axonEvent_set ──┐    ┌───┐    ┌──────────
             ───└────┘   └────┘

axonEvent_addr   100  200  300  -    -    -
axonEvent_data   0x0001 0x0004 0x0008 -    -    -

(Register inputs for better timing)
             ────┬────┬────┬────┬────┬────
*_set_reg    ───────┐    ┌───┐    ┌──────
                ───└────┘   └────┘

*_addr_reg       -    100  200  300  -    -
*_data_reg       -    0x0001 0x0004 0x0008 -    -

bramFuture_waddr -    100  200  300  -    -
bramFuture_wdata -    0x0001 0x0004 0x0008 -    -
bramFuture_wren  -    1    1    1    -    -

BRAM Write:      -    100  200  300  -    -
                      0x0001 0x0004 0x0008

No hazard handling! Single-core only.
If same address written twice within pipeline depth, later write overwrites earlier.
```

### V2 Debug Read Sequence

```
Cycle        0    1    2    3    4    5    6    7
             ────┬────┬────┬────┬────┬────┬────┬────
DBG_State    IDLE│WAIT│WAIT│WAIT│WAIT│DONE│IDLE│...
                 │  0 │  1 │  2 │  3 │    │    │

ci2eep_empty ────┐                                  ┌───
             ────└──────────────────────────────────┘

ci2eep_dout      0x1234 (stays valid until rden)

ci2eep_rden  ────────────────────────────────────┐  ┌───
             ────────────────────────────────────└──┘

bram*_raddr_dbg  0x1234 (always driven)

eep2ci_wren  ───────┐    ┌───┐    ┌───┐    ┌───┐  ┌───
             ───────└────┘   └────┘   └────┘   └──┘
                 (writes in WAIT_0 through WAIT_3 if not full)

eep2ci_din       X    {0x1234,D}  (data D emerges after latency)

Debug transaction:
  Cycle 0: Detect command available
  Cycle 1-4: Issue reads, wait for pipeline, write response
  Cycle 5: Pop command FIFO
  Cycle 6: Return to IDLE
```

---

## Cross-References

### Upstream Modules
- **command_interpreter.v** (`command_interpreter.md`):
  - Generates `setArray_go`, `setArray_addr`, `setArray_data` signals
  - V2: Provides debug FIFO interfaces (`ci2eep_*`, `eep2ci_*`)
  - Controls when external events are injected

- **pcie2fifos.v** (`pcie2fifos.md`):
  - Ultimate source of external events from host
  - Events flow: PCIe → Command Interpreter → External Events Processor

### Downstream Modules
- **hbm_processor.v** (`hbm_processor.md`):
  - Receives `exec_bram_spiked` (spike mask)
  - Provides `exec_hbm_rvalidready` (synchronization signal)
  - Uses spike masks to fetch pointer chains from HBM

- **internal_events_processor.v** (`internal_events_processor.md`):
  - Receives `exec_bram_phase1_done` (completion signal)
  - Coordinates two-phase execution (external then internal events)

### Peer Modules
- **pointer_fifo_controller.v** (`pointer_fifo_controller.md`):
  - Works with spike masks from this module
  - Controls flow of pointer data to HBM processor

---

## Module Comparison: When to Use Each Variant

### Use Base Version When:
- **Multi-core architecture** with multiple cores writing to same future BRAM
- **Concurrent writes** to the same BRAM address are expected
- **Data integrity** is critical and no events can be lost
- **Pipeline hazards** need explicit detection and merging
- **8 axons per row** organization preferred

**Trade-offs:**
- ✅ Full hazard handling
- ✅ No data loss in multi-core scenarios
- ❌ More complex logic
- ❌ Higher resource usage (pipeline registers)
- ⚠️ Debugging modifications present (lines 95-96, 103-104)

### Use Simple Version When:
- **Single-core architecture** with only one writer to future BRAM
- **Lower resource usage** is priority
- **Wider data paths** (16-bit) preferred for bandwidth
- **No concurrent writes** to same address expected
- **Simpler logic** easier to verify and debug

**Trade-offs:**
- ✅ Minimal resource usage
- ✅ 2× data width (16 vs 8 bits)
- ✅ Simpler state machine (4 vs 5 states)
- ✅ Better timing due to registered inputs
- ❌ No hazard protection
- ❌ Data loss if concurrent writes occur
- ❌ Single-core only

### Use V2 Version When:
- **Debug and verification** required
- **BRAM inspection** needed during runtime
- **Command interpreter interface** for test patterns
- **Read-modify-write** approach acceptable
- **Production debugging** of neuromorphic algorithms

**Trade-offs:**
- ✅ Debug capabilities (FIFO interface)
- ✅ Simplified future write logic (RMW vs explicit hazards)
- ✅ Direct BRAM inspection via debug ports
- ❌ Additional debug FSM (more resources)
- ❌ Extra FIFO interfaces
- ❌ Not optimized for performance

---

## Performance Characteristics

### Base and V2 Versions

**Throughput:**
- **Read Rate**: 1 BRAM address per `exec_hbm_rvalidready` cycle
- **Effective Rate**: Limited by HBM bandwidth (~450 MHz possible, typically 225 MHz)
- **Pipeline Fill**: 3 cycles (one-time cost per time step)
- **Total Time**: `3 + num_inputs[16:3]` cycles per time step

**Example** (131,072 neurons):
```
BRAM addresses = 131072 / 8 = 16384
Pipeline fill  = 3 cycles
Total cycles   = 3 + 16384 = 16387 cycles
At 225 MHz     = 16387 / 225e6 = 72.8 µs
```

**Future Write Latency:**
- **Base**: 3 cycles (pipeline depth) from `setArray_go` to BRAM write
- **V2**: 3 cycles (pipeline depth) from `setArray_go` to BRAM write
- **Hazard Penalty**: 0 cycles (merged in pipeline)

### Simple Version

**Throughput:**
- **Read Rate**: 1 BRAM address per `exec_hbm_rvalidready` cycle
- **Effective Rate**: 225 MHz typical
- **Pipeline Fill**: 3 cycles
- **Total Time**: `3 + num_inputs[16:4]` cycles per time step

**Example** (131,072 neurons):
```
BRAM addresses = 131072 / 16 = 8192
Pipeline fill  = 3 cycles
Total cycles   = 3 + 8192 = 8195 cycles
At 225 MHz     = 8195 / 225e6 = 36.4 µs  (2× faster than base!)
```

**Future Write Latency:**
- **Direct**: 1 cycle from `axonEvent_set` to registered write
- **Total**: 2 cycles (register + BRAM write)

**Resource Usage Comparison:**

| Resource | Base | Simple | V2 |
|----------|------|--------|-----|
| LUTs (approx.) | 500 | 250 | 600 |
| Flip-Flops | 200 | 120 | 280 |
| BRAM18K | 2 | 2 | 2 |
| Pipeline Regs | 3×(14+8+1) | 0 | 3×(14+1) |

---

## Common Issues and Debugging

### Issue 1: Events Lost During Time Step Transition

**Symptoms:**
- External events written near `exec_run` pulse disappear
- Inconsistent spike counts between time steps

**Root Cause:**
- Writing to future BRAM while `bram_select` is toggling
- Race condition between write and buffer swap

**Debug:**
```verilog
// Check timing of setArray_go relative to exec_run
// Add ILA probe:
ila_0 your_ila (
    .clk(clk),
    .probe0(exec_run),
    .probe1(setArray_go),
    .probe2(setArray_addr),
    .probe3(bram_select)
);
```

**Solution:**
- Ensure `setArray_go` never occurs within 3 cycles of `exec_run`
- Add FIFO between command interpreter and external events processor
- Stall writes during buffer swap

### Issue 2: Pipeline Hazards Not Detected (Base Version)

**Symptoms:**
- Expected spike data doesn't appear
- OR of multiple writes shows only one bit set

**Root Cause:**
- Hazard detection logic not functioning
- Debugging modifications (lines 95-96, 103-104) bypass merging

**Debug:**
```verilog
// Monitor pipeline stages
(* mark_debug = "true" *) reg [13:0] bramFuture_waddr_dbg [2:0];
(* mark_debug = "true" *) reg  [7:0] bramFuture_wdata_dbg [2:0];
(* mark_debug = "true" *) reg        bramFuture_wren_dbg  [2:0];

always @(posedge clk) begin
    bramFuture_waddr_dbg <= bramFuture_waddr;
    bramFuture_wdata_dbg <= bramFuture_wdata;
    bramFuture_wren_dbg  <= bramFuture_wren;
end
```

**Solution:**
- Restore original BRAM write logic:
```verilog
// Change:
assign bram0_wdata = ~bram_select ? bramPresent_wdata : bramFuture_wdata[0];

// To:
assign bram0_wdata = ~bram_select ? bramPresent_wdata :
                     bramFuture_wdata[0] | bramFuture_rdata | setArray_data_pipe;
```

### Issue 3: Address Limit Calculation Wrong

**Symptoms:**
- Phase 1 completes too early or too late
- Not all neurons receive input events

**Root Cause:**
- Incorrect calculation of BRAM address limit
- Mismatch between `num_inputs` and actual neuron count

**Debug:**
```verilog
// Check address limit
// Base:   BRAM_ADDR_LIMIT = num_inputs[16:3]  (divide by 8)
// Simple: axon_addr_limit = num_inputs[16:4]  (divide by 16)

// Add assertion:
assert property (@(posedge clk) disable iff (~resetn)
    (curr_state == STATE_READ_INPUTS && bramPresent_waddr == BRAM_ADDR_LIMIT)
    |=> (curr_state == STATE_PHASE1_DONE)
);
```

**Solution:**
- Verify `num_inputs` matches neuron configuration
- Base: Ensure multiple of 8
- Simple: Ensure multiple of 16
- Add `+1` if rounding needed:
```verilog
// If num_inputs not exact multiple
assign BRAM_ADDR_LIMIT = (num_inputs[16:3]) + |num_inputs[2:0];  // Round up
```

### Issue 4: BRAM Read Latency Mismatch

**Symptoms:**
- Data appears corrupted or delayed
- `exec_bram_spiked` shows wrong values

**Root Cause:**
- BRAM configured with latency ≠ PIPE_DEPTH (3)
- Pipeline depth parameter doesn't match actual BRAM

**Debug:**
```verilog
// Verify BRAM configuration in IP customization:
// - Read Latency: should be 3
// - Primitive Type: should match PIPE_DEPTH

// Check if raddr/waddr maintain proper offset:
assert property (@(posedge clk) disable iff (~resetn)
    (curr_state == STATE_READ_INPUTS && exec_bram_phase1_ready)
    |-> (bramPresent_raddr == bramPresent_waddr + PIPE_DEPTH)
);
```

**Solution:**
- Reconfigure BRAM IP for 3-cycle latency
- Or update PIPE_DEPTH parameter to match BRAM:
```verilog
external_events_processor #(
    .PIPE_DEPTH(2)  // If BRAM has 2-cycle latency
) eep_inst (
    // ...
);
```

### Issue 5: V2 Debug Reads Return Stale Data

**Symptoms:**
- Debug responses show old/incorrect BRAM data
- Debug state machine stuck in WAIT states

**Root Cause:**
- Insufficient wait cycles for BRAM read latency
- Debug FSM transitions too quickly

**Debug:**
```verilog
// Monitor debug state progression
(* mark_debug = "true" *) reg [2:0] dbg_state_history [7:0];

always @(posedge clk) begin
    dbg_state_history[7:1] <= dbg_state_history[6:0];
    dbg_state_history[0]   <= dbg_curr_state;
end
```

**Solution:**
- Ensure 4 WAIT states (WAIT_0→WAIT_1→WAIT_2→WAIT_3)
- Add extra wait state if needed:
```verilog
localparam [2:0] DBG_STATE_WAIT_4 = 3'd7;

// In state machine:
DBG_STATE_WAIT_3: begin
    if (~eep2ci_full)
        eep2ci_wren = 1'b1;
    dbg_next_state = DBG_STATE_WAIT_4;  // Extra cycle
end
DBG_STATE_WAIT_4: begin
    dbg_next_state = DBG_STATE_DONE;
end
```

---

## Safety and Edge Cases

### Edge Case 1: num_inputs = 0

**Behavior:**
- `BRAM_ADDR_LIMIT = 0`
- State machine immediately transitions FILL_PIPE → READ_INPUTS → PHASE1_DONE
- No BRAM accesses occur

**Safety:**
- ✅ No undefined behavior
- ✅ Module functions correctly (zero inputs processed)
- ⚠️ Wastes cycles (should be caught at system level)

### Edge Case 2: num_inputs Not Multiple of 8 (or 16)

**Example:** `num_inputs = 17'd100`

**Base Version:**
```verilog
BRAM_ADDR_LIMIT = 100 >> 3 = 12
Actual coverage   = 12 × 8 = 96 axons
Missing           = 4 axons (96-99 not processed)
```

**Fix:**
```verilog
// Round up to nearest multiple
assign BRAM_ADDR_LIMIT = (num_inputs + 7) >> 3;  // Ceiling division
```

### Edge Case 3: Concurrent setArray_go and exec_run

**Scenario:**
```
Cycle N:   exec_run = 1 (toggle bram_select)
Cycle N:   setArray_go = 1 (write to future BRAM)
```

**Problem:**
- `bram_select` changes, may write to wrong BRAM

**Current Design:**
- `bram_select` registered on `exec_run` edge
- `setArray_go` writes on same edge
- **Race condition!** Indeterminate which BRAM receives write

**Solution:**
- Pipeline `exec_run` by 1 cycle:
```verilog
reg exec_run_pipe;

always @(posedge clk) begin
    if (~resetn)
        exec_run_pipe <= 1'b0;
    else
        exec_run_pipe <= exec_run;
end

// Use exec_run_pipe for bram_select toggle
always @(posedge clk) begin
    if (~resetn)
        bram_select <= 1'b0;
    else if (exec_run_pipe)  // Changed from exec_run
        bram_select <= ~bram_select;
end
```

### Edge Case 4: BRAM Write During Pipeline Fill

**Scenario:**
```
STATE_FILL_PIPE: bramPresent_wren = 0 (not asserted yet)
Future writes:    bramFuture_wren[0] = 1 (trying to write)
```

**Problem (Multi-core):**
- If multiple cores write to future BRAM during present BRAM pipeline fill
- Potential for lost writes if exceeding BRAM write bandwidth

**Current Design:**
- Single write port per BRAM
- Future writes serialized through pipeline
- **Safe** as long as write rate ≤ 1 per 3 cycles

**Solution (if needed):**
- Use dual-port BRAM (separate read/write ports)
- Or implement write FIFO to buffer concurrent writes

### Safety Check: Phase 1 Completion Detection

**Assertion:**
```verilog
// Ensure phase1_done only asserted when all addresses processed
property phase1_done_check;
    @(posedge clk) disable iff (~resetn)
    (exec_bram_phase1_done) |-> (bramPresent_waddr == BRAM_ADDR_LIMIT);
endproperty
assert_phase1: assert property (phase1_done_check);
```

### Safety Check: No Writes During Buffer Swap

**Assertion:**
```verilog
// Ensure no future writes during exec_run
property no_write_during_swap;
    @(posedge clk) disable iff (~resetn)
    (exec_run) |-> (bramFuture_wren[0] == 1'b0);
endproperty
assert_no_write: assert property (no_write_during_swap);
```

---

## Future Enhancement Opportunities

### 1. Configurable Data Width

Allow parameterization of axons per row:

```verilog
module external_events_processor #(
    parameter PIPE_DEPTH = 3,
    parameter AXONS_PER_ROW = 8  // 8, 16, 32, etc.
)(
    // Derive address and data widths
    localparam ADDR_BITS = 17 - $clog2(AXONS_PER_ROW);
    localparam DATA_BITS = AXONS_PER_ROW;

    input [ADDR_BITS-1:0] setArray_addr,
    input [DATA_BITS-1:0] setArray_data,
    // ...
);
```

### 2. Burst Mode for Faster Pipeline Fill

Current: Fill pipeline sequentially (3 cycles)
Enhancement: Issue all 3 reads in 1 cycle (if BRAM supports)

```verilog
STATE_FILL_PIPE: begin
    if (bramPresent_raddr == 0) begin
        // Issue all 3 reads at once
        bram_raddr[0] = 14'd0;
        bram_raddr[1] = 14'd1;
        bram_raddr[2] = 14'd2;
        bram_rden[0]  = 1'b1;
        bram_rden[1]  = 1'b1;
        bram_rden[2]  = 1'b1;
        next_state = STATE_READ_INPUTS;
    end
end
```

### 3. Event Timestamping

Add timestamp to each event for precise temporal resolution:

```verilog
// Expand data width: [7:0] data + [15:0] timestamp
input [23:0] setArray_data,  // {timestamp, spike_mask}

// BRAM organization: 24 bits per row
```

### 4. Event Compression

Sparse events (few spikes per row) waste bandwidth:

```verilog
// Instead of full bit mask, store indices
// Example: Spikes at axons 5, 17, 42
// Compressed: {3'b011, 6'd42, 6'd17, 6'd5}  // Count + indices
```

### 5. Multi-Buffer (>2 BRAMs)

Allow more than 2 time steps in flight:

```verilog
parameter NUM_BUFFERS = 4;  // Quad buffering

reg [1:0] bram_select;      // 2-bit select (4 buffers)

always @(posedge clk) begin
    if (exec_run)
        bram_select <= (bram_select + 1) & 2'b11;  // Circular
end
```

### 6. AXI4-Stream Interface

Replace custom interface with standard AXI4-Stream:

```verilog
// Input events
input         s_axis_tvalid,
output        s_axis_tready,
input  [31:0] s_axis_tdata,  // {addr, data}
input         s_axis_tlast,

// Output spikes
output        m_axis_tvalid,
input         m_axis_tready,
output [31:0] m_axis_tdata,  // Spike mask + metadata
```

### 7. Configurable Pipeline Depth

Auto-detect BRAM latency at synthesis:

```verilog
// Query BRAM IP for latency
localparam BRAM_LATENCY = bram0.READ_LATENCY_A;  // From BRAM IP

external_events_processor #(
    .PIPE_DEPTH(BRAM_LATENCY)  // Match automatically
) eep (
    // ...
);
```

---

## Key Terms and Definitions

| Term | Definition |
|------|------------|
| **Axon** | Input neuron connection; source of spike events |
| **Double Buffering** | Two-buffer scheme (present/future) allowing simultaneous read and write |
| **Present BRAM** | BRAM being read during current time step (then cleared) |
| **Future BRAM** | BRAM accumulating events for next time step |
| **bram_select** | Toggle bit selecting which physical BRAM is present vs. future |
| **Pipeline Depth** | Number of cycles between BRAM read request and data availability (typically 3) |
| **Pipeline Fill** | Initial phase where read pipeline is populated before writes begin |
| **Leading Address** | Read address (raddr) - advances pipeline depth ahead of write address |
| **Lagging Address** | Write address (waddr) - clears data after it emerges from pipeline |
| **Spike Mask** | Bit vector where each bit represents spike (1) or no-spike (0) for an axon |
| **Phase 1** | External event processing (vs. Phase 2: internal/synaptic events) |
| **exec_run** | Control pulse starting new time step, toggling present ←→ future BRAMs |
| **exec_hbm_rvalidready** | Synchronization signal from HBM indicating data consumed, advance BRAM |
| **setArray_go** | Write pulse for external event (from command interpreter or other source) |
| **Pipeline Hazard** | Conflict when concurrent writes target same BRAM address within pipeline depth |
| **RMW (Read-Modify-Write)** | Pattern of reading current value, modifying, then writing back |
| **Hazard Detection** | Logic identifying when new write conflicts with in-flight writes |
| **Data Merging** | Combining multiple writes to same address via OR operation |
| **Time Step** | Discrete computation cycle in neuromorphic algorithm (milliseconds typically) |
| **Axon Event** | External spike arriving at input neuron |
| **Axons Per Row** | Number of axons packed into single BRAM address (8 or 16 bits) |
| **Address Limit** | Maximum BRAM address to read/write (depends on num_inputs) |

---

## Conclusion

The **External Events Processor** family provides flexible solutions for managing input spike events in neuromorphic systems:

- **Base version**: Full-featured with pipeline hazard handling for multi-core
- **Simple version**: Streamlined single-core variant with lower resource usage
- **V2 version**: Debug-enhanced variant for verification and development

**Key Design Principles:**
1. Double buffering prevents event loss during time step transitions
2. Pipeline management ensures correct synchronization with BRAM latency
3. Hazard detection/merging (base) or simplified RMW (V2) prevents data corruption
4. State machine coordinates read-clear cycles with downstream modules

**Selection Guide:**
- Multi-core system with concurrent writes → **Base version**
- Single-core system, resource-constrained → **Simple version**
- Debug/verification needed → **V2 version**

For questions or issues, cross-reference with `command_interpreter.md` (upstream) and `hbm_processor.md` (downstream) for complete system understanding.
