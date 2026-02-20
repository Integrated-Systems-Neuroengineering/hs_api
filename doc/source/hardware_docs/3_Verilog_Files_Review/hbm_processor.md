# hbm_processor.v

## Module Overview

### Purpose and Role in Stack

The **hbm_processor** is the **HBM (High Bandwidth Memory) controller and synapse data manager**, responsible for fetching synaptic connectivity data from off-chip HBM. This module:

- **Implements AXI4 master interface** to HBM (256-bit data width)
- **Orchestrates two-phase data retrieval:**
  - **Phase 1:** Fetch pointer data for external inputs (BRAM) and internal neurons (URAM)
  - **Phase 2:** Follow pointer chains to fetch actual synapse data
- **Manages pointer FIFO** for synapse chain traversal
- **Provides CI access** for reading/writing HBM during configuration
- **Coordinates with spike generation** by writing spike addresses directly to spike FIFOs
- **Combines 256-bit HBM reads into 512-bit packets** for 16 neuron groups

In the software/hardware stack:
```
Command Interpreter ──► HBM read/write requests
                         │
External Events Proc ──► Triggers Phase 1 execution
Internal Events Proc ──► Receives pointer/synapse data
                         │
                         ▼
                   hbm_processor
                   (AXI4 Master)
                         │
                         ▼
                    HBM Memory
                  (Synapse Storage)
                         │
                         ▼
           Pointer FIFO ◄─ Pointer chains
                         │
                         ▼
              Spike FIFOs ◄─ Spike addresses
```

This module is critical for **network connectivity**, translating sparse synaptic connections into efficient memory accesses.

---

## Module Architecture

### High-Level Block Diagram

```
                    hbm_processor
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  ┌───────────────────────────────────────────────┐     │
    │  │   TX (Transmit) State Machine                 │     │
    │  │   - Sends read/write commands to HBM          │     │
    │  │   - Manages address generation                │     │
    │  └────────────┬──────────────────────────────────┘     │
    │               │                                         │
    │  ┌────────────▼────────────────────────────────────┐   │
    │  │   Address Multiplexer                           │   │
    │  │   Phase 0: {0, tx_select, tx_addr, 4'b0}       │   │
    │  │   Phase 1: {ptr_addr, 5'b0}                    │   │
    │  │   CI mode: {ci2hbm_dout[278:256], 5'b0}        │   │
    │  └────────────┬────────────────────────────────────┘   │
    │               │                                         │
    │  ┌────────────▼────────────────────────────────────┐   │
HBM │  │   AXI4 Master Interface                         │   │
AXI4│◄─┤   - araddr, arvalid, arready (Read Address)    │   │
    │  │   - rdata, rvalid, rready (Read Data)          │   │
    │  │   - awaddr, awvalid, awready (Write Address)   │   │
    │  │   - wdata, wvalid, wready (Write Data)         │   │
    │  │   - bvalid, bready (Write Response)            │   │
    │  │   - Burst mode: INCR, size=256-bit             │   │
    │  └────────────┬────────────────────────────────────┘   │
    │               │                                         │
    │  ┌────────────▼────────────────────────────────────┐   │
    │  │   RX (Receive) State Machine                    │   │
    │  │   - Collects HBM read responses                 │   │
    │  │   - Routes data to appropriate destination      │   │
    │  └─┬──────┬──────────┬────────┬──────┬────────────┘   │
    │    │      │          │        │      │                │
    │    │      │          │        │      │                │
Pointer   │      │          │        │      │ Spikes (Phase 1)
FIFO      │      │          │        │      └──► spk0-7_wren
    ◄─────┘      │          │        │
                 │          │        │
Command          │          │        └──► hbm2ci (CI responses)
Interpreter      │          │
                 │          │
                 │          └──► exec_hbm_rdata (512-bit)
Internal/        │                  [511:0] = {upper256, lower256}
External         │                  Phase 1: Pointer data
Events Procs     │                  Phase 2: Synapse data
                 │
                 ▼
          ┌────────────────────────────────┐
          │   256→512 bit Converter         │
          │   - hbm_count toggles           │
          │   - Combines 2 × 256-bit reads  │
          │   - Outputs on 2nd read         │
          └────────────────────────────────┘
    │                                                         │
    │  ┌───────────────────────────────────────────────┐     │
    │  │   Pointer Chain Management                    │     │
    │  │   - ptrFIFO_dout[31:23]: Length (9 bits)      │     │
    │  │   - ptrFIFO_dout[22:0]:  Address (23 bits)    │     │
    │  │   - ptr_burst: Dynamic burst calculation      │     │
    │  │   - ptr_ctr: Tracks progress through chain   │     │
    │  └───────────────────────────────────────────────┘     │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

---

## Interface Specification

### Module Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HBM_ADDR_BITS` | 33 | HBM address width (8 GB addressable) |
| `HBM_DATA_WIDTH` | 256 | HBM data bus width |
| `HBM_BYTE_COUNT` | 32 | Bytes per transaction (256/8) |

### Clock and Reset

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `clk` | Input | 1 | 225 MHz system clock |
| `resetn` | Input | 1 | Active-low synchronous reset |

### Network Configuration

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `num_inputs` | Input | 17 | Number of input axons (external events) |
| `num_outputs` | Input | 17 | Number of output neurons (internal events) |
| `core_number` | Input | 5 | Core identifier (0-31) for multi-core systems |

### Execution Control

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `exec_run` | Input | 1 | Start new timestep execution |
| `exec_bram_phase1_ready` | Input | 1 | External events processor pipeline filled |
| `exec_uram_phase1_ready` | Input | 1 | Internal events processor pipeline filled |
| `exec_hbm_rvalidready` | Output (wire) | 1 | HBM data valid for IEP/EEP (every 2nd read) |
| `exec_hbm_tx_phase1_done` | Output (wire) | 1 | TX completed Phase 1 command sending |
| `exec_hbm_tx_phase2_done` | Output (wire) | 1 | TX completed Phase 2 command sending |
| `exec_hbm_rx_phase1_done` | Output (wire) | 1 | RX completed Phase 1 data collection |
| `exec_hbm_rx_phase2_done` | Output (wire) | 1 | RX completed Phase 2 data collection |

### HBM Data Output

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `exec_hbm_rdata` | Output (wire) | 512 | Combined HBM data for 16 neuron groups |
| `hbmFIFO_full` | Input | 1 | Backpressure from downstream FIFO |

**Data Format (`exec_hbm_rdata[511:0]`):**

```
[511:256] = Upper 256-bit read (most recent)
[255:0]   = Lower 256-bit read (latched from previous cycle)

Each 256-bit word contains data for 8 neuron groups (32 bits each)
Two 256-bit reads → 16 neuron groups → 512-bit output
```

### Pointer FIFO Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `ptrFIFO_empty` | Input | 1 | Pointer FIFO empty flag |
| `ptrFIFO_dout` | Input | 32 | Pointer FIFO data output |
| `ptrFIFO_rden` | Output (reg) | 1 | Pointer FIFO read enable |

**Pointer Format (`ptrFIFO_dout[31:0]`):**
```
[31:23] = Chain length (9 bits, max 511 synapses)
[22:0]  = HBM address (23 bits, byte address >> 5)
```

### Command Interpreter Interface

**Input (CI to HBM):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `ci2hbm_empty` | Input | 1 | Command FIFO empty flag |
| `ci2hbm_dout` | Input | 280 | Command data |
| `ci2hbm_rden` | Output (reg) | 1 | Command FIFO read enable |

**Command Format (`ci2hbm_dout[279:0]`):**
```
[279]       = R/W (0=read, 1=write)
[278:256]   = HBM address (23 bits)
[255:0]     = Write data (256 bits)
```

**Output (HBM to CI):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `hbm2ci_full` | Input | 1 | Response FIFO full flag |
| `hbm2ci_din` | Output (wire) | 256 | Response data (= hbm_rdata) |
| `hbm2ci_wren` | Output (reg) | 1 | Response FIFO write enable |

### Spike FIFO Interface (8 FIFOs)

**Per FIFO (0-7):**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `spkN_full` | Input | 1 | Spike FIFO full flag |
| `spkN_din` | Output (wire) | 17 | Spike neuron address |
| `spkN_wren` | Output (wire) | 1 | Spike FIFO write enable |

**Spike Data Extraction:**
```verilog
spk0_din = hbm_rdata[016:000];  // 17-bit neuron address
spk0_wren = !spk0_full & exec_hbm_rx_phase1_done &
            exec_hbm_rvalidready_2x & hbm_rdata[031];  // Spike flag

// Similar for spk1-7 from hbm_rdata[048:032] through [240:224]
```

**Note:** Spike data embedded in HBM pointer reads during Phase 1.

### HBM AXI4 Master Interface

**Read Address Channel:**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `hbm_araddr` | Output (reg) | 33 | Read address |
| `hbm_arburst` | Output (wire) | 2 | Burst type (2'b01 = INCR) |
| `hbm_arid` | Output (wire) | 6 | Transaction ID (always 6'd0) |
| `hbm_arlen` | Output (reg) | 4 | Burst length (beats - 1) |
| `hbm_arready` | Input | 1 | Address channel ready |
| `hbm_arsize` | Output (wire) | 3 | Beat size (3'd5 = 32 bytes = 256 bits) |
| `hbm_arvalid` | Output (reg) | 1 | Address valid |

**Read Data Channel:**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `hbm_rdata` | Input | 256 | Read data |
| `hbm_rid` | Input | 6 | Transaction ID |
| `hbm_rlast` | Input | 1 | Last beat of burst |
| `hbm_rready` | Output (reg) | 1 | Data channel ready |
| `hbm_rresp` | Input | 2 | Read response (ignored) |
| `hbm_rvalid` | Input | 1 | Read data valid |

**Write Address Channel:**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `hbm_awaddr` | Output (wire) | 33 | Write address (from ci2hbm) |
| `hbm_awburst` | Output (wire) | 2 | Burst type (2'b01 = INCR) |
| `hbm_awid` | Output (wire) | 6 | Transaction ID (always 6'd0) |
| `hbm_awlen` | Output (wire) | 4 | Burst length (always 4'd0 = 1 beat) |
| `hbm_awready` | Input | 1 | Address channel ready |
| `hbm_awsize` | Output (wire) | 3 | Beat size (3'd5 = 256 bits) |
| `hbm_awvalid` | Output (reg) | 1 | Address valid |

**Write Data Channel:**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `hbm_wdata` | Output (wire) | 256 | Write data (from ci2hbm) |
| `hbm_wlast` | Output (wire) | 1 | Last beat (always 1 for single-beat) |
| `hbm_wready` | Input | 1 | Data channel ready |
| `hbm_wstrb` | Output (wire) | 32 | Write strobes (all 1's) |
| `hbm_wvalid` | Output (reg) | 1 | Write data valid |

**Write Response Channel:**

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `hbm_bid` | Input | 6 | Transaction ID |
| `hbm_bready` | Output (reg) | 1 | Response channel ready |
| `hbm_bresp` | Input | 2 | Write response (ignored) |
| `hbm_bvalid` | Input | 1 | Response valid |

### Debug Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `hbm_curr_state` | Output (wire) | 4 | TX state machine state (for VIO) |

---

## Detailed Logic Description

### TX (Transmit) State Machine

**States:**
```verilog
TX_STATE_RESET                          (4'd0)
TX_STATE_IDLE                           (4'd1)
TX_STATE_SEND_INPUT_READ_COMMANDS       (4'd2)  // Phase 1a
TX_STATE_SEND_OUTPUT_READ_COMMANDS      (4'd3)  // Phase 1b
TX_STATE_PHASE1_DONE                    (4'd4)
TX_STATE_POP_POINTER_FIFO               (4'd5)  // Phase 2 prep
TX_STATE_SEND_POINTER_READ_COMMANDS     (4'd6)  // Phase 2
TX_STATE_PHASE2_DONE                    (4'd7)
TX_STATE_READ_HBM_ADDR                  (4'd8)  // CI read
TX_STATE_WRITE_HBM_ADDR                 (4'd9)  // CI write address
TX_STATE_WRITE_HBM_DATA                 (4'd10) // CI write data
TX_STATE_WRITE_HBM_RESP                 (4'd11) // CI write response
```

**State Transition Diagram:**

```
        ┌──────────────┐
        │ TX_RESET     │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
    ┌──▶│ TX_IDLE      │◄─────────────────────────────┐
    │   └──┬───────┬───┘                              │
    │      │       │                                  │
    │ exec │       │ !ci2hbm_empty                    │
    │ _run │       ├─ R/W=0 ──> READ_HBM_ADDR ────────┤
    │      │       │                                  │
    │      │       └─ R/W=1 ──> WRITE_HBM_ADDR ──>   │
    │      │                      WRITE_HBM_DATA ──>  │
    │      │                      WRITE_HBM_RESP ─────┘
    │      │
    │      ▼
    │ SEND_INPUT_READ_COMMANDS
    │  (Phase 1a: External inputs)
    │  tx_addr: 0 → INPUT_ADDR_LIMIT
    │      │
    │      ▼
    │ SEND_OUTPUT_READ_COMMANDS
    │  (Phase 1b: Internal neurons)
    │  tx_addr: 0 → OUTPUT_ADDR_LIMIT
    │      │
    │      ▼
    │ PHASE1_DONE
    │  (toggle tx_phase, tx_select)
    │      │
    │      ▼
    │ POP_POINTER_FIFO
    │  (wait for ptrFIFO data)
    │  (255-cycle timeout)
    │      │
    │      ├─ !empty ──> SEND_POINTER_READ_COMMANDS
    │      │              (follow pointer chain)
    │      │                      │
    │      │              ptr_ctr reaches ptr_len
    │      │                      │
    │      │              ◄───────┘ (loop for next pointer)
    │      │
    │      └─ timeout ──> PHASE2_DONE
    │                          │
    └──────────────────────────┘
```

**Phase 1 Addressing:**

```verilog
// Phase 0 (tx_phase = 0):
hbm_araddr = {5'd0, {8'd0, tx_select, tx_addr, 4'd0}, 5'd0};

Breakdown:
  [32:28] = 5'd0 (upper padding)
  [27:5]  = {8'd0, tx_select, tx_addr, 4'd0}
            [22:15] = 8'd0 (reserved/bank select)
            [14]    = tx_select (0=inputs/BRAM, 1=outputs/URAM)
            [13:4]  = tx_addr (10 bits)
            [3:0]   = 4'd0 (8 pointers per row × 4 bytes = 32 bytes = 5 bits)
  [4:0]   = 5'd0 (byte offset within 32-byte row)
```

**Phase 2 Addressing:**

```verilog
// Phase 1 (tx_phase = 1):
hbm_araddr = {5'd0, ptr_addr, 5'd0};

Breakdown:
  [32:28] = 5'd0
  [27:5]  = ptr_addr (23 bits from ptrFIFO_dout)
  [4:0]   = 5'd0
```

### RX (Receive) State Machine

**States:**
```verilog
RX_STATE_RESET                (4'd0)
RX_STATE_IDLE                 (4'd1)
RX_STATE_WAIT_BRAM_PIPELINE   (4'd2)  // Wait for EEP ready
RX_STATE_READ_INPUT_POINTERS  (4'd3)  // Collect external pointer data
RX_STATE_WAIT_URAM_PIPELINE   (4'd4)  // Wait for IEP ready
RX_STATE_READ_OUTPUT_POINTERS (4'd5)  // Collect internal pointer data
RX_STATE_PHASE1_DONE          (4'd6)
RX_STATE_READ_SYNAPSE_DATA    (4'd7)  // Collect synapse data (Phase 2)
RX_STATE_PHASE2_DONE          (4'd8)
RX_STATE_READ_HBM_RESP        (4'd9)  // CI read response
```

**State Transition Diagram:**

```
        ┌──────────────┐
        │ RX_RESET     │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
    ┌──▶│ RX_IDLE      │◄────────────────────────┐
    │   └──┬───────┬───┘                         │
    │      │       │                             │
    │ exec │       │ TX → READ_HBM_ADDR          │
    │ _run │       └──> READ_HBM_RESP ───────────┘
    │      │
    │      ▼
    │ WAIT_BRAM_PIPELINE
    │  (wait exec_bram_phase1_ready)
    │      │
    │      ▼
    │ READ_INPUT_POINTERS
    │  (collect HBM reads for inputs)
    │  rx_addr: 0 → {INPUT_ADDR_LIMIT, INPUT_ADDR_MOD}
    │      │
    │      ▼
    │ WAIT_URAM_PIPELINE
    │  (wait exec_uram_phase1_ready)
    │      │
    │      ▼
    │ READ_OUTPUT_POINTERS
    │  (collect HBM reads for outputs)
    │  rx_addr: 0 → {OUTPUT_ADDR_LIMIT, OUTPUT_ADDR_MOD}
    │      │
    │      ▼
    │ PHASE1_DONE
    │      │
    │      ▼
    │ READ_SYNAPSE_DATA
    │  (collect Phase 2 reads)
    │  wait: rx_ptr_ctr == tx_ptr_ctr
    │      │
    │      ▼
    │ PHASE2_DONE
    │      │
    └──────┘
```

### Pointer Chain Management

**Pointer FIFO Data Structure:**

```
ptrFIFO_dout[31:0]:
  [31:23] = Length (9 bits) → max 511 synapses in chain
  [22:0]  = Start address (23 bits) → HBM address >> 5
```

**Burst Calculation:**

```verilog
// Determine burst length for AXI transaction
ptr_burst = (ptr_ctr[8:4] == ptr_len[8:4]) ?
            ptr_len[3:0] :  // Last burst (partial)
            4'hF;           // Full burst (16 beats)

// Example:
// ptr_len = 9'd35 (36 synapses)
// Burst 1: ptr_ctr=0,  burst=15 (16 synapses)
// Burst 2: ptr_ctr=16, burst=15 (16 synapses)
// Burst 3: ptr_ctr=32, burst=3  (4 synapses, total=36)
```

**Address Increment:**

```verilog
// After each burst completes:
ptr_addr <= ptr_addr + ptr_burst + 1'b1;
ptr_ctr  <= ptr_ctr + ptr_burst + 1'b1;

// When ptr_ctr[8:4] == ptr_len[8:4], chain complete
// Pop next pointer from ptrFIFO
```

### 256 → 512 Bit Converter

**Purpose:** HBM provides 256-bit data, but 16 neuron groups require 512 bits.

**Logic:**

```verilog
reg hbm_count;  // Toggles between 0 and 1
reg [255:0] hbm_rdata_lower;

always @(posedge clk) begin
    if (hbm_rvalid && hbm_rready) begin
        hbm_count <= ~hbm_count;
        if (~hbm_count)
            hbm_rdata_lower <= hbm_rdata;  // Latch 1st read
        // else: 2nd read available on hbm_rdata
    end
end

// Output combines latched and current data
assign exec_hbm_rdata = {hbm_rdata, hbm_rdata_lower};

// Assert rvalidready only on 2nd read
assign exec_hbm_rvalidready = hbm_rvalid & hbm_rready & hbm_count & ~hbmFIFO_full;

// For spike writes (need both reads)
assign exec_hbm_rvalidready_2x = hbm_rvalid & hbm_rready;
```

**Timeline:**

```
Cycle: 0      1      2      3      4      5
       │      │      │      │      │      │
rvalid ▔▔▔▔▔▔▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁│▔▔▔▔▔▔▁▁
       │      │      │      │      │      │
rdata  DATA_L│      │DATA_U│      │DATA_L'│
       │      │      │      │      │      │
hbm_   0      │1      │0      │1      │0     │
count  │      │      │      │      │      │
       │      │      │      │      │      │
rdata  XXXX   │DATA_L│DATA_L│DATA_U│DATA_U│
_lower │      │      │      │      │      │
       │      │      │      │      │      │
exec_  XXXX   │XXXX  │{U,L} │{U,L} │{U',L'}│
hbm_   │      │      │      │      │      │
rdata  │      │      │      │      │      │
       │      │      │      │      │      │
rvalid ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁
ready  │      │      │      │      │      │
```

### Wait Clock Counter (Phase 2 Timeout)

**Purpose:** Ensure all pointers transmitted from ptrFIFO before ending Phase 2.

**Implementation:**

```verilog
reg [7:0] wait_clks_cnt;
wire [7:0] wait_clks_limit = 8'd255;

always @(posedge clk) begin
   if ((tx_curr_state == TX_STATE_POP_POINTER_FIFO) &&
       rx_phase1_done && ptrFIFO_empty)
      wait_clks_cnt <= wait_clks_cnt + 1'b1;
   else
      wait_clks_cnt <= 8'd0;
end

// Transition to PHASE2_DONE when timeout reached
if (wait_clks_cnt == wait_clks_limit)
   tx_next_state <= TX_STATE_PHASE2_DONE;
```

**Rationale:**
- Round-robin pointer FIFO controller may take up to 16 cycles to send last pointer
- 255-cycle wait provides generous margin
- Prevents premature phase completion

---

## Memory Map

### HBM Address Space

**Total:** 8 GB (33-bit address)

**Layout:**

```
┌─────────────────────────────────────────────────────────┐
│  Address Range        │  Purpose                        │
├───────────────────────┼─────────────────────────────────┤
│  [32:28] (upper 5)    │  Padding (always 0)             │
├───────────────────────┼─────────────────────────────────┤
│  [27:5]  (23 bits)    │  Row address                     │
│                       │  - Phase 0: Structured layout    │
│                       │  - Phase 1: Pointer chain addr   │
├───────────────────────┼─────────────────────────────────┤
│  [4:0]   (lower 5)    │  Byte offset (always 0 for      │
│                       │  32-byte aligned accesses)      │
└─────────────────────────────────────────────────────────┘
```

**Phase 0 Address Structure (`[27:5]` = 23 bits):**

```
[27:20] = Reserved / Bank select (8 bits, unused)
[19]    = Input/Output select (tx_select)
          0 = External inputs (BRAM)
          1 = Internal neurons (URAM)
[18:9]  = Address within input/output space (tx_addr, 10 bits)
[8:5]   = Padding (4 bits, for pointer granularity)

Example:
  Input row 100:   {8'd0, 1'b0, 10'd100, 4'd0} → Addr 0x006400
  Output row 500:  {8'd0, 1'b1, 10'd500, 4'd0} →0x087D00
```

**Pointer Data Structure (256-bit HBM row):**

```
Per pointer (32 bits × 8 pointers = 256 bits):
  [31:23] = Next pointer length (9 bits)
  [22:0]  = Next pointer address (23 bits)

Row contains 8 pointers, indexed by [8:5] of address
```

**Synapse Data Structure (256-bit HBM row):**

Depends on network configuration, but typically:
```
Per synapse (variable size, often 16-32 bits):
  - Weight (signed, 8-16 bits)
  - Target neuron ID (13-17 bits)
  - Delay (optional)
  - Other metadata
```

During Phase 2, pointer chains lead to synapse data rows.

---

## Timing Diagrams

### Phase 1a: Input Pointer Reads

```
Cycle:   0      1      2      3      4      5      ...
         │      │      │      │      │      │      │
TX State IDLE   │SEND_ │SEND_ │SEND_ │SEND_ │SEND_ │
         │      │INPUT │INPUT │INPUT │INPUT │INPUT │
         │      │      │      │      │      │      │
hbm_     ▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│
arvalid  │      │      │      │      │      │      │
         │      │      │      │      │      │      │
hbm_     ▔▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│
arready  │      │      │      │      │      │      │
         │      │      │      │      │      │      │
tx_addr  0      │1      │2      │3      │4      │5      │
         │      │      │      │      │      │      │
hbm_     0x0000 │0x0100 │0x0200 │0x0300 │0x0400 │0x0500 │
araddr   │      │      │      │      │      │      │
(low23)  │      │      │      │      │      │      │
         │      │      │      │      │      │      │
hbm_     15     │15     │15     │15     │15     │15     │
arlen    │(16   │(16   │(16   │(16   │(16   │(16   │
(burst-1)│beats)│beats)│beats)│beats)│beats)│beats)│
```

**Notes:**
- Each araddr issues a burst of 16 beats (4'hF + 1)
- tx_addr increments on each arready handshake
- Continues until tx_addr == INPUT_ADDR_LIMIT

### Phase 1b → Phase 2 Transition

```
Cycle:   N      N+1    N+2    N+3    N+4    N+5
         │      │      │      │      │      │
TX State SEND   │PHASE1│POP   │POP   │SEND  │
         OUTPUT │_DONE │_PTR  │_PTR  │_PTR  │
         │      │      │      │      │      │
tx_phase 0      │0      │1      │1      │1     │
         │      │      │      │      │      │
ptrFIFO  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁
_empty   │      │      │      │      │      │
         │      │      │      │      │      │
ptrFIFO  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁
_rden    │      │      │      │      │      │
         │      │      │      │      │      │
ptr_addr XXXX   │XXXX  │XXXX  │ADDR1 │ADDR1 │
         │      │      │      │(set) │      │
         │      │      │      │      │      │
hbm_     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│
arvalid  │      │      │      │      │      │
```

**Notes:**
- PHASE1_DONE toggles tx_phase to 1
- POP_PTR waits for ptrFIFO not empty
- ptr_addr loaded from ptrFIFO_dout
- SEND_PTR issues AXI read with ptr_burst length

### Phase 2: Pointer Chain Traversal

```
Cycle:   0      1      2      ...    16     17     18
         │      │      │      │      │      │      │
TX State SEND   │SEND  │SEND  │SEND  │POP   │SEND  │
         _PTR   │_PTR  │_PTR  │_PTR  │_PTR  │_PTR  │
         │      │      │      │      │      │      │
hbm_     ▔▔▔▔▔▔▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁
arvalid  │      │      │      │      │      │      │
         │      │      │      │      │      │      │
ptr_ctr  0      │16     │32     │48     │48     │48    │
         │      │      │      │      │      │      │
ptr_len  35     │35     │35     │35     │35     │100   │
         │      │      │      │      │      │(new) │
         │      │      │      │      │      │      │
ptr_     15     │15     │3      │15     │15     │15    │
burst    │      │      │(final)│(new   │      │      │
         │      │      │       │chain) │      │      │
         │      │      │      │      │      │      │
ptrFIFO  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁
_rden    │      │      │      │      │      │      │
```

**Notes:**
- First chain: 36 synapses (len=35) → 3 bursts
- Burst 1: 16 beats, Burst 2: 16 beats, Burst 3: 4 beats
- After completion, pop next pointer and start new chain
- Continues until ptrFIFO empty for 255 cycles

### RX: Data Collection with 256→512 Conversion

```
Cycle:   0      1      2      3      4      5      6
         │      │      │      │      │      │      │
hbm_     ▔▔▔▔▔▔▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁│▔▔▔▔▔▔▁▁
rvalid   │      │      │      │      │      │      │
         │      │      │      │      │      │      │
hbm_     DATA0L│      │DATA0U│      │DATA1L│      │
rdata    │      │      │      │      │      │      │
         │      │      │      │      │      │      │
hbm_     0      │1      │0      │1      │0      │1     │
count    │      │      │      │      │      │      │
         │      │      │      │      │      │      │
rdata_   XXXX   │DATA0L│DATA0L│DATA0U│DATA0U│DATA1L│
lower    │      │      │      │      │      │      │
         │      │      │      │      │      │      │
exec_hbm XXXX   │XXXX  │{0U,0L│{0U,0L│{1L,0U│{1L,0U│
_rdata   │      │      │}     │}     │}     │}     │
         │      │      │      │      │      │      │
exec_hbm ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁│▔▔▔▔▔▔▁▁
_rvalid  │      │      │      │      │      │      │
ready    │      │      │(2nd) │      │(2nd) │      │
```

**Notes:**
- Only every 2nd read triggers exec_hbm_rvalidready
- Data latched on odd reads, combined on even reads
- Downstream modules receive 512-bit packets

---

## Cross-References

### Related Modules

| Module | Relationship | Interface |
|--------|--------------|-----------|
| **command_interpreter.v** | Bidirectional | `ci2hbm_*` and `hbm2ci_*` FIFOs for CI access |
| **external_events_processor.v** | Coordination | `exec_bram_phase1_ready` signal |
| **internal_events_processor.v** | Downstream | `exec_hbm_rdata`, `exec_hbm_rvalidready` |
| **pointer_fifo_controller.v** | Upstream | Provides `ptrFIFO_*` interface |
| **spike_fifo_controller.v** | Downstream | `spk0-7_*` spike outputs |
| **HBM (Xilinx IP)** | Memory | AXI4 master interface |

### Software Integration

**Python (hs_bridge):**
- `compile_network.compile()` → Generates HBM memory layout
- `network.load_weights()` → Writes synapse data via CI to HBM
- `network.read_hbm(address)` → Debug HBM contents
- `utils.create_pointer_chains()` → Organizes synapses into linked lists

---

## Key Terms and Definitions

| Term | Definition |
|------|------------|
| **HBM** | High Bandwidth Memory - Off-chip DRAM with 400+ GB/s bandwidth |
| **Pointer Chain** | Linked-list structure storing variable-length synapse lists |
| **Phase 0 / Phase 1** | TX phases: 0=fetch pointers, 1=fetch synapses |
| **tx_phase** | Toggles between Phase 0 and Phase 1 |
| **tx_select** | In Phase 0: 0=inputs (BRAM), 1=outputs (URAM) |
| **ptr_addr** | HBM address for synapse data (from ptrFIFO) |
| **ptr_len** | Number of synapses in chain (from ptrFIFO) |
| **ptr_burst** | AXI burst length for current read (max 16 beats) |
| **hbm_count** | Toggles 0/1 to combine two 256-bit reads into 512-bit output |
| **exec_hbm_rvalidready** | Data valid signal (asserts every 2nd HBM read) |
| **exec_hbm_rvalidready_2x** | Data valid at HBM rate (every read) |
| **AXI4** | ARM Advanced eXtensible Interface - High-performance protocol |
| **Burst** | Multi-beat AXI transaction (up to 16 beats) |
| **INCR** | Incrementing burst type (addresses increment by size) |

---

## Performance Characteristics

### Throughput

**HBM Bandwidth:**
- **Interface:** 256-bit @ 225 MHz = 57.6 Gb/s = 7.2 GB/s per channel
- **System Total:** 32 channels (HBM2) × 7.2 GB/s = 230 GB/s theoretical

**Pointer Fetch Rate (Phase 1):**
- **Burst size:** 16 beats × 256 bits = 4096 bits = 512 bytes
- **Pointers per burst:** 512 bytes / 32 bytes = 16 pointers
- **Rate:** 16 pointers per ~20 cycles (burst + overhead) = ~180M pointers/sec

**Synapse Fetch Rate (Phase 2):**
- **Variable:** Depends on ptr_len (chain length)
- **Typical:** 1-10 synapses per neuron
- **Rate:** Limited by network connectivity, not HBM bandwidth

### Latency

| Operation | Cycles | Time @ 225 MHz |
|-----------|--------|----------------|
| AXI Address Handshake | 1 | 4.4 ns |
| HBM Read Latency | ~100-200 | 0.4-0.9 µs |
| 16-beat Burst Transfer | 16 | 71 ns |
| Total per burst | ~120-220 | 0.5-1.0 µs |

**Phase 1 Duration:**
- Depends on `num_inputs` and `num_outputs`
- Typical: 1000-10,000 cycles = 4-44 µs

**Phase 2 Duration:**
- Depends on total synapses across all active neurons
- Typical: 10,000-1,000,000 cycles = 44 µs - 4.4 ms

---

## Common Issues and Debugging

### Problem: Stuck in POP_POINTER_FIFO State

**Symptoms:** TX never reaches PHASE2_DONE

**Debug Steps:**
1. Check `ptrFIFO_empty` - should eventually assert
2. Check `wait_clks_cnt` - should increment when empty
3. Verify `pointer_fifo_controller` is writing to ptrFIFO

**Common Cause:** Pointer FIFO controller not generating pointers (upstream issue)

### Problem: exec_hbm_rvalidready Never Asserts

**Symptoms:** IEP/EEP waiting indefinitely for HBM data

**Debug Steps:**
1. Check `hbm_rvalid` - should pulse from HBM
2. Check `hbm_count` - should toggle 0→1→0
3. Check `hbmFIFO_full` - may be blocking output
4. Verify RX state machine in correct state

**Common Cause:** HBM not responding, or FIFO backpressure

### Problem: Spike FIFOs Not Receiving Data

**Symptoms:** No spikes generated during Phase 1

**Debug Steps:**
1. Check `exec_hbm_rx_phase1_done` - should assert during Phase 1 reads
2. Check `hbm_rdata[31, 63, 95, ...]` - spike flags should be set
3. Check `spkN_full` - may be blocking writes
4. Verify pointer data contains spike information

**Common Cause:** HBM pointer data doesn't include spike flags

### VIO/ILA Probes (Recommended)

```verilog
(*mark_debug = "true"*) reg [3:0] tx_curr_state;
(*mark_debug = "true"*) reg [3:0] rx_curr_state;
(*mark_debug = "true"*) wire exec_hbm_rvalidready;
(*mark_debug = "true"*) wire [22:0] ptr_addr;
(*mark_debug = "true"*) wire [8:0] ptr_len;
(*mark_debug = "true"*) wire [22:0] rx_ptr_ctr;
(*mark_debug = "true"*) wire [22:0] tx_ptr_ctr;
(*mark_debug = "true"*) wire ptrFIFO_empty;
(*mark_debug = "true"*) wire hbm_rvalid;
(*mark_debug = "true"*) wire hbm_count;
```

---

## Safety and Edge Cases

### Reset Behavior

On `resetn` deassertion:
- All state machines → RESET → IDLE
- Phase flags → done (ready for exec_run)
- Counters → 0
- Address registers → 0

### Burst Length Edge Cases

**Last Burst in Chain:**
- `ptr_burst` calculated as `ptr_len[3:0]` when on final segment
- Ensures exact number of synapses read, no over-fetch

**Empty Input/Output:**
- If `num_inputs=0` or `num_outputs=0`, respective phase skipped
- Address limit check immediately true

### AXI Protocol Compliance

**Write Transactions:**
- Single-beat only (awlen=0, wlast=1)
- No burst writes implemented

**Read Transactions:**
- Supports bursts up to 16 beats
- No support for wrap or fixed-address bursts (only INCR)

---

## Future Enhancement Opportunities

1. **Prefetching:** Begin Phase 2 pointer fetches before Phase 1 completes

2. **Burst Optimization:** Merge adjacent pointer chains into single burst

3. **Multi-Channel HBM:** Distribute addresses across HBM channels for parallelism

4. **Error Detection:** Monitor `hbm_rresp` and `hbm_bresp` for errors

5. **Performance Counters:** Track HBM utilization, stall cycles

6. **Adaptive Timeout:** Adjust wait_clks_limit based on ptrFIFO depth

7. **Write Bursts:** Support multi-beat writes for faster HBM initialization

---

**Document Version:** 1.0
**Last Updated:** December 2025
**Module File:** `hbm_processor.v`
**Module Location:** `CRI_proj/cri_fpga/code/new/hyddenn2/vivado/single_core.srcs/sources_1/new/`
**Purpose:** HBM memory controller and synapse data manager
**HBM Bandwidth:** 400+ GB/s (theoretical)
**AXI4 Interface:** 256-bit data width, 33-bit address
**Clock Frequency:** 225 MHz
