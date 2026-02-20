# Pointer FIFO Controller Module

## Overview

The **Pointer FIFO Controller** is a critical datapath component that manages the flow of synaptic pointer data during the two-phase neuromorphic execution cycle. It demultiplexes 512-bit HBM pointer data into 16 parallel FIFOs (one per neuron group), then arbitrates between these FIFOs to feed the HBM processor during Phase 2 (synaptic weight fetch).

### Role in the Software/Hardware Stack

```
                    Phase 1: External/Internal Events
                           (Spike Detection)
                                  |
    ┌─────────────────────────────┼─────────────────────────────┐
    |                             v                             |
    |              [External Events Processor]                  |
    |                      |                                    |
    |              exec_bram_spiked[15:0]                       |
    |                      |                                    |
    |              [Internal Events Processor]                  |
    |                      |                                    |
    |              exec_uram_spiked[15:0]                       |
    |                      |                                    |
    |                      v                                    |
    |         ┌────────────────────────────┐                    |
    |         │ Pointer FIFO Controller    │                    |
    |         │                            │                    |
    | HBM ───>│ 512b → 16×32b demux       │                    |
    |  Data   │ 16 Pointer FIFOs (ptr0-15)│                    |
    |         │ Round-robin arbiter        │                    |
    |         └────────────┬───────────────┘                    |
    |                      |                                    |
    |                ptrFIFO (32b)                              |
    |                      |                                    |
    |                      v                                    |
    |              [HBM Processor]                              |
    |                      |                                    |
    |              Synaptic Weights                             |
    |                      |                                    |
    |                      v                                    |
    |              [Spike FIFOs] ──> Phase 2 Synaptic Updates   |
    └───────────────────────────────────────────────────────────┘
```

**Function**:
- **Demultiplex HBM Pointer Data**: Split 512-bit HBM read into 16×32-bit pointer records
- **Spike-Gated Buffering**: Only store pointers for neurons that spiked (sparse event handling)
- **Fair Arbitration**: Round-robin scheduler ensures all neuron groups get equal service
- **Phase Coordination**: Handle both external (BRAM) and internal (URAM) spike events

**Key Innovation**: By buffering pointers in 16 parallel FIFOs, the module decouples HBM read bandwidth from pointer processing, allowing efficient handling of sparse neural activity.

---

## Module Architecture

```
                           HBM Data Path (512 bits)
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │  exec_hbm_rdata[511:0]       │
                    │  exec_hbm_rvalidready        │
                    └──────────────┬───────────────┘
                                   |
         ┌─────────────────────────┼─────────────────────────┐
         |                         v                         |
         |              Demux to 16 Groups                   |
         |    [31:0]  [63:32]  [95:64] ... [511:480]        |
         |       |       |        |            |             |
         |       v       v        v            v             |
         |   ┌─────┐ ┌─────┐  ┌─────┐      ┌─────┐         |
         |   │FIFO0│ │FIFO1│  │FIFO2│ ...  │FIFO15│         |
         |   │32b  │ │32b  │  │32b  │      │32b   │         |
         |   │FWFT │ │FWFT │  │FWFT │      │FWFT  │         |
         |   └──┬──┘ └──┬──┘  └──┬──┘      └──┬───┘         |
         |      ^        ^        ^             ^            |
         |      |        |        |             |            |
         |   wren0    wren1    wren2         wren15         |
         |      |        |        |             |            |
         |      └────────┴────────┴─────────────┘            |
         |                      |                            |
         |         Spike-Gated Write Enable Logic            |
         |    (bram_spiked[i] | uram_spiked[i]) & !full     |
         |                      ^                            |
         |      ┌───────────────┴───────────────┐            |
         |      |                               |            |
         |  exec_bram_spiked[15:0]   exec_uram_spiked[15:0] |
         |      |                               |            |
         |  ┌───┴────┐                  ┌───────┴──────┐    |
         |  │External│                  │  Internal    │    |
         |  │Events  │                  │  Events      │    |
         |  │Proc.   │                  │  Proc.       │    |
         |  └────────┘                  └──────────────┘    |
         └───────────────────────────────────────────────────┘

                      Round-Robin Arbiter
                             |
                   addr[3:0] counter (0→15)
                             |
         ┌───────────────────┼───────────────────┐
         |                   v                   |
         |        16:1 Multiplexer               |
         |    (Select ptr_dout[addr])            |
         |                   |                   |
         |                   v                   |
         |            ┌─────────────┐            |
         |            │  ptrFIFO    │            |
         |            │  (32-bit)   │            |
         |            │  To HBM Proc│            |
         |            └─────────────┘            |
         └───────────────────────────────────────┘
```

### Two-Phase Operation

**Phase 1a: External Events (BRAM Reading)**
```
1. bram_reading = 1 (set on exec_run)
2. For each HBM read (exec_hbm_rvalidready):
   - Split 512b data into ptr0_din...ptr15_din
   - Write to FIFO[i] if exec_bram_spiked[i]==1 and !ptr_full[i]
3. Continue until exec_bram_phase1_done
4. Transition to Phase 1b
```

**Phase 1b: Internal Events (URAM Reading)**
```
1. uram_reading = 1 (set on exec_bram_phase1_done)
2. For each HBM read (exec_hbm_rvalidready):
   - Split 512b data into ptr0_din...ptr15_din
   - Write to FIFO[i] if exec_uram_spiked[i]==1 and !ptr_full[i]
3. Continue until exec_uram_phase1_done
4. End of Phase 1
```

**Phase 2: Pointer Drain (Concurrent with Phase 1)**
```
1. Round-robin arbiter cycles addr 0→1→2→...→15→0
2. Every cycle:
   - If ptr[addr]_empty==0 and ptrFIFO_full==0:
     * Read from ptr[addr] (rden=1)
     * Write to ptrFIFO (wren=1, din=ptr_dout)
3. HBM processor consumes pointers, fetches synapses
4. Continues until all FIFOs empty
```

---

## Interface Specification

### Clock and Reset
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | Input | 1 | System clock (225 MHz typical) |
| `resetn` | Input | 1 | Active-low asynchronous reset |

### Execution Control
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `exec_run` | Input | 1 | Start new time step (sets bram_reading=1) |

### External Events Processor Interface
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `exec_bram_spiked` | Input | 16 | Spike mask from external events (16 neuron groups) |
| `exec_bram_phase1_done` | Input | 1 | External events complete, transition to internal |

### Internal Events Processor Interface
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `exec_uram_spiked` | Input | 16 | Spike mask from internal events (16 neuron groups) |
| `exec_uram_phase1_done` | Input | 1 | Internal events complete, end Phase 1 |

### HBM Processor Input Interface
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `exec_hbm_rvalidready` | Input | 1 | HBM read data valid and ready |
| `exec_hbm_rdata` | Input | 512 | HBM read data (16 pointers × 32 bits) |
| `hbm2pfc_rden` | Output | 1 | FIFO read enable (FWFT mode) |

**Note**: Comments indicate `hbm2pfc_dout` and `hbm2pfc_empty` are wired at top wrapper level.

### Pointer FIFO Interfaces (16 instances: ptr0-ptr15)

Each pointer FIFO has identical interface (example for ptr0):

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `ptr0_full` | Input | 1 | FIFO full flag (backpressure) |
| `ptr0_din` | Output | 32 | Data input to FIFO (pointer record) |
| `ptr0_wren` | Output | 1 | Write enable (gated by spike and full) |
| `ptr0_empty` | Input | 1 | FIFO empty flag |
| `ptr0_dout` | Input | 32 | Data output from FIFO |
| `ptr0_rden` | Output | 1 | Read enable (from arbiter) |

**Pointer FIFOs**: ptr1, ptr2, ..., ptr15 (identical interfaces)

### HBM Processor Output Interface (Aggregated Pointer FIFO)
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `ptrFIFO_full` | Input | 1 | Aggregated FIFO full flag |
| `ptrFIFO_din` | Output | 32 | Pointer data to HBM processor |
| `ptrFIFO_wren` | Output | 1 | Write enable (from arbiter) |

---

## Detailed Logic Description

### Phase Tracking State Machine

The module uses two registers to track execution phase:

```verilog
reg bram_reading;  // Phase 1a: External events
reg uram_reading;  // Phase 1b: Internal events

always @(posedge clk) begin
    if (!resetn) begin
        bram_reading <= 1'b0;
        uram_reading <= 1'b0;
    end else if (exec_run) begin
        // Start of new time step: begin external event processing
        bram_reading <= 1'b1;
    end else if (exec_bram_phase1_done & !uram_reading) begin
        // Transition from external to internal event processing
        bram_reading <= 1'b0;
        uram_reading <= 1'b1;
    end else if (exec_uram_phase1_done) begin
        // End of Phase 1
        uram_reading <= 1'b0;
    end
end
```

**State Transitions:**
```
IDLE (both=0)
    |
    | exec_run
    v
BRAM_READING (bram=1, uram=0)
    |
    | exec_bram_phase1_done
    v
URAM_READING (bram=0, uram=1)
    |
    | exec_uram_phase1_done
    v
IDLE (both=0)
```

**Note**: During idle, the round-robin arbiter continues draining pointer FIFOs (Phase 2).

### HBM Data Demultiplexing

The 512-bit HBM data is split into 16 groups of 32 bits:

```verilog
// Direct bit-slice assignments
assign ptr0_din  = exec_hbm_rdata[031:000];  // Bits 0-31
assign ptr1_din  = exec_hbm_rdata[063:032];  // Bits 32-63
assign ptr2_din  = exec_hbm_rdata[095:064];  // Bits 64-95
assign ptr3_din  = exec_hbm_rdata[127:096];  // Bits 96-127
// ... (pattern continues)
assign ptr15_din = exec_hbm_rdata[511:480];  // Bits 480-511
```

**Data Layout** (each 32-bit pointer):
```
Bits [31:23] = Length (9 bits, max 511 synapses)
Bits [22:0]  = Start address in HBM (23 bits, byte address)
```

**Example**:
```
exec_hbm_rdata = 512'h...AB12_3456_CD78_9ABC_...

ptr0_din  = 32'hCD78_9ABC  → Length=0x1AF, Addr=0x389ABC
ptr1_din  = 32'hAB12_3456  → Length=0x156, Addr=0x523456
...
```

### Spike-Gated Write Enable Logic

Each pointer FIFO write is conditional on:
1. HBM data valid and ready
2. Corresponding spike bit asserted
3. FIFO not full

```verilog
assign ptr0_wren = !ptr0_full & exec_hbm_rvalidready &
                   ((bram_reading & exec_bram_spiked[0]) |
                    (uram_reading & exec_uram_spiked[0]));

assign ptr1_wren = !ptr1_full & exec_hbm_rvalidready &
                   ((bram_reading & exec_bram_spiked[1]) |
                    (uram_reading & exec_uram_spiked[1]));

// ... (pattern repeats for ptr2-ptr15)
```

**Logic Breakdown**:
```
ptr_wren[i] = !ptr_full[i]           // FIFO has space
            & exec_hbm_rvalidready   // HBM data available
            & (
                (bram_reading & exec_bram_spiked[i])  // External spike
                |
                (uram_reading & exec_uram_spiked[i])  // Internal spike
              )
```

**Example Scenarios**:

**Scenario 1: External spike on neuron group 5**
```
Cycle N:
  bram_reading = 1
  exec_bram_spiked = 16'b0000_0000_0010_0000  (bit 5 set)
  exec_hbm_rvalidready = 1
  ptr5_full = 0

Result:
  ptr5_wren = 1  → Write exec_hbm_rdata[191:160] to ptr5 FIFO
  ptr0-4,6-15_wren = 0  → No write to other FIFOs
```

**Scenario 2: Multiple spikes (groups 0, 3, 7)**
```
Cycle N:
  uram_reading = 1
  exec_uram_spiked = 16'b0000_0000_1000_1001  (bits 0,3,7 set)
  exec_hbm_rvalidready = 1
  ptr0_full = 0, ptr3_full = 0, ptr7_full = 1  (ptr7 full!)

Result:
  ptr0_wren = 1  → Write to ptr0
  ptr3_wren = 1  → Write to ptr3
  ptr7_wren = 0  → Blocked by full (data lost!)
  Others = 0
```

**Backpressure Handling**: If any FIFO is full when its spike arrives, that pointer is **lost**. System must ensure FIFOs drain fast enough.

### Round-Robin Arbiter

A 4-bit counter cycles through FIFOs 0-15, servicing one per cycle:

```verilog
reg [3:0] addr;  // 4 bits for 16 FIFOs (0-15)

always @(posedge clk) begin
    if (~resetn)
        addr <= 4'd0;
    else
        addr <= addr + 1'b1;  // Wraps 15→0 automatically
end
```

**Arbitration Cycle**:
```
Cycle 0:  addr=0  → Check ptr0
Cycle 1:  addr=1  → Check ptr1
Cycle 2:  addr=2  → Check ptr2
...
Cycle 15: addr=15 → Check ptr15
Cycle 16: addr=0  → Back to ptr0
...
```

**Arbitration Logic** (combinational):

```verilog
always @(*) begin
    // Default: No reads, no writes
    ptr0_rden = 1'b0;
    ptr1_rden = 1'b0;
    // ... (all ptr*_rden = 0)
    ptrFIFO_din = 32'dX;
    ptrFIFO_wren = 1'b0;

    case (addr)
        4'd0: begin
            if (~ptr0_empty & ~ptrFIFO_full) begin
                ptr0_rden    = 1'b1;
                ptrFIFO_din  = ptr0_dout;
                ptrFIFO_wren = 1'b1;
            end
        end
        4'd1: begin
            if (~ptr1_empty & ~ptrFIFO_full) begin
                ptr1_rden    = 1'b1;
                ptrFIFO_din  = ptr1_dout;
                ptrFIFO_wren = 1'b1;
            end
        end
        // ... (pattern repeats for 4'd2 through 4'd15)

        default: begin
            // All outputs stay at default (0 or X)
        end
    endcase
end
```

**Arbitration Example**:

```
Cycle  | addr | ptr0_empty | ptr1_empty | ptr5_empty | ptrFIFO_full | Action
-------|------|------------|------------|------------|--------------|------------------
   0   |  0   |      0     |      0     |      0     |      0       | Read ptr0
   1   |  1   |      0     |      0     |      0     |      0       | Read ptr1
   2   |  2   |      1     |      0     |      0     |      0       | Skip (empty)
   3   |  3   |      1     |      1     |      0     |      0       | Skip (empty)
   4   |  4   |      1     |      1     |      0     |      0       | Skip (empty)
   5   |  5   |      1     |      1     |      0     |      0       | Read ptr5
   6   |  6   |      1     |      1     |      1     |      0       | Skip (empty)
   7   |  7   |      1     |      1     |      1     |      0       | Skip (empty)
  ...  | ...  |     ...    |     ...    |     ...    |     ...      | ...
  15   | 15   |      1     |      1     |      1     |      0       | Skip (empty)
  16   |  0   |      0     |      1     |      1     |      0       | Read ptr0 again
```

**Fairness**: Each FIFO gets equal opportunity (once per 16 cycles), regardless of occupancy.

**Starvation**: If a FIFO is always full, other FIFOs continue to be serviced. No single FIFO can block others.

### FWFT (First-Word Fall-Through) Mode

The FIFOs operate in FWFT mode, meaning data appears on `dout` immediately when `empty` deasserts:

```
Traditional FIFO:
  Cycle N:   rden=1  (issue read)
  Cycle N+1: dout valid  (1 cycle latency)

FWFT FIFO:
  Cycle N:   empty=0, dout already valid
  Cycle N:   rden=1  (consume word, advance to next)
  Cycle N+1: dout shows next word (if available)
```

**Why FWFT?**: Reduces latency - arbiter can read and forward pointer in single cycle.

**HBM FIFO Read Enable**:

```verilog
assign hbm2pfc_rden = exec_hbm_rvalidready;
```

Every time HBM data is consumed (`exec_hbm_rvalidready=1`), the FIFO is advanced to present next 512-bit word. This assumes FWFT mode on the HBM data FIFO.

---

## Timing Diagrams

### Phase Transition: BRAM → URAM

```
Cycle    0    1    2    3    4    5    6    7    8    9
         ────┬────┬────┬────┬────┬────┬────┬────┬────┬────
exec_run ───┐    ┌─────────────────────────────────────────
         ───└────┘

bram_reading ────┐                        ┌────────────────
         ────────└────────────────────────┘

uram_reading ─────────────────────────┐              ┌─────
         ─────────────────────────────└──────────────┘

exec_bram_phase1_done ────────────┐    ┌─────────────────
                      ────────────└────┘

exec_uram_phase1_done ─────────────────────────┐    ┌─────
                      ─────────────────────────└────┘

Phase        IDLE   BRAM  BRAM  BRAM  BRAM URAM URAM URAM IDLE
```

### Pointer FIFO Write (Spike-Gated)

```
Cycle        0    1    2    3    4    5    6
             ────┬────┬────┬────┬────┬────┬────
bram_reading ───────────────────────────────────
             ───┐
                └───────────────────────────────

exec_hbm_rvalidready ──┐    ┌───┐    ┌───┐    ┌
                   ────└────┘   └────┘   └────┘

exec_bram_spiked    0x0005   0x0003   0x0000
                    (bits 0,2)(bits 0,1) (none)

ptr0_full       ───────────────────────────────  (always room)

ptr0_wren       ───┐         ┌───┐
                ───└─────────┘   └─────────────  (spike bit 0)

ptr1_wren       ────────────────┐
                ────────────────└───────────────  (spike bit 1)

ptr2_wren       ───┐
                ───└───────────────────────────  (spike bit 2)

ptr0_din            P0        P0'
                    ↓         ↓
ptr0 FIFO       [empty] → [P0] → [P0,P0']

Explanation:
  Cycle 1: exec_bram_spiked=0x0005 (bits 0 and 2)
           → ptr0_wren=1, ptr2_wren=1
           → Write to ptr0 and ptr2 FIFOs

  Cycle 3: exec_bram_spiked=0x0003 (bits 0 and 1)
           → ptr0_wren=1, ptr1_wren=1
           → Write to ptr0 (again) and ptr1 FIFOs

  Cycle 5: exec_bram_spiked=0x0000 (no spikes)
           → All ptr*_wren=0
           → No writes (HBM data ignored)
```

### Round-Robin Arbiter Operation

```
Cycle    0    1    2    3    4    5    6    7    8
         ────┬────┬────┬────┬────┬────┬────┬────┬────
addr         0    1    2    3    4    5    6    7    8

ptr0_empty   ────┐                             ┌─────
             ────└─────────────────────────────┘
             (has data cycles 0-7, empty at 8)

ptr1_empty   ───────────────────────────────────────
             (empty throughout)

ptr2_empty   ──────────┐                   ┌────────
             ──────────└───────────────────┘
             (has data cycles 2-6)

ptrFIFO_full ───────────────────────────────────────
             (never full)

ptr0_rden    ───┐                             ┌─────
             ───└─────────────────────────────┘

ptr2_rden    ──────────┐
             ──────────└───────────────────────────

ptrFIFO_wren ───┐       ┌─────────────────────┐
             ───└───────┘                     └─────

ptrFIFO_din      D0      D2                    X

Explanation:
  Cycle 0 (addr=0): ptr0 not empty → read ptr0, write ptrFIFO
  Cycle 1 (addr=1): ptr1 empty → skip
  Cycle 2 (addr=2): ptr2 not empty → read ptr2, write ptrFIFO
  Cycle 3-7: All empty → skip
  Cycle 8 (addr=8): Continue round-robin (wraps at 15)
```

### FIFO Full Backpressure

```
Cycle        0    1    2    3    4    5
             ────┬────┬────┬────┬────┬────
exec_hbm_rvalidready ┐    ┌───┐    ┌───┐
                 ────└────┘   └────┘   └

exec_bram_spiked  0x0001 0x0001 0x0001
                  (bit 0)(bit 0)(bit 0)

ptr0_full     ────────────┐         ┌────
              ────────────└─────────┘
              (becomes full at cycle 2)

ptr0_wren     ───┐    ┌───┐         ┌────
              ───└────┘   └─────────┘

ptr0_din          D0   D1   X     D2

ptr0 contents [D0] [D0,D1] [D0,D1] [D1,D2]

Explanation:
  Cycle 1: Write D0 to ptr0 (wren=1)
  Cycle 2: ptr0 becomes full
  Cycle 3: D1 written, but ptr0_full=1 → wren=0 → D1 LOST!
  Cycle 4: ptr0 not full again
  Cycle 5: D2 written (wren=1)

  Result: D1 was lost due to FIFO full condition!
```

**Prevention**: Ensure arbiter drains FIFOs faster than they fill, or increase FIFO depth.

---

## Memory and Resource Usage

### FIFO Depth Considerations

**Minimum FIFO Depth** (to avoid loss):

Assume:
- Max neurons per group: 8192 (131,072 / 16)
- Worst case: All neurons in one group spike
- Arbiter services each FIFO once per 16 cycles

**Fill Rate** (during bram_reading or uram_reading):
- 1 pointer per HBM read (exec_hbm_rvalidready)
- Max rate: 1 per cycle (if HBM always ready)

**Drain Rate**:
- 1 pointer per 16 cycles (round-robin)

**Net Accumulation**:
- Fill: +1 per cycle (worst case)
- Drain: +1 per 16 cycles
- Net: +15 pointers per 16 cycles

**Depth Calculation**:
```
Time to process 8192 neurons @ 225 MHz:
  8192 / 16 (axons per HBM read) = 512 HBM reads
  512 cycles @ 225 MHz = 2.27 µs

Pointers accumulated in one FIFO (worst case):
  All 8192 neurons in one group spike
  = 8192 / 16 = 512 pointers
  (Each HBM read provides 1 pointer for that group)

Pointers drained during 512 cycles:
  512 / 16 = 32 pointers

Net FIFO occupancy:
  512 - 32 = 480 pointers

Required FIFO depth: ~512 (power of 2 for FPGA FIFOs)
```

**Typical FIFO Configuration**:
- **Depth**: 512 or 1024 entries
- **Width**: 32 bits
- **Type**: Distributed RAM (for small depth) or Block RAM
- **Mode**: FWFT (First-Word Fall-Through)

### Resource Estimates

**Per Pointer FIFO** (16 instances):
- **Depth 512 × 32b** = 16 Kb = 0.89 BRAM18K (use 1 BRAM18K)
- **FWFT logic**: ~50 LUTs, ~30 FFs

**Total for 16 FIFOs**:
- **BRAM18K**: 16 (one per FIFO)
- **LUTs**: ~800 (FIFOs) + ~200 (arbiter) = ~1000
- **FFs**: ~500 (FIFOs) + ~50 (arbiter/control) = ~550

**Controller Logic**:
- **Demux**: 16 × 32-bit slices (wiring only, ~0 LUTs)
- **Write Enable**: 16 × (4-input AND + OR) = ~96 LUTs
- **Arbiter**: 16-way mux + control = ~150 LUTs
- **Phase Control**: ~20 LUTs, ~3 FFs

---

## Cross-References

### Upstream Modules

- **external_events_processor.v** (`external_events_processor.md`):
  - Provides `exec_bram_spiked[15:0]` (external spike mask)
  - Asserts `exec_bram_phase1_done` to signal phase transition

- **internal_events_processor.v** (`internal_events_processor.md`):
  - Provides `exec_uram_spiked[15:0]` (internal spike mask)
  - Asserts `exec_uram_phase1_done` to signal phase 1 complete

- **hbm_processor.v** (`hbm_processor.md`):
  - Provides `exec_hbm_rdata[511:0]` (pointer data from HBM)
  - Provides `exec_hbm_rvalidready` (data valid signal)
  - Receives `ptrFIFO_din`, `ptrFIFO_wren` (aggregated pointers for Phase 2)

### Downstream Modules

- **hbm_processor.v** (`hbm_processor.md`):
  - Consumes pointers from `ptrFIFO`
  - Uses pointers to fetch synaptic weights during Phase 2
  - Sends fetched synapses to spike FIFOs

### Peer Modules

- **spike_fifo_controller.v** (`spike_fifo_controller.md`):
  - Similar architecture (demux + arbiter)
  - Handles synaptic weight data instead of pointers
  - Works in Phase 2 alongside this module's pointer drain

---

## Common Issues and Debugging

### Issue 1: Pointers Lost (FIFO Overflow)

**Symptoms:**
- Neurons don't receive expected synaptic updates
- FIFO full flags assert frequently
- Spike counts don't match expected connectivity

**Root Cause:**
- Arbiter can't drain FIFOs fast enough
- FIFO depth too small for burst activity

**Debug:**
```verilog
// Add probes for FIFO occupancy
(* mark_debug = "true" *) wire [9:0] ptr0_count;  // Assuming 512-deep FIFO
(* mark_debug = "true" *) wire       ptr0_overflow;

// Monitor overflow events
always @(posedge clk) begin
    if (ptr0_full & ptr0_wren)
        ptr0_overflow <= 1'b1;  // Overflow detected!
end
```

**Solution:**
- Increase FIFO depth (512 → 1024 or 2048)
- Optimize arbiter (see Enhancement #1 below)
- Add priority arbitration for fuller FIFOs

### Issue 2: Unfair Arbitration (Starvation)

**Symptoms:**
- Some neuron groups process much slower than others
- Uneven latency across different spike patterns

**Root Cause:**
- Round-robin gives equal slots, but some FIFOs have more data
- FIFO[0] with 100 entries gets same service as FIFO[15] with 1 entry

**Debug:**
```verilog
// Track arbitration wins per FIFO
(* mark_debug = "true" *) reg [15:0] arb_wins [15:0];

always @(posedge clk) begin
    if (ptr0_rden) arb_wins[0] <= arb_wins[0] + 1;
    if (ptr1_rden) arb_wins[1] <= arb_wins[1] + 1;
    // ... (repeat for all FIFOs)
end
```

**Solution:**
- Implement weighted round-robin (award more slots to fuller FIFOs)
- Use priority encoder favoring non-empty FIFOs
- Skip empty FIFOs faster (see Enhancement #2)

### Issue 3: Phase Transition Glitch

**Symptoms:**
- Pointers written with wrong spike mask during phase boundary
- Corruption at transition from BRAM to URAM reading

**Root Cause:**
- Race condition between `exec_bram_phase1_done` and last HBM read
- Write enable uses old phase flags

**Debug:**
```verilog
// Monitor phase transition timing
(* mark_debug = "true" *) reg phase_transition;

always @(posedge clk) begin
    if (exec_bram_phase1_done & !uram_reading)
        phase_transition <= 1'b1;
    else
        phase_transition <= 1'b0;
end

// Check if any writes occur during transition
assert property (@(posedge clk)
    phase_transition |-> (|{ptr0_wren, ptr1_wren, ..., ptr15_wren} == 0)
);
```

**Solution:**
- Pipeline phase flags by one cycle
- Add guard time between phases (no writes for 1 cycle)
- Use registered versions of bram_reading/uram_reading for write enables

### Issue 4: HBM FIFO Not Advancing

**Symptoms:**
- Same HBM data appears multiple times
- Pointer FIFOs fill with duplicate entries

**Root Cause:**
- `hbm2pfc_rden` not properly connected or not asserting
- FWFT mode misconfigured on HBM FIFO

**Debug:**
```verilog
// Verify read enable toggles
(* mark_debug = "true" *) wire hbm2pfc_rden;
(* mark_debug = "true" *) wire exec_hbm_rvalidready;
(* mark_debug = "true" *) wire [511:0] exec_hbm_rdata;

// Check for stuck data
reg [511:0] prev_hbm_rdata;
always @(posedge clk) begin
    if (exec_hbm_rvalidready)
        prev_hbm_rdata <= exec_hbm_rdata;
end

// Assert: consecutive reads should have different data (usually)
// (unless network connectivity happens to repeat, rare)
```

**Solution:**
- Verify FWFT mode enabled on HBM FIFO IP
- Check that `hbm2pfc_rden` is wired to FIFO's read enable
- Confirm FIFO has data (not empty)

### Issue 5: Address Counter Wrapping Incorrectly

**Symptoms:**
- Some FIFOs never serviced
- Arbiter stuck on certain addresses

**Root Cause:**
- 4-bit counter not wrapping correctly (should wrap 15→0)
- Synthesis optimization error

**Debug:**
```verilog
// Monitor counter progression
(* mark_debug = "true" *) reg [3:0] addr;
(* mark_debug = "true" *) reg [3:0] prev_addr;

always @(posedge clk) begin
    prev_addr <= addr;
    // Check for proper increment (with wrap)
    assert ((addr == (prev_addr + 1'b1)) || (!resetn));
end
```

**Solution:**
- Explicitly handle wrap:
```verilog
always @(posedge clk) begin
    if (~resetn)
        addr <= 4'd0;
    else if (addr == 4'd15)
        addr <= 4'd0;  // Explicit wrap
    else
        addr <= addr + 1'b1;
end
```

---

## Performance Characteristics

### Throughput Analysis

**HBM Read Bandwidth**:
- **Peak**: 512 bits per cycle @ 225 MHz = 14.4 GB/s
- **Typical**: Limited by HBM latency and contention (~50% efficiency) = 7.2 GB/s
- **Pointers per Second**: (7.2 GB/s) / (32 bits) = 1.8 billion pointers/s

**Arbiter Throughput**:
- **Max**: 1 pointer per cycle @ 225 MHz = 225 million pointers/s
- **Typical** (50% FIFO occupancy): ~112 million pointers/s
- **Bottleneck**: Arbiter is **NOT** the bottleneck (HBM fill rate >> drain rate in Phase 1)

**Phase 1 Duration** (example: 131,072 neurons):
```
External Events:
  Input axons: 16,384 (assuming 16 per HBM read)
  HBM reads: 16,384 / 16 = 1,024 reads
  Time @ 225 MHz: 1,024 cycles = 4.55 µs

Internal Events:
  URAM neurons: 131,072
  URAM rows: 131,072 / 2 = 65,536 (2 neurons per row)
  URAM banks: 16
  Rows per bank: 65,536 / 16 = 4,096
  HBM reads per bank: 4,096 / 16 = 256 (if 16 neurons spike per read)
  Total HBM reads: ~16,384 (worst case, all banks active)
  Time @ 225 MHz: 16,384 cycles = 72.8 µs

Total Phase 1: ~77 µs
```

**Phase 2 Duration** (pointer drain):
```
Assume 10% neurons spike (13,107 neurons):
  Pointers to process: 13,107
  Arbiter rate: 1 per 16 cycles (round-robin overhead)
  Effective drain: 225 MHz / 16 = 14.06 million pointers/s

  Time: 13,107 pointers / 14.06M/s = 0.93 ms

But Phase 2 overlaps with next Phase 1!
  Phase 1 and 2 pipeline, so overall latency = max(Phase1, Phase2)
  Typical: Phase 2 >> Phase 1, so Phase 2 dominates
```

**Latency** (pointer from HBM to ptrFIFO):
- **Best Case** (FIFO empty, arbiter on correct address):
  - FWFT mode: 0 cycles (immediate)
  - Write to ptrFIFO: 1 cycle
  - **Total**: 1 cycle @ 225 MHz = 4.4 ns

- **Worst Case** (FIFO full, arbiter just passed):
  - Wait for FIFO space: N cycles (depends on drain rate)
  - Wait for arbiter: 15 cycles (worst case, just missed)
  - **Total**: ~16 cycles @ 225 MHz = 71 ns (ignoring FIFO drain time)

### Resource Utilization Summary

| Resource | Usage | Notes |
|----------|-------|-------|
| LUTs | ~1,200 | Demux, arbiter, control, FIFO logic |
| FFs | ~550 | Phase control, arbiter, FIFO pointers |
| BRAM18K | 16 | One per pointer FIFO (512×32b each) |
| DSPs | 0 | No arithmetic operations |

**Percentage of Typical FPGA** (e.g., Xilinx UltraScale+ VU9P):
- LUTs: 1,200 / 1,182,240 = 0.1%
- FFs: 550 / 2,364,480 = 0.02%
- BRAM18K: 16 / 2,160 = 0.74%

**Conclusion**: Very lightweight module, dominated by FIFO storage.

---

## Safety and Edge Cases

### Edge Case 1: All Neurons Spike Simultaneously

**Scenario**: Every neuron in every group spikes in same cycle.

**Behavior**:
```
exec_bram_spiked = 16'hFFFF  (all bits set)
All 16 pointer FIFOs receive write:
  ptr0_wren = 1, ptr1_wren = 1, ..., ptr15_wren = 1

Each FIFO receives 1 pointer per HBM read.
```

**Safety**:
- ✅ All writes occur in parallel (16 separate FIFOs)
- ✅ No conflicts (each FIFO independent)
- ⚠️ FIFO depth must handle burst (512+ pointers)
- ⚠️ Arbiter drain rate becomes critical (1 per 16 cycles)

**Result**: System handles correctly if FIFO depth adequate.

### Edge Case 2: No Neurons Spike (Quiescent Network)

**Scenario**: No spikes in entire time step.

**Behavior**:
```
exec_bram_spiked = 16'h0000  (all bits clear)
exec_uram_spiked = 16'h0000

All ptr*_wren = 0  (no writes to any FIFO)
HBM reads still occur, but data discarded.
```

**Safety**:
- ✅ No FIFO writes (correct behavior)
- ✅ Arbiter continues cycling (no-op, all FIFOs empty)
- ✅ Phase transitions occur normally
- ⚠️ HBM bandwidth wasted (reading data that's discarded)

**Optimization Opportunity**: Gate HBM reads based on spike mask (see Enhancements).

### Edge Case 3: Single Bit Spike (Minimal Activity)

**Scenario**: Only one neuron in one group spikes.

**Behavior**:
```
exec_bram_spiked = 16'h0001  (only bit 0 set)

Only ptr0_wren = 1  (one FIFO active)
Other 15 FIFOs idle.
```

**Safety**:
- ✅ Correct - only relevant FIFO updated
- ✅ Arbiter cycles through all, only reads from ptr0
- ✅ Minimal resource usage

**Result**: Efficient sparse event handling.

### Edge Case 4: ptrFIFO Full (Downstream Backpressure)

**Scenario**: HBM processor can't consume pointers fast enough.

**Behavior**:
```
ptrFIFO_full = 1

Arbiter logic:
  if (~ptr[addr]_empty & ~ptrFIFO_full)  → Condition false!
    ptr[addr]_rden = 0  (no read)
    ptrFIFO_wren = 0    (no write)
```

**Safety**:
- ✅ Arbiter stalls (doesn't read from any pointer FIFO)
- ✅ Upstream pointer FIFOs continue to fill
- ⚠️ If pointer FIFOs also fill, writes are lost (see Issue 1)

**Required**: System must ensure ptrFIFO drains faster than it fills.

### Safety Check: Write Enable Conflicts

**Assertion**: Verify only one arbiter read per cycle
```verilog
wire [15:0] rdens = {ptr15_rden, ptr14_rden, ..., ptr0_rden};

property one_hot_rdens;
    @(posedge clk) disable iff (~resetn)
    $onehot0(rdens);  // At most one bit set
endproperty
assert_rdens: assert property (one_hot_rdens);
```

### Safety Check: Phase Mutual Exclusion

**Assertion**: Ensure bram_reading and uram_reading never both asserted
```verilog
property phases_mutex;
    @(posedge clk) disable iff (~resetn)
    !(bram_reading & uram_reading);
endproperty
assert_phases: assert property (phases_mutex);
```

---

## Future Enhancement Opportunities

### 1. Priority Arbiter

Replace round-robin with priority-based arbitration:

```verilog
// Calculate occupancy for each FIFO (requires rd_data_count from FIFO IP)
wire [9:0] ptr0_count, ptr1_count, ..., ptr15_count;

// Find fullest FIFO (priority encoder)
reg [3:0] priority_addr;
always @(*) begin
    if      (ptr0_count > threshold) priority_addr = 4'd0;
    else if (ptr1_count > threshold) priority_addr = 4'd1;
    // ... (priority order 0→1→2→...→15)
    else priority_addr = addr;  // Fall back to round-robin
end

// Use priority_addr instead of addr in arbiter mux
```

**Benefit**: Prevents FIFO overflow by draining fuller FIFOs first.

### 2. Skip-Empty Optimization

Current arbiter wastes cycles checking empty FIFOs:

```verilog
// Add empty flag aggregation
wire [15:0] ptrs_empty = {ptr15_empty, ..., ptr0_empty};

// Fast-forward to next non-empty FIFO
reg [3:0] next_addr;
always @(*) begin
    next_addr = addr;
    for (int i = 1; i <= 16; i++) begin
        if (!ptrs_empty[(addr + i) & 4'hF]) begin
            next_addr = (addr + i) & 4'hF;
            break;
        end
    end
end

always @(posedge clk) begin
    if (~resetn)
        addr <= 4'd0;
    else
        addr <= next_addr;  // Jump to next non-empty
end
```

**Benefit**: Reduces latency by ~50% when many FIFOs empty.

### 3. Gated HBM Reads

Don't read HBM when no spikes:

```verilog
// Compute OR of spike mask
wire any_spikes = |(exec_bram_spiked | exec_uram_spiked);

// Gate HBM read enable
assign hbm2pfc_rden = exec_hbm_rvalidready & any_spikes;
```

**Benefit**: Saves HBM bandwidth during quiescent periods.

### 4. Configurable FIFO Count

Parameterize number of FIFOs:

```verilog
module pointer_fifo_controller #(
    parameter NUM_FIFOS = 16,
    parameter FIFO_DEPTH = 512
)(
    input [NUM_FIFOS-1:0] exec_bram_spiked,
    // ... (generate FIFO instances and arbiter)
);

// Use generate blocks for FIFO instantiation
genvar i;
generate
    for (i = 0; i < NUM_FIFOS; i++) begin : fifo_gen
        fifo_32x512 ptr_fifo (
            .din(exec_hbm_rdata[(i+1)*32-1 : i*32]),
            .wr_en(ptr_wren[i]),
            // ...
        );
    end
endgenerate
```

**Benefit**: Flexible configuration for different neuron group sizes.

### 5. Multi-Port Arbiter

Read from multiple FIFOs per cycle:

```verilog
// Dual-port arbiter (2 pointers per cycle)
reg [3:0] addr_a, addr_b;

always @(posedge clk) begin
    addr_a <= addr_a + 2;  // Even addresses
    addr_b <= addr_b + 2;  // Odd addresses
end

// Mux for addr_a and addr_b, write to ptrFIFO twice per cycle
```

**Benefit**: 2× drain rate, halves FIFO depth requirements.

**Trade-off**: Requires wider ptrFIFO or double-pumped downstream.

### 6. Adaptive FIFO Depth

Dynamically adjust FIFO depth based on activity:

```verilog
// Use distributed RAM for shallow portion, spill to BRAM when full
// Requires custom FIFO controller with dual-tier storage
```

**Benefit**: Saves BRAM when network activity is sparse.

### 7. Burst Write to ptrFIFO

Instead of one pointer per cycle, burst multiple:

```verilog
// If ptrFIFO has depth, write up to 4 pointers per cycle
// Requires ptrFIFO to accept burst writes (wider interface)

assign ptrFIFO_din[127:0] = {ptr[addr+3]_dout, ptr[addr+2]_dout,
                             ptr[addr+1]_dout, ptr[addr]_dout};
assign ptrFIFO_wren = burst_valid;
```

**Benefit**: 4× drain rate (if downstream supports).

---

## Key Terms and Definitions

| Term | Definition |
|------|------------|
| **Pointer FIFO** | Buffer storing 32-bit pointer records (length + address) for synaptic lists |
| **Round-Robin** | Arbitration scheme giving equal service time to each FIFO in cyclic order |
| **Spike-Gated** | Write enable conditional on neuron spike (sparse event handling) |
| **Demultiplexing** | Splitting wide HBM data (512b) into narrow pointer streams (16×32b) |
| **FWFT (First-Word Fall-Through)** | FIFO mode where data appears immediately on `dout` when not empty |
| **Phase 1a** | External event processing (BRAM reading, external axon spikes) |
| **Phase 1b** | Internal event processing (URAM reading, neuron-to-neuron spikes) |
| **Phase 2** | Synaptic weight fetch (pointer drain, HBM synaptic reads) |
| **Neuron Group** | Set of 16 neurons mapped to one pointer FIFO |
| **Backpressure** | Flow control mechanism where full FIFO blocks upstream writes |
| **Arbiter** | Logic deciding which FIFO gets access to shared resource (ptrFIFO) |
| **ptrFIFO** | Aggregated pointer FIFO feeding HBM processor for Phase 2 |
| **Starvation** | Condition where some FIFOs never serviced (not possible in round-robin) |
| **Overflow** | Condition where pointer write lost due to FIFO full |
| **Pointer Record** | 32-bit datum: [31:23]=length (9b), [22:0]=start address (23b) |
| **HBM rvalidready** | Signal indicating HBM read data valid and consumer ready |
| **exec_run** | Control pulse starting new time step, initiating Phase 1a |

---

## Conclusion

The **Pointer FIFO Controller** is a well-designed datapath component that efficiently manages sparse neural spike events through:

1. **Parallel Buffering**: 16 independent FIFOs decouple HBM read from pointer consumption
2. **Spike-Gated Writes**: Only buffer pointers for neurons that actually spiked (sparse efficiency)
3. **Fair Arbitration**: Round-robin ensures no FIFO monopolizes downstream bandwidth
4. **Two-Phase Coordination**: Seamlessly handles both external and internal event sources

**Design Strengths**:
- Simple, proven architecture (demux + FIFOs + arbiter)
- Minimal logic (mostly wiring and control)
- FWFT mode reduces latency
- Phase control cleanly separates external and internal events

**Potential Improvements**:
- Priority arbitration to prevent overflow
- Skip-empty optimization to reduce latency
- Gated HBM reads to save bandwidth
- Multi-port arbiter for higher drain rate

**Critical Parameters**:
- FIFO depth must accommodate worst-case burst (512-1024 entries)
- Arbiter must drain faster than fill rate (or FIFOs overflow)
- Round-robin period (16 cycles) limits drain rate

For complete understanding, see cross-referenced modules: `external_events_processor.md`, `internal_events_processor.md`, `hbm_processor.md`, and `spike_fifo_controller.md`.
