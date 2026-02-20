# Spike FIFO Controller Module

## Overview

The **Spike FIFO Controller** is a simple yet critical arbitration component that aggregates output spikes from 8 parallel Spike FIFOs into a single unified stream. These spikes represent post-synaptic neuron activations generated during Phase 2 (synaptic weight processing) and are fed to the Internal Events Processor for the next time step.

### Role in the Software/Hardware Stack

```
                    Phase 2: Synaptic Processing
                       (Weight Application)
                              |
    ┌─────────────────────────┼─────────────────────────┐
    |                         v                         |
    |              [HBM Processor]                      |
    |                         |                         |
    |          Fetch synaptic weights from HBM          |
    |                         |                         |
    |          Apply weights to post-synaptic neurons   |
    |                         |                         |
    |                  spk0_wren ... spk7_wren          |
    |                         |                         |
    |         ┌───────────────┼───────────────┐         |
    |         |               v               |         |
    |         | ┌─────┐ ┌─────┐ ... ┌─────┐  |         |
    |         | │spk0 │ │spk1 │     │spk7 │  |         |
    |         | │FIFO │ │FIFO │     │FIFO │  |         |
    |         | │17b  │ │17b  │     │17b  │  |         |
    |         | └──┬──┘ └──┬──┘     └──┬──┘  |         |
    |         │    |       |            |     │         |
    |         │    v       v            v     │         |
    |         │   ┌────────────────────────┐  │         |
    |         │   │  Round-Robin Arbiter   │  │         |
    |         │   │  (3-bit counter 0-7)   │  │         |
    |         │   └──────────┬─────────────┘  │         |
    |         │              |                 │         |
    |         │              v                 │         |
    |         │      ┌────────────────┐        │         |
    |         │      │  spk2ciFIFO    │        │         |
    |         │      │   (17-bit)     │        │         |
    |         │      └───────┬────────┘        │         |
    |         └──────────────┼─────────────────┘         |
    |                        |                           |
    |                        v                           |
    |           [Internal Events Processor]              |
    |                        |                           |
    |              Next time step (Phase 1b)             |
    └───────────────────────────────────────────────────┘
```

**Function**:
- **Aggregate Spikes**: Collect spike events from 8 parallel FIFOs
- **Fair Arbitration**: Round-robin scheduler ensures all spike sources get equal service
- **Unified Output**: Present consolidated spike stream to downstream processor

**Key Innovation**: By using round-robin arbitration, the module ensures fairness - no single spike source can monopolize the output, preventing starvation even under heavy load.

---

## Module Architecture

```
                     8 Spike FIFOs
                    (Parallel Inputs)
                          |
    ┌─────────────────────┼─────────────────────┐
    |                     v                     |
    | ┌─────┐ ┌─────┐ ┌─────┐ ... ┌─────┐     |
    | │spk0 │ │spk1 │ │spk2 │     │spk7 │     |
    | │empty│ │empty│ │empty│     │empty│     |
    | │dout │ │dout │ │dout │     │dout │     |
    | │[16:0││ │[16:0││ │[16:0│    │[16:0│    |
    | └──┬──┘ └──┬──┘ └──┬──┘     └──┬──┘     |
    |    |       |       |            |        |
    |    └───────┴───────┴────────────┘        |
    |                    |                     |
    |                    v                     |
    |         ┌──────────────────────┐         |
    |         │  Round-Robin Counter │         |
    |         │    addr[2:0]         │         |
    |         │    0 → 1 → ... → 7   │         |
    |         └──────────┬───────────┘         |
    |                    |                     |
    |                    v                     |
    |         ┌──────────────────────┐         |
    |         │   8:1 Multiplexer    │         |
    |         │   (Select based on   │         |
    |         │    addr & !empty &   │         |
    |         │    !spk2ciFIFO_full) │         |
    |         └──────────┬───────────┘         |
    |                    |                     |
    |    ┌───────────────┼───────────────┐     |
    |    |               v               |     |
    | spk*_rden   spk2ciFIFO_din[16:0]  |     |
    |    |         spk2ciFIFO_wren       |     |
    |    v               |               v     |
    |   FIFO            FIFO           FIFO    |
    |  Advance         Write           Enable  |
    └───────────────────┼─────────────────────┘
                        |
                        v
              ┌──────────────────┐
              │  spk2ciFIFO      │
              │  (Output FIFO)   │
              │  To Internal     │
              │  Events Proc.    │
              └──────────────────┘
```

### Data Flow

**Phase 2: Spike Generation (Concurrent with Phase 1)**
```
1. HBM Processor applies synaptic weights
2. For each weight applied:
   - Calculate post-synaptic neuron potential update
   - If neuron crosses threshold → generate spike
   - Write spike to appropriate spike FIFO (spk0-7)
3. Spike data format: [16:0] = {1'b valid, 16'b neuron_address}
4. Spikes accumulate in parallel FIFOs
```

**Phase 3: Spike Drain (After Phase 2 Complete)**
```
1. Round-robin arbiter cycles addr 0→1→...→7→0
2. Every cycle:
   - Check if spk[addr]_empty==0 and spk2ciFIFO_full==0
   - If true: read from spk[addr], write to spk2ciFIFO
   - If false: skip (no-op), continue to next address
3. Continue until all spike FIFOs empty
4. Spikes now ready for Internal Events Processor (next time step)
```

---

## Interface Specification

### Clock and Reset
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | Input | 1 | System clock (225 MHz typical) |
| `resetn` | Input | 1 | Active-low asynchronous reset |

### Spike FIFO Interfaces (8 instances: spk0-spk7)

Each spike FIFO has identical read-only interface (example for spk0):

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `spk0_empty` | Input | 1 | FIFO empty flag |
| `spk0_dout` | Input | 17 | Spike data (likely: valid + neuron address) |
| `spk0_rden` | Output | 1 | Read enable (from arbiter) |

**Spike FIFOs**: spk1, spk2, ..., spk7 (identical interfaces)

**Note**: The module has commented-out ports for spk8-spk15 (lines 40-72, 104-111, 172-229, 239-246), suggesting the design originally supported 16 FIFOs but was reduced to 8 for the current single-core implementation.

### Output Interface (Aggregated Spike FIFO)
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `spk2ciFIFO_full` | Input | 1 | Output FIFO full flag (backpressure) |
| `spk2ciFIFO_din` | Output | 17 | Spike data to output FIFO |
| `spk2ciFIFO_wren` | Output | 1 | Write enable (from arbiter) |

**Name Interpretation**: "spk2ciFIFO" likely means "Spike to Command Interpreter FIFO" or "Spike to Core Internal FIFO", though it actually feeds the Internal Events Processor.

---

## Detailed Logic Description

### Round-Robin Arbiter

A 3-bit counter cycles through FIFOs 0-7, servicing one per cycle:

```verilog
reg [2:0] addr;  // 3 bits for 8 FIFOs (0-7)

always @(posedge clk) begin
    if (!resetn)
        addr <= 3'd0;
    else
        addr <= addr + 1'b1;  // Wraps 7→0 automatically
end
```

**Arbitration Cycle**:
```
Cycle 0:  addr=0  → Check spk0
Cycle 1:  addr=1  → Check spk1
Cycle 2:  addr=2  → Check spk2
...
Cycle 7:  addr=7  → Check spk7
Cycle 8:  addr=0  → Back to spk0
...
```

**Arbitration Period**: 8 cycles (half the period of pointer_fifo_controller's 16 cycles)

### Arbitration Logic (Combinational)

```verilog
always @(*) begin
    // Default: No reads, no writes
    spk0_rden = 1'b0;
    spk1_rden = 1'b0;
    // ... (all spk*_rden = 0)
    spk2ciFIFO_din = 32'dX;  // Note: Typo - should be 17'dX
    spk2ciFIFO_wren = 1'b0;

    case (addr)
        3'd0: begin
            if (!spk0_empty & !spk2ciFIFO_full) begin
                spk0_rden       = 1'b1;
                spk2ciFIFO_din  = spk0_dout;
                spk2ciFIFO_wren = 1'b1;
            end
        end
        3'd1: begin
            if (!spk1_empty & !spk2ciFIFO_full) begin
                spk1_rden       = 1'b1;
                spk2ciFIFO_din  = spk1_dout;
                spk2ciFIFO_wren = 1'b1;
            end
        end
        // ... (pattern repeats for 3'd2 through 3'd7)

        default: begin
            // All outputs stay at default (0 or X)
        end
    endcase
end
```

**Logic Breakdown**:
```
For each cycle:
  IF addr==N AND spk[N]_empty==0 AND spk2ciFIFO_full==0
    THEN:
      spk[N]_rden = 1      (read from spike FIFO N)
      spk2ciFIFO_din = spk[N]_dout  (forward data)
      spk2ciFIFO_wren = 1  (write to output FIFO)
  ELSE:
    (all outputs stay 0, no action)
```

**Identical to pointer_fifo_controller**: Same arbitration pattern, just fewer FIFOs (8 vs 16).

### Spike Data Format (17 bits)

While the exact format isn't documented in the code, typical interpretations:

**Option 1: Valid + Address**
```
Bit [16]:    Spike valid (1=spike, 0=no spike / padding)
Bits [15:0]: Post-synaptic neuron address (0-65535)
```

**Option 2: MSB Address + LSB Metadata**
```
Bit [16]:    Bank select or overflow flag
Bits [15:0]: Neuron address within bank
```

**Option 3: Signed Weight + Address**
```
Bit [16]:    Sign bit (excitatory/inhibitory)
Bits [15:0]: Neuron address
```

**Most Likely**: Option 1 (valid + address), as this is common in sparse event systems.

**Example**:
```
spk0_dout = 17'h10ABC  → Spike valid, neuron address 0x0ABC (2748)
spk1_dout = 17'h00000  → No spike (padding entry)
spk2_dout = 17'h1FFFF  → Spike valid, neuron address 0xFFFF (65535)
```

---

## Timing Diagrams

### Round-Robin Arbiter Operation

```
Cycle    0    1    2    3    4    5    6    7    8
         ────┬────┬────┬────┬────┬────┬────┬────┬────
addr         0    1    2    3    4    5    6    7    0

spk0_empty   ────┐                             ┌─────
             ────└─────────────────────────────┘
             (has data cycles 0-7, empty at 8)

spk1_empty   ───────────────────────────────────────
             (empty throughout)

spk2_empty   ──────────┐                   ┌────────
             ──────────└───────────────────┘
             (has data cycles 2-6)

spk5_empty   ───────────────────────┐   ┌───────────
             ───────────────────────└───┘
             (has data cycles 5-6)

spk2ciFIFO_full ────────────────────────────────────
                (never full)

spk0_rden    ───┐                             ┌─────
             ───└─────────────────────────────┘

spk2_rden    ──────────┐
             ──────────└───────────────────────────

spk5_rden    ───────────────────────┐
             ───────────────────────└───────────────

spk2ciFIFO_wren ──┐    ┌───────────┐───────────┐
                ──└────┘           └───────────┘

spk2ciFIFO_din     S0   S2         S5          X

Explanation:
  Cycle 0 (addr=0): spk0 not empty → read spk0, write spk2ciFIFO
  Cycle 1 (addr=1): spk1 empty → skip
  Cycle 2 (addr=2): spk2 not empty → read spk2, write spk2ciFIFO
  Cycle 3-4: All FIFOs empty → skip
  Cycle 5 (addr=5): spk5 not empty → read spk5, write spk2ciFIFO
  Cycle 6-7: Empty → skip
  Cycle 8 (addr=0): Back to spk0
```

### Backpressure Handling

```
Cycle        0    1    2    3    4
             ────┬────┬────┬────┬────
addr             0    1    2    3    4

spk0_empty   ────────────────────────
             (has data)

spk2ciFIFO_full ───────┐         ┌───
                ───────└─────────┘
                (becomes full at cycle 1)

spk0_rden    ───┐         ┌─────┐
             ───└─────────┘     └────

spk2ciFIFO_wren ┐         ┌─────┐
             ───└─────────┘     └────

Explanation:
  Cycle 0: spk0 has data, output FIFO not full → read and write
  Cycle 1: Output FIFO becomes full → blocked (no read/write)
  Cycle 2: Output FIFO still full → still blocked
  Cycle 3: Output FIFO has space again → resume read/write
  Cycle 4: Continue normal operation

  During cycles 1-2, addr continues incrementing (1→2→3),
  but no operations occur. When spk2ciFIFO has space again,
  arbiter is at addr=3, not addr=0 (missed spk0's turn).
  spk0 will be serviced again when addr wraps back to 0.
```

---

## Resource Usage

### Logic Complexity

**Arbiter**:
- **Counter**: 3-bit register (3 FFs)
- **8:1 Mux**: ~24 LUTs (17 bits × 8-way = ~1.5 LUTs per bit)
- **Control Logic**: ~16 LUTs (empty checks, full checks, case statement)
- **Total**: ~40 LUTs, ~3 FFs

**No FIFOs Instantiated**: This module only arbitrates; FIFOs instantiated elsewhere.

### Comparison to Pointer FIFO Controller

| Metric | Pointer FIFO Controller | Spike FIFO Controller |
|--------|------------------------|----------------------|
| **Input FIFOs** | 16 (ptr0-15) | 8 (spk0-7) |
| **Data Width** | 32 bits (pointer) | 17 bits (spike) |
| **Address Bits** | 4 bits (16 FIFOs) | 3 bits (8 FIFOs) |
| **Arbiter Period** | 16 cycles | 8 cycles |
| **Logic** | ~150 LUTs, ~50 FFs | ~40 LUTs, ~3 FFs |
| **Writes Input FIFOs** | Yes (demux from HBM) | No (read-only) |
| **Complexity** | High (demux + arbiter) | Low (arbiter only) |

**Spike FIFO Controller is simpler**: No demux logic, fewer FIFOs, narrower data.

---

## Cross-References

### Upstream Modules

- **hbm_processor.v** (`hbm_processor.md`):
  - Writes spike data to spk0-7 FIFOs during Phase 2
  - Generates spikes based on synaptic weight application
  - Each spike FIFO corresponds to a subset of HBM output channels

### Downstream Modules

- **internal_events_processor.v** (`internal_events_processor.md`):
  - Receives aggregated spikes from spk2ciFIFO
  - Uses spikes to update URAM neuron potentials in next time step
  - Coordinates Phase 1b (internal event processing)

### Peer Modules

- **pointer_fifo_controller.v** (`pointer_fifo_controller.md`):
  - Similar architecture (round-robin arbiter)
  - Handles pointer data instead of spikes
  - Both operate concurrently during Phase 2

---

## Common Issues and Debugging

### Issue 1: Spikes Lost (FIFO Overflow)

**Symptoms:**
- Neurons don't receive expected updates
- Spike counts lower than expected
- spk*_full flags assert frequently (if monitored)

**Root Cause:**
- HBM processor generates spikes faster than arbiter drains
- Spike FIFO depth too small

**Debug:**
```verilog
// Add probes for FIFO occupancy (requires FIFO IP configuration)
(* mark_debug = "true" *) wire [9:0] spk0_count;
(* mark_debug = "true" *) wire       spk0_overflow;

// Monitor overflow
always @(posedge clk) begin
    if (spk0_full & spk0_wren)  // Assumes spk0_full and spk0_wren exist
        spk0_overflow <= 1'b1;
end
```

**Solution:**
- Increase spike FIFO depth (e.g., 512 → 1024)
- Optimize arbiter (see Enhancements)
- Reduce spike generation rate (network-level optimization)

### Issue 2: Unfair Arbitration

**Symptoms:**
- Some spike sources take much longer to drain
- Uneven latency across different neuron groups

**Root Cause:**
- Round-robin treats all FIFOs equally
- spk0 with 100 spikes gets same service as spk7 with 1 spike

**Debug:**
```verilog
// Track arbitration wins
(* mark_debug = "true" *) reg [15:0] arb_wins [7:0];

always @(posedge clk) begin
    if (spk0_rden) arb_wins[0] <= arb_wins[0] + 1;
    if (spk1_rden) arb_wins[1] <= arb_wins[1] + 1;
    // ... (repeat for all)
end
```

**Solution:**
- Implement weighted round-robin
- Priority arbitration based on FIFO occupancy
- Skip-empty optimization (see Enhancement #2)

### Issue 3: Counter Wrapping Error

**Symptoms:**
- Some FIFOs never serviced
- Arbiter stuck on certain addresses

**Root Cause:**
- 3-bit counter not wrapping correctly (should wrap 7→0)

**Debug:**
```verilog
(* mark_debug = "true" *) reg [2:0] addr;

// Assertion
always @(posedge clk) begin
    assert ((addr == (prev_addr + 1'b1) % 8) || (!resetn));
end
```

**Solution:**
- Explicit wrap (though automatic wrap should work):
```verilog
always @(posedge clk) begin
    if (!resetn)
        addr <= 3'd0;
    else if (addr == 3'd7)
        addr <= 3'd0;
    else
        addr <= addr + 1'b1;
end
```

### Issue 4: Data Width Mismatch

**Symptoms:**
- Compilation warnings about width mismatch
- spk2ciFIFO_din shows unexpected bit patterns

**Root Cause:**
- Line 112: `spk2ciFIFO_din <= 32'dX;` should be `17'dX`
- Typo from when module had 32-bit interface

**Debug:**
```verilog
// Check for synthesis warnings:
// WARNING: Truncating 32-bit value to 17 bits
```

**Solution:**
```verilog
// Fix line 112 and 247:
spk2ciFIFO_din <= 17'dX;  // Changed from 32'dX
```

### Issue 5: Output FIFO Never Drains

**Symptoms:**
- spk2ciFIFO_full asserts and stays high
- Spike processing stalls

**Root Cause:**
- Downstream consumer (internal events processor) not reading
- Deadlock or timing issue

**Debug:**
```verilog
// Monitor output FIFO state
(* mark_debug = "true" *) wire spk2ciFIFO_full;
(* mark_debug = "true" *) wire spk2ciFIFO_rden;  // From downstream
(* mark_debug = "true" *) wire spk2ciFIFO_empty;

// Track stall duration
reg [15:0] stall_counter;
always @(posedge clk) begin
    if (spk2ciFIFO_full & !spk2ciFIFO_rden)
        stall_counter <= stall_counter + 1;
    else
        stall_counter <= 0;
end
```

**Solution:**
- Verify downstream module (internal events processor) is enabled
- Check for deadlock conditions
- Ensure proper handshaking between modules

---

## Performance Characteristics

### Throughput Analysis

**Arbiter Throughput**:
- **Max**: 1 spike per cycle @ 225 MHz = 225 million spikes/s
- **Typical** (50% FIFO occupancy): ~112 million spikes/s
- **Effective** (accounting for empty FIFOs): Variable

**Arbiter Service Rate per FIFO**:
- **Period**: 8 cycles
- **Rate**: 225 MHz / 8 = 28.125 million spikes/s per FIFO
- **Latency** (best case): 0-7 cycles to service (avg 3.5 cycles)

**Example Scenario** (10% neurons spike):
```
Total neurons: 131,072
Neurons spiking: 13,107 (10%)
Spikes distributed across 8 FIFOs: ~1,638 per FIFO

Drain time per FIFO:
  1,638 spikes / (28.125 M spikes/s) = 58.2 µs

Total drain time (concurrent):
  All FIFOs drain in parallel (interleaved by arbiter)
  Total time ≈ 1,638 spikes × 8 FIFOs / 225 MHz = 58.2 µs

(Assuming continuous draining without stalls)
```

### Latency Analysis

**Spike Latency** (from FIFO write to output):
- **Best Case** (FIFO non-empty, arbiter on correct address, output not full):
  - 1 cycle (immediate)

- **Worst Case** (FIFO just filled, arbiter just passed, output full):
  - Wait for arbiter: 7 cycles (worst case, just missed)
  - Wait for output FIFO space: N cycles (depends on drain rate)
  - **Total**: ~8+ cycles @ 225 MHz = ~35+ ns

- **Average Case**:
  - ~4 cycles @ 225 MHz = ~18 ns

**Comparison to Pointer FIFO Controller**:
- Spike controller: 8-cycle period → average 4-cycle wait
- Pointer controller: 16-cycle period → average 8-cycle wait
- **Spike controller has 2× better average latency**

---

## Safety and Edge Cases

### Edge Case 1: All Spike FIFOs Full Simultaneously

**Scenario**: Every spike FIFO is full, new spikes arriving.

**Behavior**:
- HBM processor tries to write spikes, but FIFOs full
- Writes are **lost** (assuming standard FIFO behavior)
- No indication to upstream (unless full flags monitored)

**Safety**:
- ⚠️ Silent data loss (spikes dropped)
- ❌ No backpressure to HBM processor (design limitation)

**Required**: Ensure FIFOs sized to handle worst-case burst.

### Edge Case 2: No Spikes Generated (Quiescent Network)

**Scenario**: No neurons spike during entire Phase 2.

**Behavior**:
```
All spk*_empty = 1  (all FIFOs empty)

Arbiter cycles addr 0→1→...→7→0, but:
  All cases have condition: if (!spk*_empty & ...)
  Condition always false → no reads, no writes

spk2ciFIFO receives no data (correct behavior)
```

**Safety**:
- ✅ Correct - no spurious spikes generated
- ✅ Arbiter idles without consuming resources
- ✅ Downstream sees empty spk2ciFIFO (correct state)

### Edge Case 3: Single Spike in Single FIFO

**Scenario**: Only spk3 has one spike, all others empty.

**Behavior**:
```
Cycle 0-2 (addr=0-2): All empty, no action
Cycle 3 (addr=3): spk3 not empty → read and write
Cycle 4-7 (addr=4-7): All empty, no action
Cycle 8 (addr=0): Back to start, spk3 now empty
```

**Safety**:
- ✅ Correct - single spike processed
- ✅ Minimal overhead (7 idle cycles, 1 active)
- ⚠️ Inefficient for sparse spikes (see Enhancement #2)

### Edge Case 4: Output FIFO Full (Downstream Backpressure)

**Scenario**: spk2ciFIFO full, upstream FIFOs have data.

**Behavior**:
```
spk2ciFIFO_full = 1

For all cases:
  if (!spk*_empty & !spk2ciFIFO_full)  → Condition false
    spk*_rden = 0
    spk2ciFIFO_wren = 0

Result: No spikes drained, all FIFOs stall
```

**Safety**:
- ✅ Proper backpressure (stops draining)
- ⚠️ Upstream FIFOs may overflow if HBM processor continues writing
- 🔒 Deadlock possible if spk2ciFIFO never drains

**Required**: Ensure spk2ciFIFO downstream consumer always active.

### Safety Check: One-Hot Read Enables

**Assertion**: Verify only one FIFO read per cycle
```verilog
wire [7:0] rdens = {spk7_rden, spk6_rden, ..., spk0_rden};

property one_hot_rdens;
    @(posedge clk) disable iff (~resetn)
    $onehot0(rdens);  // At most one bit set
endproperty
assert_rdens: assert property (one_hot_rdens);
```

### Safety Check: No Spurious Writes

**Assertion**: Ensure write only when read occurs
```verilog
property write_implies_read;
    @(posedge clk) disable iff (~resetn)
    spk2ciFIFO_wren |-> |rdens;  // Write implies at least one read
endproperty
assert_write: assert property (write_implies_read);
```

---

## Future Enhancement Opportunities

### 1. Priority Arbiter (Occupancy-Based)

Favor FIFOs with more data:

```verilog
wire [9:0] spk_counts [7:0];  // Assume FIFO IP provides rd_data_count

reg [2:0] priority_addr;
always @(*) begin
    // Find fullest FIFO
    priority_addr = 0;
    for (int i = 1; i < 8; i++) begin
        if (spk_counts[i] > spk_counts[priority_addr])
            priority_addr = i;
    end
end

// Use priority_addr instead of round-robin addr (when FIFO above threshold)
```

**Benefit**: Reduces overflow risk by draining fuller FIFOs first.

### 2. Skip-Empty Optimization

Jump to next non-empty FIFO:

```verilog
wire [7:0] spks_empty = {spk7_empty, ..., spk0_empty};

reg [2:0] next_addr;
always @(*) begin
    next_addr = addr;
    for (int i = 1; i <= 8; i++) begin
        if (!spks_empty[(addr + i) % 8]) begin
            next_addr = (addr + i) % 8;
            break;
        end
    end
end

always @(posedge clk) begin
    if (!resetn)
        addr <= 3'd0;
    else
        addr <= next_addr;  // Jump to next non-empty
end
```

**Benefit**: ~4× faster draining when many FIFOs empty (worst case 8 cycles → 2 cycles avg).

### 3. Multi-Port Arbiter

Read multiple FIFOs per cycle:

```verilog
// Dual-port: service 2 FIFOs per cycle
reg [2:0] addr_a, addr_b;

always @(posedge clk) begin
    addr_a <= (addr_a + 2) % 8;  // Even addresses
    addr_b <= (addr_b + 2) % 8;  // Odd addresses
end

// Dual mux, dual write to spk2ciFIFO (requires wider interface or double-pump)
```

**Benefit**: 2× throughput (if downstream supports burst writes).

### 4. Configurable FIFO Count

Parameterize for flexibility:

```verilog
module spike_fifo_controller #(
    parameter NUM_FIFOS = 8
)(
    // Generate FIFO ports and arbiter logic
);

// Use generate blocks for scalability
```

**Benefit**: Easy to switch between 8 and 16 FIFOs (uncomment lines 40-72).

### 5. Burst Mode Output

Write multiple spikes per cycle:

```verilog
// Wider output: 4 spikes per cycle
assign spk2ciFIFO_din[67:0] = {spk[addr+3]_dout, spk[addr+2]_dout,
                                spk[addr+1]_dout, spk[addr]_dout};
```

**Benefit**: 4× throughput (requires downstream support).

### 6. Adaptive Arbitration

Switch between round-robin and priority based on load:

```verilog
wire high_load = (spk_counts[0] + spk_counts[1] + ... > THRESHOLD);

assign arb_addr = high_load ? priority_addr : round_robin_addr;
```

**Benefit**: Fair when lightly loaded, efficient when heavily loaded.

### 7. Fix Data Width Typo

Minor bug fix:

```verilog
// Line 112, 247: Change 32'dX to 17'dX
spk2ciFIFO_din <= 17'dX;  // Match actual port width
```

---

## Key Terms and Definitions

| Term | Definition |
|------|------------|
| **Spike FIFO** | Buffer storing spike events (neuron address + metadata) |
| **Round-Robin** | Fair arbitration scheme servicing each FIFO in cyclic order |
| **Post-Synaptic** | Neuron receiving input from synaptic connection (target neuron) |
| **spk2ciFIFO** | Output FIFO aggregating spikes from all spike FIFOs |
| **Arbiter** | Logic deciding which FIFO gets access to shared output |
| **Phase 2** | Synaptic weight application phase (HBM processor generates spikes) |
| **Phase 3** | Spike drain phase (this module aggregates spikes for next time step) |
| **Backpressure** | Flow control where full output FIFO blocks upstream reads |
| **FWFT (assumed)** | First-Word Fall-Through mode (data immediately available) |
| **Starvation** | Condition where some FIFOs never serviced (not possible in round-robin) |
| **17-bit Spike** | Data format: likely {valid, neuron_address} |
| **Arbiter Period** | Number of cycles to service all FIFOs once (8 cycles) |
| **Fairness** | Equal service time for all FIFOs regardless of occupancy |
| **Service Rate** | Frequency at which each FIFO gets arbitration turn (28.125 MHz per FIFO) |

---

## Conclusion

The **Spike FIFO Controller** is an elegantly simple component that performs critical aggregation:

**Design Strengths**:
- **Minimal Complexity**: Pure round-robin arbiter, no demux logic
- **Fair Service**: All spike sources get equal treatment
- **Proven Architecture**: Identical pattern to pointer_fifo_controller
- **Low Resource Usage**: ~40 LUTs, ~3 FFs (negligible)

**Design Limitations**:
- **No Backpressure to Upstream**: HBM processor can overflow spike FIFOs
- **Inefficient for Sparse Spikes**: Wastes cycles checking empty FIFOs
- **Fixed Arbitration**: No priority for fuller FIFOs
- **Minor Bug**: Data width typo (32'dX vs 17'dX)

**Optimization Opportunities**:
- Skip-empty optimization (2-4× faster for sparse spikes)
- Priority arbitration (prevent overflow)
- Multi-port arbiter (2× throughput)
- Burst mode output (4× throughput)

**Critical Parameters**:
- Spike FIFO depth must handle worst-case burst
- Arbiter period (8 cycles) limits drain rate to 225 MHz / 8 = 28.125 M spikes/s per FIFO
- Output FIFO (spk2ciFIFO) must drain faster than aggregate fill rate

For complete system understanding, see cross-referenced modules: `hbm_processor.md` (upstream spike generation), `internal_events_processor.md` (downstream spike consumption), and `pointer_fifo_controller.md` (peer arbiter).
