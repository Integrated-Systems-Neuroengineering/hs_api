# pcie2fifos.v

## Module Overview

### Purpose and Role in Stack

The **pcie2fifos** module serves as the **PCIe AXI4 to FIFO bridge**, providing the critical interface between the host computer and the FPGA neuromorphic hardware. This module:

- **Converts AXI4 memory-mapped transactions** (from PCIe controller) into simple FIFO operations
- **Implements AXI4 slave interface** for both read and write transactions
- **Manages bidirectional data flow:** Host→FPGA (writes) and FPGA→Host (reads)
- **Operates at 512-bit data width** for high-bandwidth communication
- **Provides flow control** through FIFO full/empty signals

In the software/hardware stack:
```
Host PC (Python) → PCIe DMA Driver → PCIe Controller (Xilinx IP)
                                           ↓ AXI4 512-bit
                                    ┌──────────────┐
                                    │ pcie2fifos   │
                                    └──────┬───────┘
                                           ↓ FIFO Interface
                              ┌────────────┴─────────────┐
                              │                          │
                         Input FIFO                 Output FIFO
                         (Host→FPGA)                (FPGA→Host)
                              │                          │
                              ▼                          ▼
                     command_interpreter          command_interpreter
```

This module abstracts away AXI4 protocol complexity, allowing the rest of the design to work with simple FIFO interfaces.

---

## Module Architecture

### High-Level Block Diagram

```
                        pcie2fifos
    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │  ┌─────────────────────────────────────────────┐      │
    │  │   AXI4 Write Channel State Machine          │      │
    │  │   (Host → FPGA Data Path)                   │      │
    │  │                                             │      │
    │  │   RX_RESET → RX_DATA → RX_DONE → RX_RESET   │      │
    │  │        │        ▲         │                 │      │
    │  │        │        │         │                 │      │
AXI4 │  │    AW ready  W data   B response           │      │
Write│  └────────┬──────┼─────────┼──────────────────┘      │
─────┼───────────┘      │         │                         │
     │                  │         │                         │
s_axi│  ┌───────────────▼─────────┴───────────────────┐     │
_awaddr              AXI4 Write Logic                  │    │
_awvalid             - Address channel (AW)            │    │
_awready             - Data channel (W)                │    │ Input FIFO
_wdata  │            - Response channel (B)            │───▶│ (512-bit)
_wvalid │            - Flow control via inpFIFO_full  │    │ PC → FPGA
_wready │            - Write enable: wready & wvalid   │    │
_wlast  │            └────────────────────────────────────┘    │
_bvalid │                                                     │
_bready │                                                     │
        │                                                     │
        │  ┌─────────────────────────────────────────────┐   │
        │  │   AXI4 Read Channel State Machine           │   │
        │  │   (FPGA → Host Data Path)                   │   │
        │  │                                              │   │
        │  │   TX_RESET → TX_DATA → TX_RESET             │   │
        │  │        │        ▲                            │   │
        │  │        │        │                            │   │
AXI4    │  │    AR ready  R data (multi-beat)           │   │
Read    │  └────────┬──────┼───────────────────────────┘   │
────────┼───────────┘      │                               │
        │                  │                               │
s_axi   │  ┌───────────────▼─────────────────────────┐    │
_araddr │  │       AXI4 Read Logic                   │    │ Output FIFO
_arvalid│  │    - Address channel (AR)                │◄───│ (512-bit)
_arready│  │    - Data channel (R)                    │    │ FPGA → PC
_rdata  │  │    - Beat counter (arlen)                │    │ (FWFT mode)
_rvalid │  │    - Auto-pop on rvalid & rready         │    │
_rready │  │    - rlast generation                    │    │
_rlast  │  │    └──────────────────────────────────────────┘    │
        │                                                     │
        │  ┌─────────────────────────────────────────────┐   │
        │  │   Mandatory AXI4 Signal Assignments         │   │
        │  │   - bid = 4'd0                              │   │
        │  │   - rid = 4'd0                              │   │
        │  │   - rresp = 2'd0 (OKAY)                     │   │
        │  │   - bresp = 2'd0 (OKAY)                     │   │
        │  └─────────────────────────────────────────────┘   │
        │                                                     │
        └─────────────────────────────────────────────────────┘
```

---

## Interface Specification

### Clock and Reset

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `aclk` | Input | 1 | 225 MHz AXI4 clock |
| `aresetn` | Input | 1 | Active-low asynchronous reset |

### AXI4 Read Address Channel

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `s_axi_araddr` | Input | 64 | Read address (unused in this design) |
| `s_axi_arburst` | Input | 2 | Burst type (unused) |
| `s_axi_arcache` | Input | 4 | Cache type (unused) |
| `s_axi_arid` | Input | 4 | Transaction ID (unused, returns 0) |
| `s_axi_arlen` | Input | 8 | Burst length (beats - 1) |
| `s_axi_arlock` | Input | 1 | Lock type (unused) |
| `s_axi_arprot` | Input | 3 | Protection type (unused) |
| `s_axi_arready` | Output (reg) | 1 | Read address ready |
| `s_axi_arsize` | Input | 3 | Burst size (unused) |
| `s_axi_arvalid` | Input | 1 | Read address valid |

**Note:** Address and metadata signals are ignored; this module simply pops data from output FIFO regardless of address.

### AXI4 Write Address Channel

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `s_axi_awaddr` | Input | 64 | Write address (unused) |
| `s_axi_awburst` | Input | 2 | Burst type (unused) |
| `s_axi_awcache` | Input | 4 | Cache type (unused) |
| `s_axi_awid` | Input | 4 | Transaction ID (unused, returns 0) |
| `s_axi_awlen` | Input | 8 | Burst length (unused, single-beat assumed) |
| `s_axi_awlock` | Input | 1 | Lock type (unused) |
| `s_axi_awprot` | Input | 3 | Protection type (unused) |
| `s_axi_awready` | Output (reg) | 1 | Write address ready |
| `s_axi_awsize` | Input | 3 | Burst size (unused) |
| `s_axi_awvalid` | Input | 1 | Write address valid |

### AXI4 Write Data Channel

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `s_axi_wdata` | Input | 512 | Write data (directly connected to inpFIFO_din) |
| `s_axi_wlast` | Input | 1 | Last beat of write burst |
| `s_axi_wready` | Output (reg) | 1 | Write data ready |
| `s_axi_wstrb` | Input | 64 | Write strobes (unused, all bytes written) |
| `s_axi_wvalid` | Input | 1 | Write data valid |

### AXI4 Write Response Channel

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `s_axi_bid` | Output | 4 | Response transaction ID (always 4'd0) |
| `s_axi_bready` | Input | 1 | Response ready (from master) |
| `s_axi_bresp` | Output | 2 | Write response (always 2'd0 = OKAY) |
| `s_axi_bvalid` | Output (reg) | 1 | Write response valid |

### AXI4 Read Data Channel

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `s_axi_rdata` | Output | 512 | Read data (from outFIFO_dout or default pattern) |
| `s_axi_rid` | Output | 4 | Read transaction ID (always 4'd0) |
| `s_axi_rlast` | Output | 1 | Last beat of read burst |
| `s_axi_rready` | Input | 1 | Read data ready (from master) |
| `s_axi_rresp` | Output | 2 | Read response (always 2'd0 = OKAY) |
| `s_axi_rvalid` | Output (reg) | 1 | Read data valid |

### Input FIFO Interface (PC → FPGA)

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `inpFIFO_full` | Input | 1 | FIFO full flag (backpressure) |
| `inpFIFO_din` | Output | 512 | FIFO data input |
| `inpFIFO_wren` | Output | 1 | FIFO write enable |

**Connection:**
```verilog
assign inpFIFO_din  = s_axi_wdata;
assign inpFIFO_wren = s_axi_wready & s_axi_wvalid;
```

### Output FIFO Interface (FPGA → PC)

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `outFIFO_empty` | Input | 1 | FIFO empty flag |
| `outFIFO_dout` | Input | 512 | FIFO data output |
| `outFIFO_rden` | Output | 1 | FIFO read enable |

**Connection (FWFT FIFO assumed):**
```verilog
assign s_axi_rdata = ~outFIFO_empty ? outFIFO_dout : {16{32'h89ABCDEF}};
assign outFIFO_rden = ~outFIFO_empty & s_axi_rvalid & s_axi_rready;
```

**Note:** FWFT (First-Word Fall-Through) mode means data appears on `dout` one cycle after `rden` assertion is not required; data is valid when `empty` is deasserted.

---

## Detailed Logic Description

### Mandatory Signal Assignments

```verilog
assign s_axi_bid = 4'd0;      // Transaction ID always 0
assign s_axi_rid = 4'd0;      // Transaction ID always 0
assign s_axi_rresp = 2'd0;    // OKAY response
assign s_axi_bresp = 2'd0;    // OKAY response
```

These signals are required by AXI4 protocol but carry no information in this simple bridge design.

### Input Data Path (Write Logic)

**Direct Assignments:**
```verilog
assign inpFIFO_din  = s_axi_wdata;         // 512-bit data passthrough
assign inpFIFO_wren = s_axi_wready & s_axi_wvalid;  // Write when handshake occurs
```

**Write State Machine States:**
```verilog
localparam [1:0] STATE_RX_RESET = 2'd0;
localparam [1:0] STATE_RX_DATA  = 2'd1;
localparam [1:0] STATE_RX_DONE  = 2'd3;  // Note: 2'd2 skipped
```

**State Transition Diagram:**

```
        ┌──────────────┐
        │  RX_RESET    │
        └──────┬───────┘
               │ aresetn
               ▼
        ┌──────────────┐
    ┌──▶│  RX_DATA     │
    │   │  Wait for    │
    │   │  awvalid     │──awvalid──┐
    │   └──────────────┘           │
    │                              │
    │                              ▼
    │                       ┌──────────────┐
    │                       │  RX_DATA     │
    │                       │  Push FIFO   │
    │                       │  (if !full)  │
    │                       └──────┬───────┘
    │                              │
    │                         wvalid & wlast
    │                              │
    │                              ▼
    │                       ┌──────────────┐
    │                       │  RX_DONE     │
    │                       │  Send bresp  │
    │                       └──────┬───────┘
    │                              │
    │                         bready
    │                              │
    │                              ├─ awvalid ──┐
    │                              │            │
    │                              ▼            ▼
    └──────────────────────────────┘      (back to RX_DATA)
```

**Detailed State Behavior:**

**STATE_RX_RESET:**
```verilog
s_axi_awready = 1'b1;          // Ready to accept address
if (s_axi_awvalid)
   rx_next_state = STATE_RX_DATA;
```

**STATE_RX_DATA:**
```verilog
if (!inpFIFO_full) begin
   s_axi_wready = 1'b1;        // Ready to accept data
   if (s_axi_wvalid & s_axi_wlast)
      rx_next_state = STATE_RX_DONE;
end
// Note: Backpressure applied when FIFO full
```

**STATE_RX_DONE:**
```verilog
s_axi_bvalid = 1'b1;           // Assert write response valid
if (s_axi_bready) begin
   s_axi_awready = 1'b1;       // Ready for next transaction
   if (s_axi_awvalid)
      rx_next_state = STATE_RX_DATA;
   else
      rx_next_state = STATE_RX_RESET;
end
```

### Output Data Path (Read Logic)

**Direct Assignments:**
```verilog
assign s_axi_rdata = ~outFIFO_empty ? outFIFO_dout : {16{32'h89ABCDEF}};
assign outFIFO_rden = ~outFIFO_empty & s_axi_rvalid & s_axi_rready;
```

**Default Pattern:** `0x89ABCDEF` repeated 16 times (512 bits) when FIFO empty

**Read State Machine States:**
```verilog
localparam STATE_TX_RESET = 1'b0;
localparam STATE_TX_DATA  = 1'b1;
```

**Read Length Tracking:**
```verilog
reg [7:0] s_axi_arlen_reg;  // Capture burst length
reg [7:0] tx_ctr;           // Beat counter

// Capture burst length
always @(posedge aclk)
   if (!aresetn)
      s_axi_arlen_reg <= 8'b0;
   else if (s_axi_arready & s_axi_arvalid)
      s_axi_arlen_reg <= s_axi_arlen;

// Increment beat counter
always @(posedge aclk)
   if (!aresetn | (s_axi_arready & s_axi_arvalid))
      tx_ctr <= 8'd0;
   else if (s_axi_rready & s_axi_rvalid)
      tx_ctr <= tx_ctr + 1'b1;

// Generate last signal
assign s_axi_rlast = (tx_ctr == s_axi_arlen_reg);
```

**State Transition Diagram:**

```
        ┌──────────────┐
        │  TX_RESET    │
        └──────┬───────┘
               │ aresetn
               ▼
        ┌──────────────┐
    ┌──▶│  TX_RESET    │
    │   │  Wait for    │
    │   │  arvalid     │──arvalid──┐
    │   └──────────────┘           │
    │                              │
    │                              ▼
    │                       ┌──────────────┐
    │                       │  TX_DATA     │
    │                       │  Stream data │
    │                       │  (multi-beat)│
    │                       └──────┬───────┘
    │                              │
    │                     rready & rlast
    │                              │
    │                              ├─ arvalid ──┐
    │                              │            │
    │                              ▼            ▼
    └──────────────────────────────┘      (stay in TX_DATA)
```

**Detailed State Behavior:**

**STATE_TX_RESET:**
```verilog
s_axi_arready = 1'b1;          // Ready to accept read address
if (s_axi_arvalid)
   tx_next_state = STATE_TX_DATA;
```

**STATE_TX_DATA:**
```verilog
s_axi_rvalid = 1'b1;           // Data always valid (even if FIFO empty)
if (s_axi_rready & (tx_ctr==s_axi_arlen_reg)) begin
   s_axi_arready = 1'b1;       // Ready for next transaction
   if (!s_axi_arvalid)
      tx_next_state = STATE_TX_RESET;
   // else stay in TX_DATA for back-to-back bursts
end
```

**FWFT FIFO Behavior:**
- Data valid on `outFIFO_dout` when `outFIFO_empty==0`
- No read latency: `rden` pulse pops current data and advances to next
- Simplifies `rden` logic: just AND of `rvalid` and `rready`

---

## Timing Diagrams

### Write Transaction (Single Beat)

```
Cycle:     0      1      2      3      4      5      6      7
           │      │      │      │      │      │      │      │
aclk     ──┘▔▔▔▔▔▔└┐    ┌▔┐    ┌▔┐    ┌▔┐    ┌▔┐    ┌▔┐    ┌▔
           │      │▔▔▔▔▔▔│▔│▔▔▔▔│▔│▔▔▔▔│▔│▔▔▔▔│▔│▔▔▔▔│▔│▔▔▔▔│
RX State   RESET  │DATA  │DATA  │DATA  │DONE  │RESET │
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│
awvalid    │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▔▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│
awready    │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wvalid     │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wready     │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wlast      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     XXXXXX │XXXXXX│XXXXXX│WDATA │WDATA │XXXXXX│XXXXXX│
wdata      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
inpFIFO_   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wren       │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁
bvalid     │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁
bready     │      │      │      │      │      │      │      │
```

**Notes:**
- Cycle 0: RESET state, `awready` asserted
- Cycle 1: `awvalid` handshake, transition to DATA state
- Cycle 2: Wait for `wvalid` (FIFO not full assumed)
- Cycle 3: `wvalid` & `wready` handshake, `wlast` asserted, data written to FIFO
- Cycle 4: DONE state, `bvalid` asserted for write response
- Cycle 5: `bready` handshake, return to RESET

### Write Transaction with Backpressure

```
Cycle:     0      1      2      3      4      5      6      7
           │      │      │      │      │      │      │      │
RX State   DATA   │DATA  │DATA  │DATA  │DATA  │DONE  │RESET │
           │      │      │      │      │      │      │      │
inpFIFO_   ▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
full       │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▔▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁
wvalid     │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wready     │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
inpFIFO_   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wren       │      │      │      │      │      │      │      │
```

**Notes:**
- Cycles 1-3: FIFO full, `wready` deasserted (backpressure)
- Cycle 4: FIFO no longer full, `wready` asserted, handshake occurs
- Master must hold `wvalid` and `wdata` stable during backpressure

### Read Transaction (4-Beat Burst, arlen=3)

```
Cycle:     0      1      2      3      4      5      6      7
           │      │      │      │      │      │      │      │
TX State   RESET  │DATA  │DATA  │DATA  │DATA  │RESET │
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
arvalid    │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▔▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│
arready    │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     8'hXX  │8'h03 │8'h03 │8'h03 │8'h03 │8'h03 │8'h00 │
arlen      │      │      │(=3)  │      │      │      │      │
           │      │      │      │      │      │      │      │
arlen_reg  8'h00  │8'h00 │8'h03 │8'h03 │8'h03 │8'h03 │8'h03 │
           │      │      │      │      │      │      │      │
tx_ctr     0      │0      │0      │1      │2      │3      │0
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁
rvalid     │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁
rready     │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     XXXXXX │XXXXXX│DATA0 │DATA1 │DATA2 │DATA3 │XXXXXX│
rdata      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
s_axi_     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁
rlast      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │
outFIFO_   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁
rden       │      │      │      │      │      │      │      │
```

**Notes:**
- Cycle 0: RESET state, `arready` asserted
- Cycle 1: `arvalid` handshake, `arlen=3` (4 beats total)
- Cycle 2: First beat, `arlen` registered, `tx_ctr=0`, `rlast=0`
- Cycle 3-5: Beats 2-4, `tx_ctr` increments
- Cycle 5: Last beat, `tx_ctr=3`, `rlast=1`
- Cycle 6: Return to RESET, counter resets

---

## Memory Map

This module does not implement a true memory map. All address signals (`s_axi_araddr`, `s_axi_awaddr`) are ignored. The module operates as a streaming interface:

- **Write Operations:** Data pushed to input FIFO regardless of address
- **Read Operations:** Data popped from output FIFO regardless of address

The actual memory mapping and addressing is handled by:
- **Host software:** Determines which FIFO queue to use (via PCIe BAR mapping)
- **command_interpreter:** Interprets commands from input FIFO and routes to appropriate modules

---

## Protocol Details

### AXI4 Burst Types

**Supported:** INCR (incrementing burst) - assumed by design
**Unsupported:** FIXED, WRAP bursts - ignored, module treats all as streaming

**Burst Length:**
- Write: Assumed single-beat (though `wlast` detection allows multi-beat)
- Read: Supports multi-beat bursts via `arlen` and beat counter

### AXI4 Response Codes

**Write Response (`bresp`):** Always `2'b00` (OKAY)
**Read Response (`rresp`):** Always `2'b00` (OKAY)

No error conditions reported; module assumes all transactions succeed.

### FIFO Assumptions

**Input FIFO (Standard FIFO):**
- `wren` pulse writes data on same cycle
- `full` flag provides backpressure
- Depth sufficient to absorb PCIe burst traffic

**Output FIFO (FWFT Mode):**
- Data appears on `dout` when `empty` deasserted
- `rden` pulse pops data and advances to next
- Zero read latency (data valid same cycle as `empty=0`)

**Typical FIFO Depths:**
- Input: 512 entries (256 KB) - handles large DMA transfers
- Output: 512 entries (256 KB) - buffers spike/response data

---

## Performance Characteristics

### Bandwidth

**Theoretical Maximum:**
- 512 bits @ 225 MHz = 115.2 Gb/s = 14.4 GB/s per direction

**Practical Bandwidth:**
- Write: Limited by FIFO full condition and PCIe overhead
- Read: Limited by FIFO empty condition and burst efficiency
- Expected: 8-12 GB/s (accounting for protocol overhead and command processing)

### Latency

**Write Transaction:**
- Address handshake: 1 cycle
- Data handshake: 1 cycle (if FIFO not full)
- Response handshake: 1 cycle
- **Total:** 3 cycles minimum (13.3 ns @ 225 MHz)

**Read Transaction:**
- Address handshake: 1 cycle
- Data valid: Immediate (FWFT FIFO)
- Per-beat: 1 cycle (if master ready)
- **Total:** 1 cycle address + N cycles data (N = burst length)

### Flow Control

**Backpressure Mechanisms:**

Write Path:
```
inpFIFO_full → s_axi_wready deasserted → PCIe master stalls
```

Read Path:
```
outFIFO_empty → s_axi_rdata = default pattern → PCIe master receives dummy data
```
**Note:** Read path does NOT stall when FIFO empty; returns default pattern instead. Software must track expected response count.

---

## Cross-References

### Related Modules

| Module | Relationship | Interface |
|--------|--------------|-----------|
| **PCIe Controller (Xilinx IP)** | Upstream | AXI4 master providing `s_axi_*` signals |
| **command_interpreter.v** | Downstream | Connects to `inpFIFO` (read) and `outFIFO` (write) |
| **Input FIFO (Xilinx IP)** | Internal | 512-bit async FIFO (PC→FPGA) |
| **Output FIFO (Xilinx IP)** | Internal | 512-bit async FIFO (FPGA→PC), FWFT mode |

### Software Integration

**DMA Driver (C++ wrapper: adxdma_dmadump.cpp):**
- `DMA_Write()` → Triggers AXI4 write transactions → `inpFIFO`
- `DMA_Read()` → Triggers AXI4 read transactions → `outFIFO`

**Python (hs_bridge):**
- `fpga_controller.send_command()` → Formats PCIe packet → DMA write
- `fpga_controller.receive_response()` → DMA read → Parse packet

---

## Key Terms and Definitions

| Term | Definition |
|------|------------|
| **AXI4** | ARM Advanced eXtensible Interface version 4 - High-performance memory-mapped protocol |
| **AXI4 Slave** | Device that responds to read/write requests (this module) |
| **AXI4 Master** | Device that initiates transactions (PCIe controller) |
| **Handshake** | Valid/ready protocol: transaction occurs when both asserted |
| **Burst** | Multi-beat transaction transferring multiple data words |
| **Beat** | Single data transfer within a burst |
| **FWFT** | First-Word Fall-Through - FIFO mode with zero read latency |
| **Backpressure** | Flow control mechanism to pause upstream when buffer full |
| **Transaction ID** | `arid`/`awid`/`bid`/`rid` - Allows out-of-order completion (not used here) |
| **Response Code** | `rresp`/`bresp` - Status of transaction (always OKAY here) |
| **INCR Burst** | Incrementing address burst type (addresses increment by size) |
| **wlast** | Indicates last beat of write burst |
| **rlast** | Indicates last beat of read burst |
| **arlen** | Read burst length in beats (0 = 1 beat, 1 = 2 beats, etc.) |

---

## Design Simplifications

This module makes several simplifying assumptions compared to full AXI4 protocol:

1. **No Address Decoding:** All addresses ignored; operates as pure FIFO bridge
2. **No Transaction IDs:** All IDs hardcoded to 0; no out-of-order support
3. **No Error Reporting:** All responses OKAY; no SLVERR or DECERR
4. **Single Outstanding Transaction:** State machines handle one transaction at a time
5. **No Write Strobes:** `wstrb` ignored; all bytes written
6. **No Burst Optimization:** Treats each burst independently
7. **Read Returns Dummy Data When Empty:** Does not stall; software must manage

These simplifications are valid for this use case:
- Single master (PCIe controller)
- Software controls transaction ordering
- Command-response pattern ensures data availability
- High-level protocol (in command_interpreter) handles errors

---

## Common Issues and Debugging

### Problem: Data Lost on Write

**Symptoms:** Host sends data but FPGA doesn't receive it

**Debug Steps:**
1. Check `inpFIFO_full` - if stuck high, downstream not consuming
2. Check `s_axi_wready` - should pulse when not full
3. Check `inpFIFO_wren` - should pulse on `wready & wvalid`
4. Verify `wlast` assertion - module expects it to transition to DONE state

**Common Cause:** Input FIFO overflow due to command_interpreter stalled

### Problem: Read Returns Garbage

**Symptoms:** Host reads unexpected data

**Debug Steps:**
1. Check `outFIFO_empty` - if high, FIFO has no data
2. Check `s_axi_rdata` - should be `0x89ABCDEF` pattern when empty
3. Verify command_interpreter has written response to `outFIFO`
4. Check beat counter `tx_ctr` vs `arlen_reg` - ensure correct burst length

**Common Cause:** Reading before response ready, or incorrect burst length

### Problem: Write Transaction Hangs

**Symptoms:** `s_axi_wvalid` stuck high, no progress

**Debug Steps:**
1. Check `s_axi_wready` - if stuck low, likely `inpFIFO_full`
2. Check `rx_curr_state` - should be DATA state
3. Verify `aresetn` - ensure not in reset
4. Check for missing `s_axi_wlast` - module waits for it

**Common Cause:** FIFO backpressure or missing `wlast` signal

### VIO/ILA Probes (Recommended)

```verilog
// State machines
(*mark_debug = "true"*) reg [1:0] rx_curr_state;
(*mark_debug = "true"*) reg tx_curr_state;

// Handshakes
(*mark_debug = "true"*) wire aw_hs = s_axi_awvalid & s_axi_awready;
(*mark_debug = "true"*) wire w_hs  = s_axi_wvalid & s_axi_wready;
(*mark_debug = "true"*) wire b_hs  = s_axi_bvalid & s_axi_bready;
(*mark_debug = "true"*) wire ar_hs = s_axi_arvalid & s_axi_arready;
(*mark_debug = "true"*) wire r_hs  = s_axi_rvalid & s_axi_rready;

// FIFO status
(*mark_debug = "true"*) wire inpFIFO_full;
(*mark_debug = "true"*) wire outFIFO_empty;
```

---

## Potential Enhancements

1. **Address-Based Routing:** Use addresses to multiplex between multiple logical FIFOs
2. **Error Detection:** Monitor for protocol violations (e.g., `wvalid` without prior `awvalid`)
3. **Performance Counters:** Track transactions, stall cycles, bandwidth utilization
4. **Configurable Data Width:** Parameterize for 256-bit or 1024-bit data paths
5. **Timeout Detection:** Abort transactions stuck waiting for handshakes
6. **Out-of-Order Support:** Use transaction IDs for pipelined operation
7. **Read Stalling:** Deassert `rvalid` when output FIFO empty instead of returning dummy data

---

## Safety and Reset Behavior

### Reset Assertions

On `aresetn` deassertion (active-low reset):
- All state machines → RESET state
- All `ready` signals → 1 (accepting transactions)
- All `valid` outputs → 0 (no data/responses)
- Beat counter `tx_ctr` → 0
- Burst length register `s_axi_arlen_reg` → 0

### FIFO Reset Synchronization

FIFOs must be reset consistently with this module:
- Use same `aresetn` or synchronized version
- Ensure FIFOs clear before module exits reset
- First transaction after reset should not assume FIFO state

---

**Document Version:** 1.0
**Last Updated:** December 2025
**Module File:** `pcie2fifos.v`
**Module Location:** `CRI_proj/cri_fpga/code/new/hyddenn2/vivado/single_core.srcs/sources_1/new/`
**Protocol:** AXI4 (slave interface)
**Data Width:** 512 bits
**Clock Frequency:** 225 MHz (aclk)
