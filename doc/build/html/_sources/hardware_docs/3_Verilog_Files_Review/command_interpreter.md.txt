# command_interpreter.v

## Module Overview

### Purpose and Role in Stack

The **command_interpreter** module serves as the central command router and execution controller for the neuromorphic FPGA core. It acts as the primary interface between the host computer (via PCIe) and all internal processing modules. This module:

- **Decodes and routes PCIe commands** to appropriate processors (HBM, external events, internal events)
- **Manages network execution flow** including time-step sequencing and execution counters
- **Handles axon event distribution** by parsing 512-bit PCIe packets into individual axon events
- **Collects and batches spike outputs** for transmission back to the host
- **Maintains execution statistics** including time-step counters and FPGA cycle timers

In the software/hardware stack:
```
Host (Python hs_bridge) → PCIe DMA → pcie2fifos → command_interpreter → Processing Modules
                                                   ↓
                                         [External Events Processor]
                                         [HBM Processor]
                                         [Internal Events Processor]
```

---

## Module Architecture

### High-Level Block Diagram

```
                    ┌────────────────────────────────────────────────┐
                    │         command_interpreter                    │
                    │                                                 │
    PCIe RX FIFO    │  ┌──────────────────────────────────────┐     │
    512-bit         │  │      RX State Machine                │     │
    ────────────────┼──►  Command Decoder & Router            │     │
                    │  │  - CMD_EEP_W: Load axon events       │     │
                    │  │  - CMD_HBM_RW: R/W synapses         │     │
                    │  │  - CMD_IEP_RW: R/W neurons          │     │
                    │  │  - CMD_NTWK_PARAM_W: Set params     │     │
                    │  │  - CMD_EXEC_STEP: Run 1 timestep    │     │
                    │  │  - CMD_EXEC_CONT: Continuous run    │     │
                    │  └────┬─────────────┬──────────┬────────┘     │
                    │       │             │          │               │
                    │   ┌───▼────┐   ┌───▼────┐  ┌─▼────────┐      │
                    │   │ Axon   │   │  HBM   │  │ Internal │      │
                    │   │ Event  │   │  FIFO  │  │  Events  │      │
                    │   │ Shifter│   │ ci2hbm │  │   FIFO   │      │
    External Events │   └────┬───┘   └───┬────┘  │ ci2iep   │      │
    Processor       │        │           │       └─┬────────┘      │
    ◄───────────────┼────────┘           │         │               │
                    │                    │         │               │
    HBM Processor   │                    │         │               │
    ◄───────────────┼────────────────────┘         │               │
                    │                              │               │
    Internal Events │                              │               │
    Processor       │                              │               │
    ◄───────────────┼──────────────────────────────┘               │
                    │                                               │
                    │  ┌──────────────────────────────────────┐    │
                    │  │      TX State Machine                │    │
    PCIe TX FIFO    │  │  Spike Collection & Batching         │    │
    512-bit         │  │  - Batches 14 spikes per packet      │◄───┼── Spike FIFO
    ◄───────────────┼──┤  - Includes timestamp (execRun_ctr)  │    │   (from spike_
                    │  │  - Opcode: 0xEEEE_EEEE               │    │    fifo_controller)
                    │  └──────────────────────────────────────┘    │
                    │                                               │
                    │  ┌──────────────────────────────────────┐    │
                    │  │   Execution Control Registers        │    │
                    │  │  - execRun_ctr (time-step counter)   │    │
                    │  │  - execRun_limit (max timesteps)     │    │
                    │  │  - execRun_timer (FPGA cycle count)  │    │
                    │  │  - execRun_running / execRun_done    │    │
                    │  └──────────────────────────────────────┘    │
                    └────────────────────────────────────────────────┘
```

---

## Interface Specification

### Module Parameters

| Parameter | Width | Default | Description |
|-----------|-------|---------|-------------|
| `AXI_ADDR_BITS` | - | 32 | AXI address width (currently unused) |
| `AXI_DATA_WIDTH` | - | 32 | AXI data width (currently unused) |
| `HBM_ADDR_BITS` | - | 33 | HBM address width (8 GB addressable) |
| `HBM_DATA_WIDTH` | - | 256 | HBM data width |
| `HBM_BYTE_COUNT` | - | 32 | HBM bytes per transaction (256 bits / 8) |

### Clock and Reset

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `aclk` | Input | 1 | 225 MHz system clock |
| `aresetn` | Input | 1 | Active-low asynchronous reset |

### PCIe Interface (via pcie2fifos)

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| **RX FIFO (Host → Card)** | | | |
| `rxFIFO_empty` | Input | 1 | RX FIFO empty flag |
| `rxFIFO_dout` | Input | 512 | RX FIFO data output |
| `rxFIFO_rden` | Output | 1 | RX FIFO read enable |
| | | | |
| **TX FIFO (Card → Host)** | | | |
| `txFIFO_full` | Input | 1 | TX FIFO full flag |
| `txFIFO_din` | Output | 512 | TX FIFO data input |
| `txFIFO_wren` | Output | 1 | TX FIFO write enable |

**PCIe Data Format:**

RX FIFO (512-bit packet from host):
```
[511:504] = 8-bit command opcode
[503:0]   = Command-specific data payload
```

TX FIFO (512-bit packet to host):
```
Spike packet:
  [511:480] = 0xEEEE_EEEE (spike opcode)
  [479:32]  = 14 × 32-bit spike events (448 bits)
              Each spike: [31:24]=sub-timestamp, [23]=flag, [22:16]=zeros, [15:0]=neuron addr
  [31:0]    = execRun_ctr (timestep counter)

HBM read response:
  [511:496] = 0xBBBB
  [495:256] = zeros (240 bits)
  [255:0]   = HBM data

Neuron read response:
  [511:496] = 0xCCCC
  [495:53]  = zeros (443 bits)
  [52:0]    = Neuron data (36-bit value + 17-bit address)
```

### External Events Processor Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `axonEvent_set` | Output | 1 | Axon event valid/write enable |
| `axonEvent_addr` | Output | 13 | Row address in BRAM (0 to 8191) |
| `axonEvent_data` | Output | 16 | Data mask (1 bit per neuron group) |

**Address Calculation:**
- 16 neuron groups → 16 axons per row
- `num_inputs[16:4]` = number of rows (ignoring lower 4 bits)
- Each 512-bit PCIe packet contains 512/16 = 32 axon events

### HBM Processor Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| **Command Interface (CI → HBM)** | | | |
| `ci2hbm_full` | Input | 1 | Command FIFO full flag |
| `ci2hbm_din` | Output | 280 | Command data |
| `ci2hbm_wren` | Output | 1 | Command write enable |
| | | | |
| **Response Interface (HBM → CI)** | | | |
| `hbm2ci_empty` | Input | 1 | Response FIFO empty flag |
| `hbm2ci_dout` | Input | 256 | Response data |
| `hbm2ci_rden` | Output | 1 | Response read enable |

**Command Format (`ci2hbm_din[279:0]`):**
```
[279]     = R/W command (0=read, 1=write)
[278:256] = 23-bit row address
[255:0]   = 256-bit data (for writes)
```

### Internal Events Processor Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| **Command Interface (CI → IEP)** | | | |
| `ci2iep_full` | Input | 1 | Command FIFO full flag |
| `ci2iep_din` | Output | 54 | Command data |
| `ci2iep_wren` | Output | 1 | Command write enable |
| | | | |
| **Response Interface (IEP → CI)** | | | |
| `iep2ci_empty` | Input | 1 | Response FIFO empty flag |
| `iep2ci_dout` | Input | 53 | Response data |
| `iep2ci_rden` | Output | 1 | Response read enable |

**Command Format (`ci2iep_din[53:0]`):**
```
[53]      = R/W command (0=read, 1=write)
[52:36]   = 17-bit neuron address (128K neurons addressable)
[35:0]    = 36-bit neuron data (membrane potential or other state)
```

**Note:** Comments in code show previous 16-bit data format; upgraded to 36-bit.

### Spike Event Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `spk2ciFIFO_dout` | Input | 17 | Spiked neuron address |
| `spk2ciFIFO_empty` | Input | 1 | Spike FIFO empty flag |
| `spk2ciFIFO_rden` | Output | 1 | Spike FIFO read enable |

### Network Execution Control

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `exec_iep_phase2_done` | Input | 1 | Internal events processor phase 2 completion flag |
| `exec_run` | Output | 1 | Execute time-step command (1-cycle pulse) |
| `execRun_running` | Output | 1 | Execution in progress flag |
| `execRun_done` | Output | 1 | Execution completed flag |
| `execRun_limit` | Output | 32 | Maximum time steps (0 = single step) |
| `execRun_ctr` | Output | 32 | Current time-step counter |
| `execRun_timer` | Output | 64 | FPGA clock cycle counter during execution |

### Network Parameters

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `num_inputs` | Output | 17 | Number of input axons (131,072 max) |
| `num_outputs` | Output | 17 | Number of output neurons to monitor |
| `threshold` | Output | 36 | Neuron spike threshold (signed) |
| `exec_neuron_model` | Output | 2 | Neuron model selection (0-3) |

### Debug Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `vio_rx_curr_state` | Output | 3 | RX state machine state (for VIO debugging) |
| `vio_tx_curr_state` | Output | 2 | TX state machine state (for VIO debugging) |

---

## Detailed Logic Description

### Command Opcodes

```verilog
localparam [7:0] CMD_EEP_W        = 8'd1;  // Write axon events to BRAM
localparam [7:0] CMD_HBM_RW       = 8'd2;  // Read/write HBM synapse data
localparam [7:0] CMD_IEP_RW       = 8'd3;  // Read/write neuron state
localparam [7:0] CMD_NTWK_PARAM_W = 8'd4;  // Write network parameters
localparam [7:0] CMD_EXEC_STEP    = 8'd6;  // Execute single time step
localparam [7:0] CMD_EXEC_CONT    = 8'd7;  // Execute continuous (multiple timesteps)
```

Command extracted from: `rx_command = rxFIFO_dout[511:504]`

### RX State Machine

**States:**
```
RX_STATE_RESET                   (3'd0) - Reset state
RX_STATE_IDLE                    (3'd1) - Wait for commands
RX_STATE_REGISTER_PCIE_AXON_DATA (3'd2) - Register 512-bit axon packet
RX_STATE_SET_AXON_DATA           (3'd3) - Shift and distribute axon events
RX_STATE_EXEC_STEP               (3'd4) - Execute time step
RX_STATE_WAIT_RUN                (3'd5) - Wait for execution completion
RX_STATE_EXEC_DONE               (3'd6) - Execution finished
```

**State Transition Diagram:**

```
                   ┌─────────────┐
                   │ RX_RESET    │
                   └──────┬──────┘
                          │
                          v
      ┌───────────────────────────────────────────┐
      │          RX_IDLE                          │
      │   Wait for rxFIFO commands                │
      └──┬──┬──┬──┬──┬──────────────────────────┬─┘
         │  │  │  │  │                          │
    EEP_W│  │  │  │  │EXEC_STEP                │
      ───┼──┼──┼──┼──┼──                        │
         │  │  │  │  │  │                       │
         │  │  │  │  v  v                       │
         │  │  │  │  RX_EXEC_STEP               │
         │  │  │  │      │                      │
         │  │  │  │      v                      │
         │  │  │  │  RX_WAIT_RUN                │
         │  │  │  │   (wait_clks_cnt)           │
         │  │  │  │      │                      │
         │  │  │  │      ├─ ctr<limit ──────────┘
         │  │  │  │      │
         │  │  │  │      └─ ctr==limit
         │  │  │  │            │
         │  │  │  │            v
         │  │  │  │      RX_EXEC_DONE
         │  │  │  │            │
         │  │  │  └────────────┴────────────┐
         │  │  │                            │
         │  │  │HBM_RW / IEP_RW / NTWK_W    │
         │  │  └───(immediate)──────────────┤
         │  │                               │
         │  │EXEC_CONT                      │
         v  v                               v
    RX_REGISTER_PCIE_AXON_DATA         RX_IDLE
         │
         v
    RX_SET_AXON_DATA
      (shift & increment)
         │
         ├─ addr[4:0]==31 ─────> loop to REGISTER
         │
         └─ addr==limit ───┬─ !running ──> RX_IDLE
                           │
                           └─  running ──> RX_EXEC_STEP
```

**Key Logic:**

1. **CMD_EEP_W (Write Axon Events):**
   - Resets `axonEvent_addr` to 0
   - Fetches 512-bit packet into `axon_data_sr`
   - Shifts out 16 bits at a time, incrementing address
   - After every 32 events (one packet), fetches next packet
   - Continues until `axonEvent_addr == num_inputs[16:4]`

2. **CMD_HBM_RW / CMD_IEP_RW:**
   - Direct passthrough to respective FIFOs
   - Waits for FIFO not full before writing
   - Returns to IDLE immediately

3. **CMD_NTWK_PARAM_W:**
   - Writes network parameters from PCIe packet:
     ```
     num_inputs[16:0]        = rxFIFO_dout[16:0]
     num_outputs[16:0]       = rxFIFO_dout[33:17]
     threshold[35:0]         = rxFIFO_dout[69:34]
     exec_neuron_model[1:0]  = rxFIFO_dout[71:70]
     ```

4. **CMD_EXEC_STEP / CMD_EXEC_CONT:**
   - Resets execution counters and timer
   - CMD_EXEC_CONT sets `execRun_limit` from packet `[31:0]`
   - Loads axon events (if EXEC_CONT) then executes
   - Waits for `exec_iep_phase2_done` signal
   - Additional 31-cycle wait to ensure all spikes collected
   - Repeats until `execRun_ctr == execRun_limit`

### TX State Machine

**States:**
```
TX_STATE_RESET           (2'd0) - Reset state
TX_STATE_IDLE            (2'd1) - Check execution status & FIFOs
TX_STATE_WAIT_FOR_SPIKES (2'd2) - Collect spikes during execution
TX_STATE_SEND_SPIKES     (2'd3) - Send batched spikes to host
```

**State Transition Diagram:**

```
         ┌─────────────┐
         │  TX_RESET   │
         └──────┬──────┘
                │
                v
         ┌────────────────────────────────┐
         │  TX_IDLE                       │
         │  Check execution & FIFOs       │
         └──┬──────┬──────────────────────┘
            │      │
execRun     │      │ !execRun & !txFIFO_full
  ──────────┘      └───────┬──────────────
            │              │
            v              ├─ !hbm2ci_empty ──> send HBM data ──┐
   TX_WAIT_FOR_SPIKES      │                                    │
      Collect spikes       └─ !iep2ci_empty ──> send IEP data ─┤
            │                                                   │
            ├─ spike_ctr==14 ──────────────────────────────────┤
            │                                                   │
            ├─ !spk2ciFIFO_empty ─> spike_inc (read & shift)   │
            │                                                   │
            └─ execRun_done ────┬─ spikes_sent ────────────────┤
                                │                               │
                                └─ !spikes_sent                 │
                                        │                       │
                                        v                       │
                                TX_SEND_SPIKES                  │
                                  (batch of 14)                 │
                                        │                       │
                                        └───────────────────────┘
                                                                │
                                                                v
                                                            TX_IDLE
```

**Spike Collection Logic:**

- Spikes stored in 448-bit shift register `spike_sr`
- Counter `spike_ctr` tracks number of spikes (max 14)
- Each spike formatted as 32 bits:
  ```
  [31:24] = execRun_ctr[7:0]  (sub-timestamp)
  [23]    = 1'b1              (valid flag)
  [22:16] = 7'd0              (padding)
  [15:0]  = spk2ciFIFO_dout   (17-bit neuron address truncated?)
  ```
  **Note:** Bit allocation seems inconsistent (17-bit addr in 16 bits + 1 flag?)

- Batch sent when:
  - `spike_ctr == 14`, OR
  - `execRun_done` and spikes pending

- Final packet format (512 bits):
  ```
  [511:480] = 32'hEEEE_EEEE (opcode)
  [479:32]  = spike_sr[447:0] (14 spikes)
  [31:0]    = execRun_ctr (timestamp)
  ```

### Axon Event Shifter

**Purpose:** Convert 512-bit PCIe packets into sequential 16-bit axon events.

**Registers:**
- `axon_data_sr[511:0]` - Shift register holding PCIe packet
- `axonEvent_addr[12:0]` - Current row address (0 to 8191)

**Operation:**

1. **Load:** `axon_data_set` loads `rxFIFO_dout` into `axon_data_sr`
2. **Shift & Increment:** `axon_addr_inc` triggers:
   ```verilog
   axonEvent_addr <= axonEvent_addr + 1'b1;
   axon_data_sr   <= {16'd0, axon_data_sr[511:16]};  // Right shift 16 bits
   ```
3. **Output:** `axonEvent_data = axon_data_sr[15:0]` (LSBs)
4. **Limit:** `axon_addr_limit = num_inputs[16:4]`
   - Divide by 16 since each row handles 16 axons
5. **Reload:** After 32 shifts (one packet exhausted), fetch next packet

**Timeline Example:**
```
Cycle 0: Load packet → axon_data_sr = [511:0]
Cycle 1: Shift #1    → axonEvent_addr=0, data=sr[15:0]
Cycle 2: Shift #2    → axonEvent_addr=1, data=sr[31:16] (original)
...
Cycle 32: Shift #32  → axonEvent_addr=31, data=sr[511:496] (original)
Cycle 33: Load next packet
```

### Execution Control Registers

**Controlled Signals:**

1. **execRun_limit[31:0]:**
   - Set by `exec_run_set` from `rxFIFO_dout[31:0]`
   - Reset by `exec_run_rst` (when not simultaneously setting)
   - Value of 0 means single time step

2. **execRun_ctr[31:0]:**
   - Reset by `exec_run_rst`
   - Incremented by `exec_run_inc`
   - Represents current time step

3. **execRun_timer[63:0]:**
   - Reset by `exec_run_rst`
   - Increments every cycle while `execRun_running==1`
   - Provides FPGA cycle count for performance measurement

4. **execRun_running:**
   - Set by `exec_run_rst`
   - Cleared by `exec_run_done`

5. **execRun_done:**
   - Set by `exec_run_done`
   - Cleared by `exec_run_rst`

**Control Flow:**
```
exec_run_rst  ──┬──> execRun_ctr = 0
                ├──> execRun_timer = 0
                ├──> execRun_running = 1
                └──> execRun_done = 0

exec_run_set  ──> execRun_limit = rxFIFO_dout[31:0]

exec_run_inc  ──> execRun_ctr++

execRun_running ──> execRun_timer++ (every cycle)

exec_run_done ──┬──> execRun_running = 0
                └──> execRun_done = 1
```

### Wait Clock Counter

**Purpose:** Ensure all intermediate spikes have been transmitted before advancing to next time step.

**Register:** `wait_clks_cnt[4:0]`
**Limit:** 31 cycles (5'd31)

**Logic:**
```verilog
if ((rx_curr_state==RX_STATE_WAIT_RUN) &
    exec_iep_phase2_done &
    spk2ciFIFO_empty)
   wait_clks_cnt <= wait_clks_cnt + 1'b1;
else
   wait_clks_cnt <= 5'd0;
```

**Rationale:**
- Spike FIFO controller uses round-robin across 8 FIFOs
- Up to 8 cycles for a spike to propagate to `spk2ciFIFO`
- 31-cycle wait provides safety margin (4× worst case)
- Only starts counting when phase 2 done AND spike FIFO empty

---

## Memory Map

### Network Parameter Registers

These parameters are configured via `CMD_NTWK_PARAM_W` command:

| Register | Bits | Address in PCIe Packet | Description |
|----------|------|------------------------|-------------|
| `num_inputs` | 17 | [16:0] | Number of input axons (0 to 131,071) |
| `num_outputs` | 17 | [33:17] | Number of output neurons to monitor |
| `threshold` | 36 (signed) | [69:34] | Neuron spike threshold |
| `exec_neuron_model` | 2 | [71:70] | Neuron model selection |

**Neuron Model Encoding:**
```
2'b00 = Model 0 (e.g., Leaky Integrate-and-Fire)
2'b01 = Model 1 (e.g., Izhikevich)
2'b10 = Model 2 (e.g., Hodgkin-Huxley approximation)
2'b11 = Model 3 (reserved/custom)
```
*Note: Actual model semantics defined in internal_events_processor*

---

## Timing Diagrams

### CMD_EEP_W: Writing Axon Events

```
Cycle:    0      1      2      3      4      5      6      7      8      9
          │      │      │      │      │      │      │      │      │      │
aclk    ──┘▔▔▔▔▔▔└┐    ┌▔┐    ┌▔┐    ┌▔┐    ┌▔┐    ┌▔┐    ┌▔┐    ┌▔┐    ┌▔
          │      │▔▔▔▔▔▔│▔│▔▔▔▔│▔│▔▔▔▔│▔│▔▔▔▔│▔│▔▔▔▔│▔│▔▔▔▔│▔│▔▔▔▔│▔│▔▔▔▔│
State     IDLE   │REG_D │SET_D │SET_D │SET_D │SET_D │SET_D │SET_D │SET_D │
          │      │      │      │      │      │      │      │      │      │
rxFIFO    ▔▔CMD▔▔▔X▔▔PKT1▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
_dout     │      │      │      │      │      │      │      │      │      │
          │      │      │      │      │      │      │      │      │      │
rxFIFO    ▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
_rden     │      │      │      │      │      │      │      │      │      │
          │      │      │      │      │      │      │      │      │      │
axon_data ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
_set      │      │      │      │      │      │      │      │      │      │
          │      │      │      │      │      │      │      │      │      │
axon_addr ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│
_inc      │      │      │      │      │      │      │      │      │      │
          │      │      │      │      │      │      │      │      │      │
axonEvent ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│
_set      │      │      │      │      │      │      │      │      │      │
          │      │      │      │      │      │      │      │      │      │
axonEvent 0x0000 │0x0000 │D[15:0│D[31:16│D[47:32│D[63:48│D[79:64│D[95:80│
_data     │      │      │]     │]     │]     │]     │]     │]     │
          │      │      │      │      │      │      │      │      │      │
axonEvent 0      │0      │0     │1     │2     │3     │4     │5     │6     │
_addr     │      │      │      │      │      │      │      │      │      │
```

**Notes:**
- IDLE state: Command detected (CMD_EEP_W)
- REG_D state: Register 512-bit PCIe packet
- SET_D state: Shift out 16 bits per cycle, increment address
- After 32 shifts, return to REG_D to fetch next packet

### CMD_EXEC_STEP: Single Time-Step Execution

```
Cycle:     0      1      2      3     ...    N     N+1    N+2   ...  N+32  N+33   N+34
           │      │      │      │      │      │      │      │      │      │      │
State      IDLE   │EXEC_ │WAIT_ │WAIT_ │WAIT_ │WAIT_ │WAIT_ │WAIT_ │WAIT_ │IDLE  │
           │      │STEP  │RUN   │RUN   │RUN   │RUN   │RUN   │RUN   │RUN   │      │
           │      │      │      │      │      │      │      │      │      │      │
exec_run   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
           │      │      │      │      │      │      │      │      │      │      │
execRun    ▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁
_running   │      │      │      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │      │      │      │
exec_iep   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔▁▁▁▁▁▁
_phase2    │      │      │      │      │      │      │      │      │      │      │
_done      │      │      │      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │      │      │      │
spk2ci     ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔│▔▔▔▔▔▔
FIFO_empty │      │      │      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │      │      │      │
wait_clks  0      │0      │0      │0      │0      │0      │1      │2      │31    │0
_cnt       │      │      │      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │      │      │      │
execRun    0      │0      │0      │0      │0      │0      │0      │0      │0      │0
_timer     │      │      │      │      │      │N     │N+1   │N+2   │N+32  │N+32  │
```

**Notes:**
- N cycles: Time for external + internal event processing
- Phase 2 done: Internal events processor completes neuron updates
- 31-cycle wait: Ensures all spikes collected before next time step
- Single time step: `execRun_limit=0`, no increment of `execRun_ctr`

### Spike Batching and Transmission

```
Cycle:     0      1      2      3     ...    14     15     16     17     18
           │      │      │      │      │      │      │      │      │      │
TX State   WAIT   │WAIT  │WAIT  │WAIT  │WAIT  │SEND  │WAIT  │WAIT  │WAIT  │
           _SPIKES│_SPKS │_SPKS │_SPKS │_SPKS │_SPKS │_SPIKES│_SPKS │_SPKS │
           │      │      │      │      │      │      │      │      │      │
spk2ciFIFO ▔▔▔▔▔▔▔│▁▁▁▁▁▁│▔▔▔▔▔▔│▁▁▁▁▁▁│▁▁▁▁▁▁│▁▁▁▁▁▁│▁▁▁▁▁▁│▔▔▔▔▔▔│▁▁▁▁▁▁│
_empty     │      │      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │      │      │
spk2ciFIFO ▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁
_rden      │      │      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │      │      │
spike_inc  ▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁
           │      │      │      │      │      │      │      │      │      │
spike_ctr  0      │1      │1      │2      │2      │14     │0      │1      │1
           │      │      │      │      │      │      │      │      │      │
spike_sr   XXXX   │SPK1  │SPK1  │SPK2  │SPK2  │SPK14 │0     │SPK1' │SPK1' │
           │      │      │      │      │      │      │      │      │      │
txFIFO     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁│▔▔▔▔▔▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
_wren      │      │      │      │      │      │      │      │      │      │
           │      │      │      │      │      │      │      │      │      │
txFIFO     XXXXXX │XXXXXX│XXXXXX│XXXXXX│XXXXXX│0xEEEE│XXXXXX│XXXXXX│XXXXXX│
_din       │      │      │      │      │      │_EEEE │      │      │      │
           │      │      │      │      │      │+14SPK│      │      │      │
```

**Notes:**
- Spikes collected as they arrive (non-empty FIFO)
- After 14 spikes: Transition to SEND_SPIKES
- Packet sent with opcode 0xEEEE_EEEE
- Counter and shift register reset after send
- Process repeats for next batch

---

## Cross-References

### Related Modules

| Module | Relationship | Interface |
|--------|--------------|-----------|
| **pcie2fifos.v** | Upstream | Provides rxFIFO/txFIFO interfaces from PCIe AXI4 |
| **external_events_processor.v** | Downstream | Receives axon events via `axonEvent_*` signals |
| **hbm_processor.v** | Bidirectional | Commands via `ci2hbm_*`, responses via `hbm2ci_*` |
| **internal_events_processor.v** | Bidirectional | Commands via `ci2iep_*`, responses via `iep2ci_*` |
| **spike_fifo_controller.v** | Upstream | Provides spike events via `spk2ciFIFO_*` |

### Software Integration

**Python (hs_bridge) Functions:**

- `fpga_controller.write_axon_events()` → Sends `CMD_EEP_W` commands
- `fpga_controller.read_hbm()` / `write_hbm()` → Sends `CMD_HBM_RW` commands
- `fpga_controller.read_neuron()` / `write_neuron()` → Sends `CMD_IEP_RW` commands
- `fpga_controller.set_network_params()` → Sends `CMD_NTWK_PARAM_W` command
- `fpga_controller.execute_step()` → Sends `CMD_EXEC_STEP` command
- `fpga_controller.execute_continuous()` → Sends `CMD_EXEC_CONT` command
- `fpga_controller.read_spikes()` → Reads TX FIFO for spike packets (opcode 0xEEEE_EEEE)

---

## Key Terms and Definitions

| Term | Definition |
|------|------------|
| **Command Opcode** | 8-bit identifier in PCIe packet `[511:504]` specifying operation type |
| **Axon Event** | External input spike represented as row address + 16-bit mask |
| **Time Step** | One iteration of network simulation (external events → internal updates) |
| **execRun_ctr** | Time-step counter, incremented after each iteration |
| **execRun_timer** | FPGA clock cycle counter for performance profiling |
| **Spike Batching** | Grouping 14 spikes into single 512-bit PCIe packet for efficiency |
| **Sub-timestamp** | Lower 8 bits of `execRun_ctr`, provides intra-timestep spike ordering |
| **Round-Robin** | Fair scheduling used in spike FIFO controller (external to this module) |
| **Wait Clock Counter** | 31-cycle delay ensuring all spikes transmitted before next time step |
| **Shift Register** | `axon_data_sr` and `spike_sr` used for serial data conversion |
| **Phase 2** | Internal events processor phase updating neuron states |
| **FWFT** | First-Word Fall-Through FIFO mode (used in pcie2fifos) |

---

## Design Evolution and Commented Code

### Evidence of Scaling (8 → 16 Neuron Groups)

**Axon Event Data Width:**
```verilog
// OLD (8 groups):  output [7:0] axonEvent_data
// NEW (16 groups): output [15:0] axonEvent_data

// OLD shift: axon_data_sr <= {8'd0, axon_data_sr[511:8]};
// NEW shift: axon_data_sr <= {16'd0, axon_data_sr[511:16]};
```

**Address Calculation:**
```verilog
// OLD: wire [13:0] axon_addr_limit = num_inputs[16:3];  // 8 axons per row
// NEW: wire [12:0] axon_addr_limit = num_inputs[16:4];  // 16 axons per row
```

**Shift Detection:**
```verilog
// OLD: if (axonEvent_addr[5:0]==6'd63)  // 64 x 8-bit events per packet
// NEW: if (axonEvent_addr[4:0]==5'd31)  // 32 x 16-bit events per packet
```

### Neuron Data Width Upgrade

**Internal Events Processor Interface:**
```verilog
// OLD (16-bit neuron data):
// output [1+17+15:0] ci2iep_din   // 34 bits total
// input  [17+15:0]   iep2ci_dout  // 33 bits total

// NEW (36-bit neuron data):
output [1+17+35:0] ci2iep_din   // 54 bits total
input  [17+35:0]   iep2ci_dout  // 53 bits total
```

This suggests upgrade to higher-precision membrane potentials or additional neuron state variables.

---

## Performance Considerations

### Throughput

**Axon Event Loading:**
- 32 events per 512-bit PCIe packet
- 1 event per FPGA cycle (225 MHz)
- 32 cycles to exhaust packet + 1-2 cycles fetch overhead
- **Throughput:** ~6.75M axon events/second per core

**Spike Output:**
- 14 spikes per 512-bit PCIe packet
- Batching amortizes packet overhead
- **Throughput:** Depends on spike rate; typically 1-10% of neurons spike per time step

**Time Step Execution:**
- Variable duration based on:
  - Number of active axons (external events phase)
  - HBM access latency for synapse fetching
  - Number of neurons to update (internal events phase)
  - 31-cycle safety margin for spike collection
- Typical: 1000-10,000 FPGA cycles per time step

### Latency

**Command Response:**
- HBM read: ~100-200 ns (HBM latency + processing)
- Neuron read: ~50-100 ns (URAM access + FIFO transfer)

**Execution Completion:**
- Minimum: ~5 µs (1000 cycles @ 225 MHz)
- Typical: ~20-50 µs depending on network activity

---

## Debugging and Verification

### VIO Signals

```verilog
output [2:0] vio_rx_curr_state,  // Monitor RX state machine
output [1:0] vio_tx_curr_state,  // Monitor TX state machine
```

**State Encodings for Debugging:**

RX States:
```
3'd0 = RESET
3'd1 = IDLE
3'd2 = REGISTER_PCIE_AXON_DATA
3'd3 = SET_AXON_DATA
3'd4 = EXEC_STEP
3'd5 = WAIT_RUN
3'd6 = EXEC_DONE
```

TX States:
```
2'd0 = RESET
2'd1 = IDLE
2'd2 = WAIT_FOR_SPIKES
2'd3 = SEND_SPIKES
```

### Common Debugging Scenarios

**Problem:** Network doesn't execute
- Check: `execRun_running` should assert after `CMD_EXEC_*` command
- Check: `exec_iep_phase2_done` should eventually assert
- Check: `wait_clks_cnt` should count to 31

**Problem:** Spikes not received by host
- Check: `spk2ciFIFO_empty` - should toggle during execution
- Check: `spike_ctr` - should increment when spikes detected
- Check: `txFIFO_wren` - should pulse when batches sent

**Problem:** Axon events not loaded
- Check: `rxFIFO_dout[511:504]` == `CMD_EEP_W` (8'd1)
- Check: `axonEvent_addr` should increment from 0 to `num_inputs[16:4]`
- Check: `axonEvent_set` should pulse for each event

---

## Safety and Edge Cases

### Reset Behavior
- All counters and state machines reset to safe states
- Asynchronous reset (`~aresetn`) ensures immediate response
- Execution flags cleared to prevent spurious runs

### FIFO Full/Empty Handling
- RX state machine waits for `!ci2hbm_full`, `!ci2iep_full` before writing
- TX state machine waits for `!txFIFO_full` before writing
- Spike collection waits for `!spk2ciFIFO_empty` before reading

### Execution Limit Edge Case
- `execRun_limit == 0`: Single time step execution
- Comparison `execRun_ctr == execRun_limit` correctly handles both cases

### Axon Address Limit
- Prevents writing beyond allocated BRAM rows
- Correctly handles non-multiple-of-16 input counts (rounds down via `[16:4]`)

---

## Future Enhancement Opportunities

1. **Pipelined Command Processing:** Currently processes one command at a time; could overlap execution with data loading

2. **Variable Spike Batch Size:** Fixed 14-spike batches may be inefficient for low spike rates; adaptive sizing could reduce latency

3. **Compression:** Sparse spike patterns could benefit from run-length encoding or similar compression

4. **Multi-Core Coordination:** For N_cores implementation, add inter-core communication commands

5. **Error Reporting:** Add status/error codes to TX packets for invalid commands or failed operations

6. **Performance Counters:** Instrument critical paths (HBM access time, execution phase durations) for profiling

---

**Document Version:** 1.0
**Last Updated:** December 2025
**Module File:** `command_interpreter.v`
**Module Location:** `CRI_proj/cri_fpga/code/new/hyddenn2/vivado/single_core.srcs/sources_1/new/`
