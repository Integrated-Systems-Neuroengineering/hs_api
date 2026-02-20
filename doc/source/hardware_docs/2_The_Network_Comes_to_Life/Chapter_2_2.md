# 2.2 State Machine Coordination and Control Signals

## Introduction

The FPGA neuromorphic system executes neural network timesteps through the coordinated operation of **four independent state machines** across multiple Verilog modules. Understanding how these state machines communicate and synchronize is key to understanding how the hardware processes spiking neural networks in real-time.

### The Four State Machines

1. **Command Interpreter (CI)** - The Orchestrator
   - **Module:** `command_interpreter.v`
   - **Role:** Receives commands from the host PC, initiates timesteps, coordinates all other modules, batches output spikes
   - **State Machines:** RX (receive commands) and TX (send responses)
   - **Analogy:** The conductor of an orchestra, setting tempo and cuing sections

2. **HBM Processor** - The Memory Access Engine
   - **Module:** `hbm_processor.v`
   - **Role:** Reads connectivity pointers and synaptic data from High Bandwidth Memory, routes spikes
   - **State Machines:** TX (send memory read commands) and RX (receive and distribute data)
   - **Analogy:** The database engine that fetches network structure and weights

3. **External Events Processor (EEP)** - The Input Handler
   - **Module:** `external_events_processor.v`
   - **Role:** Manages external spike inputs in BRAM, streams them to HBM processor during Phase 1
   - **State Machine:** Pipeline controller for BRAM read/write/clear operations
   - **Analogy:** The input buffer that receives and distributes external stimuli

4. **Internal Events Processor (IEP)** - The Neuron State Manager
   - **Module:** `internal_events_processor.v`
   - **Role:** Stores neuron membrane potentials in URAM, accumulates synaptic inputs, generates spikes
   - **State Machine:** Pipeline controller for URAM read/modify/write operations
   - **Analogy:** The computational core that updates neuron states

### Why Coordination Is Critical

Each state machine runs independently but must synchronize at specific points because:

- **Pipeline Latency:** BRAM has 3-cycle read latency, URAM has 3-cycle read latency, HBM has 100+ cycle latency
- **Data Dependencies:** Phase 2 cannot start until Phase 1 pointers are collected
- **Resource Sharing:** Multiple modules access shared resources (HBM, spike FIFOs)
- **Timing Constraints:** All must complete before the next timestep can begin

The coordination is achieved through **handshake signals** (like `exec_bram_phase1_ready`, `exec_hbm_rx_phase1_done`) that allow modules to signal their status and wait for others.

---

## Part 1: Conceptual Overview

### The Symphony Orchestra Analogy

Think of timestep execution like a symphony orchestra performance:

- **Conductor (Command Interpreter):** Raises baton (`exec_run` pulse), cues sections, keeps time
- **String Section (External Events Processor):** Starts immediately, fills the air with melody (streams input spikes)
- **Brass Section (HBM Processor):** Waits for strings to establish rhythm, then adds harmony (fetches pointers, then synapses)
- **Percussion (Internal Events Processor):** Listens to all sections, adds accents (updates neurons, generates spikes)
- **Stage Manager (FIFOs):** Passes music sheets between sections (pointer FIFOs, spike FIFOs)

The conductor must wait for all sections to finish their parts before starting the next movement (timestep).

### Visual Timeline of a Complete Timestep

```
CYCLE 0: exec_run pulse ━━━━━┓ (Conductor raises baton)
                             ┃
         ┏━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┓
         ┃                   ┃                    ┃
         ▼ EEP               ▼ IEP                ▼ HBM
    Fill BRAM pipe      Fill URAM pipe      Wait for sections
    (3 cycles)          (3 cycles)          to signal ready
         │                   │                    │
         ▼                   ▼                    ▼
    exec_bram_          exec_uram_          RX waits for
    phase1_ready        phase1_ready        exec_bram_phase1_ready
         │                   │                    │
         │                   │                    ▼
         │                   │               TX sends input
         │                   │               pointer read cmds
         │                   │                    │
         ├───────────────────┴────────────────────┤
         │                                        │
         │         PHASE 1: POINTER COLLECTION    │
         │         (~200-500 cycles)              │
         │                                        │
         │  EEP: Stream input spikes ────────────>│
         │  IEP: Stream neuron pointers ─────────>│ HBM: Collect
         │                                        │ pointers in
         │                                        │ ptrFIFO
         │                                        │
         ├────────────────────────────────────────┤
         │                                        ▼
         │                                   rx_phase1_done
         │                                        │
         │         PHASE 2: SYNAPSE PROCESSING    │
         │         (~300-1000 cycles)             │
         │                                        │
         │                         HBM TX: Pop ptrFIFO,
         │                         send synapse read cmds
         │                                 │
         │                                 ▼
         │                         HBM RX: Receive
         │                         synapse data
         │                                 │
         │                     ┌───────────┴──────────┐
         │                     ▼                      ▼
         │              IEP: Accumulate         Spike FIFOs:
         │              synaptic inputs         Collect output
         │              Update neurons          spikes
         │              Generate spikes              │
         │                     │                     │
         ▼                     ▼                     ▼
    exec_bram_          exec_iep_              CI TX: Batch
    phase1_done         phase2_done            spikes, send
                                               to host
                             │
                             ▼
                        Wait 31 cycles
                        (safety margin)
                             │
                             ▼
                        execRun_ctr++
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
          More timesteps?            All done
          Loop to CYCLE 0            exec_run_done=1
```

---

### Phase 1: Pointer Collection (Conceptual Flow)

**Goal:** Identify which neurons (input and internal) have connectivity data to fetch, collect their HBM addresses.

#### Step 1: Initiation (Cycles 0-3)
```
Command Interpreter:
  - Receives exec_run pulse (from CMD_EXEC_STEP or CMD_EXEC_CONT command)
  - Sets execRun_running = 1
  - Starts performance timer (execRun_timer++)

External Events Processor (EEP):
  - Toggles bram_select (swap present/future buffers)
  - Enters FILL_PIPE state
  - Issues 3 consecutive BRAM reads (addresses 0, 1, 2)
  - Fills 3-stage pipeline

Internal Events Processor (IEP):
  - Resets uram_raddr = 0
  - Enters FILL_PIPE_PHASE1 state
  - Issues 3 consecutive URAM reads (addresses 0, 1, 2)
  - Fills 3-stage pipeline

HBM Processor:
  - TX clears completion flags (tx_done_rst = 1)
  - TX enters STATE_SEND_INPUT_READ_COMMANDS
  - RX enters STATE_WAIT_BRAM_PIPELINE
  - Both wait for pipeline ready signals
```

**Why 3 cycles?** BRAM and URAM have 3-cycle read latency. The pipeline must be "primed" before continuous reading can begin.

#### Step 2: BRAM Pipeline Ready (Cycle 4+)
```
EEP:
  - Pipeline filled, data from address 0 now available
  - Sets exec_bram_phase1_ready = 1
  - Enters STATE_READ_INPUTS (continuous streaming mode)

HBM RX:
  - Sees exec_bram_phase1_ready signal
  - Enters STATE_READ_INPUT_POINTERS
  - Ready to receive pointer data from HBM

HBM TX:
  - Already sending read commands to HBM
  - Address format: {8'd0, 1'b0, tx_addr[9:0], 4'd0, 5'd0}
  - tx_addr increments from 0 to INPUT_ADDR_LIMIT
  - Burst length = 15 (reads 16 consecutive 256-bit words per address)
```

**Synchronized Streaming:** EEP reads next BRAM address only when `exec_hbm_rvalidready = 1`, ensuring HBM keeps up.

#### Step 3: Input Pointer Collection (Cycles 5-200+)
```
Parallel Operation:

EEP (Foreground):
  - bramPresent_rden = 1 (when HBM signals ready)
  - Reads neuron group from BRAM
  - Outputs exec_bram_spiked (8-bit mask showing which groups spiked)
  - Sends to pointer_fifo_controller
  - Simultaneously: bramPresent_wren = 1 (clears read data to zero)

HBM TX (Sending):
  - Loops through tx_addr = 0 to INPUT_ADDR_LIMIT
  - Sends burst read requests to HBM
  - Each request fetches 16 consecutive pointer words

HBM RX (Receiving):
  - Receives 256-bit pointer words from HBM
  - Each word contains 8 pointers: [31:23]=length, [22:0]=address
  - Forwards to pointer_fifo_controller
  - pointer_fifo_controller demuxes to 16 ptrFIFOs based on spike mask

Completion:
  - When bramPresent_waddr == BRAM_ADDR_LIMIT
  - EEP sets exec_bram_phase1_done = 1
```

**Why Demuxing?** The 16 ptrFIFOs correspond to the 16 neuron groups (URAM banks). Each FIFO receives only the pointers for neurons in its group that spiked.

#### Step 4: URAM Pipeline Ready (After BRAM Complete)
```
IEP:
  - Waits for exec_bram_phase1_done = 1
  - Transitions to STATE_PUSH_PTR_FIFO
  - Continues incrementing uram_raddr
  - Sets exec_uram_phase1_ready = 1
  - Outputs neuron pointers for internal neurons that have connections

HBM RX:
  - Enters STATE_WAIT_URAM_PIPELINE
  - Waits for exec_uram_phase1_ready signal
  - Then enters STATE_READ_OUTPUT_POINTERS

HBM TX:
  - When tx_addr == INPUT_ADDR_LIMIT (inputs complete)
  - Sets tx_select_inc = 1 (switch from inputs to outputs)
  - Enters STATE_SEND_OUTPUT_READ_COMMANDS
  - Address format changes: {8'd0, 1'b1, tx_addr[9:0], 4'd0, 5'd0}
  - tx_select bit [18] = 1 indicates internal neurons
```

#### Step 5: Output Pointer Collection (Cycles 200-500)
```
Similar to input pointer collection:

IEP (Streaming):
  - Reads from all 16 URAM banks in parallel
  - For each neuron: checks if it has outgoing connections
  - Outputs pointer addresses to HBM processor

HBM TX:
  - Loops through output neuron groups
  - Sends burst read requests for output pointers

HBM RX:
  - Receives output pointer data
  - Forwards to pointer_fifo_controller
  - Demuxes to 16 ptrFIFOs (now containing both input and output pointers)

Completion:
  - When rx_addr == {OUTPUT_ADDR_LIMIT, OUTPUT_ADDR_MOD}
  - RX sets rx_phase1_done = 1
  - When tx_addr == OUTPUT_ADDR_LIMIT
  - TX sets tx_phase1_done = 1
  - TX toggles tx_phase (0 → 1, entering Phase 2)
```

**Result of Phase 1:** The 16 ptrFIFOs now contain all HBM addresses for synaptic data that needs to be fetched.

---

### Phase 2: Synapse Processing (Conceptual Flow)

**Goal:** Fetch synaptic data from HBM, accumulate inputs in neurons, detect threshold crossings, generate output spikes.

#### Step 1: Phase Transition (Cycle ~500)
```
HBM TX:
  - Enters STATE_POP_POINTER_FIFO
  - Waits for rx_phase1_done = 1 (already set)
  - Waits for ptrFIFO not empty (OR safety timer expires)

HBM RX:
  - Enters STATE_PHASE1_DONE briefly
  - Then STATE_READ_SYNAPSE_DATA
  - Sets rx_phase1_done = 1 (enables TX to proceed)

IEP:
  - Transitions to Phase 2 processing state
  - Ready to receive synaptic inputs from HBM
```

#### Step 2: Pointer-Driven Synapse Fetching (Cycles 500-1500)
```
HBM TX (Loop):
  For each non-empty ptrFIFO:
    1. ptrFIFO_rden = 1 (pop next pointer)
    2. ptr_addr_set = 1 (load pointer into registers)
       - ptr_addr = pointer.base_address [22:0]
       - ptr_len = pointer.length [8:0] (number of synapse rows)
       - ptr_ctr = 0 (progress counter)

    3. Calculate burst length:
       - ptr_burst = (remaining synapse rows < 16) ? remainder : 15

    4. Send HBM read command:
       - hbm_arvalid = 1
       - hbm_araddr = {5'd0, ptr_addr[22:0], 5'd0}
       - hbm_arlen = ptr_burst (0-15)

    5. When HBM accepts (hbm_arready = 1):
       - ptr_addr_inc = 1
       - ptr_addr += (ptr_burst + 1)
       - ptr_ctr += (ptr_burst + 1)

    6. If ptr_ctr[8:4] == ptr_len[8:4]:
       - Current pointer exhausted
       - Return to step 1 (get next pointer)

  Continue until:
    - All ptrFIFOs empty
    - Safety timer (wait_clks_cnt = 255) expires

HBM TX Completion:
  - Sets tx_phase2_done = 1
  - tx_ptr_ctr = total 256-bit words requested
```

**Why Round-Robin?** The pointer_fifo_controller uses round-robin arbitration to fairly select from the 16 ptrFIFOs, ensuring all neuron groups get processed.

#### Step 3: Synapse Data Reception and Distribution (Parallel)
```
HBM RX:
  - hbm_rready = 1 (always accept synapse data)
  - When hbm_rvalid = 1 (data arrives from HBM):

    Step 3a: Assemble 512-bit packets
      - 1st 256-bit word: Store in hbm_rdata_lower
      - 2nd 256-bit word: Current hbm_rdata
      - Combined = 512-bit exec_hbm_rdata
      - Set hbm_count = 1 (second word marker)
      - Assert exec_hbm_rvalidready = 1 (packet ready)

    Step 3b: Route to IEP
      - IEP sees exec_hbm_rvalidready = 1
      - Reads exec_hbm_rdata (512 bits = 16 groups × 32 bits each)
      - Extracts 16 synaptic input values
      - For each group with input ≠ 0:
          • Read current neuron potential from URAM
          • Add synaptic input: new_V = old_V + input
          • Apply neuron model (leak, reset, etc.)
          • Check: if new_V > threshold → spike!
          • Write updated potential back to URAM
          • Set bit in exec_uram_spiked if spiked

    Step 3c: Route to Spike FIFOs
      - Extract 8 synapse entries from 256-bit word
      - For synapse 0-7:
          • Check spike_flag = hbm_rdata[N*32 + 31]
          • If flag = 1:
            - Extract neuron_address = hbm_rdata[N*32 + 16:N*32]
            - Route to spike FIFO based on address[2:0]
            - spkN_wren = 1, spkN_din = neuron_address

    Step 3d: Count progress
      - rx_ptr_ctr++ (count 256-bit words received)

HBM RX Completion:
  - When tx_phase2_done = 1 AND rx_ptr_ctr == tx_ptr_ctr
  - All synapse data received and processed
  - Sets rx_phase2_done = 1
```

**Dual Distribution:** Each 256-bit HBM word is used for TWO purposes:
1. Synaptic inputs routed to IEP (for neuron updates)
2. Spike flags routed to spike FIFOs (for output collection)

#### Step 4: Spike Output Aggregation (Parallel with Step 3)
```
Spike FIFO Controller:
  - Receives spikes from 8 parallel spike FIFOs (spk0-7)
  - Round-robin reads from non-empty FIFOs
  - Forwards to spk2ciFIFO (single aggregated FIFO)

Command Interpreter TX:
  - State: TX_STATE_WAIT_FOR_SPIKES
  - While execRun_running = 1:
      • spk2ciFIFO_rden = 1 (when spike available)
      • spike_inc = 1 (add to batch)
      • Batch size = 14 spikes

  - When spike_ctr == 14:
      • Send 512-bit packet to host:
        [511:480] = 0xEEEEEEEE (opcode)
        [479:32]  = 14 × 32-bit spike events
        [31:0]    = execRun_ctr (timestamp)
      • spike_rst = 1 (clear batch)
```

#### Step 5: Phase 2 Completion and Timestep Finalization
```
IEP:
  - When exec_hbm_rx_phase2_done = 1
  - Sets exec_iep_phase2_done = 1
  - Neuron updates complete

Command Interpreter RX:
  - State: RX_STATE_WAIT_RUN
  - Checks completion conditions:
      • exec_iep_phase2_done = 1
      • spk2ciFIFO_empty = 1 (all spikes drained)
      • wait_clks_cnt == 31 (safety margin)

  - When all satisfied:
      • exec_run_inc = 1
      • execRun_ctr++ (increment timestep counter)

      • If execRun_ctr < execRun_limit:
        - Load next timestep input data
        - Loop back to CYCLE 0

      • Else:
        - exec_run_done = 1
        - execRun_running = 0
        - All timesteps complete!
```

**Safety Margin:** The 31-cycle wait ensures all spikes have propagated through the round-robin arbiters and FIFOs before starting the next timestep.

---

## Part 2: Technical Reference

### Execution Control Signals Reference

Complete table of all `exec_*` signals that coordinate the state machines:

| Signal | Width | Origin | Destination(s) | Purpose | Asserted When | Cleared When |
|--------|-------|--------|----------------|---------|---------------|--------------|
| **`exec_run`** | 1 bit | Command Interpreter RX | EEP, IEP, HBM TX/RX | Initiate new timestep | CI parses CMD_EXEC_STEP or starts timestep in CMD_EXEC_CONT | Automatically (pulse signal) |
| **`exec_bram_phase1_ready`** | 1 bit | External Events Processor | HBM Processor RX | BRAM pipeline filled, ready to stream | EEP enters STATE_READ_INPUTS (after FILL_PIPE) | EEP enters STATE_FILL_PIPE (next timestep) |
| **`exec_bram_phase1_done`** | 1 bit | External Events Processor | Internal Events Processor | External spike processing complete | bramPresent_waddr == BRAM_ADDR_LIMIT | EEP enters STATE_FILL_PIPE |
| **`exec_uram_phase1_ready`** | 1 bit | Internal Events Processor | HBM Processor RX | URAM pipeline filled, ready to output | IEP enters STATE_PUSH_PTR_FIFO | IEP enters FILL_PIPE_PHASE1 |
| **`exec_uram_phase1_done`** | 1 bit | Internal Events Processor | HBM Processor TX | Internal neuron pointer output complete | uram_raddr == URAM_ADDR_LIMIT | Next exec_run |
| **`exec_hbm_tx_phase1_done`** | 1 bit | HBM Processor TX | HBM Processor RX (internal) | All pointer read commands sent | tx_curr_state == TX_STATE_PHASE1_DONE | tx_done_rst on next exec_run |
| **`exec_hbm_tx_phase2_done`** | 1 bit | HBM Processor TX | HBM Processor RX | All synapse read commands sent | tx_curr_state == TX_STATE_PHASE2_DONE | tx_done_rst on next exec_run |
| **`exec_hbm_rx_phase1_done`** | 1 bit | HBM Processor RX | HBM Processor TX, Spike routing logic | All pointers received from HBM | rx_next_state == RX_STATE_READ_SYNAPSE_DATA | rx_done_rst on next exec_run |
| **`exec_hbm_rx_phase2_done`** | 1 bit | HBM Processor RX | Internal Events Processor, CI TX | All synapse data received and processed | rx_curr_state == RX_STATE_PHASE2_DONE | rx_done_rst on next exec_run |
| **`exec_hbm_rvalidready`** | 1 bit | HBM Processor RX | EEP, IEP | HBM data valid (512-bit packet complete) | hbm_rvalid & hbm_rready & hbm_count & ~hbmFIFO_full | Every other 256-bit HBM read |
| **`exec_hbm_rdata`** | 512 bits | HBM Processor RX | Internal Events Processor | Synaptic input data (2 × 256-bit HBM words) | When exec_hbm_rvalidready = 1 | Continuous data stream |
| **`exec_uram_spiked`** | 16 bits | Internal Events Processor | Pointer FIFO Controller | Bit mask of neuron groups that spiked | Any neuron in group exceeds threshold | Cleared each timestep |
| **`exec_bram_spiked`** | 8 bits | External Events Processor | Pointer FIFO Controller | Spike mask from BRAM (8 groups) | Read from bramPresent_rdata | Continuous stream during Phase 1 |
| **`exec_iep_phase2_done`** | 1 bit | Internal Events Processor | Command Interpreter RX | Neuron updates complete | IEP finishes processing all synaptic inputs | Next exec_run |
| **`execRun_running`** | 1 bit | Command Interpreter RX | Command Interpreter TX | Execution active flag | exec_run_rst in CI RX | exec_run_done in CI RX |
| **`execRun_done`** | 1 bit | Command Interpreter RX | Command Interpreter TX | All timesteps complete | execRun_ctr == execRun_limit | Next execution sequence |
| **`execRun_ctr`** | 32 bits | Command Interpreter RX | Command Interpreter TX | Current timestep number | exec_run_inc after each timestep | exec_run_rst at start |
| **`execRun_limit`** | 32 bits | Command Interpreter RX | Command Interpreter RX | Target number of timesteps | exec_run_set from CMD_EXEC_CONT | Next execution |
| **`execRun_timer`** | 64 bits | Command Interpreter RX | Debug/monitoring | Performance counter (clock cycles) | Every cycle while execRun_running = 1 | exec_run_rst at start |

---

### Module-by-Module State Machine Details

#### 1. Command Interpreter - The Orchestrator

**File:** `command_interpreter.v`

##### RX State Machine (Command Processing and Timestep Control)

**States:**
```
RX_STATE_RESET (0)                   → Initialization
RX_STATE_IDLE (1)                    → Waiting for command from rxFIFO
RX_STATE_REGISTER_PCIE_AXON_DATA (2) → Loading external event data
RX_STATE_SET_AXON_DATA (3)           → Unpacking and sending to EEP
RX_STATE_EXEC_STEP (4)               → Pulse exec_run signal
RX_STATE_WAIT_RUN (5)                → Wait for timestep completion
RX_STATE_EXEC_DONE (6)               → Execution finished
```

**Key Transitions:**
```
IDLE → REGISTER_PCIE_AXON_DATA:
  When: rx_command == CMD_EEP_W
  Action: axon_addr_rst = 1, rxFIFO_rden = 1

IDLE → EXEC_STEP:
  When: rx_command == CMD_EXEC_STEP
  Action: axon_addr_rst = 1, exec_run_rst = 1, rxFIFO_rden = 1

IDLE → REGISTER_PCIE_AXON_DATA:
  When: rx_command == CMD_EXEC_CONT
  Action: axon_addr_rst = 1, exec_run_rst = 1, exec_run_set = 1, rxFIFO_rden = 1

EXEC_STEP → WAIT_RUN:
  When: Always (after pulsing exec_run = 1)
  Action: exec_run = 1 (pulse)

WAIT_RUN → EXEC_DONE:
  When: wait_clks_cnt == 31 AND execRun_ctr == execRun_limit
  Action: exec_run_done = 1

WAIT_RUN → REGISTER_PCIE_AXON_DATA:
  When: wait_clks_cnt == 31 AND execRun_ctr < execRun_limit
  Action: exec_run_inc = 1, axon_addr_rst = 1

EXEC_DONE → IDLE:
  When: Always
  Action: exec_run_done = 1 (latched)
```

**Key Variables:**

| Variable | Width | Purpose | Update Condition |
|----------|-------|---------|------------------|
| `execRun_ctr` | 32 bits | Current timestep (0 to limit) | exec_run_inc = 1 |
| `execRun_limit` | 32 bits | Total timesteps to execute | exec_run_set = 1, loaded from rxFIFO_dout[31:0] |
| `execRun_timer` | 64 bits | Performance counter (cycles) | Every cycle while execRun_running = 1 |
| `execRun_running` | 1 bit | Execution active flag | Set by exec_run_rst, cleared by exec_run_done |
| `execRun_done` | 1 bit | Completion flag | Set when execRun_ctr == execRun_limit |
| `wait_clks_cnt` | 5 bits | Safety margin counter (0-31) | Increments in WAIT_RUN when conditions met |
| `axonEvent_addr` | 13 bits | Current axon row address | Increments in SET_AXON_DATA state |
| `axon_data_sr` | 512 bits | Shift register for axon data | Loaded in REGISTER state, shifted in SET state |

**Wait Conditions in WAIT_RUN:**
```verilog
if (wait_clks_cnt == wait_clks_limit) begin
    // Safety margin reached (31 cycles)
    if (execRun_ctr == execRun_limit)
        // All timesteps complete
        next_state = EXEC_DONE;
    else
        // More timesteps to process
        exec_run_inc = 1;
        next_state = REGISTER_PCIE_AXON_DATA;
end

// Increment counter when:
if ((curr_state==WAIT_RUN) & exec_iep_phase2_done & spk2ciFIFO_empty)
    wait_clks_cnt++;
else
    wait_clks_cnt = 0;
```

##### TX State Machine (Response and Spike Batching)

**States:**
```
TX_STATE_RESET (0)           → Initialization
TX_STATE_IDLE (1)            → Check for data to send
TX_STATE_WAIT_FOR_SPIKES (2) → Accumulate spike batch
TX_STATE_SEND_SPIKES (3)     → Send 512-bit spike packet
```

**Key Transitions:**
```
IDLE → WAIT_FOR_SPIKES:
  When: execRun_running = 1
  Action: spike_rst = 1 (reset batch counter)

WAIT_FOR_SPIKES → SEND_SPIKES:
  When: spike_ctr == 14 (batch full)
  Action: Prepare 512-bit packet

WAIT_FOR_SPIKES → IDLE:
  When: execRun_done = 1 AND spikes_sent = 1
  Action: All spikes transmitted

WAIT_FOR_SPIKES → SEND_SPIKES:
  When: execRun_done = 1 AND spikes_sent = 0
  Action: Send final partial batch

SEND_SPIKES → WAIT_FOR_SPIKES:
  When: !txFIFO_full
  Action: txFIFO_wren = 1, spike_rst = 1
```

**Key Variables:**

| Variable | Width | Purpose | Update Condition |
|----------|-------|---------|------------------|
| `spike_sr` | 448 bits | Spike batch shift register (14 × 32 bits) | spike_inc = 1, shifted left |
| `spike_ctr` | 4 bits | Current batch size (0-14) | Increments with spike_inc |
| `spikes_sent` | 1 bit | Final batch sent flag | Cleared by spike_inc, set by spike_rst |

**Spike Packet Format (512 bits):**
```
[511:480] = 32'hEEEE_EEEE       // Opcode (identifies spike packet)
[479:32]  = 448-bit spike batch // 14 × 32-bit spike events
[31:0]    = execRun_ctr         // Timestamp (current timestep number)

Each 32-bit spike event:
[31:24] = execRun_ctr[7:0]      // Sub-timestamp (lower 8 bits of timestep)
[23:17] = 7'd0                  // Padding
[16:0]  = neuron_address        // Which neuron spiked
```

---

#### 2. HBM Processor - The Memory Access Engine

**File:** `hbm_processor.v`

##### TX State Machine (Memory Read Command Generator)

**States:**
```
TX_STATE_RESET (0)                          → Initialization
TX_STATE_IDLE (1)                           → Wait for work
TX_STATE_SEND_INPUT_READ_COMMANDS (2)       → Phase 1a: Read input pointers
TX_STATE_SEND_OUTPUT_READ_COMMANDS (3)      → Phase 1b: Read output pointers
TX_STATE_PHASE1_DONE (4)                    → Phase 1 complete, prepare Phase 2
TX_STATE_POP_POINTER_FIFO (5)               → Phase 2: Get next pointer
TX_STATE_SEND_POINTER_READ_COMMANDS (6)     → Phase 2: Read synapse data
TX_STATE_PHASE2_DONE (7)                    → Phase 2 complete
TX_STATE_READ_HBM_ADDR (8)                  → Host read command
TX_STATE_WRITE_HBM_ADDR (9)                 → Host write address phase
TX_STATE_WRITE_HBM_DATA (10)                → Host write data phase
TX_STATE_WRITE_HBM_RESP (11)                → Host write response phase
```

**Key Transitions (Execution Path):**
```
IDLE → SEND_INPUT_READ_COMMANDS:
  When: exec_run = 1
  Action: tx_done_rst = 1, tx_addr_rst = 1

SEND_INPUT_READ_COMMANDS → SEND_OUTPUT_READ_COMMANDS:
  When: hbm_arready = 1 AND tx_addr == INPUT_ADDR_LIMIT
  Action: tx_addr_inc = 1, tx_addr_rst = 1, tx_select_inc = 1

SEND_OUTPUT_READ_COMMANDS → PHASE1_DONE:
  When: hbm_arready = 1 AND tx_addr == OUTPUT_ADDR_LIMIT
  Action: tx_addr_inc = 1

PHASE1_DONE → POP_POINTER_FIFO:
  When: Always
  Action: tx_phase_inc = 1 (toggle to Phase 2), tx_ptr_ctr_rst = 1

POP_POINTER_FIFO → SEND_POINTER_READ_COMMANDS:
  When: !ptrFIFO_empty AND rx_phase1_done = 1
  Action: ptr_addr_set = 1, ptrFIFO_rden = 1

POP_POINTER_FIFO → PHASE2_DONE:
  When: wait_clks_cnt == 255 AND ptrFIFO_empty
  Action: Safety timer expired, all pointers processed

SEND_POINTER_READ_COMMANDS → POP_POINTER_FIFO:
  When: hbm_arready = 1 AND ptr_ctr[8:4] == ptr_len[8:4]
  Action: ptr_addr_inc = 1 (current pointer exhausted)

PHASE2_DONE → IDLE:
  When: Always
  Action: tx_phase_inc = 1 (toggle back to Phase 1)
```

**Key Variables:**

| Variable | Width | Purpose | Update Condition | Range |
|----------|-------|---------|------------------|-------|
| `tx_phase` | 1 bit | Phase selector (0=pointers, 1=synapses) | tx_phase_inc = 1 | 0-1 |
| `tx_select` | 1 bit | Phase 1: input (0) or output (1) | tx_select_inc = 1 | 0-1 |
| `tx_addr` | 10 bits | Phase 1: group address counter | tx_addr_inc = 1 | 0-1023 |
| `ptr_addr` | 23 bits | Phase 2: synapse block address | ptr_addr_inc, ptr_addr_set | 0-8M |
| `ptr_len` | 9 bits | Phase 2: synapses in current pointer | ptr_addr_set (from ptrFIFO) | 0-511 |
| `ptr_ctr` | 9 bits | Phase 2: progress within pointer | ptr_addr_inc | 0-511 |
| `ptr_burst` | 4 bits | Phase 2: current burst length | Calculated | 0-15 |
| `tx_ptr_ctr` | 23 bits | Phase 2: total synapse blocks sent | ptr_addr_inc | 0-8M |
| `tx_phase1_done` | 1 bit | Phase 1 commands complete flag | State = PHASE1_DONE | 0-1 |
| `tx_phase2_done` | 1 bit | Phase 2 commands complete flag | State = PHASE2_DONE | 0-1 |

**HBM Read Address Calculation:**

```verilog
// Phase 1 (Pointer Reads):
if (~tx_phase) begin
    hbm_araddr = {5'd0, {8'd0, tx_select, tx_addr, 4'd0}, 5'd0};
    // Breakdown:
    // [32:28] = 5'd0           (upper padding)
    // [27:20] = 8'd0           (padding)
    // [19]    = tx_select      (0=inputs, 1=outputs)
    // [18:9]  = tx_addr[9:0]   (group address)
    // [8:5]   = 4'd0           (within-group offset)
    // [4:0]   = 5'd0           (32-byte alignment)
end

// Phase 2 (Synapse Reads):
else begin
    hbm_araddr = {5'd0, ptr_addr[22:0], 5'd0};
    // Breakdown:
    // [32:28] = 5'd0           (upper padding)
    // [27:5]  = ptr_addr       (synapse block address)
    // [4:0]   = 5'd0           (32-byte alignment)
end
```

**Burst Length Calculation:**

```verilog
// Phase 1:
if (~tx_phase) begin
    if (~tx_select) begin
        // Reading input pointers
        if (tx_addr == INPUT_ADDR_LIMIT)
            hbm_arlen = INPUT_ADDR_MOD;  // Last burst (partial)
        else
            hbm_arlen = 4'hF;  // 15 = 16 words (full burst)
    end else begin
        // Reading output pointers
        if (tx_addr == OUTPUT_ADDR_LIMIT)
            hbm_arlen = OUTPUT_ADDR_MOD;
        else
            hbm_arlen = 4'hF;
    end
end

// Phase 2:
else begin
    hbm_arlen = ptr_burst;
    // Where ptr_burst = (ptr_ctr[8:4] == ptr_len[8:4]) ? ptr_len[3:0] : 4'hf
    // Meaning: Last burst gets remainder, others get 15 (16 words)
end
```

##### RX State Machine (Memory Read Response Handler)

**States:**
```
RX_STATE_RESET (0)                → Initialization
RX_STATE_IDLE (1)                 → Wait for work
RX_STATE_WAIT_BRAM_PIPELINE (2)   → Wait for EEP ready
RX_STATE_READ_INPUT_POINTERS (3)  → Phase 1a: Receive input pointers
RX_STATE_WAIT_URAM_PIPELINE (4)   → Wait for IEP ready
RX_STATE_READ_OUTPUT_POINTERS (5) → Phase 1b: Receive output pointers
RX_STATE_PHASE1_DONE (6)          → Phase 1 complete
RX_STATE_READ_SYNAPSE_DATA (7)    → Phase 2: Receive synapse data
RX_STATE_PHASE2_DONE (8)          → Phase 2 complete
RX_STATE_READ_HBM_RESP (9)        → Host read response
```

**Key Transitions:**
```
IDLE → WAIT_BRAM_PIPELINE:
  When: exec_run = 1
  Action: rx_done_rst = 1, rx_addr_rst = 1

WAIT_BRAM_PIPELINE → READ_INPUT_POINTERS:
  When: exec_bram_phase1_ready = 1
  Action: None (just wait for EEP pipeline to fill)

READ_INPUT_POINTERS → WAIT_URAM_PIPELINE:
  When: hbm_rvalid = 1 AND rx_addr == {INPUT_ADDR_LIMIT, INPUT_ADDR_MOD}
  Action: rx_addr_inc = 1, rx_addr_rst = 1

WAIT_URAM_PIPELINE → READ_OUTPUT_POINTERS:
  When: exec_uram_phase1_ready = 1
  Action: None (wait for IEP pipeline to fill)

READ_OUTPUT_POINTERS → PHASE1_DONE:
  When: hbm_rvalid = 1 AND rx_addr == {OUTPUT_ADDR_LIMIT, OUTPUT_ADDR_MOD}
  Action: rx_addr_inc = 1

PHASE1_DONE → READ_SYNAPSE_DATA:
  When: Always
  Action: rx_phase1_done = 1

READ_SYNAPSE_DATA → PHASE2_DONE:
  When: tx_phase2_done = 1 AND rx_ptr_ctr == tx_ptr_ctr
  Action: All synapse data received

PHASE2_DONE → IDLE:
  When: Always
  Action: rx_phase2_done = 1
```

**Key Variables:**

| Variable | Width | Purpose | Update Condition |
|----------|-------|---------|------------------|
| `rx_addr` | 14 bits | Phase 1: pointer word counter | rx_addr_inc = 1 |
| `rx_ptr_ctr` | 23 bits | Phase 2: synapse blocks received | Increments with each hbm_rvalid |
| `rx_phase1_done` | 1 bit | Phase 1 complete flag | State = READ_SYNAPSE_DATA |
| `rx_phase2_done` | 1 bit | Phase 2 complete flag | State = PHASE2_DONE |
| `hbm_count` | 1 bit | 512-bit packet assembly (0=1st word, 1=2nd word) | Toggles with each hbm_rvalid |
| `hbm_rdata_lower` | 256 bits | Stores 1st 256-bit word for packet | hbm_count = 0 |

**Data Flow in RX:**

```
hbm_rdata (256 bits from HBM)
    │
    ├─> (If hbm_count = 0) Store in hbm_rdata_lower
    │
    └─> (If hbm_count = 1) Combine with hbm_rdata_lower
            │
            ▼
        exec_hbm_rdata (512 bits) = {hbm_rdata, hbm_rdata_lower}
            │
            ├─> Internal Events Processor (synaptic inputs)
            │   └─> 16 groups × 32 bits each
            │
            └─> Spike Routing Logic (extract spike flags)
                └─> 8 synapse entries × spike_flag bit
```

---

#### 3. External Events Processor - The Input Handler

**File:** `external_events_processor.v`

##### BRAM Pipeline State Machine

**States:**
```
STATE_RESET (0)       → Initialization
STATE_IDLE (1)        → Wait for exec_run
STATE_FILL_PIPE (2)   → Fill 3-stage BRAM read pipeline
STATE_READ_INPUTS (3) → Stream spike data to HBM
STATE_PHASE1_DONE (4) → Processing complete
```

**Key Transitions:**
```
RESET → IDLE:
  When: Always
  Action: bramPresent_addr_rst = 1

IDLE → FILL_PIPE:
  When: exec_run = 1
  Action: bramPresent_addr_rst = 1, bram_select toggles

FILL_PIPE → READ_INPUTS:
  When: bramPresent_raddr >= PIPE_DEPTH (3)
  Action: exec_bram_phase1_ready = 1

READ_INPUTS → PHASE1_DONE:
  When: exec_hbm_rvalidready = 1 AND bramPresent_waddr == BRAM_ADDR_LIMIT
  Action: exec_bram_phase1_done = 1

PHASE1_DONE → IDLE:
  When: Always
  Action: None
```

**Key Variables:**

| Variable | Width | Purpose | Update Condition |
|----------|-------|---------|------------------|
| `bram_select` | 1 bit | Double-buffer toggle (0=bram0 present, 1=bram1 present) | Toggles on exec_run |
| `bramPresent_raddr` | 14 bits | Read address (leads by PIPE_DEPTH) | Increments during FILL_PIPE and READ_INPUTS |
| `bramPresent_waddr` | 14 bits | Write address (clears after reading) | Lags behind raddr by 3 cycles |
| `bramFuture_waddr[2:0]` | 14 bits × 3 | 3-stage write pipeline for incoming spikes | Shifts each cycle when setArray_go = 1 |
| `bramFuture_wdata[2:0]` | 8 bits × 3 | Spike mask data in pipeline | Shifts each cycle, OR on collision |

**Double-Buffering Mechanism:**

```
Timestep N:
  Present BRAM = bram0 (reading and clearing)
  Future BRAM = bram1 (receiving new spikes for N+1)

exec_run pulse:
  bram_select toggles

Timestep N+1:
  Present BRAM = bram1 (now reading what was written during N)
  Future BRAM = bram0 (now receiving new spikes for N+2)
```

**Pipeline Hazard Resolution:**

The 3-stage write pipeline prevents data loss when multiple spikes arrive for the same address while a write is in progress:

```verilog
if (setArray_go) begin
    // Check for address collisions in pipeline
    if (setArray_addr == bramFuture_waddr[2])
        // Collision at stage 2: OR new spike with pending
        bramFuture_wdata[1] <= bramFuture_wdata[2] | setArray_data;

    else if (setArray_addr == bramFuture_waddr[1])
        // Collision at stage 1: OR new spike
        bramFuture_wdata[0] <= bramFuture_wdata[1] | setArray_data;

    else if (setArray_addr == bramFuture_waddr[0])
        // Collision at stage 0: will be ORed at BRAM level
        // (commented logic in production code)

    else
        // No collision: add new entry to pipeline
        bramFuture_wdata[2] <= setArray_data;
```

---

#### 4. Internal Events Processor - The Neuron State Manager

**File:** `internal_events_processor.v`

##### URAM State Machine

**States:**
```
STATE_RESET (0)              → Initialization
STATE_IDLE (1)               → Wait for exec_run
STATE_FILL_PIPE_PHASE1 (2)   → Fill URAM read pipeline
STATE_WAIT_BRAM_PHASE1_DONE (3) → Wait for external events complete
STATE_PUSH_PTR_FIFO (4)      → Output neuron pointers
STATE_PHASE1_DONE (5)        → Phase 1 complete
STATE_POP_PTR_FIFO (6)       → Get pointer from FIFO (Phase 2)
STATE_WAIT_SYNAPSE (7)       → Wait for synapse data
STATE_PHASE2_DONE (8)        → Phase 2 complete, updates written
```

**Key Transitions:**
```
RESET → IDLE:
  When: Always
  Action: uram_raddr = 0

IDLE → FILL_PIPE_PHASE1:
  When: exec_run = 1
  Action: uram_raddr_rst = 1

FILL_PIPE_PHASE1 → WAIT_BRAM_PHASE1_DONE:
  When: uram_raddr >= PIPE_DEPTH (3)
  Action: exec_uram_phase1_ready = 1

WAIT_BRAM_PHASE1_DONE → PUSH_PTR_FIFO:
  When: exec_bram_phase1_done = 1
  Action: None (can start outputting pointers)

PUSH_PTR_FIFO → PHASE1_DONE:
  When: uram_raddr == URAM_ADDR_LIMIT
  Action: exec_uram_phase1_done = 1

PHASE1_DONE → WAIT_SYNAPSE (or continuous processing):
  When: Always
  Action: Prepare for Phase 2

WAIT_SYNAPSE → PHASE2_DONE:
  When: exec_hbm_rx_phase2_done = 1
  Action: exec_uram_phase2_done = 1, exec_iep_phase2_done = 1

PHASE2_DONE → IDLE:
  When: Always
  Action: Ready for next timestep
```

**Key Variables:**

| Variable | Width | Purpose | Update Condition |
|----------|-------|---------|------------------|
| `uram_raddr` | 13 bits | Global read address (all 16 groups) | Increments during Phase 1 |
| `uram_rden` | 1 bit | Global read enable | 1 during Phase 1 |
| `uram_waddr[15:0]` | 12 bits × 16 | Individual write addresses per group | Set during Phase 2 updates |
| `uram_wren[15:0]` | 1 bit × 16 | Individual write enables per group | 1 when updating group |
| `uram_wdata_reg[15:0]` | 72 bits × 16 | Updated neuron data (2 neurons per word) | Calculated during Phase 2 |
| `exec_uram_spiked` | 16 bits | Spike mask (bit per group) | Set when neuron exceeds threshold |

**Neuron Update Process (Phase 2):**

```verilog
When exec_hbm_rvalidready = 1:
    // Extract 16 group inputs from 512-bit packet
    for (group = 0 to 15) {
        input_upper[group] = exec_hbm_rdata[group*32 + 31:group*32 + 16];
        input_lower[group] = exec_hbm_rdata[group*32 + 15:group*32 + 0];

        // Read current neuron potential from URAM
        old_potential = uram_rdata[group];

        // Add synaptic input
        new_potential_upper = old_potential[71:36] + sign_extend(input_upper);
        new_potential_lower = old_potential[35:0]  + sign_extend(input_lower);

        // Apply neuron model (leak, reset, etc.)
        // Model 0: Memoryless
        // Model 1: Incremental
        // Model 2: Leaky I&F
        // Model 3: Non-leaky I&F

        // Check threshold
        if (new_potential_upper > threshold)
            exec_uram_spiked[group] = 1;
        if (new_potential_lower > threshold)
            exec_uram_spiked[group] = 1;

        // Write back
        uram_wdata_reg[group] = {new_potential_upper, new_potential_lower};
        uram_wren[group] = 1;
    }
```

---

### FIFO Usage and Data Flow

Complete reference of all FIFOs in the system:

#### Input Path (Host → FPGA)

| FIFO Name | Width | Depth | Source | Consumer | Format | Purpose |
|-----------|-------|-------|--------|----------|--------|---------|
| **rxFIFO** | 512 bits | 512 | pcie2fifos (Host PC) | Command Interpreter RX | [511:504]=opcode, [503:0]=payload | Incoming command queue |
| **ci2hbm** | 280 bits | Variable | Command Interpreter RX | HBM Processor TX | [279]=R/W, [278:256]=addr, [255:0]=data | Host HBM access commands |
| **ci2iep** | 54 bits | Variable | Command Interpreter RX | Internal Events Processor | [53]=R/W, [52:36]=addr, [35:0]=data | Host neuron access commands |

#### Internal Data Pipes

| FIFO Name | Width | Depth | Source | Consumer | Format | Purpose |
|-----------|-------|-------|--------|----------|--------|---------|
| **ptrFIFO[0-15]** | 32 bits | 512 each | HBM Processor RX | HBM Processor TX | [31:23]=length, [22:0]=address | 16 parallel pointer queues (one per neuron group) |
| **hbmFIFO** | 512 bits | Variable | HBM Processor RX | Internal Events Processor | 16 groups × 32 bits | Synaptic inputs to neurons |

**Pointer FIFO Details:**
- 16 independent FIFOs, one per neuron group (URAM bank)
- Populated during Phase 1 by pointer_fifo_controller demuxing
- Consumed during Phase 2 by HBM TX using round-robin selection
- Pointer format: `{synapse_count[8:0], hbm_base_address[22:0]}`

#### Output Path (FPGA → Host)

| FIFO Name | Width | Depth | Source | Consumer | Format | Purpose |
|-----------|-------|-------|--------|----------|--------|---------|
| **spk0-7** | 17 bits | 512 each | HBM Processor RX spike logic | Spike FIFO Controller | Neuron address (17 bits) | 8 parallel spike distribution FIFOs |
| **spk2ciFIFO** | 17 bits | Variable | Spike FIFO Controller | Command Interpreter TX | Neuron address | Aggregated spike queue |
| **txFIFO** | 512 bits | 512 | Command Interpreter TX | pcie2fifos (Host PC) | [511:480]=opcode, [479:0]=payload | Outgoing response queue |
| **hbm2ci** | 256 bits | Variable | HBM Processor RX | Command Interpreter TX | HBM read data | Host HBM read responses |
| **iep2ci** | 53 bits | Variable | Internal Events Processor | Command Interpreter TX | [52:36]=addr, [35:0]=data | Host neuron read responses |

**Spike FIFO Routing:**
```
HBM RX extracts 8 synapses from 256-bit word
    │
    ├─> Synapse 0 (addr[2:0]=0) → spk0
    ├─> Synapse 1 (addr[2:0]=1) → spk1
    ├─> Synapse 2 (addr[2:0]=2) → spk2
    ├─> Synapse 3 (addr[2:0]=3) → spk3
    ├─> Synapse 4 (addr[2:0]=4) → spk4
    ├─> Synapse 5 (addr[2:0]=5) → spk5
    ├─> Synapse 6 (addr[2:0]=6) → spk6
    └─> Synapse 7 (addr[2:0]=7) → spk7
            │
            ▼
    Spike FIFO Controller (round-robin arbitration)
            │
            ▼
        spk2ciFIFO (aggregated)
            │
            ▼
    Command Interpreter TX (batch into 14-spike packets)
            │
            ▼
        txFIFO → Host PC
```

---

### Complete Timing Sequence

Cycle-by-cycle breakdown of a complete timestep execution:

#### Setup and Phase 1 Initiation (Cycles 0-3)

```
CYCLE 0: HOST PC → exec_run command received
    │
    ├─> Command Interpreter RX:
    │   • Parses CMD_EXEC_STEP or starts timestep loop in CMD_EXEC_CONT
    │   • exec_run_rst = 1 (reset counters)
    │   • execRun_ctr = 0
    │   • execRun_timer = 0
    │   • execRun_running = 1 (next cycle)
    │
    ├─> External Events Processor:
    │   • bram_select toggles (swap present/future buffers)
    │   • bramPresent_addr_rst = 1
    │   • bramPresent_raddr = 0
    │   • State → FILL_PIPE
    │
    ├─> Internal Events Processor:
    │   • uram_raddr_rst = 1
    │   • uram_raddr = 0
    │   • State → FILL_PIPE_PHASE1
    │
    └─> HBM Processor:
        • TX: tx_done_rst = 1, tx_addr_rst = 1
              State → SEND_INPUT_READ_COMMANDS
        • RX: rx_done_rst = 1, rx_addr_rst = 1
              State → WAIT_BRAM_PIPELINE

CYCLE 1: Pipeline Filling Begins
    │
    ├─> EEP: FILL_PIPE state
    │   • bramPresent_rden = 1
    │   • bramPresent_raddr = 0 (reading address 0)
    │   • BRAM latency = 3 cycles (data arrives cycle 4)
    │
    ├─> IEP: FILL_PIPE_PHASE1 state
    │   • uram_rden_0-15 = 1 (all groups in parallel)
    │   • uram_raddr = 0
    │   • URAM latency = 3 cycles
    │
    └─> HBM TX: SEND_INPUT_READ_COMMANDS
        • hbm_arvalid = 1 (send first read request)
        • hbm_araddr = {5'd0, 8'd0, 1'b0, 10'd0, 4'd0, 5'd0}
        • hbm_arlen = 15 (request 16 words)
        • tx_addr = 0

CYCLE 2: Pipeline Filling (2nd address)
    │
    ├─> EEP: bramPresent_raddr = 1
    ├─> IEP: uram_raddr = 1
    └─> HBM TX:
        • If hbm_arready = 1: tx_addr_inc, tx_addr = 1
        • Send next read command

CYCLE 3: Pipeline Filling (3rd address)
    │
    ├─> EEP: bramPresent_raddr = 2
    ├─> IEP: uram_raddr = 2
    └─> HBM TX: tx_addr = 2
```

#### Phase 1a: Input Pointer Collection (Cycles 4-200)

```
CYCLE 4: Pipelines Ready, Synchronized Streaming Begins
    │
    ├─> EEP:
    │   • Data from BRAM address 0 now valid (after 3-cycle latency)
    │   • bramPresent_raddr >= PIPE_DEPTH (3)
    │   • exec_bram_phase1_ready = 1 ✓ (SIGNAL TO HBM)
    │   • State → READ_INPUTS
    │   • bramPresent_rden = 1 (when exec_hbm_rvalidready = 1)
    │   • exec_bram_spiked = bramPresent_rdata (8-bit spike mask)
    │   • bramPresent_wren = 1 (clear BRAM address 0 to zero)
    │
    ├─> HBM RX:
    │   • Sees exec_bram_phase1_ready = 1
    │   • State → READ_INPUT_POINTERS
    │   • hbm_rready = 1 (ready to accept pointer data)
    │
    └─> HBM TX:
        • Continues sending read commands
        • tx_addr increments each cycle (when hbm_arready = 1)
        • Loop: tx_addr = 0 → INPUT_ADDR_LIMIT

CYCLES 5-100+: HBM Data Arrives (High Latency)
    │
    ├─> HBM Memory:
    │   • First pointer data arrives after ~100 cycles
    │   • Each HBM read returns 256-bit word
    │   • Each 256-bit word contains 8 pointers
    │   • Burst of 16 words arrives consecutively
    │
    └─> HBM RX:
        • hbm_rvalid = 1 (data valid)
        • hbm_rready = 1 (accept data)
        • rx_addr_inc = 1
        • Forward pointer word to pointer_fifo_controller

        Pointer Format (32 bits per pointer):
        • [31:23] = length (number of synapse rows, 0-511)
        • [22:0]  = base_address (HBM location, 0-8M)

CYCLES 100-200: Parallel Streaming
    │
    ├─> EEP:
    │   • bramPresent_rden = 1 (synchronized with exec_hbm_rvalidready)
    │   • Incrementing bramPresent_raddr and bramPresent_waddr
    │   • exec_bram_spiked continuously updated
    │   • When bramPresent_waddr == BRAM_ADDR_LIMIT:
    │     exec_bram_phase1_done = 1 ✓
    │
    ├─> HBM RX:
    │   • Receiving pointer data bursts from HBM
    │   • Forwarding to pointer_fifo_controller
    │
    └─> Pointer FIFO Controller:
        • Demuxes pointers to 16 ptrFIFOs based on exec_bram_spiked
        • Example: exec_bram_spiked = 0b00000111
          → Groups 0, 1, 2 have spikes
          → Distribute pointers accordingly
        • ptrFIFO[0-15] fill up with addresses
```

#### Phase 1b: Output Pointer Collection (Cycles 200-500)

```
CYCLE ~200: Transition to Output Pointers
    │
    ├─> EEP:
    │   • exec_bram_phase1_done = 1 (latched)
    │   • State → PHASE1_DONE → IDLE
    │
    ├─> IEP:
    │   • Sees exec_bram_phase1_done = 1
    │   • State → PUSH_PTR_FIFO
    │   • exec_uram_phase1_ready = 1 ✓
    │   • Continues reading URAM (uram_raddr incrementing)
    │   • Outputs neuron pointers for internal connections
    │
    ├─> HBM RX:
    │   • Finishes receiving input pointers
    │   • rx_addr == {INPUT_ADDR_LIMIT, INPUT_ADDR_MOD}
    │   • State → WAIT_URAM_PIPELINE
    │   • Waits for exec_uram_phase1_ready = 1
    │
    └─> HBM TX:
        • tx_addr == INPUT_ADDR_LIMIT
        • tx_select_inc = 1 (toggle to outputs)
        • tx_addr_rst = 1
        • State → SEND_OUTPUT_READ_COMMANDS

CYCLES 201-500: Output Pointer Streaming
    │
    ├─> IEP:
    │   • uram_rden_0-15 = 1 (all 16 groups in parallel)
    │   • Reads neuron potentials from URAM
    │   • For each neuron: check if has outgoing connections
    │   • Output pointers to HBM processor
    │   • When uram_raddr == URAM_ADDR_LIMIT:
    │     exec_uram_phase1_done = 1 ✓
    │
    ├─> HBM RX:
    │   • Sees exec_uram_phase1_ready = 1
    │   • State → READ_OUTPUT_POINTERS
    │   • Receives output pointer data from HBM
    │   • Forwards to pointer_fifo_controller
    │   • When rx_addr == {OUTPUT_ADDR_LIMIT, OUTPUT_ADDR_MOD}:
    │     State → PHASE1_DONE
    │     rx_phase1_done = 1 ✓
    │
    ├─> HBM TX:
    │   • Sends output pointer read commands
    │   • tx_addr loops: 0 → OUTPUT_ADDR_LIMIT
    │   • When complete:
    │     State → PHASE1_DONE
    │     tx_phase1_done = 1 ✓
    │     tx_phase_inc = 1 (toggle tx_phase: 0 → 1)
    │     State → POP_POINTER_FIFO
    │
    └─> Pointer FIFO Controller:
        • Continues demuxing output pointers to ptrFIFOs
        • ptrFIFO[0-15] now contain both input and output pointers
```

**Phase 1 Complete:**
- EEP: exec_bram_phase1_done = 1
- IEP: exec_uram_phase1_done = 1
- HBM TX: tx_phase1_done = 1, tx_phase = 1 (Phase 2 mode)
- HBM RX: rx_phase1_done = 1
- All 16 ptrFIFOs filled with connectivity pointers

---

#### Phase 2: Synapse Data Fetching and Processing (Cycles 500-1500)

```
CYCLE ~500: Phase 2 Initiation
    │
    ├─> HBM TX:
    │   • State = POP_POINTER_FIFO
    │   • Checks: rx_phase1_done = 1 ✓
    │   • Checks: ptrFIFO not empty (at least one group has pointers)
    │   • ptrFIFO_rden = 1 (pop first pointer)
    │   • ptr_addr_set = 1 (load pointer)
    │   • ptr_addr = ptrFIFO_dout[22:0]
    │   • ptr_len = ptrFIFO_dout[31:23]
    │   • ptr_ctr = 0
    │   • State → SEND_POINTER_READ_COMMANDS
    │
    ├─> HBM RX:
    │   • State = READ_SYNAPSE_DATA
    │   • hbm_rready = 1 (always accept synapse data)
    │   • rx_ptr_ctr = 0
    │
    └─> IEP:
        • State = Phase 2 processing
        • Ready to receive synaptic inputs

CYCLES 501-1500: Pointer-Driven Synapse Fetch Loop
    │
    └─> For each pointer in ptrFIFOs:

        HBM TX:
        • Calculate burst length:
          ptr_burst = (ptr_ctr[8:4] == ptr_len[8:4]) ? ptr_len[3:0] : 4'hf
        • Send HBM read command:
          hbm_arvalid = 1
          hbm_araddr = {5'd0, ptr_addr[22:0], 5'd0}
          hbm_arlen = ptr_burst
        • When hbm_arready = 1:
          ptr_addr_inc = 1
          ptr_addr += (ptr_burst + 1)
          ptr_ctr += (ptr_burst + 1)
          tx_ptr_ctr += (ptr_burst + 1)

        • If ptr_ctr[8:4] == ptr_len[8:4]:
          Current pointer exhausted
          State → POP_POINTER_FIFO (get next)

        • Else:
          Continue fetching remaining bursts for current pointer

        HBM RX (Parallel):
        • Receives 256-bit synapse words from HBM
        • Assembles into 512-bit packets:
          Cycle N:   hbm_count = 0, store in hbm_rdata_lower
          Cycle N+1: hbm_count = 1, combine with current hbm_rdata
                     exec_hbm_rdata = {hbm_rdata, hbm_rdata_lower}
                     exec_hbm_rvalidready = 1 ✓

        • Extract spike flags and route to spike FIFOs:
          For synapse 0-7:
            spike_flag = hbm_rdata[N*32 + 31]
            If spike_flag = 1:
              neuron_addr = hbm_rdata[N*32 + 16:N*32]
              Route to spkN based on neuron_addr[2:0]
              spkN_wren = 1, spkN_din = neuron_addr

        • rx_ptr_ctr++ (count 256-bit words)

        IEP (Parallel):
        • When exec_hbm_rvalidready = 1:

          Extract 16 group inputs:
          For group 0-15:
            input_upper = exec_hbm_rdata[group*32 + 31:group*32 + 16]
            input_lower = exec_hbm_rdata[group*32 + 15:group*32 + 0]

          Read current neuron potential from URAM:
            old_potential = uram_rdata[group]

          Accumulate synaptic input:
            new_V_upper = old_potential[71:36] + sign_extend(input_upper)
            new_V_lower = old_potential[35:0]  + sign_extend(input_lower)

          Apply neuron model (leak, reset, etc.):
            (depends on exec_neuron_model setting)

          Check threshold:
            if (new_V_upper > threshold):
              exec_uram_spiked[group] = 1
              new_V_upper = reset_value (depending on model)
            if (new_V_lower > threshold):
              exec_uram_spiked[group] = 1
              new_V_lower = reset_value

          Write updated potential back to URAM:
            uram_waddr[group] = current_word_address
            uram_wdata_reg[group] = {new_V_upper, new_V_lower}
            uram_wren[group] = 1

        Spike FIFO Controller (Parallel):
        • Round-robin reads from spk0-7
        • Aggregates to spk2ciFIFO

        Command Interpreter TX (Parallel):
        • Batches spikes from spk2ciFIFO
        • When spike_ctr == 14:
          Send 512-bit packet:
          [511:480] = 0xEEEEEEEE
          [479:32]  = 14 × 32-bit spikes
          [31:0]    = execRun_ctr
          txFIFO_wren = 1
          spike_rst = 1

CYCLE ~1500: Phase 2 Completion
    │
    ├─> HBM TX:
    │   • All ptrFIFOs empty (or safety timer expired)
    │   • wait_clks_cnt == 255 AND ptrFIFO_empty
    │   • State → PHASE2_DONE
    │   • tx_phase2_done = 1 ✓
    │   • tx_ptr_ctr = final count (e.g., 5000 synapse blocks)
    │
    ├─> HBM RX:
    │   • Checks: tx_phase2_done = 1 ✓
    │   • Checks: rx_ptr_ctr == tx_ptr_ctr ✓
    │   • State → PHASE2_DONE
    │   • rx_phase2_done = 1 ✓
    │
    └─> IEP:
        • Sees exec_hbm_rx_phase2_done = 1
        • All neuron updates complete
        • exec_uram_phase2_done = 1 ✓
        • exec_iep_phase2_done = 1 ✓
```

---

#### Timestep Finalization (Cycles 1500-1535)

```
CYCLE 1501: Command Interpreter Checks Completion
    │
    └─> CI RX (State = WAIT_RUN):
        • Checks: exec_iep_phase2_done = 1 ✓
        • Checks: spk2ciFIFO_empty = 1 (all spikes drained)
        • wait_clks_cnt starts incrementing

CYCLES 1502-1531: Safety Margin
    │
    └─> CI RX:
        • wait_clks_cnt increments each cycle
        • Ensures all spikes propagated through arbiters
        • Ensures FIFOs fully drained

CYCLE 1532: Timestep Complete (wait_clks_cnt == 31)
    │
    └─> CI RX:
        • exec_run_inc = 1
        • execRun_ctr++ (e.g., 0 → 1)
        • execRun_timer continues counting

        Check continuation:
        • If execRun_ctr < execRun_limit:
          axon_addr_rst = 1
          State → REGISTER_PCIE_AXON_DATA
          Load next timestep's input data
          Loop back to CYCLE 0 (exec_run pulse)

        • Else (execRun_ctr == execRun_limit):
          State → EXEC_DONE
          exec_run_done = 1
          execRun_running = 0 (stop timer)
          All timesteps complete!
```

**Typical Timing Summary:**
- Phase 1 (Pointer Collection): ~200-500 cycles
- Phase 2 (Synapse Processing): ~300-1000 cycles
- Safety Margin: 31 cycles
- **Total per Timestep: ~500-1500 cycles**
- At 225 MHz: **2.2-6.7 microseconds per timestep**

---

### Handshake Protocols

Detailed sequence diagrams for each inter-module handshake:

#### 1. EEP ↔ HBM: Input Spike Coordination

**Problem:** HBM needs to know when EEP is ready to stream spike data.

**Solution:** 3-stage handshake

```
Step 1: Pipeline Initialization
    EEP: Receives exec_run pulse
    EEP: Toggles bram_select (swap buffers)
    EEP: Enters FILL_PIPE state
    EEP: Issues 3 consecutive BRAM reads

Step 2: Pipeline Ready Signal
    EEP: After 3 cycles (bramPresent_raddr >= 3)
    EEP: exec_bram_phase1_ready = 1 ✓
    HBM RX: Waiting in WAIT_BRAM_PIPELINE
    HBM RX: Sees exec_bram_phase1_ready = 1
    HBM RX: State → READ_INPUT_POINTERS

Step 3: Synchronized Streaming
    EEP: bramPresent_rden = 1 ONLY when exec_hbm_rvalidready = 1
    EEP: Reads next BRAM address
    EEP: Outputs exec_bram_spiked
    HBM: Receives pointer data from HBM memory
    HBM: Forwards to pointer_fifo_controller
    HBM: Sets exec_hbm_rvalidready = 1 on every 2nd 256-bit word

    Loop: Synchronized 1:1 until BRAM exhausted

Step 4: Completion
    EEP: When bramPresent_waddr == BRAM_ADDR_LIMIT
    EEP: exec_bram_phase1_done = 1 ✓
    IEP: Waiting for this signal before proceeding
```

**Key Signal Timing:**
```
         exec_run
             │
CYCLE 0      ▼
             ├─ bram_select toggles
             │
CYCLE 1-3    ├─ FILL_PIPE (reading addresses 0, 1, 2)
             │
CYCLE 4      ├─ exec_bram_phase1_ready = 1
             │  └─> HBM RX sees signal, enters READ_INPUT_POINTERS
             │
CYCLE 5-200  ├─ Synchronized streaming:
             │  • bramPresent_rden = exec_hbm_rvalidready
             │  • exec_bram_spiked updated each valid cycle
             │
CYCLE 200    └─ exec_bram_phase1_done = 1
```

---

#### 2. IEP ↔ HBM: Output Neuron Coordination

**Problem:** HBM must wait for IEP URAM pipeline to fill AND for EEP to finish before reading output pointers.

**Solution:** 4-stage handshake

```
Step 1: Wait for External Events Complete
    IEP: State = WAIT_BRAM_PHASE1_DONE
    IEP: Waits for exec_bram_phase1_done = 1
    IEP: Cannot proceed until external spikes processed

Step 2: URAM Pipeline Filling
    IEP: After exec_bram_phase1_done received
    IEP: State → PUSH_PTR_FIFO
    IEP: uram_rden_0-15 = 1 (all groups reading in parallel)
    IEP: Reads neuron potentials, outputs pointers

Step 3: URAM Ready Signal
    IEP: After uram_raddr >= PIPE_DEPTH (3)
    IEP: exec_uram_phase1_ready = 1 ✓
    HBM RX: Waiting in WAIT_URAM_PIPELINE
    HBM RX: Sees exec_uram_phase1_ready = 1
    HBM RX: State → READ_OUTPUT_POINTERS

Step 4: Pointer Streaming
    IEP: Continues reading URAM (uram_raddr incrementing)
    IEP: Outputs neuron pointers
    HBM: Receives pointer data from HBM memory
    HBM: Forwards to pointer_fifo_controller

    Loop: Until uram_raddr == URAM_ADDR_LIMIT

Step 5: Completion
    IEP: exec_uram_phase1_done = 1 ✓
    HBM RX: When rx_addr == {OUTPUT_ADDR_LIMIT, OUTPUT_ADDR_MOD}
    HBM RX: rx_phase1_done = 1 ✓
```

**Dependency Chain:**
```
exec_run
   │
   ├─> EEP starts (immediate)
   │   └─> exec_bram_phase1_done = 1
   │       └─> IEP can proceed
   │           └─> exec_uram_phase1_ready = 1
   │               └─> HBM RX can read output pointers
   │                   └─> rx_phase1_done = 1
   │                       └─> HBM TX can start Phase 2
   │
   └─> HBM TX starts sending input pointer commands (immediate)
```

---

#### 3. HBM TX ↔ RX: Phase Coordination

**Problem:** RX must know when TX finishes Phase 1 to enable Phase 2. TX must know when RX finishes Phase 1 to start popping ptrFIFO.

**Solution:** Cross-phase completion flags

```
Phase 1:
    TX: Sends input pointer read commands
    TX: When tx_addr == INPUT_ADDR_LIMIT:
        tx_select_inc = 1, State → SEND_OUTPUT_READ_COMMANDS
    TX: Sends output pointer read commands
    TX: When tx_addr == OUTPUT_ADDR_LIMIT:
        State → PHASE1_DONE
        tx_phase1_done = 1 ✓
        tx_phase_inc = 1 (toggle tx_phase: 0 → 1)
        State → POP_POINTER_FIFO

    RX: Receives input pointer data
    RX: When rx_addr == {INPUT_ADDR_LIMIT, INPUT_ADDR_MOD}:
        State → WAIT_URAM_PIPELINE
    RX: Waits for exec_uram_phase1_ready = 1
    RX: Receives output pointer data
    RX: When rx_addr == {OUTPUT_ADDR_LIMIT, OUTPUT_ADDR_MOD}:
        State → PHASE1_DONE
        rx_phase1_done = 1 ✓

Phase 1 → Phase 2 Transition:
    TX: State = POP_POINTER_FIFO
    TX: Checks: rx_phase1_done = 1 (RX finished receiving pointers)
    TX: Checks: ptrFIFO not empty (has pointers to fetch)
    TX: If both conditions met:
        ptrFIFO_rden = 1, ptr_addr_set = 1
        State → SEND_POINTER_READ_COMMANDS

    RX: State = READ_SYNAPSE_DATA
    RX: hbm_rready = 1 (always accept synapse data)

Phase 2:
    TX: Loops through all ptrFIFO pointers
    TX: Sends synapse read commands
    TX: tx_ptr_ctr increments with each burst
    TX: When all ptrFIFOs empty (or timer expires):
        State → PHASE2_DONE
        tx_phase2_done = 1 ✓

    RX: Receives synapse data
    RX: rx_ptr_ctr increments with each 256-bit word
    RX: When tx_phase2_done = 1 AND rx_ptr_ctr == tx_ptr_ctr:
        State → PHASE2_DONE
        rx_phase2_done = 1 ✓

Phase 2 Complete:
    Both TX and RX return to IDLE
    Ready for next exec_run pulse
```

**Counter Synchronization:**
```
TX maintains: tx_ptr_ctr (total 256-bit words requested)
RX maintains: rx_ptr_ctr (total 256-bit words received)

Completion condition: (tx_phase2_done = 1) AND (rx_ptr_ctr == tx_ptr_ctr)

This ensures ALL synapse data has been received before proceeding.
```

---

#### 4. HBM ↔ IEP: Synaptic Data Streaming

**Problem:** IEP needs continuous stream of synaptic inputs, synchronized with HBM data arrival.

**Solution:** Data-valid handshake with automatic flow control

```
HBM RX Phase 2:
    • State = READ_SYNAPSE_DATA
    • hbm_rready = 1 (always ready to accept from HBM)
    • When hbm_rvalid = 1 (HBM data arrives):

        Packet Assembly:
        Cycle N:   hbm_count = 0
                   hbm_rdata_lower = hbm_rdata (256 bits)
        Cycle N+1: hbm_count = 1
                   exec_hbm_rdata = {hbm_rdata, hbm_rdata_lower} (512 bits)
                   exec_hbm_rvalidready = 1 ✓

        • Check backpressure: !hbmFIFO_full
        • If full: wait (stall pipeline)

IEP Phase 2:
    • Continuously monitors exec_hbm_rvalidready
    • When exec_hbm_rvalidready = 1:

        hbm2iep_rden = 1 (consume packet)

        Extract 16 group inputs:
        • exec_hbm_rdata[31:0]    → group 0 (upper/lower)
        • exec_hbm_rdata[63:32]   → group 1
        • ...
        • exec_hbm_rdata[511:480] → group 15

        For each group:
        • Read URAM (current neuron potential)
        • Add synaptic input
        • Apply neuron model
        • Check threshold → spike?
        • Write updated potential back to URAM

        • If spike: exec_uram_spiked[group] = 1

Completion:
    HBM RX: exec_hbm_rx_phase2_done = 1
    IEP: Sees signal, sets exec_iep_phase2_done = 1
```

**Timing Diagram:**
```
HBM Memory             HBM RX                 IEP
    │                     │                    │
    ├─ 256-bit word ────> ├─ Store lower      │
    │   (hbm_rvalid=1)    │   hbm_count=0     │
    │                     │                    │
    ├─ 256-bit word ────> ├─ Combine          │
    │   (hbm_rvalid=1)    │   hbm_count=1     │
    │                     │   exec_hbm_rdata  │
    │                     │   rvalidready=1 ──>├─ Process 16 groups
    │                     │                    │   Read URAM
    │                     │                    │   Update neurons
    │                     │                    │   Write URAM
    │                     │                    │   hbm2iep_rden=1
    │                     │                    │
    ├─ 256-bit word ────> ├─ Store lower      │
    │                     │   hbm_count=0     │
    │                     │                    │
    └─ Continuous...      └─ Continuous...    └─ Continuous...
```

---

### Quick Reference Tables

#### Control Signal Summary

| Signal | Type | Asserted By | Consumed By | Meaning | Typical Cycle |
|--------|------|-------------|-------------|---------|---------------|
| `exec_run` | Pulse | CI RX | All modules | Start new timestep | 0 |
| `exec_bram_phase1_ready` | Level | EEP | HBM RX | BRAM pipeline filled | 4 |
| `exec_bram_phase1_done` | Level | EEP | IEP | External spikes processed | 200 |
| `exec_uram_phase1_ready` | Level | IEP | HBM RX | URAM pipeline filled | 210 |
| `exec_uram_phase1_done` | Level | IEP | HBM TX | Internal pointers output | 500 |
| `exec_hbm_tx_phase1_done` | Level | HBM TX | Internal | Pointer commands sent | 500 |
| `exec_hbm_rx_phase1_done` | Level | HBM RX | HBM TX, Spike logic | Pointers received | 500 |
| `exec_hbm_tx_phase2_done` | Level | HBM TX | HBM RX | Synapse commands sent | 1500 |
| `exec_hbm_rx_phase2_done` | Level | HBM RX | IEP, CI | Synapses received | 1500 |
| `exec_hbm_rvalidready` | Pulse | HBM RX | EEP, IEP | 512-bit packet ready | Every 2nd HBM read |
| `exec_iep_phase2_done` | Level | IEP | CI RX | Neuron updates complete | 1500 |
| `execRun_running` | Level | CI RX | CI TX | Execution active | 1-1530 |
| `execRun_done` | Level | CI RX | CI TX | All timesteps complete | After last timestep |

---

#### State Transition Summary

**Command Interpreter RX:**
```
RESET → IDLE → (on command) → REGISTER_AXON_DATA → SET_AXON_DATA → IDLE
                            → EXEC_STEP → WAIT_RUN → EXEC_DONE → IDLE
                                               └─→ (loop) REGISTER_AXON_DATA
```

**Command Interpreter TX:**
```
RESET → IDLE → WAIT_FOR_SPIKES ⟷ SEND_SPIKES → IDLE
```

**HBM Processor TX:**
```
RESET → IDLE → SEND_INPUT_CMDS → SEND_OUTPUT_CMDS → PHASE1_DONE
             → POP_PTR_FIFO ⟷ SEND_PTR_CMDS → PHASE2_DONE → IDLE
```

**HBM Processor RX:**
```
RESET → IDLE → WAIT_BRAM_PIPE → READ_INPUT_PTRS → WAIT_URAM_PIPE
             → READ_OUTPUT_PTRS → PHASE1_DONE → READ_SYNAPSE_DATA
             → PHASE2_DONE → IDLE
```

**External Events Processor:**
```
RESET → IDLE → FILL_PIPE → READ_INPUTS → PHASE1_DONE → IDLE
```

**Internal Events Processor:**
```
RESET → IDLE → FILL_PIPE_PH1 → WAIT_BRAM_DONE → PUSH_PTR
             → PHASE1_DONE → (Phase 2 processing) → PHASE2_DONE → IDLE
```

---

#### Variable Reference by Module

**Command Interpreter:**

| Variable | Width | Initialize | Update | Purpose |
|----------|-------|------------|--------|---------|
| execRun_ctr | 32 | 0 (exec_run_rst) | exec_run_inc | Current timestep |
| execRun_limit | 32 | 0 | exec_run_set | Target timesteps |
| execRun_timer | 64 | 0 | +1 each cycle | Performance counter |
| wait_clks_cnt | 5 | 0 | +1 in WAIT_RUN | Safety margin |
| spike_ctr | 4 | 0 (spike_rst) | spike_inc | Batch size |

**HBM Processor TX:**

| Variable | Width | Initialize | Update | Purpose |
|----------|-------|------------|--------|---------|
| tx_phase | 1 | 0 | tx_phase_inc | Phase selector |
| tx_select | 1 | 0 | tx_select_inc | Input/output toggle |
| tx_addr | 10 | 0 (tx_addr_rst) | tx_addr_inc | Group address |
| ptr_addr | 23 | 0 | ptr_addr_set/inc | Synapse address |
| ptr_len | 9 | 0 | ptr_addr_set | Synapse count |
| ptr_ctr | 9 | 0 | ptr_addr_inc | Progress counter |
| tx_ptr_ctr | 23 | 0 | ptr_addr_inc | Total sent |

**HBM Processor RX:**

| Variable | Width | Initialize | Update | Purpose |
|----------|-------|------------|--------|---------|
| rx_addr | 14 | 0 (rx_addr_rst) | rx_addr_inc | Pointer word count |
| rx_ptr_ctr | 23 | 0 | +1 per hbm_rvalid | Synapse blocks received |
| hbm_count | 1 | 0 | Toggle | Packet assembly |

**External Events Processor:**

| Variable | Width | Initialize | Update | Purpose |
|----------|-------|------------|--------|---------|
| bram_select | 1 | 0 | Toggle on exec_run | Buffer selector |
| bramPresent_raddr | 14 | 0 | +1 | Read address (leading) |
| bramPresent_waddr | 14 | 0 | +1 | Write address (lagging) |

**Internal Events Processor:**

| Variable | Width | Initialize | Update | Purpose |
|----------|-------|------------|--------|---------|
| uram_raddr | 13 | 0 | +1 | Global read address |
| exec_uram_spiked | 16 | 0 | Set per group | Spike mask output |

---

## Conclusion

The state machine coordination in this neuromorphic system is a carefully orchestrated dance of four independent modules, synchronized through handshake signals and completion flags. The key insights are:

1. **Pipeline Priming:** BRAM and URAM require 3-cycle pipeline fills before streaming can begin
2. **Two-Phase Architecture:** Phase 1 collects connectivity pointers, Phase 2 processes synaptic data
3. **Handshake Synchronization:** Modules wait for `exec_*_ready` and `exec_*_done` signals
4. **FIFO Buffering:** Pointer FIFOs and spike FIFOs decouple pipeline stages
5. **Safety Margins:** Wait counters prevent race conditions and ensure complete data drainage

Understanding this coordination is essential for:
- **Debugging:** Knowing where to look when execution stalls
- **Optimization:** Identifying bottlenecks in the pipeline
- **Extension:** Adding new modules or modifying behavior
- **Verification:** Ensuring timing requirements are met

**Related Documentation:**
- [Chapter 2.1](Chapter_2_1.md) - Conceptual overview of execution flow
- [Chapter 2.2](Chapter_2_2.md) - Python and Verilog code walkthrough
- [Chapter 3: Verilog Files Review](../chapter_3.md) - Module-specific implementation details
- [command_interpreter.v](../3_Verilog_Files_Review/command_interpreter.md) - Orchestrator details
- [hbm_processor.v](../3_Verilog_Files_Review/hbm_processor.md) - Memory access details

---

**Total Duration:** ~500-1500 cycles per timestep (2.2-6.7 μs @ 225 MHz)

**Key Performance Factors:**
- HBM latency (~100 cycles for first word)
- Synaptic fanout (more connections = more Phase 2 time)
- Spike rate (more spikes = more pointer fetches)
- FIFO depths (prevent backpressure stalls)
