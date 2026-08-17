# 4.3 Hardware Implementation of R-STDP

This chapter explains how R-STDP is implemented in the FPGA hardware by extending the existing framework from Chapters 1-2. The key insight is **reusing existing logic**: eligibility traces are updated like neuron membrane potentials (but without spiking), and weight updates reuse the existing HBM write infrastructure.

---

## Memory Architecture: Adding HBM Region 4 for Eligibility Traces

Recall from Chapter 1 that the existing system uses three HBM regions:
- **Region 1**: Axon pointers (where each axon's synapses are located)
- **Region 2**: Neuron pointers (where each neuron's output synapses are located)
- **Region 3**: Synapse data (OpCode, Target address, Weight)

For R-STDP, we add a fourth region:
- **Region 4**: Eligibility traces (one value per synapse, stored like membrane potentials)

### Address Mapping: Region 3 ↔ Region 4

Each synapse in Region 3 has a corresponding eligibility trace in Region 4. The mapping uses simple address arithmetic:

```
Synapse address (Region 3)  →  Eligibility trace address (Region 4)
──────────────────────────────────────────────────────────────────
0x8000                      →  0x8000 + REGION4_OFFSET
0x8001                      →  0x8001 + REGION4_OFFSET
0x8002                      →  0x8002 + REGION4_OFFSET
```

**REGION4_OFFSET** = 1000 (or approximately half the number of rows in Region 3)

**Example:**
- Synapse at HBM row `0x8001` in Region 3
- Its eligibility trace at row `0x8001 + 1000 = 0x8FA9` in Region 4

**Why this simple mapping?**
- Easy to compute in hardware (just add/subtract offset)
- Leverages existing HBM read/write logic
- Reversible: `Region_3_addr = Region_4_addr - REGION4_OFFSET`

### Data Format

Each eligibility trace is stored as a **36-bit signed fixed-point value**, identical in format to neuron membrane potentials:

```
[35:0] = Eligibility trace value
  - Starts at 0
  - Increases when STDP event occurs (coincidence detection)
  - Decays over time (like membrane potential leak)
  - NO threshold comparison (doesn't spike)
```

This reuse of the membrane potential format means we can use the **same hardware logic** that updates neurons to also update eligibility traces.

---

## The Reward Register: Dopamine Signal

A new 1-bit register stores the reward signal (dopamine modulation):

**Reward Register:**
- **Width**: 1 bit (binary on/off)
- **Set by**: New command opcode (CMD_SET_REWARD = 0x0A)
- **Broadcast to**: internal_events_processor during Phase 2
- **Purpose**: Gates whether weight updates occur

**Simplified for initial implementation:**
- Binary signal: 1 = reward on, 0 = reward off
- Can be toggled per timestep or set once for entire simulation
- Future enhancement: Multi-bit for graded reward levels

---

## Phase 2 Flow with R-STDP: Step-by-Step

We extend the existing Phase 2 (Synaptic Processing) to include eligibility trace handling and weight updates.

### Step 1: Pointer Processing with Dual Address Computation

Recall from Chapter 2 that Phase 2 begins with `pointer_fifo_controller` reading synapse pointers from the 16 ptrFIFOs and sending them to `hbm_processor`.

**Extension:** When `hbm_processor` receives a synapse pointer, it now computes **two addresses**:

1. **Region 3 address** (existing): Where the synapse data is stored
2. **Region 4 address** (new): Where the eligibility trace is stored
   - Computed as: `Region_3_addr + REGION4_OFFSET`

**New FIFO introduced: etFIFO (Eligibility Trace FIFO)**

| Property | Value |
|----------|-------|
| **Width** | 23 bits (HBM address) |
| **Depth** | 512 |
| **Written by** | hbm_processor (when processing synapse pointers) |
| **Read by** | internal_events_processor (during neuron updates) |
| **Contains** | Region 4 addresses (eligibility trace locations) |

**What happens:**
- `hbm_processor` pops a synapse pointer from ptrFIFO
- Extracts Region 3 address: `0x8001`
- Computes Region 4 address: `0x8001 + 1000 = 0x8FA9`
- Pushes `0x8FA9` to **etFIFO**
- Issues HBM read for synapse data at Region 3 address `0x8001` (existing behavior)

**Result:** The etFIFO now holds eligibility trace addresses in the **same order** as synapses are being processed. This synchronization is critical for the next step.

---

### Step 2: Neuron Update with Coincidence Detection

As synapses are read from HBM Region 3, they flow to `internal_events_processor` for neuron updates (existing behavior from Chapter 2.2).

**Extension:** While processing each synapse, the internal_events_processor now also:

1. **Pops the corresponding eligibility trace address from etFIFO**
   - Because addresses were pushed in the same order, this stays synchronized

2. **Performs the neuron update** (existing):
   - Read current membrane potential from URAM
   - Add synaptic weight: `V_new = V_old + weight`
   - Check threshold: `spike = (V_new >= threshold)`
   - Write updated potential to URAM

3. **Performs coincidence detection** (new):
   - Check: Did the post-synaptic neuron spike?
   - If YES: We have an STDP event (pre fired → caused input → post fired)
   - This is **coincidence detection**: pre and post activity temporally correlated

4. **Checks reward signal** (new):
   - Read the reward register (broadcast to all processing units)
   - Check: Is dopamine ON?

5. **Conditional weight update trigger** (new):
   - If (coincidence detected) AND (reward signal = 1):
     - Push the Region 4 address to **et2FIFO**

**New FIFO introduced: et2FIFO (Eligibility Trace to Weight Update FIFO)**

| Property | Value |
|----------|-------|
| **Width** | 23 bits (HBM Region 4 address) |
| **Depth** | 512 |
| **Written by** | internal_events_processor (when coincidence + reward detected) |
| **Read by** | hbm_processor weight update logic |
| **Contains** | Region 4 addresses of synapses that should have weights updated |

**What this achieves:**
- Only synapses with **recent STDP events** (recorded in eligibility traces) that also receive **reward** will trigger weight updates
- The et2FIFO acts as a queue of "synapses to update"

---

### Step 3: Weight Updates via Eligibility Traces

While Phase 2 is running (or immediately after), `hbm_processor` monitors the **et2FIFO**. When entries appear, it performs weight updates.

**Process for each entry:**

1. **Pop Region 4 address from et2FIFO**
   - Example: `0x8FA9` (eligibility trace address)

2. **Read eligibility trace value from HBM Region 4**
   - Issue HBM read to address `0x8FA9`
   - Receive 36-bit eligibility trace value: `c(t)`

3. **Compute corresponding synapse address**
   - Reverse the mapping: `Region_3_addr = 0x8FA9 - 1000 = 0x8001`

4. **Read current synapse weight from HBM Region 3**
   - Issue HBM read to address `0x8001`
   - Receive 32-bit synapse data
   - Extract weight: `current_weight = synapse_data[15:0]`

5. **Compute new weight using R-STDP rule**
   - R-STDP: `Δw = R(t) × c(t)`
   - Since we're in this state, `R(t) = 1` (reward is on)
   - Therefore: `Δw = c(t)` (the eligibility trace value)
   - New weight: `w_new = w_old + c(t)`
   - Apply clamping to prevent overflow: `w_new ∈ [-32768, 32767]`

6. **Write updated weight to HBM Region 3**
   - Reconstruct synapse data: `{OpCode, Target, w_new}`
   - Issue HBM write to address `0x8001`
   - **This reuses the same HBM write logic** that's called when the host updates weights via `write_synapse` commands

**Key insight:** The weight update path reuses existing infrastructure. We just need to:
- Cut into the write_synapse function at the point where it has the Region 3 address and new weight value
- Provide those values from our R-STDP computation instead of from the host

---

### Step 4: Eligibility Trace Maintenance

Eligibility traces need to be updated to:
1. **Decay over time** (like membrane potential leak)
2. **Increase when STDP events occur** (coincidence detection)

**Mathematical update:**

$$\dot{c}(t) = -\frac{c}{\tau_c} + \delta_{\text{STDP event}}$$

**Implementation approach: Reuse neuron membrane potential logic**

The key insight is that eligibility trace updates are **nearly identical** to neuron updates:
- Both are 36-bit values stored in memory
- Both decay exponentially (leak)
- Both accumulate inputs
- **Difference:** Eligibility traces don't spike (no threshold comparison)

**Method:**
- Read eligibility trace from Region 4: `c_old`
- Apply decay: `c_new = c_old - (c_old >> leak_shift)`
  - This is the same right-shift leak used for neurons
  - `leak_shift` determines τ_c (longer decay than neurons)
- If STDP event detected for this synapse: `c_new += STDP_INCREMENT`
- Write back to Region 4: `c_new`

**When to perform these updates:**
- **Option A**: During Phase 2 (parallel with neuron updates)
- **Option B**: Separate Phase 3 after neuron updates complete
- **Option C**: Lazy update - only when accessed for weight updates
  - More efficient: don't read/write all eligibility traces every timestep
  - Only update the ones being used

---

## Complete Phase 2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 2 WITH R-STDP                      │
└─────────────────────────────────────────────────────────────┘

Step 1: Pointer Processing
─────────────────────────
pointer_fifo_controller
    │
    └──> ptrFIFO (pop synapse pointer)
           │
           └──> hbm_processor
                  │
                  ├──> Compute Region 3 address (synapse)
                  ├──> Compute Region 4 address (eligibility trace)
                  ├──> Push Region 4 addr to etFIFO ───┐
                  └──> Read synapse from HBM Region 3  │
                                                        │
Step 2: Neuron Update + Coincidence Detection          │
───────────────────────────────────────────            │
hbm_processor forwards synapse data                    │
    │                                                   │
    └──> internal_events_processor                     │
           │                                            │
           ├──> Pop Region 4 addr from etFIFO <────────┘
           ├──> Update neuron (URAM):
           │      V_new = V_old + weight
           ├──> Check threshold: spike?
           ├──> Coincidence detection: post fired?
           ├──> Check reward register: DA on?
           │
           └──> If (coincidence AND reward):
                  Push Region 4 addr to et2FIFO ───┐
                                                    │
Step 3: Weight Updates                              │
──────────────────────                              │
hbm_processor (parallel state machine)              │
    │                                                │
    ├──> Pop Region 4 addr from et2FIFO <──────────┘
    ├──> Read eligibility trace from Region 4
    ├──> Compute Region 3 addr (subtract offset)
    ├──> Read current weight from Region 3
    ├──> Compute: w_new = w_old + et_value
    └──> Write w_new to Region 3 (reuse write_synapse logic)

Step 4: Eligibility Trace Maintenance (parallel or separate)
────────────────────────────────────────────────────────────
For each synapse's eligibility trace:
    ├──> Read from Region 4
    ├──> Apply decay: c_new = c_old - (c_old >> leak_shift)
    ├──> If STDP event: c_new += INCREMENT
    └──> Write back to Region 4
```

---

## Summary of New Components

### Memory Regions
- **HBM Region 4**: Eligibility trace storage (one 36-bit value per synapse)

### FIFOs (introduced as needed in the flow)
- **etFIFO**: Queues eligibility trace addresses during pointer processing
- **et2FIFO**: Queues eligibility traces that should trigger weight updates

### Control Signals
- **Reward Register**: 1-bit dopamine signal (set by new opcode CMD_SET_REWARD)
- **exec_reward**: Broadcast signal from command_interpreter to internal_events_processor

### Reused Logic
- **Neuron update pipeline** → Eligibility trace updates (disable threshold check)
- **HBM write infrastructure** → Weight updates
- **Phase 2 synapse processing** → Coincidence detection point

---

## Key Design Decisions

1. **Simple address mapping**: Region4 = Region3 + constant offset
   - Makes address computation trivial (one addition)
   - Easily reversible for going from eligibility trace back to synapse

2. **Binary reward signal**: On/off only (for now)
   - Simplifies initial implementation
   - Future: Could be multi-bit for graded reward

3. **Coincidence-based STDP**: Post fires in same timestep as pre input
   - Simplified from asymmetric STDP window
   - Good enough for initial learning demonstrations

4. **Reuse existing hardware**: Eligibility traces use neuron logic
   - Minimal new hardware required
   - Proven, tested infrastructure

5. **FIFO-based pipeline**: Maintains synchronization
   - etFIFO keeps addresses aligned with synapse processing
   - et2FIFO queues weight updates for parallel processing

---

## Testing the Implementation

### Simple Test Case

**Network:**
- Neuron A (pre-synaptic) → Neuron B (post-synaptic)
- Single synapse with initial weight `W = 500`
- Threshold = 2000

**Experiment sequence:**

1. **Timestep 0**: Fire A repeatedly without B firing
   - No coincidence → eligibility trace stays at 0

2. **Timestep 5**: Fire A enough to make B spike
   - B receives input, crosses threshold, spikes
   - Coincidence detected: pre (A) was active, post (B) fired
   - Eligibility trace increases (but reward is still OFF)

3. **Timesteps 6-10**: No activity
   - Eligibility trace decays: `c(t) × exp(-Δt/τ_c)`

4. **Timestep 11**: Turn on reward (set reward register = 1)

5. **Timestep 12**: Fire A, B spikes again
   - Coincidence detected
   - Reward is ON
   - → Weight update triggered!
   - Read eligibility trace from Region 4: `c(t) = 100` (example)
   - Compute: `W_new = 500 + 100 = 600`
   - Write updated weight to Region 3

**Validation:**
- Read back synapse weight from HBM → should be 600
- On next input from A, B should spike with less input (stronger synapse)

---

## Potential Issues and Mitigations

### 1. FIFO Overflow
**Issue**: etFIFO or et2FIFO fills up if too many synapses processed

**Mitigation**:
- Size FIFOs same as ptrFIFO (512 entries proven sufficient)
- Add overflow detection flags
- Backpressure: stall pointer processing if etFIFO full

### 2. HBM Bandwidth
**Issue**: Additional reads/writes for Region 4 and weight updates

**Impact analysis**:
- Existing: ~200-300 HBM transactions per timestep
- New: +1 read per synapse (eligibility trace), +2 reads + 1 write per weight update
- For network with 1000 active synapses, 10% coincidence: +100 transactions

**Mitigation**:
- Weight updates run in parallel with ongoing Phase 2
- Burst mode for sequential eligibility trace reads
- Lazy decay (only update accessed traces)

### 3. Address Mapping Collision
**Issue**: Region 4 might overlap with Region 3 if offset too small

**Solution**:
- Choose REGION4_OFFSET > maximum Region 3 size
- Example: If Region 3 uses rows 0x8000-0xFFFF (32K rows)
  - Set REGION4_OFFSET = 0x8000 (32768 in decimal)
  - Region 4 spans: 0x10000-0x17FFF
  - No overlap ✓

### 4. Coincidence Detection Granularity
**Issue**: Current design: "post fired this timestep" = coincidence

**Limitation**: Doesn't distinguish if post fired 1ms or 10ms after pre

**Future enhancement**:
- Track spike times with sub-timestep resolution
- Implement asymmetric STDP window (separate τ+ and τ-)
- Different eligibility trace increments for potentiation vs depression

### 5. Eligibility Trace Decay Timing
**Issue**: When to apply decay? Every timestep for all traces is expensive

**Recommended approach**: Lazy decay
- Store last_update_timestamp with each trace
- When reading for weight update:
  - Calculate time elapsed: `Δt = current_time - last_update_time`
  - Apply accumulated decay: `c = c_stored × exp(-Δt/τ_c)`
- Only write back when updating (not every read)

---

## Python API Usage

