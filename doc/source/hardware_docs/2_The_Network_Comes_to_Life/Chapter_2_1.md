# 2.1 Visualizing the Network's Signal Processing in Hardware

### Key Terms

Before diving into the execution flow, let's define some important hardware terms used throughout this chapter:

**Group (Neuron Group)**
A collection of neurons managed by a single URAM bank. The hardware has 16 URAM banks (URAM0-URAM15), so there are 16 groups numbered 0-15. Each group can hold approximately 8,192 neurons (131,072 total ÷ 16 banks). When we say "group 0 has spikes," we mean neurons stored in URAM bank 0 have spiked.

**Mask (Spike Mask)**
A bit vector where each bit indicates whether a group has spiking activity. For example, `exec_bram_spiked[15:0] = 0x0007` means bits 0, 1, and 2 are set, indicating groups 0, 1, and 2 have spiking neurons. Each bit corresponds to one of the 16 URAM banks (neuron groups).

**Muxing/Demuxing (Multiplexing/Demultiplexing)**
- **Demuxing**: Taking one input and routing it to one of multiple outputs based on a selector. In Phase 1, `pointer_fifo_controller` demuxes HBM data (containing 16 pointers) to 16 separate pointer FIFOs based on the spike mask.
- **Muxing**: Taking multiple inputs and selecting one as the output. In Phase 2, `pointer_fifo_controller` muxes (round-robin reads) from 16 pointer FIFOs to select which pointer to process next.

**Round-Robin Reading**
A fair scheduling algorithm that cycles through resources in sequential order. The `pointer_fifo_controller` reads from ptr0 → ptr1 → ptr2 → ... → ptr15 → ptr0 (looping back), ensuring all groups get equal processing time.

**FIFO (First In, First Out)**
A queue data structure where the first item added is the first item removed. The hardware uses multiple types of FIFOs:
- **Pointer FIFOs (ptr0-ptr15)**: 16 FIFOs that store 32-bit HBM addresses pointing to synapse data locations. Each FIFO corresponds to one neuron group.
- **Spike FIFOs (spk0-spk7)**: 8 FIFOs that store 17-bit neuron addresses (neuron IDs that spiked). These aggregate spikes before sending to the host.
- **rxFIFO/txFIFO**: PCIe communication buffers (512-bit wide) for receiving commands from and sending results to the host.

**One-Hot / Multi-Hot Encoding**
A binary representation where bits indicate active states:
- **One-hot**: Exactly one bit is set to 1. Example: `0x0004` (only bit 2 set) means only group 2 is active.
- **Multi-hot**: Multiple bits can be set to 1. Example: `exec_bram_spiked = 0x0007` (bits 0, 1, 2 set) means groups 0, 1, and 2 all have spiking neurons.
The spike masks (`exec_bram_spiked`, `exec_uram_spiked`) use multi-hot encoding because multiple groups can have spikes simultaneously.

**Pointer (HBM Address)**
A 32-bit value that tells the hardware WHERE to find synapse data in HBM memory. Format: `{length[31:23], base_address[22:0]}`.
- `length`: How many 256-bit rows to read (number of synapses ÷ 8)
- `base_address`: Starting location in HBM
Example: `0x02001000` means "read 2 rows starting at HBM address 0x1000"

**Important distinction**: Pointer FIFOs store ADDRESSES (where to find data), NOT the actual synapse data. The synapse data is only read in Phase 2 using these addresses.

---

### Overview: The Two-Phase Execution Model

Every timestep of network execution goes through **two main phases**, with each phase potentially executing multiple times if neurons spike and trigger recurrent processing.

```
SETUP: INPUT RECEPTION (Host → FPGA)
┌──────────────────────────────────────────────────────────────┐
│ Host sends input spikes via network.step(['a0', 'a1', ...]) │
│ Written to: BRAM (Future buffer, 256-bit spike masks)       │
│ Time: ~1-10 microseconds (depends on PCIe transfer)          │
│ Status: Network is IDLE, waiting for execute command         │
│ Next: Buffer role reassignment (bram_select flip)            │
└──────────────────────────────────────────────────────────────┘
                           │
                           │ execute() command sent
                           │ Buffer roles reassigned: Future buffer → Present, Present buffer → Future
                           ▼
PHASE 1: POINTER COLLECTION
┌──────────────────────────────────────────────────────────────┐
│ Goal: Find which synapses need to be processed               │
│                                                               │
│ PHASE 1a: External Events (Axon Spikes)                      │
│   - external_events_processor reads Present BRAM              │
│   - Generates exec_bram_spiked[15:0] mask                    │
│   - hbm_processor reads axon pointers from HBM Region 1      │
│   - pointer_fifo_controller demuxes pointers → 16 FIFOs      │
│                                                               │
│ PHASE 1b: Internal Events (Neuron Spikes from previous cycle)│
│   - internal_events_processor reads URAM states              │
│   - Generates exec_uram_spiked[15:0] mask                    │
│   - hbm_processor reads neuron pointers from HBM Region 2    │
│   - pointer_fifo_controller continues filling 16 FIFOs       │
│                                                               │
│ Output: 16 pointer FIFOs filled with addresses               │
│ Time: ~1-5 microseconds (depends on # of spiking sources)    │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
PHASE 2: SYNAPTIC PROCESSING & NEURON UPDATES
┌──────────────────────────────────────────────────────────────┐
│ Goal: Read synapses, accumulate weights, update neurons      │
│                                                               │
│ - pointer_fifo_controller: Round-robin read from 16 FIFOs    │
│ - hbm_processor: Pop pointer → read synapses from Region 3   │
│ - hbm_processor: Parse synapses → forward to IEP             │
│ - internal_events_processor: Accumulate weights into neurons │
│ - internal_events_processor: Apply neuron model (leak, etc.) │
│ - internal_events_processor: Threshold check → spike?        │
│ - internal_events_processor: Write updated state to URAM     │
│ - Spikes → spike_fifo_controller → command_interpreter       │
│                                                               │
│ Output: Updated neuron states in URAM, new spikes generated  │
│ Time: ~1-10 microseconds (depends on # of synapses)          │
│                                                               │
│ If neurons spiked: Loop back to Phase 1b                     │
│ If no more spikes: Timestep complete                         │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
SPIKE OUTPUT (FPGA → Host)
┌──────────────────────────────────────────────────────────────┐
│ command_interpreter batches spikes (14 per 512-bit packet)   │
│ Sends via txFIFO → pcie2fifos → PCIe → Host                  │
│ Host reads via flush_spikes()                                │
│ Time: ~1-5 microseconds                                      │
└──────────────────────────────────────────────────────────────┘
```

**Total time per timestep:** ~5-30 microseconds depending on network activity
- Small networks with few spikes: ~5 µs
- Large networks with many spikes: ~30 µs
- Target: 1 millisecond per timestep → can run 30-200 timesteps in real-time

**Key insight:** Phase 1 collects *addresses* of synapses to process. Phase 2 fetches and processes those synapses. This two-phase approach allows efficient bulk memory access.

---

### The Data Journey: What Moves Where

Let's visualize where data lives at each phase, using our example where **axons a0, a1, a2 fire**.

#### Before Setup: Network is Idle

```
┌─────────────────────────────────────────────────────────────┐
│                        HOST                                 │
│  Python code:                                               │
│    network.step(['a0', 'a1', 'a2'])                         │
│                                                             │
│  In memory: List of axon names to send                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        FPGA                                 │
│                                                             │
│  BRAM (Double Buffer):                                     │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Present: [0x00...00] - Processing last timestep      │ │
│  │ Future:  [0x00...00] - Ready to receive new inputs   │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  HBM: Contains network structure (from initialization)     │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Region 1: Axon pointers (where each axon's synapses) │ │
│  │ Region 2: Neuron pointers (where each neuron's outs) │ │
│  │ Region 3: Synapse data (target addresses + weights)  │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  URAM: All neurons at V=0 (just cleared)                   │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ h0: V=0, h1: V=0, h2: V=0, h3: V=0, h4: V=0          │ │
│  │ o0: V=0, o1: V=0, o2: V=0, o3: V=0, o4: V=0          │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### After Setup: Inputs Written to BRAM Future Buffer

The host writes a 256-bit spike mask to BRAM Future buffer:

```
┌─────────────────────────────────────────────────────────────┐
│                        FPGA BRAM                            │
│                                                             │
│  Future Buffer (Row 0): 256-bit spike mask                 │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Bit 0 = 1: Axon a0 is firing                          │ │
│  │  Bit 1 = 1: Axon a1 is firing                          │ │
│  │  Bit 2 = 1: Axon a2 is firing                          │ │
│  │  Bit 3 = 0: Axon a3 is silent                          │ │
│  │  Bit 4 = 0: Axon a4 is silent                          │ │
│  │  Bits 5-255 = 0: No other axons (network only has 5)   │ │
│  │                                                          │ │
│  │  As hex bytes (little-endian):                         │ │
│  │  Byte [0] = 0x07 = 0b00000111 (bits 0,1,2 set)         │ │
│  │  Bytes [1]-[31] = 0x00 (all zeros)                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  Present Buffer: Still contains last timestep's data       │
│  Status: Waiting for execute() command                     │
└─────────────────────────────────────────────────────────────┘
```

When execute() is called:
1. BRAM buffer roles reassigned: `bram_select` flips (no data movement)
2. Future buffer cleared for next timestep
3. Phase 1a begins

---

#### During Phase 1a: External Event Processing (Pointer Collection)

```
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 1a DATA FLOW                        │
│                                                             │
│  Step 1: external_events_processor reads Present BRAM       │
│    Row 0 data: 0x00...00000007                             │
│    Detected: Bits 0,1,2 are set                            │
│    Output: exec_bram_spiked[15:0] = 0x0007                 │
│            (Neuron group 0 has 3 active axons)             │
│                                                             │
│  Step 2: hbm_processor reads axon pointers from HBM         │
│                                                             │
│    HBM Region 1, Row 0 (Axon Pointer Storage):             │
│    Each pointer = 32 bits: [31:23]=length, [22:0]=address  │
│                                                             │
│    ┌─ Axon a0 pointer ─────────────────────────────────┐   │
│    │ Read HBM[Region1 + 0]:                           │   │
│    │ Pointer = {length: 1 row, start: 0x8000}         │   │
│    │ Meaning: "a0's synapses are at HBM[0x8000]"      │   │
│    └───────────────────────────────────────────────────┘   │
│                                                             │
│    ┌─ Axon a1 pointer ─────────────────────────────────┐   │
│    │ Read HBM[Region1 + 4 bytes]:                     │   │
│    │ Pointer = {length: 1 row, start: 0x8001}         │   │
│    │ Meaning: "a1's synapses are at HBM[0x8001]"      │   │
│    └───────────────────────────────────────────────────┘   │
│                                                             │
│    ┌─ Axon a2 pointer ─────────────────────────────────┐   │
│    │ Read HBM[Region1 + 8 bytes]:                     │   │
│    │ Pointer = {length: 1 row, start: 0x8002}         │   │
│    └───────────────────────────────────────────────────┘   │
│                                                             │
│  Step 3: pointer_fifo_controller demuxes pointers           │
│                                                             │
│    Input: 512-bit HBM data with 16 pointers                │
│    Input: exec_bram_spiked[15:0] = 0x0007 (group 0 only)   │
│    Action: Write pointers to ptr0 FIFO (others unused)     │
│                                                             │
│    Pointer FIFO 0 now contains:                            │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ Entry 0: 0x8000 (a0's synapse row address)         │ │
│    │ Entry 1: 0x8001 (a1's synapse row address)         │ │
│    │ Entry 2: 0x8002 (a2's synapse row address)         │ │
│    └─────────────────────────────────────────────────────┘ │
│                                                             │
│    Pointer FIFOs 1-15: Empty (no activity in other groups) │
└─────────────────────────────────────────────────────────────┘
```

**Critical understanding:** Phase 1a ONLY collects addresses (pointers). The actual synapse data (weights, targets) is NOT read yet. This allows efficient batching of memory accesses in Phase 2.

---

#### During Phase 1b: Internal Event Processing (If neurons spiked)

In the first timestep with V=0 everywhere, no neurons have spiked yet, so Phase 1b is skipped initially. But let's see what happens when neurons DO spike:

```
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 1b DATA FLOW                        │
│         (Executed when neurons spiked in previous cycle)    │
│                                                             │
│  Step 1: internal_events_processor checks for spikes        │
│    Reads URAM states, applies threshold                    │
│    If neuron V >= threshold: mark as spiked                │
│    Output: exec_uram_spiked[15:0] mask                     │
│                                                             │
│  Example: If h0, h1, h2, h3, h4 all spiked:                │
│    exec_uram_spiked[0] = 0x001F (bits 0-4 set)             │
│    exec_uram_spiked[1-15] = 0x0000 (no neurons in groups)  │
│                                                             │
│  Step 2: hbm_processor reads neuron pointers from HBM       │
│                                                             │
│    HBM Region 2, Group 0 entries (Neuron Pointer Storage): │
│                                                             │
│    ┌─ Neuron h0 pointer ───────────────────────────────┐   │
│    │ Read HBM[Region2 + h0_offset]:                   │   │
│    │ Pointer = {length: 1 row, start: 0xA000}         │   │
│    │ Meaning: "h0's output synapses at HBM[0xA000]"   │   │
│    └───────────────────────────────────────────────────┘   │
│                                                             │
│    Similarly for h1, h2, h3, h4...                         │
│                                                             │
│  Step 3: pointer_fifo_controller adds to FIFOs             │
│                                                             │
│    Pointer FIFO 0 now contains (appended):                 │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ Entry 0: 0x8000 (a0's synapses) ← from Phase 1a    │ │
│    │ Entry 1: 0x8001 (a1's synapses) ← from Phase 1a    │ │
│    │ Entry 2: 0x8002 (a2's synapses) ← from Phase 1a    │ │
│    │ Entry 3: 0xA000 (h0's synapses) ← from Phase 1b    │ │
│    │ Entry 4: 0xA001 (h1's synapses) ← from Phase 1b    │ │
│    │ Entry 5: 0xA002 (h2's synapses) ← from Phase 1b    │ │
│    │ Entry 6: 0xA003 (h3's synapses) ← from Phase 1b    │ │
│    │ Entry 7: 0xA004 (h4's synapses) ← from Phase 1b    │ │
│    └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Key point:** Phase 1b adds MORE pointers to the same FIFOs. This allows processing both external inputs (axons) and internal recurrent connections (neuron spikes) in a single Phase 2 sweep.

---

#### During Phase 2: Synaptic Read and Neuron Updates

Now the actual work happens:

```
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: SYNAPTIC PROCESSING                   │
│                                                             │
│  Step 1: pointer_fifo_controller round-robin reads FIFOs    │
│                                                             │
│    Cycle 0: Check FIFO 0 → not empty                       │
│             Pop entry: 0x8000 (a0's synapse row)           │
│             Send to hbm_processor                          │
│                                                             │
│  Step 2: hbm_processor reads synapse data from HBM Region 3 │
│                                                             │
│    ┌─ Read HBM[0x8000] (a0's synapse row) ──────────────┐  │
│    │ HBM returns 256-bit data = 8 × 32-bit synapses    │  │
│    │                                                     │  │
│    │ Synapse format: [31:29]=OpCode, [28:16]=Target,   │  │
│    │                 [15:0]=Weight                      │  │
│    │                                                     │  │
│    │ Synapse 0: {op=0, target=h0 (addr 0), wt=1000}    │  │
│    │ Synapse 1: {op=0, target=h1 (addr 1), wt=1000}    │  │
│    │ Synapse 2: {op=0, target=h2 (addr 2), wt=1000}    │  │
│    │ Synapse 3: {op=0, target=h3 (addr 3), wt=1000}    │  │
│    │ Synapse 4: {op=0, target=h4 (addr 4), wt=1000}    │  │
│    │ Synapses 5-7: Unused (padding)                    │  │
│    └─────────────────────────────────────────────────────┘  │
│                                                             │
│  Step 3: hbm_processor forwards synapse data to IEP         │
│                                                             │
│    Signal: exec_hbm_rdata[511:0] (512 bits = 16 synapses)  │
│    Contains: 5 valid synapses from a0                      │
│                                                             │
│  Step 4: internal_events_processor accumulates              │
│                                                             │
│    Process Synapse 0: {target=h0, weight=1000}             │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ Cycle 0: Read URAM[h0] → V_old = 0                 │ │
│    │ Cycle 1: Compute V_new = 0 + 1000 = 1000           │ │
│    │ Cycle 2: Check threshold: 1000 < 2000 → no spike   │ │
│    │ Cycle 3: Write URAM[h0] = 1000                     │ │
│    └─────────────────────────────────────────────────────┘ │
│                                                             │
│    Process Synapse 1: {target=h1, weight=1000}             │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ Cycle 4: Read URAM[h1] → V_old = 0                 │ │
│    │ Cycle 5: Compute V_new = 0 + 1000 = 1000           │ │
│    │ Cycle 6: Write URAM[h1] = 1000                     │ │
│    └─────────────────────────────────────────────────────┘ │
│                                                             │
│    ... (similar for h2, h3, h4)                            │
│                                                             │
│    After a0's synapses processed:                          │
│    URAM: h0=1000, h1=1000, h2=1000, h3=1000, h4=1000       │
│                                                             │
│  Step 5: Continue round-robin, pop next pointer            │
│                                                             │
│    Cycle N: Pop entry: 0x8001 (a1's synapse row)          │
│             Read HBM[0x8001] → 5 more synapses             │
│             Accumulate into h0-h4                          │
│                                                             │
│    Process a1's Synapse 0: {target=h0, weight=1000}        │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ Read URAM[h0] → V_old = 1000                        │ │
│    │ Compute V_new = 1000 + 1000 = 2000                  │ │
│    │ Check threshold: 2000 >= 2000 → SPIKE!              │ │
│    │ Write URAM[h0] = 0 (reset after spike)              │ │
│    │ Generate spike: neuron_id = h0                      │ │
│    │ Send to spike FIFO                                  │ │
│    └─────────────────────────────────────────────────────┘ │
│                                                             │
│    Similarly, h1-h4 all reach threshold and spike          │
│                                                             │
│    After a1's synapses processed:                          │
│    URAM: h0=0, h1=0, h2=0, h3=0, h4=0 (all reset)          │
│    Spikes: h0, h1, h2, h3, h4 sent to spike FIFOs          │
│                                                             │
│  Step 6: Continue with a2's synapses                       │
│                                                             │
│    Cycle M: Pop entry: 0x8002 (a2's synapse row)          │
│             Read HBM[0x8002] → 5 more synapses             │
│             Accumulate into h0-h4                          │
│                                                             │
│    After a2's synapses processed:                          │
│    URAM: h0=1000, h1=1000, h2=1000, h3=1000, h4=1000       │
│          (Each got 1 more input after spiking)             │
│                                                             │
│  Result after first Phase 2 pass:                          │
│    - Hidden neurons spiked once (at V=2000)                │
│    - Now at V=1000 (accumulated 3 inputs total)            │
│    - Spikes generated: h0, h1, h2, h3, h4                  │
└─────────────────────────────────────────────────────────────┘
```

**Critical insight:** Neurons can spike in the MIDDLE of Phase 2 processing. When h0 receives input from a0 (V=1000), then from a1 (V=2000 → spike → reset), then from a2 (V=1000 again), all within the same phase.

---

#### Recurrent Processing: Hidden Neuron Spikes

When neurons spike in Phase 2, they trigger another round of Phase 1b → Phase 2:

```
┌─────────────────────────────────────────────────────────────┐
│         RECURRENT CYCLE (Hidden → Output Layer)             │
│                                                             │
│  Hidden neurons h0-h4 spiked during Phase 2                 │
│  These spikes trigger Phase 1b again                        │
│                                                             │
│  ═══════════════════════════════════════════════════════    │
│  PHASE 1b (Recurrent): Collect neuron output pointers       │
│  ═══════════════════════════════════════════════════════    │
│                                                             │
│  internal_events_processor:                                 │
│    exec_uram_spiked[0] = 0x001F (h0-h4 spiked)             │
│                                                             │
│  hbm_processor reads neuron pointers from HBM Region 2:     │
│    h0 pointer → 0xA000                                     │
│    h1 pointer → 0xA001                                     │
│    h2 pointer → 0xA002                                     │
│    h3 pointer → 0xA003                                     │
│    h4 pointer → 0xA004                                     │
│                                                             │
│  pointer_fifo_controller writes to FIFOs:                   │
│    Pointer FIFO 0 gets: 0xA000, 0xA001, 0xA002, 0xA003,    │
│                         0xA004                              │
│                                                             │
│  ═══════════════════════════════════════════════════════    │
│  PHASE 2 (Recurrent): Process hidden → output synapses      │
│  ═══════════════════════════════════════════════════════    │
│                                                             │
│  Pop 0xA000 from FIFO → Read HBM[0xA000] (h0's outputs):   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Synapse 0: {op=0,   target=o0, weight=1000}        │   │
│  │ Synapse 1: {op=0,   target=o1, weight=1000}        │   │
│  │ Synapse 2: {op=0,   target=o2, weight=1000}        │   │
│  │ Synapse 3: {op=0,   target=o3, weight=1000}        │   │
│  │ Synapse 4: {op=100, target=o4, weight=1000}        │   │
│  │    ↑ OpCode 100 = "send spike to host"             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  internal_events_processor accumulates h0's outputs:        │
│    o0: V = 0 + 1000 = 1000  (OpCode 0: just accumulate)    │
│    o1: V = 0 + 1000 = 1000  (OpCode 0: just accumulate)    │
│    o2: V = 0 + 1000 = 1000  (OpCode 0: just accumulate)    │
│    o3: V = 0 + 1000 = 1000  (OpCode 0: just accumulate)    │
│    o4: V = 0 + 1000 = 1000  (OpCode 100: accumulate)       │
│                                                             │
│  Pop 0xA001, read HBM[0xA001] (h1's outputs):              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Synapse 0: {op=0,   target=o0, weight=1000}        │   │
│  │ Synapse 1: {op=0,   target=o1, weight=1000}        │   │
│  │ Synapse 2: {op=0,   target=o2, weight=1000}        │   │
│  │ Synapse 3: {op=0,   target=o3, weight=1000}        │   │
│  │ Synapse 4: {op=100, target=o4, weight=1000}        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Processing h1's synapses:                                  │
│    o0: V = 1000 + 1000 = 2000 → SPIKE! → V=0               │
│       OpCode 0: No host output, spike only triggers Phase1b│
│    o1: V = 1000 + 1000 = 2000 → SPIKE! → V=0               │
│       OpCode 0: No host output                              │
│    o2: V = 1000 + 1000 = 2000 → SPIKE! → V=0               │
│       OpCode 0: No host output                              │
│    o3: V = 1000 + 1000 = 2000 → SPIKE! → V=0               │
│       OpCode 0: No host output                              │
│    o4: V = 1000 + 1000 = 2000 → SPIKE! → V=0               │
│       OpCode 100: hbm_processor writes to spike FIFO!       │
│       ┌───────────────────────────────────────────────┐     │
│       │ spike_id = o4 (17-bit neuron address)        │     │
│       │ group = o4[16:14] % 8 → spk0                 │     │
│       │ Write {17'd4} to spk0 FIFO                   │     │
│       └───────────────────────────────────────────────┘     │
│                                                             │
│  Continue with h2, h3, h4 outputs (all similar):           │
│    Each processes 5 synapses (4 OpCode 0, 1 OpCode 100)    │
│    When o0-o4 spike again from h2's input:                  │
│      o0-o3: OpCode 0 → internal only, no host output       │
│      o4: OpCode 100 → write to spk0 (second time)          │
│    When o0-o4 spike again from h3's input:                  │
│      o4: OpCode 100 → write to spk0 (third time)           │
│    When o0-o4 spike again from h4's input:                  │
│      o4: OpCode 100 → write to spk0 (fourth time)          │
│                                                             │
│  Final state after recurrent processing:                   │
│    URAM output neurons: o0=3000, o1=3000, o2=3000,         │
│                         o3=3000, o4=3000                   │
│    (Each received 5 inputs × 1000 = 5000, spiked at 2000,  │
│     reset to 0, then accumulated 3 more inputs = 3000)     │
│                                                             │
│  Spike FIFOs after recurrent processing:                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ spk0: [o4, o4, o4, o4]  (4 spikes from o4)         │   │
│  │       ↑ o4 spiked 4 times, each recorded            │   │
│  │ spk1-spk7: Empty (no neurons in those groups)       │   │
│  │                                                      │   │
│  │ Note: o0-o3 also spiked 4 times each, but OpCode 0  │   │
│  │       means "internal only" - not sent to host      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key difference from external events:**
- External axon spikes → Phase 1a (read from BRAM)
- Internal neuron spikes → Phase 1b (read from URAM state)
- Both types feed into the same pointer FIFOs
- Phase 2 processes them identically

---

#### Spike Aggregation and Transmission to Host

After Phase 2 completes, the spike FIFOs contain all neuron IDs that spiked with OpCode 100. These spikes now need to be aggregated and sent to the host:

```
┌─────────────────────────────────────────────────────────────┐
│           SPIKE COLLECTION AND HOST TRANSMISSION            │
│                                                             │
│  Current state: spk0 contains [o4, o4, o4, o4]             │
│                 spk1-spk7 are empty                         │
│                                                             │
│  Step 1: spike_fifo_controller round-robin aggregation      │
│                                                             │
│    Round-robin read from spk0 → spk1 → ... → spk7:        │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ Cycle 0: Check spk0 → not empty → pop first o4     │  │
│    │          Write {17'd4} to spk2ciFIFO                │  │
│    │ Cycle 1: Check spk1 → empty → skip                 │  │
│    │ Cycle 2: Check spk2 → empty → skip                 │  │
│    │ ...                                                  │  │
│    │ Cycle 7: Check spk7 → empty → skip                 │  │
│    │ Cycle 8: Check spk0 → not empty → pop second o4    │  │
│    │          Write {17'd4} to spk2ciFIFO                │  │
│    │ ...continues for all 4 spikes...                    │  │
│    └─────────────────────────────────────────────────────┘  │
│                                                             │
│    spk2ciFIFO now contains: [o4, o4, o4, o4]               │
│                                                             │
│  Step 2: command_interpreter batches spikes into packets    │
│                                                             │
│    Reads from spk2ciFIFO and packs into 512-bit packets:    │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ Packet format:                                      │  │
│    │   [511:480] = 0xEEEEEEEE  (spike output opcode)    │  │
│    │   [479:463] = spike_0  (17-bit neuron ID)          │  │
│    │   [462:446] = spike_1  (17-bit neuron ID)          │  │
│    │   ...                                               │  │
│    │   [241:225] = spike_13 (17-bit neuron ID)          │  │
│    │   [224:0]   = padding  (unused for < 14 spikes)    │  │
│    │                                                      │  │
│    │ Our packet with 4 o4 spikes:                        │  │
│    │   [511:480] = 0xEEEEEEEE                            │  │
│    │   [479:463] = 17'd4  (first o4 spike)              │  │
│    │   [462:446] = 17'd4  (second o4 spike)             │  │
│    │   [445:429] = 17'd4  (third o4 spike)              │  │
│    │   [428:412] = 17'd4  (fourth o4 spike)             │  │
│    │   [411:0]   = 0 (padding, spikes 5-13 unused)      │  │
│    └─────────────────────────────────────────────────────┘  │
│                                                             │
│  Step 3: Transmit packet to host via PCIe                   │
│                                                             │
│    command_interpreter → txFIFO → pcie2fifos → PCIe → Host  │
│                                                             │
│  Step 4: Host reads spikes with flush_spikes()              │
│                                                             │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ # In Python (hs_bridge/network.py)                 │  │
│    │ spikes = fpga_controller.flush_spikes(coreID)      │  │
│    │                                                      │  │
│    │ # FPGA sends: 512-bit packet with opcode 0xEEEEEEEE│  │
│    │ # Python unpacks: [17'd4, 17'd4, 17'd4, 17'd4]     │  │
│    │ # Looks up neuron names: neuron_id=4 → "o4"        │  │
│    │ # Returns: ['o4', 'o4', 'o4', 'o4']                │  │
│    │                                                      │  │
│    │ # Result: o4 spiked 4 times this timestep          │  │
│    └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Why multiple spikes from the same neuron?**
Neuron o4 spiked 4 times during this timestep because it received 5 inputs (one from each h0-h4). It spiked when it reached threshold (2000), reset to 0, then accumulated more inputs and potentially spiked again. Each spike is recorded separately, allowing the host to see the temporal spike count.

**Why 8 spike FIFOs?**
The 8 parallel spike FIFOs (spk0-spk7) allow `hbm_processor` to write spikes from different neuron groups without contention. Neurons are distributed across 16 URAM banks (groups 0-15), and routing them to 8 FIFOs (group % 8) reduces write conflicts when multiple neurons spike simultaneously.

**Why batch 14 spikes per packet?**
The PCIe interface uses 512-bit packets. With a 32-bit opcode (0xEEEEEEEE) and 17-bit spike IDs, we can fit: (512 - 32) ÷ 17 = 28.2 → 14 spikes per packet (rounded down, with some padding).

---

#### Final State After One Complete Timestep

```
┌─────────────────────────────────────────────────────────────┐
│                  FINAL STATE (Timestep 0)                   │
│                                                             │
│  URAM (Neuron States):                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ h0: V = 1000  (got 3 inputs, spiked at 2000, reset,│   │
│  │                then 1 more input)                   │   │
│  │ h1: V = 1000                                        │   │
│  │ h2: V = 1000                                        │   │
│  │ h3: V = 1000                                        │   │
│  │ h4: V = 1000                                        │   │
│  │                                                      │   │
│  │ o0: V = 3000  (got 5 inputs, spiked at 2000, reset,│   │
│  │                then 3 more inputs)                  │   │
│  │ o1: V = 3000                                        │   │
│  │ o2: V = 3000                                        │   │
│  │ o3: V = 3000                                        │   │
│  │ o4: V = 3000                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Spike Output (sent to host):                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Host received: ['o4', 'o4', 'o4', 'o4']             │   │
│  │                                                      │   │
│  │ o4 spiked 4 times (OpCode 100 - send to host)       │   │
│  │ o0-o3 spiked 4 times each (OpCode 0 - internal only)│   │
│  │                                                      │   │
│  │ Spike routing summary:                               │   │
│  │ • Hidden neurons h0-h4: Internal (not sent to host) │   │
│  │ • Output neurons o0-o3: OpCode 0 (internal only)    │   │
│  │ • Output neuron o4: OpCode 100 (sent to host)       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

### Timing: How Long Does This Take?

Let's estimate the time for one complete timestep:

```
SETUP: Input Writing (Host → FPGA BRAM)
├─ Python: Convert axon names → 256-bit mask: ~100 ns (CPU)
├─ DMA command: Queue PCIe write: ~500 ns
├─ PCIe transfer: 256 bits ÷ 14 GB/s = 2 ns
├─ BRAM write: 1 cycle @ 225 MHz = 4.4 ns
└─ Total: ~600 nanoseconds

PHASE 1a: External Events (First Pass - Axon Spikes)
├─ Buffer role reassignment: 1 cycle = 4.4 ns
├─ BRAM read pipeline fill: 3 cycles = 13.2 ns
├─ BRAM read: 1 row @ 225 MHz = 4.4 ns
├─ Generate exec_bram_spiked: 1 cycle = 4.4 ns
├─ HBM read axon pointers:
│   ├─ AXI4 address phase: ~10 ns
│   ├─ HBM latency: ~100 ns
│   ├─ AXI4 data phase: ~10 ns
│   └─ Total per read: ~120 ns
│   └─ 3 axons (can pipeline): ~150 ns
├─ Demux to pointer FIFOs: 3 pointers × 4.4ns = 13.2 ns
└─ Total: ~200 nanoseconds

PHASE 2: First Pass (Process Axon Synapses)
├─ Round-robin FIFO reads: 3 pointers × 4.4ns = 13.2 ns
├─ HBM read synapses (Region 3):
│   ├─ 3 rows × 120ns HBM latency (pipelined) = ~180 ns
├─ Parse synapse data: 15 synapses × 2.2ns = 33 ns
├─ Neuron updates @ 450 MHz (URAM operations):
│   ├─ Each synapse: Read URAM (3 cycles) + Compute (1 cycle)
│   │                + Write URAM (1 cycle) = 5 cycles
│   ├─ 5 cycles @ 450 MHz = 11.1 ns per synapse
│   └─ 15 synapses × 11.1 ns = 167 ns
│       (Can overlap with HBM reads via pipelining: ~100ns actual)
├─ Spike detection & FIFO write: 5 spikes × 2.2ns = 11 ns
└─ Total: ~340 nanoseconds

PHASE 1b: Internal Events (Hidden Neuron Spikes)
├─ URAM read pipeline fill: 3 cycles @ 450 MHz = 6.7 ns
├─ URAM reads for spike detection: 10 neurons ÷ 16 banks
│   ├─ 1 parallel read × 2.2ns = 2.2 ns
├─ Generate exec_uram_spiked: 1 cycle = 2.2 ns
├─ HBM read neuron pointers (Region 2):
│   ├─ 5 neurons × 120ns (pipelined) = ~180 ns
├─ Write to pointer FIFOs: 5 pointers × 4.4ns = 22 ns
└─ Total: ~210 nanoseconds

PHASE 2: Second Pass (Process Hidden → Output Synapses)
├─ Round-robin FIFO reads: 5 pointers × 4.4ns = 22 ns
├─ HBM read synapses: 5 rows × 120ns (pipelined) = ~200 ns
├─ Parse synapse data: 25 synapses × 2.2ns = 55 ns
├─ Neuron updates: 25 synapses × 11.1 ns = 278 ns
│   (Pipelined with HBM: ~150ns actual)
├─ Spike detection: 5 output spikes × 2.2ns = 11 ns
└─ Total: ~440 nanoseconds

SPIKE OUTPUT (FPGA → Host)
├─ spike_fifo_controller aggregation:
│   ├─ Round-robin from 8 FIFOs: ~10 ns
├─ command_interpreter batching:
│   ├─ Batch 5 spikes into 512-bit packet: ~20 ns
│   ├─ Add opcode (0xEEEEEEEE): 1 cycle = 4.4 ns
│   ├─ Write to txFIFO: 1 cycle = 4.4 ns
├─ PCIe transfer: 512 bits ÷ 14 GB/s = 3.7 ns
└─ Total: ~40 nanoseconds

═══════════════════════════════════════════════════════════
TOTAL TIME PER TIMESTEP: ~1.8 microseconds
═══════════════════════════════════════════════════════════
```

**1.8 microseconds per timestep!** This means the hardware can theoretically run **550,000 timesteps per second**.

For our small network:
- Setup: 0.6 µs
- Phase 1a: 0.2 µs
- Phase 2 (pass 1): 0.34 µs
- Phase 1b: 0.21 µs
- Phase 2 (pass 2): 0.44 µs
- Spike output: 0.04 µs
- **Total: ~1.8 µs**

In practice, you're limited by:
- PCIe overhead and host processing: ~5-10 µs per timestep
- Python overhead in host code: ~100-500 µs per timestep
- Network size: larger networks take longer (more HBM reads)

But you can easily achieve **1000 timesteps/second** (1 millisecond/timestep) for real-time operation, with room to spare for much larger networks.

---

### Key Hardware Mechanisms Explained

#### 1. The Two-Phase Architecture

**Why separate pointer collection (Phase 1) from synapse processing (Phase 2)?**

This allows **efficient memory coalescing**:
- Phase 1: Collect all addresses we'll need
- Phase 2: Batch read from HBM in bursts

Without this separation, we'd have to:
1. Read pointer for axon 0
2. Read synapses for axon 0
3. Read pointer for axon 1
4. Read synapses for axon 1
5. ...

With two phases, we can:
1. Read ALL pointers in one burst
2. Read ALL synapses in one burst

This reduces HBM access latency from ~100ns per pointer to ~20ns amortized (burst mode).

#### 2. The 16 Pointer FIFOs

**Why 16 FIFOs instead of one?**

The FPGA has **16 parallel neuron groups** (URAM banks 0-15). Each group can process neurons independently. By maintaining 16 separate pointer queues:

- Each URAM bank can pull work from its own FIFO
- No contention between groups
- Allows parallel processing: all 16 groups update neurons simultaneously

**Why round-robin instead of priority?**

Round-robin ensures fairness. If we used priority (always read FIFO 0 first), then:
- Group 0 neurons would always be processed first
- Group 15 neurons might starve if Group 0 is very active
- Non-uniform latency across groups

Round-robin gives all groups equal opportunity.

#### 3. Pipeline Latency and Hazard Detection

URAM has a **3-cycle read latency**:
- Cycle 0: Send address
- Cycle 1: URAM decodes address, accesses memory
- Cycle 2: Data emerges from URAM
- Cycle 3: Data available to logic

This creates a **read-modify-write hazard**:

```
Cycle 0: Start read h0 (for synapse A)
Cycle 3: Data arrives: V=1000
Cycle 4: Compute: 1000 + 500 = 1500
Cycle 5: Write h0: V=1500

But what if another synapse B for h0 starts at Cycle 2?

Cycle 2: Start read h0 (for synapse B)  ← Problem!
Cycle 5: Data arrives: V=1000           ← Stale! Should be 1500
Cycle 6: Compute: 1000 + 300 = 1300     ← Wrong!
Cycle 7: Write h0: V=1300               ← Lost the first update!
```

**Solution: Read-after-write bypass**

The `internal_events_processor` tracks recent writes:
```verilog
// Track the last 3 writes
reg [16:0] recent_write_addr[2:0];
reg [35:0] recent_write_data[2:0];

// When reading, check for bypass
if (read_addr == recent_write_addr[0])
    data = recent_write_data[0];  // Bypass URAM
else if (read_addr == recent_write_addr[1])
    data = recent_write_data[1];
else if (read_addr == recent_write_addr[2])
    data = recent_write_data[2];
else
    data = uram_dout;  // Normal path
```

This ensures we always get the most recent value.

#### 4. Double-Buffered BRAM

**Why two BRAM banks (Present and Future)?**

This allows **seamless timestep transitions**:

```
Timestep N execution:
├─ Read from Present buffer (contains timestep N inputs)
└─ Write to Future buffer (receives timestep N+1 inputs)

Timestep N+1 execution:
├─ Reassign buffer roles: bram_select flips (no data movement)
├─ Read from Present buffer (was Future, now has N+1 inputs)
└─ Write to Future buffer (receives timestep N+2 inputs)
```

Without double-buffering:
- Must wait for timestep N to finish
- Then clear BRAM
- Then write timestep N+1 inputs
- Then start execution

With double-buffering:
- Write timestep N+1 inputs while N executes
- Just flip bram_select when ready (role reassignment)
- No wait time

This allows **pipelining** of host input preparation and FPGA execution.

#### 5. Clock Domain Crossing

The FPGA has two clock domains:
- **225 MHz (aclk)**: Used by most modules (HBM, PCIe, pointer controller)
- **450 MHz (aclk450)**: Used by URAM for 2× throughput

Data crosses between domains via **asynchronous FIFOs** that use:
- Gray code counters (prevents glitches during clock crossing)
- Multi-stage synchronizers (prevents metastability)
- Full/empty flags synchronized to both clock domains

This is why the pointer FIFOs and spike FIFOs can safely transfer data between 225 MHz modules and the 450 MHz neuron processor.

---

### Summary: The Complete Execution Picture

```
┌─────────────────────────────────────────────────────────┐
│  ONE TIMESTEP = SETUP + (PHASE1 → PHASE2) × N_ROUNDS   │
└─────────────────────────────────────────────────────────┘

Round 1:
  Setup: Host writes input spikes to BRAM Future
  Reassign: bram_select flips (Future buffer → Present, Present buffer → Future)
  Phase 1a: Read BRAM → Get axon pointers → Fill FIFOs
  Phase 2:  Read synapses → Accumulate → Hidden neurons spike

Round 2 (if neurons spiked):
  Phase 1b: Read URAM → Get neuron pointers → Fill FIFOs
  Phase 2:  Read synapses → Accumulate → Output neurons spike

Round 3 (if neurons spiked again):
  Phase 1b: Read URAM → Get neuron pointers → Fill FIFOs
  Phase 2:  Read synapses → Accumulate → More spikes

... Continue until no more spikes ...

Output: Send accumulated spikes to host via PCIe
```

**Key insights:**
1. **Pointers are addresses, not data** - Phase 1 collects *where* to look, Phase 2 does the looking
2. **FIFOs are queues of HBM addresses** - Not synapse data, just pointers to rows
3. **External vs internal events use different paths** - BRAM for axons (Phase 1a), URAM for neurons (Phase 1b)
4. **Recurrent processing is automatic** - Neuron spikes trigger Phase 1b → Phase 2 loops
5. **Parallelism at every level** - 16 neuron groups, pipelined HBM, concurrent URAM banks

This architecture allows the FPGA to process hundreds of thousands of synapses per microsecond, making real-time spiking neural network simulation possible.
