# Appendix: Hardware Glossary for Software Experts

This appendix defines all hardware terms and concepts used throughout the documentation. Each entry includes a software analogy where applicable to bridge the gap between hardware and software thinking.

---

## A

### Address Passthrough
**What it is:** A technique where a memory controller echoes back the requested address along with the returned data.

**Software analogy:** Like a database query that returns both the data and the primary key you searched for. Useful when multiple reads are in flight and you need to correlate responses with requests.

**Why it matters:** In pipelined hardware, you might issue multiple reads before the first one completes. Address passthrough tells you which read just finished.

---

### Arbiter
**What it is:** Hardware logic that decides which of multiple competing requesters gains access to a shared resource (memory, bus, FIFO).

**Software analogy:** Like a mutex or semaphore, but implemented in hardware logic gates. Multiple threads want the same resource; the arbiter picks one based on priority, round-robin fairness, or other policy.

**Common types:**
- **Priority arbiter:** Always picks highest-priority requester (like VIP queue)
- **Round-robin arbiter:** Cycles through requesters fairly (like `itertools.cycle()`)
- **First-come-first-served:** Picks whoever requested first (like a queue)

**Why it matters:** Prevents conflicts when multiple hardware modules try to access the same memory or bus simultaneously.

---

### AXI4 (Advanced eXtensible Interface)
**What it is:** A memory-mapped communication protocol designed by ARM for on-chip data transfer. It's the "language" FPGA modules speak to transfer data.

**Software analogy:** Like a REST API for hardware—defines request/response formats, handshake protocols, and data structures. Instead of HTTP methods (GET, POST), you have read and write transactions.

**Five channels:**
1. **Read Address (AR):** "I want to read from address X"
2. **Read Data (R):** "Here's the data you requested"
3. **Write Address (AW):** "I want to write to address X"
4. **Write Data (W):** "Here's the data to write"
5. **Write Response (B):** "Write complete, here's the status"

**Key signals:**
- `VALID`: "I have valid data/request"
- `READY`: "I'm ready to accept data/request"
- **Transaction occurs only when both are high (handshake)**

**Why it matters:** All on-chip communication in the FPGA uses AXI4. Understanding it is like understanding HTTP for web development.

**See:** hardware_map.md Section 4, Chapter 1 (DMA transfers)

---

### AXI4 Master
**What it is:** A hardware module that initiates AXI4 transactions (sends read/write requests).

**Software analogy:** Like an HTTP client that makes requests to servers. The FPGA's DMA engine is a master; it requests data from host memory.

**Why it matters:** Determines who "drives" the bus. Masters control data flow; slaves respond.

---

### AXI4 Slave
**What it is:** A hardware module that responds to AXI4 transactions (serves read/write requests).

**Software analogy:** Like an HTTP server that responds to client requests. Host memory is a slave; it serves data when the FPGA requests it.

**Why it matters:** Slaves must be ready to handle requests at any time. They implement the actual storage or computation.

---

### Axon
**What it is:** In neuromorphic systems, an external input to the network. In hardware terms, axons don't have internal state (no membrane potential)—they're just spike sources.

**Software analogy:** Like a publisher in a pub/sub system. When an axon fires, it publishes a spike event that gets delivered to all subscribed neurons.

**Storage:** Axon spike masks are stored in BRAM as one-hot bit vectors (256 bits per row).

**Why it matters:** Distinguishing axons (stateless inputs) from neurons (stateful computations) determines memory organization and processing flow.

**See:** Chapter 1 (BRAM organization), Chapter 2 (Phase 1 processing)

---

## B

### Backpressure
**What it is:** A flow control mechanism where a downstream module signals "I'm full, stop sending data" to an upstream module.

**Software analogy:** Like `TCP windowing` or a bounded queue's `.full()` flag. When a buffer fills up, backpressure tells the sender to pause.

**Implementation:** Typically via a `FULL` or `!READY` signal. When asserted, upstream must stop writing.

**Why it matters:** Prevents data loss when fast producers overwhelm slow consumers. Critical in pipelined systems.

**See:** Chapter 2 (FIFO interfaces), hardware_map.md (AXI4 handshakes)

---

### Beat
**What it is:** A single data transfer within a burst transaction. A burst is composed of multiple beats.

**Software analogy:** Like a single element in a batch API request. If you request 16 records in one query, each record is one "beat."

**Example:** A 16-beat burst transfers 16 data words sequentially.

**Why it matters:** Bursts amortize transaction overhead. Instead of 16 separate requests (expensive), send one request for 16 beats (efficient).

---

### BRAM (Block RAM)
**What it is:** Fast on-chip SRAM built into the FPGA fabric. Organized into dual-port blocks (can read and write simultaneously from different ports).

**Specifications:**
- **Capacity:** 1 MB total (256 rows × 256 bits per row)
- **Latency:** 3 clock cycles @ 225 MHz (~13 ns)
- **Technology:** 6-transistor SRAM cells (fast, power-hungry)

**Software analogy:** Like L1 cache—very fast, very expensive (in silicon area), small capacity. Use for frequently accessed data.

**Usage in neuromorphic system:**
- Stores axon spike masks (one-hot encoded, 256 bits per row)
- Dual-port allows host PC to write new spikes while FPGA reads current spikes

**Why it matters:** BRAM is the fastest memory accessible to FPGA logic (except registers). Ideal for low-latency, small datasets.

**See:** Chapter 1 Section 1.1, Chapter 2 Phase 0

---

### Burst
**What it is:** An AXI4 transaction that transfers multiple consecutive data words in a single request.

**Software analogy:** Like a batch database query. Instead of:
```python
for i in range(16):
    data[i] = read(addr + i)  # 16 separate requests
```
You do:
```python
data = read_burst(addr, length=16)  # 1 request, 16 data transfers
```

**Why it matters:** Reduces protocol overhead. Each request has setup time (address, handshake); bursts amortize this over multiple data words.

**See:** hardware_map.md (AXI4 timing), Chapter 1 (HBM bursts)

---

### Burst Length
**What it is:** The number of beats (data words) in a burst transaction.

**AXI4 limits:** 1-256 beats for INCR bursts.

**Example:** Burst length 16 means the transaction transfers 16 consecutive data words.

**Why it matters:** Longer bursts → higher bandwidth efficiency, but require contiguous addresses.

---

## C

### Clock Domain
**What it is:** A region of a circuit where all flip-flops are clocked by the same clock signal. Everything in a clock domain updates simultaneously on clock edges.

**Software analogy:** Like a thread with its own event loop. All state updates happen synchronously at clock ticks (like `await` points in async Python).

**Example in neuromorphic system:**
- **225 MHz domain:** BRAM, HBM interfaces, most control logic
- **450 MHz domain:** URAM, internal events processor (2x faster for neuron updates)

**Why it matters:** Data transfer between clock domains requires special synchronization (FIFOs with gray code) to avoid metastability.

**See:** hardware_map.md (clock generation), Chapter 2 (dual clock operation)

---

### Clock Domain Crossing (CDC)
**What it is:** Transferring data from one clock domain to another (e.g., 225 MHz → 450 MHz).

**Software analogy:** Like passing data between threads with different event loops. You need thread-safe mechanisms (mutex, queue) to avoid race conditions.

**Hardware solution:** Asynchronous FIFOs with gray-code counters to prevent metastability.

**Why it matters:** Improper CDC causes random bit flips, data corruption, or circuit hangs. Must use proven synchronization techniques.

**See:** hardware_map.md Section 3 (FIFO synchronization)

---

### Combinational Logic
**What it is:** Digital circuits where outputs depend only on current inputs (no memory, no state). Examples: AND gates, multiplexers, adders.

**Software analogy:** Pure functions in functional programming—same inputs always produce same outputs, no side effects, no state.

```python
def combinational(a, b):
    return a & b  # Output depends only on a, b (no self.state)
```

**Verilog:**
```verilog
assign output = (a & b) | c;  // Combinational: output updates whenever a, b, c change
```

**Why it matters:** Combinational logic is fast (no clock delay) but creates no memory. Contrast with sequential logic (flip-flops, registers) which stores state.

**See:** hardware_map.md (logic gates), Chapter 2 (state machines use both)

---

### Command Opcode
**What it is:** An 8-bit identifier in PCIe command packets that specifies the operation type.

**Software analogy:** Like HTTP methods (GET, POST, PUT, DELETE) or opcodes in assembly language (MOV, ADD, JMP).

**Example opcodes in neuromorphic system:**
- `0x01`: Write to BRAM (axon spikes)
- `0x02`: Write to HBM (network structure)
- `0x03`: Write to URAM (neuron states)
- `0x04`: Read from URAM
- `0xC8`: Execute (run one timestep)

**Why it matters:** The command interpreter decodes opcodes to route data to correct hardware modules.

**See:** Chapter 1 Section 1.2, Chapter 2 Phase 0

---

## D

### DMA (Direct Memory Access)
**What it is:** A technique where an FPGA (or peripheral) transfers data to/from host memory without involving the CPU. The FPGA becomes a "bus master" and directly reads/writes RAM.

**Software analogy:** Like `mmap()` or shared memory. Instead of:
```python
# Slow: CPU copies data
data = host_memory.read()
fpga.write(data)
```
You have:
```python
# Fast: FPGA directly accesses host memory
fpga.dma_read(host_address, size)
```

**Why it matters:** DMA bypasses the CPU bottleneck, achieving 10-100x higher bandwidth. Essential for large data transfers (network initialization, spike outputs).

**How it works:**
1. FPGA sends PCIe TLP (Transaction Layer Packet) with memory address
2. Host memory controller responds with data
3. No CPU involvement—fully offloaded

**See:** Chapter 1 Section 1.2 (initialization), hardware_map.md Section 2 (PCIe)

---

### Double Buffering
**What it is:** Using two buffers that alternate roles—while one is being read (consumed), the other is being written (produced).

**Software analogy:**
```python
class DoubleBuffer:
    def __init__(self):
        self.buffer_a = [0] * 1000
        self.buffer_b = [0] * 1000
        self.active = 'a'  # Which buffer is being read

    def swap(self):
        # Swap roles: reader becomes writer, writer becomes reader
        self.active = 'b' if self.active == 'a' else 'a'
```

**Usage in neuromorphic system:** External events processor uses double-buffered BRAMs:
- **Present BRAM:** Being read/cleared (processing current timestep spikes)
- **Future BRAM:** Being written (collecting spikes for next timestep)
- After execute, they swap roles

**Why it matters:** Enables pipelining—overlap computation with I/O. No need to wait for reads to finish before accepting new writes.

**See:** Verilog sources (external_events_processor_simple.v)

---

### DRAM (Dynamic RAM)
**What it is:** Memory technology that stores bits as charge in capacitors. "Dynamic" because charge leaks, requiring periodic refresh.

**Structure:** 1 transistor + 1 capacitor per bit (1T1C). Compact but slow.

**Software analogy:** Like disk storage—high capacity, slow access, requires maintenance (refresh = defragmentation).

**Types:**
- **DDR4:** Desktop/server RAM (~20 GB/s per channel)
- **HBM2:** 3D-stacked high-bandwidth RAM (~400 GB/s, used in neuromorphic system)

**Contrast with SRAM:**
- DRAM: 1T1C, dense, slow (50-100 ns), needs refresh
- SRAM: 6T, fast (1-3 ns), expensive, no refresh

**Why it matters:** DRAM provides bulk storage (8 GB HBM) for synaptic weights. Too slow for direct neuron computation (hence URAM for neuron states).

**See:** hardware_map.md Section 1.3 (HBM structure)

---

## E

### execRun_ctr (Execution Run Counter)
**What it is:** A hardware register that counts timesteps. Increments after each `execute()` command completes.

**Software analogy:**
```python
class NeuromorphicEngine:
    def __init__(self):
        self.timestep = 0  # Like execRun_ctr

    def execute(self):
        # Run one timestep
        self.timestep += 1
```

**Why it matters:** Used to tag spike events with their timestep. When spikes return to host, you know when they occurred.

**See:** Chapter 2 (spike packet format)

---

### execRun_timer (Execution Run Timer)
**What it is:** A hardware counter that measures clock cycles elapsed during a timestep.

**Software analogy:**
```python
import time
start = time.perf_counter()
execute()
elapsed_cycles = (time.perf_counter() - start) * clock_frequency
```

**Why it matters:** Profiling tool. Tells you how many clock cycles (and thus microseconds) a timestep took. Useful for optimization.

---

## F

### FIFO (First-In-First-Out)
**What it is:** A hardware buffer that stores data in order received and outputs data in the same order.

**Software analogy:** `queue.Queue()` in Python—`.put()` adds to tail, `.get()` removes from head.

**Hardware interface:**
```verilog
// Write side
input  full,       // 1 = FIFO full, cannot write
output wren,       // 1 = write data_in this cycle
output data_in,    // Data to write

// Read side
input  empty,      // 1 = FIFO empty, no data
output rden,       // 1 = read data_out this cycle
input  data_out,   // Data read
```

**FWFT mode (First-Word Fall-Through):** `data_out` always shows next item (zero latency). Like `.peek()` in Python.

**Why it matters:** FIFOs decouple clock domains, buffer mismatched rates (fast producer, slow consumer), and enable pipelining.

**See:** hardware_map.md Section 3, Chapter 2 (pointer FIFOs, spike FIFOs)

---

### Flip-Flop
**What it is:** A basic sequential logic element that stores one bit. Updates on clock edge.

**Software analogy:** Like a single-bit instance variable:
```python
class FlipFlop:
    def __init__(self):
        self.q = 0  # Current state

    def clock_edge(self, d):
        self.q = d  # Update state on clock tick
```

**Verilog:**
```verilog
always @(posedge clk) begin
    q <= d;  // q updates to d on rising clock edge
end
```

**D-type flip-flop:** Most common. Has one data input (D) and one output (Q). On clock edge, Q ← D.

**Why it matters:** Flip-flops are the fundamental unit of memory in digital circuits. Registers, counters, state machines—all built from flip-flops.

**See:** hardware_map.md Section 1.2 (FPGA fabric)

---

### FPGA (Field-Programmable Gate Array)
**What it is:** A chip containing millions of configurable logic blocks (LUTs, flip-flops) connected by programmable routing. You write code (Verilog/VHDL) that gets "compiled" into a configuration bitstream that rewires the chip.

**Software analogy:** Imagine if you could rewrite the CPU's microarchitecture at runtime. An FPGA is like a blank canvas where you design custom hardware for your application.

**Neuromorphic system uses:** Xilinx XCVU37p UltraScale+ VU37P
- 1.1 million LUTs (programmable logic gates)
- 2.2 million flip-flops (registers)
- 50 MB BRAM (on-chip cache)
- 4.5 MB URAM (ultra-dense cache)
- 16 lanes PCIe Gen3 (host communication)
- 32 HBM2 channels (memory bandwidth)

**Why it matters:** FPGAs offer specialized, massively parallel computation. Neuromorphic networks need to update 100,000+ neurons in microseconds—impossible on CPU, possible on FPGA.

**See:** hardware_map.md Section 1.2, Chapter 1 (compilation)

---

### FWFT (First-Word Fall-Through)
**What it is:** A FIFO operating mode where the output data is always valid (showing the next item) without requiring a read strobe first.

**Normal FIFO:**
```python
if not fifo.empty():
    fifo.read_enable = True  # Request data
    # Wait 1 cycle
    data = fifo.data_out     # Data appears next cycle
```

**FWFT FIFO:**
```python
if not fifo.empty():
    data = fifo.data_out     # Data already available (0 latency)
    fifo.read_enable = True  # Advance to next item
```

**Why it matters:** Reduces latency by 1 cycle. Important in high-speed pipelines where every cycle counts.

---

## H

### Handshake
**What it is:** A two-signal protocol (`VALID` and `READY`) where data transfer occurs only when both signals are high.

**Protocol:**
- **Sender:** Asserts `VALID` when data is available, holds data stable
- **Receiver:** Asserts `READY` when it can accept data
- **Transfer occurs:** When `VALID && READY` on clock edge

**Software analogy:**
```python
while True:
    sender.valid = sender.has_data()
    receiver.ready = not receiver.is_full()

    if sender.valid and receiver.ready:
        receiver.data = sender.data  # Transfer!
        sender.pop()
        break
```

**Why it matters:** Prevents data loss. Sender can't send until receiver is ready; receiver doesn't miss data because sender holds it stable.

**See:** hardware_map.md (AXI4 channels), Chapter 2 (HBM read valid/ready)

---

### Hazard
**What it is:** A conflict in pipelined hardware when consecutive operations access the same resource (typically same memory address) in ways that could cause data corruption.

**Types:**
- **Read-after-write (RAW):** Read tries to fetch old value before write completes
- **Write-after-read (WAR):** Write might corrupt data before read completes
- **Write-after-write (WAW):** Second write might complete before first

**Software analogy:** Like race conditions in multithreaded code:
```python
# Thread 1
x = x + 1  # Read x, add 1, write back

# Thread 2 (starts during Thread 1)
y = x      # Which value of x do I see?
```

**Hardware solution:** Hazard detection logic checks if addresses match in pipeline stages. If match, either:
- **Stall:** Pause until pipeline clears
- **Bypass/Forward:** Route fresh data directly from pipeline register

**Why it matters:** URAM has 3-cycle read latency. If you read address A, then write address A within 3 cycles, hazard detection prevents reading stale data.

**See:** Chapter 2 Section 2.2 (internal_events_processor hazard logic)

---

### HBM (High Bandwidth Memory)
**What it is:** 3D-stacked DRAM technology providing extreme bandwidth (~400 GB/s). Multiple DRAM dies stacked vertically, connected by Through-Silicon Vias (TSVs).

**Specifications:**
- **Capacity:** 8 GB (for synaptic connectivity data)
- **Bandwidth:** 400+ GB/s (32 channels × 14 GB/s per channel)
- **Latency:** ~100-200 ns (slow compared to BRAM/URAM, fast compared to DDR4)
- **Organization:** 2 pseudo-channels × 16 banks per channel

**Software analogy:** Like a distributed database with 32 shards. Each shard (channel) serves 256 MB and can handle requests independently (parallel access).

**Usage in neuromorphic system:**
- **Region 1:** Axon pointers (maps axon ID → synapse list address)
- **Region 2:** Neuron pointers (maps neuron ID → synapse list address)
- **Region 3:** Synapses (actual connection data: target neuron + weight)

**Why it matters:** Stores the entire network structure (billions of synapses for large networks). Bandwidth supports updating millions of neurons per millisecond.

**See:** hardware_map.md Section 1.3, Chapter 1 Section 1.1, Chapter 2 Phase 1

---

## I

### INCR Burst
**What it is:** An AXI4 burst type where addresses increment sequentially for each beat.

**Example:**
```
Burst start address: 0x1000
Burst length: 4
Addresses accessed: 0x1000, 0x1004, 0x1008, 0x100C (assuming 32-bit words)
```

**Contrast with WRAP bursts:** Addresses wrap around within a boundary (used for cache lines).

**Why it matters:** Simplest burst type. Used for sequential data access (reading synapse lists, writing neuron states).

---

## L

### Latency
**What it is:** The time delay between when a request is issued and when the response is received.

**Software analogy:** Like ping time in networking, or function call overhead.

**Examples in neuromorphic system:**
- **BRAM latency:** 3 cycles @ 225 MHz = ~13 ns
- **URAM latency:** 1 cycle @ 450 MHz = ~2 ns
- **HBM latency:** ~100-200 ns
- **PCIe latency:** ~1-10 µs (depends on packet size, distance)

**Why it matters:** Latency determines pipeline depth. 3-cycle BRAM latency means you need 3 pipeline stages between read request and data availability.

**Contrast with throughput:** Latency is "time per operation," throughput is "operations per time."

**See:** hardware_map.md (memory specifications), Chapter 2 (pipeline filling)

---

### LUT (Lookup Table)
**What it is:** A programmable logic element in an FPGA. Typically a 6-input, 1-output function implemented as a 64-bit SRAM.

**How it works:**
- 6 inputs → 2^6 = 64 possible input combinations
- 64-bit SRAM stores the output value for each combination
- LUT acts as a 6-input truth table

**Software analogy:**
```python
# 2-input LUT (simplified)
lut_contents = [0, 0, 0, 1]  # 4-bit SRAM for 2 inputs
def lut(a, b):
    index = (a << 1) | b  # Convert inputs to index
    return lut_contents[index]

# This LUT implements: output = a AND b
# 00 → 0, 01 → 0, 10 → 0, 11 → 1
```

**Why it matters:** LUTs are the fundamental building blocks of FPGA logic. Any combinational function of up to 6 inputs can be implemented in 1 LUT. Complex functions use multiple LUTs connected together.

**See:** hardware_map.md Section 1.2

---

## M

### Metastability
**What it is:** An undefined state in digital circuits where a flip-flop's output oscillates between 0 and 1, unable to settle. Occurs during clock domain crossings when setup/hold time requirements are violated.

**Software analogy:** Like a race condition where a variable reads as garbage because two threads wrote simultaneously.

**Cause:** When a flip-flop's input changes too close to the clock edge, the output can become metastable (neither 0 nor 1, or fluctuating).

**Solution:** Multi-stage synchronizers (2-3 flip-flops in series) give time for metastability to resolve. First flip-flop might be metastable, but probability that second flip-flop is also metastable is astronomically low.

**Why it matters:** Metastability causes random bit flips, data corruption, system crashes. Must use proven synchronization techniques for clock domain crossings (async FIFOs with gray code).

**See:** hardware_map.md Section 3 (FIFO synchronization)

---

## N

### Neuron Group
**What it is:** A set of 8,192 neurons that share a URAM bank and processing pipeline. The neuromorphic system has 16 neuron groups (0-15), totaling 131,072 neurons.

**Organization:**
- **Group 0:** Neurons 0-8,191 (URAM bank 0)
- **Group 1:** Neurons 8,192-16,383 (URAM bank 1)
- ...
- **Group 15:** Neurons 122,880-131,071 (URAM bank 15)

**Why groups?** Parallel processing. All 16 groups can be updated simultaneously (16-way parallelism).

**Software analogy:** Like database sharding. Each shard (neuron group) has its own storage (URAM bank) and compute pipeline.

**Why it matters:** Determines memory addresses, routing logic, and parallelism. Understanding groups is key to understanding performance scaling.

**See:** Chapter 1 Section 1.1, Chapter 2 Section 2.2

---

## P

### PCIe (Peripheral Component Interconnect Express)
**What it is:** A high-speed serial communication bus connecting the CPU to peripherals (GPUs, FPGAs, SSDs). Uses differential signaling over point-to-point links.

**Neuromorphic system:** PCIe Gen3 x16
- **Gen3:** Third generation (8 GT/s per lane)
- **x16:** 16 lanes (parallel connections)
- **Bandwidth:** ~14 GB/s bidirectional

**Software analogy:** Like USB or Ethernet, but much faster and lower latency. From software, you use it via memory-mapped I/O or DMA.

**How it works:**
1. **Physical layer:** Differential pairs (16 lanes × 2 directions = 32 pairs)
2. **Data link layer:** Packets with CRC error detection
3. **Transaction layer:** TLP (Transaction Layer Packets) for reads/writes

**Why it matters:** PCIe is the bridge between host CPU (software) and FPGA (hardware). All data transfer (inputs, outputs, configuration) flows through PCIe.

**See:** hardware_map.md Section 2, Chapter 1 Section 1.2

---

### Pipeline
**What it is:** A technique where operations are broken into stages, with multiple operations executing simultaneously at different stages.

**Software analogy:**
```python
# No pipeline: 3 cycles per item
for item in data:
    stage1(item)  # 1 cycle
    stage2(item)  # 1 cycle
    stage3(item)  # 1 cycle
# Total: 3N cycles for N items

# Pipeline: 1 cycle per item after initial fill
# Cycle 0: stage1(item[0])
# Cycle 1: stage1(item[1]), stage2(item[0])
# Cycle 2: stage1(item[2]), stage2(item[1]), stage3(item[0])
# Cycle 3: stage1(item[3]), stage2(item[2]), stage3(item[1])
# Total: N+2 cycles for N items (3x speedup for large N)
```

**Example in neuromorphic system:** BRAM read pipeline
- **Cycle 0:** Issue address for read 0
- **Cycle 1:** Issue address for read 1 (read 0 in pipeline stage 1)
- **Cycle 2:** Issue address for read 2 (read 1 in stage 1, read 0 in stage 2)
- **Cycle 3:** Read 0 data available, issue address for read 3

**Why it matters:** Pipelines enable high throughput despite high latency. BRAM has 3-cycle latency but can start a new read every cycle (throughput = 1 read/cycle).

**See:** hardware_map.md (memory timing), Chapter 2 (pipeline filling)

---

### Pipeline Depth
**What it is:** The number of stages in a pipeline, or equivalently, the number of clock cycles between input and output.

**Example:** BRAM has 3-cycle read latency → pipeline depth = 3.

**Why it matters:** Determines how many cycles you must wait before first result. Also affects hazard detection (must check all pipeline stages for address conflicts).

---

### Pointer Chain
**What it is:** A linked-list structure in HBM where each neuron/axon has a pointer to the start of its synapse list.

**Structure:**
```
Axon a0 → Pointer: 0x00080000 (address of synapse list in HBM)
HBM[0x00080000] = [synapse_0, synapse_1, ..., synapse_4]
Each synapse = {target: h0, weight: 1000}
```

**Software analogy:**
```python
# Like a dictionary of lists
synapses = {
    'a0': [('h0', 1000), ('h1', 1000), ...],
    'a1': [('h0', 1000), ('h1', 1000), ...],
}

# In hardware, stored as pointers:
pointers = {'a0': 0x80000, 'a1': 0x80005}
memory = {
    0x80000: [('h0', 1000), ('h1', 1000), ...],
    0x80005: [('h0', 1000), ('h1', 1000), ...],
}
```

**Why it matters:** Enables variable fan-out (some neurons have 10 synapses, others 10,000). Pointer indirection allows efficient memory usage.

**See:** Chapter 1 Section 1.1 (HBM layout), Chapter 2 Phase 1

---

### Priority Arbitration
**What it is:** An arbitration scheme where high-priority requesters always win over low-priority requesters.

**Software analogy:** Like a priority queue or VIP line.

**Example:** If both CPU and DMA request the bus, CPU (high priority) always wins.

**Disadvantage:** Low-priority requesters can starve (never get access).

**See also:** Round-robin (fair alternative)

---

## R

### Read Latency
**What it is:** The number of clock cycles from when a read address is issued to when the data becomes valid.

**Examples:**
- **BRAM:** 3 cycles
- **URAM:** 1 cycle
- **HBM:** ~100-200 cycles (at 225 MHz)

**Why it matters:** Determines pipeline depth and minimum loop iteration time.

**See:** Memory specifications throughout documentation

---

### Register
**What it is:** A storage element (or group of flip-flops) that holds a multi-bit value.

**Software analogy:** Like an instance variable in a class.

**Verilog:**
```verilog
reg [7:0] counter;  // 8-bit register

always @(posedge clk) begin
    counter <= counter + 1;  // Updates on every clock edge
end
```

**Why it matters:** Registers store state in sequential circuits. Counters, addresses, data buffers—all implemented as registers.

---

### Register Slice
**What it is:** A pipeline stage inserted into a data path to improve timing (break long combinational paths).

**Software analogy:** Like adding an intermediate variable to break up a complex expression:

```python
# Hard to optimize (long dependency chain)
result = f(g(h(i(j(x)))))

# Easier (can pipeline/parallelize)
temp1 = j(x)
temp2 = i(temp1)
temp3 = h(temp2)
temp4 = g(temp3)
result = f(temp4)
```

**Why it matters:** High-speed interfaces (450 MHz) require short logic paths. Register slices reduce maximum combinational delay, allowing higher clock frequencies.

**See:** internal_events_processor (450 MHz URAM access)

---

### RMW (Read-Modify-Write)
**What it is:** A pattern where you read a value, modify it, and write it back.

**Software analogy:**
```python
x = memory[addr]       # Read
x = x + 1              # Modify
memory[addr] = x       # Write
```

**Hardware challenges:** With 3-cycle read latency, must ensure no other operation accesses the same address during the RMW sequence (hazard detection).

**Example in neuromorphic system:** Masked URAM writes
- Read 72-bit word containing 2 neurons
- Modify upper 36 bits (neuron voltage)
- Write back full 72-bit word

**Why it matters:** Common pattern requiring careful hazard management.

**See:** Chapter 2 Section 2.2 (internal_events_processor)

---

### Round-Robin
**What it is:** A fair arbitration algorithm that cycles through requesters in order, giving each a turn.

**Software analogy:**
```python
requesters = [0, 1, 2, 3, 4, 5, 6, 7]
current = 0

while True:
    if requesters[current].has_request():
        service(requesters[current])
    current = (current + 1) % len(requesters)  # Cycle: 0→1→2→...→7→0
```

**Verilog:**
```verilog
reg [2:0] addr;  // 3-bit counter for 8 requesters

always @(posedge clk) begin
    addr <= addr + 1;  // Automatically wraps 7→0
end
```

**Usage in neuromorphic system:**
- **Pointer FIFO controller:** 16-way round-robin (checks ptr0, ptr1, ..., ptr15, ptr0, ...)
- **Spike FIFO controller:** 8-way round-robin (checks spk0, spk1, ..., spk7, spk0, ...)

**Why it matters:** Ensures fairness—no requester starves. Every requester gets regular turns regardless of activity.

**See:** Chapter 2 (FIFO controllers), Verilog sources

---

## S

### Sequential Logic
**What it is:** Digital circuits with memory—outputs depend on both current inputs and past state.

**Software analogy:** Objects with instance variables:
```python
class Counter:
    def __init__(self):
        self.count = 0  # State

    def increment(self):
        self.count += 1  # Output depends on previous state
```

**Verilog:**
```verilog
reg [7:0] count;

always @(posedge clk) begin
    count <= count + 1;  // State updates on clock edges
end
```

**Building blocks:** Flip-flops, registers, state machines, counters, shift registers.

**Contrast with combinational logic:** Combinational has no memory (pure functions).

**Why it matters:** All computation with memory/state requires sequential logic.

**See:** hardware_map.md (flip-flops), Chapter 2 (state machines)

---

### Sign Extension
**What it is:** Expanding a signed integer to a wider bit width by replicating the sign bit.

**Example:**
```
8-bit signed: 11111010 (-6 in two's complement)
16-bit signed: 11111111 11111010 (still -6)
```

**Rule:** Copy the most significant bit (sign bit) into all new upper bits.

**Why it matters:** Prevents corruption when mixing different bit widths in arithmetic. Hardware often needs to extend weights (16-bit) to match neuron voltages (36-bit).

**See:** Chapter 2 (synaptic accumulation)

---

### Spike
**What it is:** An action potential—a neuron firing event that occurs when membrane potential crosses threshold.

**In hardware:** Represented as a 17-bit value: `{valid_bit, neuron_address[16:0]}`

**Example:** Neuron 5 spikes → spike packet = `0x00005` with valid bit = 1

**Why it matters:** Spikes are the fundamental events in spiking neural networks. All computation revolves around spike propagation and accumulation.

**See:** Throughout documentation

---

### SRAM (Static RAM)
**What it is:** Memory technology using 6 transistors per bit (6T). "Static" because it holds state as long as powered (no refresh needed).

**Characteristics:**
- **Fast:** 1-3 ns access time
- **Expensive:** 6 transistors per bit (vs. 1T1C for DRAM)
- **Low density:** Large silicon area

**Examples:** CPU caches (L1, L2), FPGA BRAM

**Contrast with DRAM:**
- SRAM: 6T, fast, expensive, no refresh
- DRAM: 1T1C, slow, cheap, needs refresh

**Why it matters:** SRAM is used for speed-critical data (BRAM for spike masks). DRAM is used for bulk storage (HBM for synapses).

**See:** hardware_map.md Section 1.1 (BRAM), Section 1.3 (HBM comparison)

---

### State Machine
**What it is:** A sequential circuit that transitions through a defined set of states based on inputs and current state.

**Software analogy:**
```python
class StateMachine:
    def __init__(self):
        self.state = 'IDLE'

    def update(self, input):
        if self.state == 'IDLE' and input == 'start':
            self.state = 'RUNNING'
        elif self.state == 'RUNNING' and input == 'done':
            self.state = 'IDLE'
```

**Verilog:**
```verilog
reg [1:0] state;
localparam IDLE = 0, RUNNING = 1, DONE = 2;

always @(posedge clk) begin
    case (state)
        IDLE: if (start) state <= RUNNING;
        RUNNING: if (done) state <= DONE;
        DONE: state <= IDLE;
    endcase
end
```

**Example in neuromorphic system:** external_events_processor
- **IDLE:** Waiting for execute command
- **FILL_PIPE:** Filling BRAM read pipeline (3 cycles)
- **READ:** Reading spike masks and fetching synapses
- **DONE:** All spikes processed

**Why it matters:** State machines coordinate complex multi-step operations. Almost every hardware module has at least one state machine for control flow.

**See:** Chapter 2 Section 2.2 (Verilog state machines)

---

### Synapse
**What it is:** A connection between two neurons with an associated weight. When the presynaptic neuron spikes, the postsynaptic neuron's voltage increases by the synaptic weight.

**Hardware representation:** 32-bit value
- **Bits [31:29]:** Opcode (usually 0 for normal synapse)
- **Bits [28:16]:** Target neuron address (13 bits)
- **Bits [15:0]:** Synaptic weight (signed 16-bit integer)

**Example:** `0x0000_03E8` = target neuron 0, weight 1000

**Storage:** Synapses stored in HBM (billions of them for large networks).

**Why it matters:** Synapses define the network structure. All learning involves modifying synaptic weights.

**See:** Chapter 1 Section 1.1 (HBM synapse region), Chapter 2 Phase 1

---

## T

### Threshold
**What it is:** The membrane potential value at which a neuron fires (spikes).

**Example:** If threshold = 2000 and neuron voltage reaches 2000, the neuron spikes and resets to 0.

**Hardware:** Stored as a 36-bit signed integer in configuration registers.

**Software analogy:**
```python
if neuron.voltage >= threshold:
    neuron.spike()
    neuron.voltage = 0  # Reset
```

**Why it matters:** Determines network sensitivity and dynamics. Lower threshold → more spikes, higher threshold → sparse activity.

**See:** Introduction (neuron model), Chapter 2 (threshold checking)

---

### Throughput
**What it is:** The amount of data processed per unit time.

**Software analogy:** Requests per second (RPS), or bandwidth in networking.

**Examples:**
- **HBM throughput:** 400 GB/s (can transfer 400 billion bytes per second)
- **BRAM throughput:** 1 read per cycle @ 225 MHz = 225 million reads/s
- **Pipeline throughput:** 1 result per cycle (after initial fill)

**Contrast with latency:** Latency = time per operation, throughput = operations per time.

**Relationship:** High latency can still have high throughput if pipelined.

**Why it matters:** Throughput determines how fast you can process large datasets. Latency determines responsiveness for individual operations.

---

### Transaction ID (TID)
**What it is:** An identifier tag attached to requests to allow out-of-order completion.

**Software analogy:**
```python
# Send multiple requests with IDs
send_request(addr=0x1000, tid=1)
send_request(addr=0x2000, tid=2)
send_request(addr=0x3000, tid=3)

# Responses can return in any order
response = receive()  # {tid: 3, data: ...}  ← request 3 finished first
response = receive()  # {tid: 1, data: ...}  ← request 1 finished second
response = receive()  # {tid: 2, data: ...}  ← request 2 finished last
```

**Why it matters:** Allows parallelism. Without TIDs, you'd have to wait for request 1 to complete before issuing request 2. With TIDs, issue all requests immediately and match responses when they arrive.

**See:** AXI4 protocol, HBM memory controller

---

### Transistor
**What it is:** A semiconductor device that acts as an electrically-controlled switch. The fundamental building block of all digital circuits.

**Types:**
- **NMOS:** Conducts when gate voltage is high (switch closes with 1)
- **PMOS:** Conducts when gate voltage is low (switch closes with 0)

**Software analogy:** Like an `if` statement:
```python
if gate_voltage:
    output = input  # Transistor "on" (conducting)
else:
    output = disconnected  # Transistor "off" (not conducting)
```

**Scale:** Modern FPGAs contain billions of transistors. A 6T SRAM cell has 6 transistors, a DRAM cell has 1 transistor.

**Why it matters:** Everything in hardware—logic gates, memory, CPUs—is built from transistors.

**See:** hardware_map.md Section 1 (SRAM cells, DRAM cells)

---

## U

### URAM (UltraRAM)
**What it is:** High-density on-chip DRAM-like memory blocks in Xilinx UltraScale+ FPGAs. Faster and denser than BRAM.

**Specifications:**
- **Capacity:** 4.5 MB total (16 banks × 288 KB per bank)
- **Latency:** 1 cycle @ 450 MHz (~2 ns)
- **Technology:** 1T1C DRAM-like cells (denser than BRAM's 6T SRAM)
- **Organization:** 16 banks, each 4096 words × 72 bits

**Software analogy:** Like L2 cache—larger than L1 (BRAM) but still much faster than main memory (HBM).

**Usage in neuromorphic system:**
- Stores neuron membrane potentials (131,072 neurons × 36 bits)
- Each bank holds 8,192 neurons (1 neuron group)
- Dual-neuron packing: 2 neurons per 72-bit word ([71:36]=upper, [35:0]=lower)

**Why it matters:** URAM is the perfect middle ground—larger than BRAM, faster than HBM. Ideal for neuron state storage (frequent access, moderate size).

**See:** Chapter 1 Section 1.1, Chapter 2 Section 2.2 (internal_events_processor)

---

## Memory Hierarchy Summary

From fastest to slowest:

| Memory | Capacity | Latency | Bandwidth | Use Case |
|--------|----------|---------|-----------|----------|
| **Registers** | ~1 KB | 0 cycles | N/A | Pipeline state |
| **URAM** | 4.5 MB | 1 cycle (2 ns) | ~200 GB/s | Neuron states |
| **BRAM** | 1 MB | 3 cycles (13 ns) | ~50 GB/s | Spike masks |
| **HBM** | 8 GB | 100-200 ns | 400 GB/s | Synaptic weights |
| **Host DDR4** | 64+ GB | 1-10 µs | 20 GB/s | Long-term storage |

**Software analogy:**
- Registers = CPU registers
- URAM = L1 cache
- BRAM = L2 cache
- HBM = Main RAM
- Host DDR4 = Disk/SSD

---

## Key Concepts Summary

### Pipelining
Break operations into stages, overlap execution. Latency stays same, throughput increases.

### State Machines
Sequential control logic stepping through states (IDLE → RUNNING → DONE).

### Hazard Detection
Prevent read-modify-write conflicts by tracking pipeline addresses.

### Clock Domain Crossing
Synchronize data between different clock frequencies using async FIFOs.

### Handshake Protocol
`VALID && READY` ensures reliable data transfer with flow control.

### Round-Robin Arbitration
Fair scheduling by cycling through requesters in order.

### Backpressure
Downstream signals "I'm full" to pause upstream sender.

---

## Notation Conventions

Throughout the documentation, you'll see:

- **Hexadecimal:** `0x1234` or `0xABCD`
- **Binary:** `0b1010` or `4'b1010` (4-bit binary)
- **Bit ranges:** `[31:0]` means bits 31 down to 0 (32 bits total)
- **Bit indexing:** `data[7]` means bit 7 of data
- **Active-low signals:** `resetn` (n suffix) means 0=active, 1=inactive
- **Register notation:** `reg [7:0] counter` = 8-bit register named counter

---

## Further Reading

For deeper hardware understanding:
- **AXI4 specification:** ARM IHI0022 (AXI protocol)
- **Xilinx UltraScale+ architecture:** UG574 (FPGA fabric, memory)
- **PCIe specification:** PCI-SIG PCIe Base 3.0
- **Digital design fundamentals:** "Digital Design and Computer Architecture" by Harris & Harris

For neuromorphic computing:
- **Spiking neural networks:** "Neuronal Dynamics" by Gerstner et al.
- **Hardware acceleration:** "Computer Architecture: A Quantitative Approach" by Hennessy & Patterson

---

This glossary provides the foundation for understanding the hardware implementation details throughout the documentation. When you encounter an unfamiliar term, refer back to this appendix for clarification and context.
