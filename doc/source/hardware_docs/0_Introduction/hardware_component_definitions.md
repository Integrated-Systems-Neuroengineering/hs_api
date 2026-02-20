# Hardware Component Definitions (Low-Level)

This guide introduces hardware components from the ground up, starting with basic concepts and building toward more complex systems. Each section assumes only knowledge from previous sections. The rest of the chapters are also set up to introduce these ideas in the context of a network.

---

## Part 1: Foundational Concepts

### **Essential Terminology**

Before diving into hardware components, let's define some fundamental terms you'll see throughout:

**CPU (Central Processing Unit):**
- The "brain" of the computer - the chip that executes instructions
- Example: AMD EPYC 7502 (our host CPU)

**System Memory (RAM):**
- The large storage area where the CPU keeps data it's working on
- Example: DDR4 memory sticks you plug into the motherboard

**System Memory In our hs_bridge system:**
- Stores **command packets** that tell the FPGA what to do (e.g., "inject spike from axon 5", "run simulation for 10 timesteps")
- Stores **network configuration data** during initialization (before being transferred to FPGA's HBM)
- Serves as a **staging area** for data transfer: CPU writes data here, then FPGA reads it via DMA
- The FPGA communicates with system memory because:
  - It's where the CPU prepares data for the FPGA
  - Large capacity (gigabytes) - can hold entire network configurations
  - Shared between CPU and FPGA - both can access it (CPU via normal writes, FPGA via DMA)

**Buffer:**
- A temporary holding area for data
- Like a waiting room for data in transit
- **Types we'll see:**
  - **Row buffer:** Built into memory chips, holds one row's worth of data for fast repeated access
  - **DMA buffer:** Region of system memory set aside for transfers between devices
  - **FIFO buffer:** Hardware queue (First-In-First-Out) for data moving between components

**Bus:**
- A set of electrical wires that carry signals between components
- Like a highway connecting two locations
- **Examples:**
  - **DDR4 bus:** Wires connecting CPU to memory chips (64 bits wide)
  - **PCIe bus:** Wires connecting CPU to FPGA (512 bits wide in our case)
  - **AXI bus:** Wires inside the FPGA connecting modules
- **Width:** How many bits can travel in parallel (like highway lanes)

**Bus Master:**
- A device that can initiate (start) transfers on a bus
- Normally: CPU is the master, everything else responds
- With DMA: FPGA can also be a bus master (can request data without CPU help)
- Like: Normally only the manager can request files, but with DMA the assistant can too

**Packet:**
- A chunk of data wrapped with control information (headers)
- Like an envelope: has destination address, sender info, and the actual message inside
- **Examples:**
  - **PCIe TLP (Transaction Layer Packet):** Data moving over PCIe includes address, length, type
  - **Network packet:** Data over Ethernet/WiFi
- **Not all communication uses packets:** CPU talking to memory uses raw electrical signals, not packets

---

### **What is Memory?**

At its core, memory is a place to store digital information (bits: 0s and 1s). Think of it like a massive array of mailboxes, where each mailbox has:
- An **address** (which mailbox)
- **Contents** (what's stored in it)

When we say "read from memory," we're asking: "What's in mailbox #12345?"
When we say "write to memory," we're saying: "Put this value in mailbox #12345."

The two key questions for any memory technology are:
1. **How much can it store?** (capacity)
2. **How fast can we access it?** (bandwidth and latency)

Different types of memory make different trade-offs between these properties.

---

### **Host DDR4 SDRAM (System Memory) - The Basics**

**Full name:** DDR4 SDRAM = **D**ouble **D**ata **R**ate **4**th generation **S**ynchronous **D**ynamic **R**andom **A**ccess **M**emory

**What it is:** This is the main memory in your computer - the RAM that stores your programs and data while running.

**Why it exists:** CPUs (and other processors) need a place to store data they're working on. This data is too large to fit inside the processor itself, so we use external memory chips.

**Library building analogy:**

1. **DIMM (memory stick)** - The entire building. This is what you physically plug into the motherboard.
2. **Rank** - One floor of the building. Most DIMMs have chips on both sides; each side is a "rank."
3. **Chip** - One bookshelf on that floor. Each DIMM has 8-16 memory chips.
4. **Bank** - One section of a bookshelf. Each chip has 8 banks (like having 8 separate card catalogs).
5. **Row** - One shelf in that section. Each bank has ~65,000 rows.
6. **Column** - One book on that shelf. Each row has ~1,000 columns.
7. **DRAM cell** - A single page in a book. This is the smallest storage unit (1 transistor + 1 capacitor storing 1 bit).

So the full hierarchy: **Building → Floor → Bookshelf → Section → Shelf → Book → Page**
Or in hardware terms: **DIMM → Rank → Chip → Bank → Row → Column → DRAM cell**

When you read memory at [address](../supplementary_information/addresses_explained.md) 0x12345678, the memory controller breaks it down:
- "Go to DIMM #2, Rank #1, Chip #5, Bank #3, Row #42, Column #100"
- Like saying: "Building 2, Floor 1, Bookshelf 5, Section 3, Shelf 42, Book 100"

**Physical Organization:**
Think of memory like a filing system:
- **DIMMs (memory sticks):** The physical modules you plug into your motherboard
- **Chips:** Each DIMM has 8-16 chips
- **Banks:** Each chip has 8 banks (like having 8 separate filing cabinets)
- **Rows and Columns:** Each bank is organized as a grid (e.g., 65,000 rows × 1,000 columns)

---

### **PCIe (Peripheral Component Interconnect Express) - The Basics**

**What it is:** A high-speed highway that connects your CPU/memory to peripheral devices like graphics cards, network cards, and FPGAs.

**Think of it like a highway system:**
- **Lanes:** Data travels in lanes (typically 1, 4, 8, or 16 lanes)
- **Bidirectional:** Each lane has traffic going both directions (transmit and receive)
- **Point-to-point:** Each device has its own dedicated connection (not a shared bus)

**Speed:**
- PCIe Gen3: 8 gigabits/second per lane
- An x16 configuration (16 lanes): 16 × 8 = 128 Gb/s raw bandwidth
- After encoding overhead (128b/130b encoding): ~15.75 GB/s effective

**Key takeaway:** PCIe is the data highway connecting your main computer (CPU + memory) to specialized devices like our FPGA. It's fast, reliable, and provides dedicated connections.

---

## Part 2: Moving Data Between Components

Now that we understand basic memory (DDR4) and connections (PCIe), we need to understand *how* data moves between them.

### **Memory-Mapped I/O (MMIO)**

**MMIO = Memory-Mapped Input/Output**
- "Input/Output" (I/O) means communication with devices (keyboard, disk, FPGA, etc.)
- "Memory-Mapped" means we use memory addresses to talk to these devices

**The concept:** Make hardware devices look like memory.

**How it works:**
- The system has a **global address space** (like our mailbox analogy) - typically billions of addresses
  - From the **CPU's perspective:** It can access any address via load/store instructions
  - From the **FPGA's perspective:** When acting as a bus master (during DMA), it can also generate addresses to access system memory
  - The **memory controller** routes each address to the correct destination (RAM vs device registers)
- **Some addresses** refer to RAM (actual memory where data is stored)
- **Other addresses** refer to device registers (special control locations in hardware devices)
- **Who defines addresses:** The system designer/OS assigns address ranges (e.g., "addresses 0xD0000000-0xD0000FFF go to FPGA registers"). Individual FPGA modules don't generate their own address ranges - they respond to the ranges assigned to them.

**Example address map:**
```
Address 0x00000000 - 0x7FFFFFFF: System RAM (actual DDR4 memory)
Address 0xD0000000 - 0xD0000FFF: FPGA control registers (inside FPGA chip)
```

**What happens when you write to an MMIO address:**
```
CPU executes: memory[0xD0000000] = 0x00000001
```
1. CPU puts address 0xD0000000 on the address bus
2. CPU puts value 0x00000001 on the data bus
3. Memory controller sees this address is NOT in RAM range
4. Memory controller routes this over PCIe to the FPGA
5. PCIe wraps it in a TLP packet (Memory Write)
6. FPGA receives the packet, extracts the value
7. FPGA's hardware register at offset 0x0 gets the value 0x00000001
8. This might trigger: start simulation, stop simulation, reset, etc.

**In our system (hs_bridge):**
- Host software uses MMIO to send control commands to FPGA
- Example: `fpga.write_register(0xD0000000, start_flag=1)`
- This gets translated to a PCIe write that lands in FPGA control logic

**Key takeaway:** MMIO lets us control hardware devices using normal memory read/write operations. The CPU doesn't know (or care) if an address goes to RAM or a device - it just reads/writes, and the hardware routes it correctly.

---

## Part 3: Specialized Hardware

Now we understand how the host system works (DDR4 memory, PCIe connections, etc). Let's look at specialized hardware that can process data much faster than a CPU.

### **FPGA (Field-Programmable Gate Array) - The Basics**

**What it is:** An FPGA is a chip full of reconfigurable logic. Think of it as a blank canvas of digital circuits that you can reprogram to do whatever you want.

**Our FPGA:** Xilinx XCVU37p (VU37P)
- **Technology:** 20nm FinFET manufacturing process
- **Die size:** ~800 mm²
- **Power:** 50-100W typical (varies with design and clock speed)
- **Building Blocks:**
  - Configurable Logic Blocks (CLBs)
  - Programmable connection matrix for the CLBs
  - Global clock driving technology (to allow modules to run on the same clock and allow us define a clean state machine)
  - Block RAM
  - UltraRAM

---

### **FPGA Internal Structure**

**Configurable Logic Blocks (CLBs):**

The FPGA fabric is an array of CLBs connected by programmable routing.

**What's in a CLB?**
- **8 LUTs** (Look-Up Tables) - implement logic functions
- **16 Flip-Flops** - store state/register values
- **Carry logic** - for efficient arithmetic

**LUT (Look-Up Table):**

**Physical implementation:** 64-bit SRAM (Static Random Access Memory)
- **SRAM** = memory that holds data as long as power is on (see Appendix for full definition)
- **"Static"** means the module's behavior is fixed once programmed (doesn't change during execution)
  - The LUT configuration is loaded when you program the FPGA (upload the bitstream)
  - During execution, the LUT's function stays constant - it just evaluates its programmed logic
  - To change behavior, you must reprogram the entire FPGA
- **64-bit** = 64 memory cells storing 0s and 1s

**Function:** Can implement any 6-input Boolean function

**How it works - detailed explanation:**

Think of a LUT as a tiny lookup table with 64 entries:
```
┌──────────────┬────────┐
│   Address    │ Output │
│  (6 bits =   │ (1 bit)│
│   inputs)    │        │
├──────────────┼────────┤
│ 0b000000 (0) │   ?    │
│ 0b000001 (1) │   ?    │
│ 0b000010 (2) │   ?    │
│     ...      │  ...   │
│ 0b111110 (62)│   ?    │
│ 0b111111 (63)│   ?    │
└──────────────┴────────┘
```

The 6 input wires form a binary address that selects which of the 64 entries to read.

**Example: Programming a 2-input AND gate (simplified to 2 inputs for clarity)**

An AND gate outputs 1 only when BOTH inputs are 1:
```
Truth table for AND:
  A B │ Output
  ────┼────────
  0 0 │   0
  0 1 │   0
  1 0 │   0
  1 1 │   1    ← Only this outputs 1
```

To implement this in a LUT:
1. Use inputs A and B as the address (ignoring the other 4 input bits)
2. Program the SRAM contents to match the truth table:

```
Address (A,B) │ SRAM Contents │ Meaning
──────────────┼───────────────┼─────────────────
0b000000 (00) │      0        │ 0 AND 0 = 0
0b000001 (01) │      0        │ 0 AND 1 = 0
0b000010 (10) │      0        │ 1 AND 0 = 0
0b000011 (11) │      1        │ 1 AND 1 = 1 ✓
0b000100-111  │      0        │ (unused inputs)
```

**In Verilog:**
```verilog
// This Verilog code:
wire a, b, out;
assign out = a & b;  // AND gate

// Gets synthesized into a LUT where:
// - Inputs a, b connect to LUT input pins
// - LUT is programmed with the AND truth table above
// - LUT output connects to wire 'out'
// - When a=1, b=1: address=0b11, LUT outputs 1
// - When a=0, b=1: address=0b01, LUT outputs 0
```

**For a full 6-input example (like OR gate):**
```verilog
assign out = a | b | c | d | e | f;  // 6-input OR gate

// LUT programmed so:
// - Address 0b000000: outputs 0  (all inputs zero)
// - Address 0b000001: outputs 1  (at least one input is 1)
// - Address 0b000010: outputs 1
// - ...
// - Address 0b111111: outputs 1  (all inputs one)
// Total: 1 zero entry, 63 one entries
```

**Key insight:** The SRAM stores a complete lookup table mapping every possible input combination to the desired output. This is why LUTs can implement ANY 6-input Boolean function - just program the SRAM with the right truth table!

**Example:** `assign out = a & b;`
- Synthesis tool maps this to a LUT
- LUT is programmed (via SRAM bits) to implement the AND function
- Inputs a and b connect to LUT inputs
- LUT output connects to signal out

**Flip-Flop (Register):**
- **Physical implementation:** D-type register (master-slave latch pair)
- **Function:** Stores 1 bit, updates on clock edge
- **Inputs:**
  - D: Data input (value to store)
  - CLK: Clock (when to update)
  - CE: Clock Enable (enable updating)
  - RST: Reset (force to 0)
- **Operation:** On rising edge of CLK: if CE=1, then Q <= D

**Example:** `always @(posedge clk) q <= d;`
- This Verilog creates a flip-flop
- On each clock edge, q gets the value of d

**Synthesis flow:**
```
Verilog code → Logic gates → Map to LUTs + FFs

Example:
  Combinational logic: `assign out = a & b;` → Maps to LUTs
  Registers: `always @(posedge clk) q <= d;` → Maps to Flip-Flops

  (Text in `backticks` is actual Verilog code)
```

---

### **FPGA Routing and Timing**

**Programmable Interconnect:**
- **Problem:** We have thousands of LUTs and FFs that need to connect together
- **Solution:** Programmable switches (like a telephone switchboard)
  - **Switch matrix:** Crossbar at each routing junction
  - **Implementation:** Transistor pass gates controlled by SRAM bits
  - **Configuration:** SRAM bits determine which wires connect

**Routing delay:**
- Signals take time to travel through wires and switches
- Typical: 0.5-2 nanoseconds depending on distance
- This adds to the logic delay (time for LUTs to compute)

**Timing closure:**
Our design runs at 225 MHz (4.4 ns period). This means:
- **Critical path:** The longest path from one flip-flop to another must complete in < 4.4 ns
- Critical path time = LUT delay + routing delay + flip-flop setup time
- If too long: Must add pipeline registers (breaks path into shorter segments)
  - Tradeoff: Adds latency (more clock cycles) but meets timing

---

### **FPGA Clock Distribution**

**Challenge:** We have thousands of flip-flops that all need to see the clock edge at the same time.

**Solution: Global clock tree**
- **H-tree topology:** Balanced routing that fans out to all regions
  - Ensures all flip-flops see the clock edge within ~100 picoseconds (skew)
- **Clock buffers (BUFG):** Special high-fanout buffers
  - Can drive thousands of flip-flops without degradation

**PLLs and MMCMs (Clock generation):**
- **Input:** 100 MHz reference clock
- **Output:** 225 MHz and 450 MHz for our design
- **How they work:**
  - PLL: Phase-Locked Loop tracks input and generates multiples
  - VCO: Voltage-Controlled Oscillator runs at high frequency (900-2000 MHz)
  - Dividers: Divide VCO output to get desired frequencies

---

### **FPGA Memory: Block RAM (BRAM)**

**What it is:** Dedicated memory blocks built into the FPGA (separate from logic fabric)

**Why BRAMs exist:**
- Could build memory using LUTs (they're SRAMs after all)
- But: Inefficient - wastes logic resources
- Better: Dedicated memory blocks optimized for storage

**BRAM technology:**
- **6-transistor SRAM cell:** 2 cross-coupled inverters + 2 access transistors
- **Static storage:** Unlike DRAM, no refresh needed (data persists as long as powered)
- **Trade-off:** More transistors per bit than DRAM, but faster

**RAMB36E2 primitive (basic BRAM block):**
- **Capacity:** 36 Kilobits (36 Kb = 4.5 KB)
- **Configurable width:** 1 to 72 bits wide
  - Width × Depth = 36K bits
  - Examples: 36K×1, 18K×2, 9K×4, ..., 512×72
- **Dual-port:** Can read and write simultaneously on two independent ports

**Access timing:**
- **Synchronous:** Operates on clock edges (not asynchronous like CPU cache)
- **Latency:** 2-3 clock cycles
  - Cycle 0: Present address
  - Cycle 1: Internal row decode
  - Cycle 2: Data valid on output

**Our usage:**
- Configuration: 32,768 addresses × 256 bits wide
- Uses: 256 RAMB36 primitives (each configured and then address-mapped together)

---

### **FPGA Memory: UltraRAM (URAM)**

**What it is:** Higher-density memory blocks (like BRAM but bigger)

**Technology:**
- **1T1C DRAM-like cell:** 1 transistor + 1 capacitor (similar to DDR4 cells)
- **On-chip:** Integrated into FPGA die (not external)
- **Advantage:** 4× density vs BRAM (288 Kb vs 36 Kb per primitive)
- **Trade-off:** Requires refresh (but automatic, handled by primitive logic)

**URAM288 primitive:**
- **Capacity:** 288 Kilobits (36 KB)
- **Configuration:** 4096 words × 72 bits (typical for our design)

**Access timing:**
- **Synchronous:** Operates on clock edges
- **Latency:** 1 clock cycle @ 450 MHz (faster than BRAM!)
  - Cycle N: Address presented
  - Cycle N+1: Data valid

**Refresh:**
- Automatic and transparent to user logic
- Built into the primitive controller

**Our usage:**
- 16 banks × 288 Kb = 4.5 Megabits total
- Stores neuron state information

---

### **FPGA Hard IP Blocks**

**What are "Hard IP" blocks?**
- Most of the FPGA is reconfigurable fabric (LUTs, FFs, routing)
- Some functions are implemented as fixed silicon (not programmable)
- These are "Hard IP" blocks

**Why hard IP?**
- **Performance:** Dedicated circuits run faster than fabric implementation
- **Efficiency:** Use less power and less die area
- **Interfaces:** Some protocols require precise timing (hard to achieve in fabric)

**Our FPGA's Hard IP:**

**1. PCIe block:**
- **Location:** Fixed position on die corner (near pins)
- **Contains:** SerDes (serializer/deserializer), PHY, MAC layers
- **Advantage:** Meets PCIe Gen3 timing requirements reliably

**2. HBM interface controllers:**
- **Purpose:** Interface to High Bandwidth Memory (see next section)
- **Provides:** 32 independent AXI ports (one per HBM channel)
- **Why hard:** Timing-critical signaling for high-speed memory

---

## Part 4: Ultra-High-Performance Memory

We've covered DDR4 (main system memory, moderate bandwidth). Now let's look at specialized memory for extreme bandwidth.

**Where HBM fits in the system:**
- **HBM is physically attached to the FPGA** - they are packaged together on the same silicon interposer
- **It's a customization/option:** Not all FPGAs have HBM; our XCVU37p model includes 8 GB of HBM2
- **Think of it as:** The FPGA's "private" high-speed memory, while DDR4 is the host's "shared" memory
  - Host DDR4: Shared between CPU and FPGA, accessed via PCIe DMA
  - FPGA HBM2: Exclusive to FPGA, direct connection (no PCIe), much faster

### **HBM2 (High Bandwidth Memory) - Why It Exists**

**The bandwidth problem:**
- Our neural network simulation needs to read/write neuron states very quickly
- DDR4 provides ~25 GB/s per channel (good for general-purpose computing)
- But our application needs ~400-900 GB/s (much higher!)

**The solution: HBM2**
- Specialized memory technology optimized for bandwidth
- Trade-offs: More expensive, less capacity than DDR4
- Achieved bandwidth: ~920 GB/s theoretical, ~400 GB/s practical

**How HBM achieves high bandwidth:**
- **Wide buses:** 1024 bits per stack (vs 64 bits for DDR4) = 16× wider
- **High frequency:** 1800 MT/s (similar to DDR4)
- **Calculation:** 1024 bits × 1800 MT/s = 230 GB/s per stack
- **4 stacks:** 230 GB/s × 4 = 920 GB/s total

---

### **HBM2 Memory Organization**

**Hierarchy:**
```
Stack (4 total)
  └─ Channel (8 per stack)
      └─ Bank (16 per channel)
          └─ Row (16,384 per bank)
              └─ Column (1,024 per row)
```

**Row buffer concept:**
- When you activate a row, the entire row (512 bytes) is read into a buffer
- Subsequent accesses to the same row are fast (~10 ns) - "page hit"
- Accessing a different row requires:
  1. Close current row (precharge)
  2. Open new row (activate)
  - This is slower (~50 ns) - "page miss"

**Performance implications:**
- **Best case (sequential access within row):** Very fast, high bandwidth
- **Worst case (random access across rows):** Slower, reduced bandwidth
- **Our design:** Tries to access memory sequentially to maximize page hits

---

### **HBM2 AXI4 Interface**

**What is AXI4?**
- **AXI:** Advanced eXtensible Interface (ARM standard)
- **Purpose:** Standard protocol for connecting memory and devices in hardware
- **Why standard?** Different IP blocks can interoperate (like USB for internal hardware)

**Five independent channels:**

**1. Write Address (AW):** Master sends where to write
- Signals: AWADDR (address), AWLEN (burst length), AWVALID/AWREADY

**2. Write Data (W):** Master sends what to write
- Signals: WDATA (data), WSTRB (byte enables), WVALID/WREADY

**3. Write Response (B):** Slave acknowledges completion
- Signals: BRESP (response code), BVALID/BREADY

**4. Read Address (AR):** Master requests data
- Signals: ARADDR (address), ARLEN (burst length), ARVALID/ARREADY

**5. Read Data (R):** Slave returns data
- Signals: RDATA (data), RRESP (response), RVALID/RREADY

**Key features:**

**Decoupling:**
- Address and data channels are independent
- Can send multiple read addresses, then receive data later
- Enables pipelining and out-of-order completion

**Handshake protocol (VALID/READY):**
- **Source asserts VALID:** "My data is ready"
- **Destination asserts READY:** "I can accept data"
- **Transfer occurs when:** VALID AND READY (both high)
- This allows flow control (receiver can apply backpressure)

**Bursts:**
- Single address can request multiple data beats (up to 256)
- Example: Address=0x1000, Length=16 → returns 16 consecutive words
- Amortizes address overhead (one address, many data)

---

### **HBM2 Access Latency**

**Best case (row hit): ~50 ns**
- Address decode: 5 ns
- Column select: 10 ns
- Sense amplifier: 10 ns
- Data serialization: 10 ns
- AXI handshake: 15 ns

**Worst case (row miss): ~200 ns**
- Precharge old row: 30 ns
- Activate new row: 50 ns
- Column access: 50 ns
- (rest as above)

**Optimization in our design:**
- Prefetch next row during processing current data
- Pipelines operations to hide latency
- Access patterns designed for row locality

---

## Part 5: Data Movement Primitives

Finally, we need ways to move data between all these components (host memory, PCIe, FPGA logic, HBM). FIFOs are the basic building block.

### **FIFO (First-In-First-Out Buffer)**

**What it is:** A hardware queue - data comes out in the same order it went in.

**Why FIFOs exist:**
- **Problem 1:** Different components run at different speeds
  - Example: PCIe sends data in bursts, FPGA processing is continuous
  - FIFO smooths out the rate mismatch
- **Problem 2:** Different components run on different clocks
  - Example: PCIe side at 225 MHz, HBM side at 450 MHz
  - FIFO safely transfers data between clock domains

**Think of it like:**
- A line at a coffee shop (first person in line is first served)
- A pipe (data flows through, can't jump ahead or reorder)

---

### **FIFO Implementation (Xilinx FIFO36E2)**

**Storage:** Uses BRAM36 primitive (36 Kb SRAM block)

**Pointers:**
- **Write pointer (WP):** Points to next location to write
- **Read pointer (RP):** Points to next location to read
- Both are counters that increment with each operation

**Status signals:**
- **Empty:** WP == RP (no data to read)
- **Full:** (WP + 1) mod DEPTH == RP (no space to write)
- Software/hardware checks these before reading/writing

**FWFT mode (First-Word Fall-Through):**
- Normal FIFO: Must assert RD_EN, wait 1 cycle, then data appears
- FWFT FIFO: Data appears on output port as soon as EMPTY goes low
- Implementation: Extra output register + bypass mux
- Advantage: Zero-latency read (useful for streaming pipelines)

---

### **Asynchronous FIFO (Clock Domain Crossing)**

**The problem:**
- Write side: 225 MHz clock
- Read side: 450 MHz clock
- Cannot directly compare pointers (in different clock domains!)

**Why this is hard:**
- If a signal changes in one clock domain and is read in another, **metastability** can occur
- Metastability: Flip-flop input violates setup/hold time → output voltage stuck between 0 and 1
- Can take nanoseconds (or longer!) to resolve to a valid logic level
- During metastability, output can oscillate or produce glitches

**The solution: Gray code + 2-FF synchronizer**

**Gray code:**
- Special binary encoding where only 1 bit changes per increment
- Examples:
  - Binary: 3→4 is 011→100 (3 bits change)
  - Gray: 3→4 is 010→110 (only 1 bit changes)
- **Why this helps:** If we catch the pointer mid-transition, we're only off by ±1 (not random garbage)

**2-FF synchronizer:**
```verilog
always @(posedge rd_clk) begin
  wptr_gray_sync1 <= wptr_gray;      // First FF (may go metastable)
  wptr_gray_sync2 <= wptr_gray_sync1; // Second FF (stable output)
end
```

**How it works:**
1. First FF captures signal from other clock domain
   - May go metastable (voltage between 0 and 1)
2. One full clock period passes (2.2 ns @ 450 MHz)
   - Metastability has time to resolve
3. Second FF captures now-stable value
   - Guaranteed valid 0 or 1

**Empty/Full calculation:**
- **Empty:** Calculated in read domain using synchronized write pointer
  - "Is the read pointer caught up to where the writer was?"
- **Full:** Calculated in write domain using synchronized read pointer
  - "Is the write pointer about to lap the reader?"

**Timing conservative:**
- Due to synchronization, pointers are slightly "old" (2-3 cycles)
- This makes FIFO appear fuller/emptier than reality (safe direction)
- Means: Slightly less efficient, but never corrupts data

---

### **FIFOs in Our System**

**Input/Output FIFOs (PCIe ↔ FPGA fabric):**
- **Width:** 512 bits (64 bytes) - matches PCIe TLP data width
- **Depth:** 512 entries
- **Purpose:** Buffer data transfers between PCIe and processing logic
- **Async:** Crosses clock domain (PCIe clock → fabric clock)

**Pointer FIFOs (HBM data → neuron groups):**
- **Width:** 32 bits
- **Depth:** 512 entries
- **Purpose:** Distribute memory addresses/pointers to different processing units
- **Sync:** Same clock domain (can use simpler FIFO)

**Spike FIFOs (neurons → spike controller):**
- **Width:** 17 bits (neuron ID + metadata)
- **Depth:** 512 entries
- **Purpose:** Collect spike events from processing units
- **Async:** Different processing clocks converge to controller clock

**Key insight:**
- FIFOs are everywhere in the design
- They buffer, smooth rate mismatches, and cross clock domains
- Simple concept, but essential for making everything work together

---

## Summary: How It All Fits Together

**Data flow example: Host sends commands to FPGA**

1. **Host preparation:**
   - Software creates command array in DDR4 memory (system RAM)
   - Gets physical address of buffer (e.g., 0x123456000)

2. **MMIO handoff:**
   - Software writes address to FPGA register via PCIe MMIO
   - "Here's the command buffer: 0x123456000"

3. **DMA transfer:**
   - FPGA reads descriptor from its register
   - FPGA initiates PCIe memory read transactions
   - Requests data from host memory at 0x123456000
   - Host responds with command data over PCIe

4. **Buffering:**
   - PCIe data arrives in bursts → stored in input FIFO
   - FIFO crosses clock domain (225 MHz → 450 MHz)

5. **Processing:**
   - FPGA logic reads commands from FIFO
   - Uses commands to coordinate processing
   - Reads/writes neuron data from/to HBM2 via AXI4

6. **HBM access:**
   - FPGA sends AXI read request to HBM
   - HBM controller activates row, reads data
   - Data returns via AXI → FPGA processes

7. **Results:**
   - Spike events → spike FIFOs
   - Output data → output FIFO → PCIe → host DDR4

**Every component has a role:**
- **DDR4:** Large staging area for host data
- **PCIe:** High-speed highway connecting host and FPGA
- **FPGA:** Reconfigurable logic for parallel processing
- **BRAM/URAM:** Fast on-chip memory for state
- **HBM2:** Ultra-high-bandwidth memory for large datasets
- **FIFOs:** Buffers and clock domain crossings throughout

This is how modern heterogeneous computing systems work: specialized hardware (FPGA) accelerates specific tasks, while remaining integrated with general-purpose host system via high-speed interconnects.
