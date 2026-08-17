# hs_bridge Role and Responsibilities

This document clarifies what `hs_bridge` (the Python software library) actually controls versus what is handled by standard hardware/OS mechanisms.

---

## What is hs_bridge?

**hs_bridge** is a Python software library that provides a high-level interface for communicating with the FPGA-based spiking neural network simulator.

**What it is:**
- Python code running on the host computer (CPU)
- Part of the software stack, not hardware
- Provides functions like `fpga_compiler`, `dmadump`, `fpga_controller`

**What it is NOT:**
- Not a communication protocol (uses existing protocols)
- Not part of the operating system

---

## The Three Communication Paths

To understand hs_bridge's role, we need to understand the three communication paths in the system:

### **Path 1: CPU ↔ System Memory (DDR4)**

**What happens here:**
- CPU writes data to RAM
- CPU reads data from RAM
- This is standard memory operations

**Protocol used:** DDR4 (industry standard)

**Controlled by:**
- CPU hardware (memory controller)
- Operating system (Linux kernel)
- Memory management unit (MMU)

**hs_bridge's role:** **NONE**
- hs_bridge just makes normal Python memory writes
- Example: `array[0] = 0x12345678` is a standard memory operation
- The OS and CPU handle how this actually gets to DDR4

```
┌─────────────────────────────────────────────────────────────┐
│ Path 1: CPU ↔ System Memory                                │
│ Protocol: DDR4 (industry standard)                          │
│ Controlled by: CPU hardware + OS kernel                     │
│ hs_bridge involvement: NONE (just makes normal writes)      │
│                                                              │
│ ┌─────┐  DDR4 protocol  ┌──────────┐                       │
│ │ CPU │ ════════════════│ DDR4 RAM │                       │
│ └─────┘  (no packets)   └──────────┘                       │
│                                                              │
│ When hs_bridge does:                                        │
│   command_array[0] = 0x12345678                             │
│                                                              │
│ Under the hood (hs_bridge has NO control):                  │
│ 1. Python → CPU STORE instruction                           │
│ 2. CPU MMU: virtual address → physical address              │
│ 3. CPU puts address on DDR4 bus                             │
│ 4. Memory controller: ACTIVATE, WRITE, PRECHARGE            │
│ 5. Data written to DDR4 chips                               │
└─────────────────────────────────────────────────────────────┘
```

---

### **Path 2: Host ↔ FPGA (via PCIe MMIO)**

**What happens here:**
- CPU sends control commands to FPGA
- CPU writes to FPGA registers
- This uses Memory-Mapped I/O (MMIO)

**Protocol used:** PCIe TLPs (Transaction Layer Packets - industry standard)

**Controlled by:**
- PCIe hardware (in both CPU and FPGA)
- PCIe specification (industry standard)

**hs_bridge's role:** **DEFINES the register map and command meanings**

What hs_bridge DOES define:
- ✅ Which MMIO addresses map to which FPGA registers
- ✅ What writing to register 0xD0000000 means
- ✅ What writing to register 0xD0000004 means
- ✅ The command format and opcodes

What hs_bridge does NOT define:
- ❌ PCIe packet format (uses standard PCIe TLPs)
- ❌ How MMIO works (standard mechanism)

```
┌─────────────────────────────────────────────────────────────┐
│ Path 2: Host ↔ FPGA (MMIO)                                 │
│ Protocol: PCIe TLPs (industry standard)                     │
│ Controlled by: PCIe hardware                                │
│ hs_bridge involvement: DEFINES register map, meanings       │
│                                                              │
│ ┌──────────┐  PCIe TLPs  ┌──────┐                          │
│ │ Host CPU │ ════════════│ FPGA │                          │
│ └──────────┘  (packets)  └──────┘                          │
│                                                              │
│ What hs_bridge DEFINES:                                     │
│ - Register 0xD0000000 = DMA buffer address                  │
│ - Register 0xD0000004 = DMA control (start/stop)            │
│ - Register 0xD0000008 = Execution control                   │
│ - Writing value 0x1 to control reg = "start"                │
│ - Writing value 0x0 to control reg = "stop"                 │
│                                                              │
│ What hs_bridge does NOT define:                             │
│ - How the write gets turned into a PCIe packet (standard)   │
│ - The PCIe TLP format (industry standard)                   │
│ - How PCIe physical layer works (standard)                  │
└─────────────────────────────────────────────────────────────┘
```

**Example:**
```python
# hs_bridge defines what this means:
fpga.write_register(0xD0000000, buffer_address)
# "Write to register at offset 0x0 = set DMA buffer address"

# hs_bridge does NOT define how this happens:
# - OS kernel driver translates to PCIe write
# - PCIe hardware creates Memory Write TLP packet
# - Packet travels over PCIe physical lanes
# - FPGA PCIe block receives and decodes packet
# All of this uses standard PCIe mechanisms
```

---

### **Path 3: FPGA ↔ System Memory (via DMA)**

**What happens here:**
- FPGA reads data from host DDR4 memory
- FPGA writes results back to host DDR4 memory
- Uses Direct Memory Access (DMA)

**Protocol used:**
- PCIe TLPs for communication (industry standard)
- DDR4 for memory access (industry standard)

**Controlled by:**
- PCIe hardware (standard DMA mechanism)
- Host memory controller (handles DDR4 access)

**hs_bridge's role:** **DEFINES data structures and sets up the transfer**

What hs_bridge DOES define:
- ✅ The format of command structures in memory (512-bit commands)
- ✅ Opcode meanings (0x00 = input, 0x02 = HBM write, etc.)
- ✅ How to tell FPGA where to find the data

What hs_bridge does NOT define:
- ❌ PCIe DMA protocol (standard PCIe Memory Read/Write TLPs)
- ❌ How FPGA becomes a bus master (standard PCIe feature)
- ❌ DDR4 access mechanism (standard hardware)

```
┌─────────────────────────────────────────────────────────────┐
│ Path 3: FPGA ↔ System Memory (via DMA)                     │
│ Protocol: PCIe TLPs (industry standard)                     │
│ Controlled by: PCIe hardware                                │
│ hs_bridge involvement: Defines data structures, sets up     │
│                        transfer, but not the mechanism       │
│                                                              │
│ ┌──────┐  PCIe TLPs  ┌────────────┐  DDR4  ┌──────────┐   │
│ │ FPGA │ ═══════════>│ Memory Ctrl│ ═════=>│ DDR4 RAM │   │
│ └──────┘  (packets)  └────────────┘  (raw) └──────────┘   │
│                                                              │
│ What hs_bridge DOES:                                        │
│ 1. Creates command array in memory (Path 1)                 │
│ 2. Defines command format:                                  │
│    [511:504] = opcode (8 bits)                              │
│    [503:496] = core_id (8 bits)                             │
│    [495:0]   = payload (496 bits)                           │
│ 3. Tells FPGA the address via MMIO (Path 2)                 │
│                                                              │
│ What hs_bridge does NOT do:                                 │
│ - FPGA reads via standard PCIe Memory Read TLPs             │
│ - Host memory controller handles DDR4 access                │
│ - All using industry standard protocols                     │
└─────────────────────────────────────────────────────────────┘
```

---

## What hs_bridge Actually Does

### **1. Data Structure Definitions**

hs_bridge defines the FORMAT of data that gets stored in memory:

```python
# fpga_compiler.py - part of hs_bridge

def create_command_packet(opcode, core_id, payload):
    """
    Creates a 512-bit command packet

    Format (hs_bridge defines this):
    [511:504] = opcode (what operation to perform)
    [503:496] = core_id (which FPGA core)
    [495:0]   = payload (command-specific data)
    """
    packet = (opcode << 504) | (core_id << 496) | payload
    return packet

# hs_bridge defines what these opcodes mean:
OPCODE_INPUT = 0x00        # Inject external spikes
OPCODE_EXECUTE = 0x01      # Run simulation step
OPCODE_HBM_WRITE = 0x02    # Write to HBM memory
OPCODE_HBM_READ = 0x03     # Read from HBM memory
```

**Key point:** hs_bridge defines WHAT the data looks like, not HOW it gets written to memory (that's standard OS/CPU operations).

---

### **2. FPGA Interface Abstraction**

hs_bridge provides Python functions that hide the complexity of MMIO register writes:

```python
# dmadump.py - part of hs_bridge

class DMADump:
    # hs_bridge defines this register map:
    DMA_ADDRESS_REG = 0xD0000000  # Where to write buffer address
    DMA_CONTROL_REG = 0xD0000004  # Where to write control commands

    def dma_dump_write(self, buffer_address):
        """
        Tell FPGA to read data from buffer_address using DMA

        hs_bridge defines:
        - Which registers to write to
        - What order to write them
        - What values mean "start DMA"
        """
        # Write buffer address to FPGA register
        self._write_mmio(self.DMA_ADDRESS_REG, buffer_address)

        # Write start command to control register
        self._write_mmio(self.DMA_CONTROL_REG, 0x1)  # 0x1 = start

    def _write_mmio(self, address, value):
        """
        Low-level MMIO write

        hs_bridge does NOT define how this works:
        - Kernel driver handles PCIe access
        - PCIe hardware creates TLP packets
        - Standard PCIe protocol
        """
        # This uses standard OS system calls
        os.write(self.fpga_device_fd, address, value)
```

---

### **3. Workflow Orchestration**

hs_bridge defines the SEQUENCE of operations to communicate with the FPGA:

```python
# fpga_controller.py - part of hs_bridge

def run_simulation_step():
    """
    hs_bridge orchestrates the workflow
    """
    # Step 1: Create command data in system memory
    # (Uses Path 1 - standard CPU→Memory writes)
    commands = fpga_compiler.compile_commands([
        {'type': 'input', 'spikes': [1, 5, 10]},
        {'type': 'execute', 'timesteps': 100}
    ])
    # commands is now an array in DDR4 system memory

    # Step 2: Get physical address of the buffer
    physical_addr = get_physical_address(commands)

    # Step 3: Tell FPGA where to find commands
    # (Uses Path 2 - MMIO over PCIe)
    dmadump.dma_dump_write(physical_addr)

    # Step 4: FPGA reads commands via DMA
    # (Uses Path 3 - FPGA→Memory via PCIe DMA)
    # This happens automatically in hardware
    # hs_bridge just waits for completion

    # Step 5: Read results back from FPGA
    results = fpga_controller.flush_spikes()

    return results
```

---

## What hs_bridge Does NOT Define

### **CPU ↔ Memory Communication**

```python
# When hs_bridge does this:
command_array[0] = 0x12345678  # Write to memory

# Under the hood (hs_bridge has NO control over this):
# 1. Python interpreter → CPU STORE instruction
# 2. CPU's Memory Management Unit (MMU): virtual → physical address
# 3. CPU puts address on DDR4 address bus (electrical signal)
# 4. CPU puts data on DDR4 data bus (electrical signal)
# 5. Memory controller sends DDR4 commands:
#    - ACTIVATE Bank 3, Row 1000
#    - WRITE Column 50
#    - PRECHARGE
# 6. Data gets written to DRAM cells (capacitor charging)

# All of this is handled by:
# - Linux kernel (memory management)
# - CPU hardware (memory controller)
# - DDR4 protocol (JEDEC standard)

# hs_bridge just makes a normal Python array write
```

### **PCIe Protocol**

```python
# When hs_bridge does this:
fpga.write_register(0xD0000000, value)

# Under the hood (hs_bridge has NO control over this):
# 1. OS kernel driver intercepts the write
# 2. Driver creates PCIe Memory Write request
# 3. PCIe hardware creates TLP packet:
#    Header: [Address: 0xD0000000, Length: 4 bytes, Type: Write]
#    Payload: [value]
# 4. PCIe link layer adds sequence number, CRC
# 5. PCIe physical layer serializes onto differential pairs
# 6. Travels over PCIe lanes (16 lanes × 8 Gb/s each)
# 7. FPGA PCIe block receives, validates CRC, decodes
# 8. FPGA extracts value and writes to internal register

# All of this uses PCIe specification (PCI-SIG standard)
# hs_bridge just makes a system call to write to a device file
```

### **DMA Mechanism**

```python
# When hs_bridge does this:
dmadump.dma_dump_write(buffer_address)

# hs_bridge defines:
# - Which register to write the address to
# - Which register to write the "go" command to

# hs_bridge does NOT define:
# - How FPGA becomes a bus master (standard PCIe feature)
# - PCIe Memory Read TLP format (industry standard)
# - How memory controller handles the request (standard hardware)
# - How data gets packaged in Completion TLP (PCIe standard)
```

---

## Analogy: Email Application

Think of hs_bridge like an email application (Gmail, Outlook, etc.):

**What the email app DOES define:**
- ✅ Email message format (To, From, Subject, Body)
- ✅ User interface (buttons, menus)
- ✅ Workflow (click Send → email goes out)

**What the email app does NOT define:**
- ❌ SMTP protocol (how email travels between servers)
- ❌ TCP/IP (how data travels over the internet)
- ❌ Ethernet/WiFi (physical layer communication)
- ❌ How your router works

Similarly, **hs_bridge:**

**DOES define:**
- ✅ Command data format (opcodes, payload structure)
- ✅ FPGA register map (which addresses do what)
- ✅ Python API (functions to call)
- ✅ Workflow orchestration (sequence of operations)

**Does NOT define:**
- ❌ DDR4 protocol (CPU ↔ Memory)
- ❌ PCIe protocol (Host ↔ FPGA)
- ❌ DMA mechanism (standard PCIe feature)
- ❌ How memory controllers work

---

## Summary Table

| Communication Path | Protocol Used | Controlled By | hs_bridge Role |
|-------------------|--------------|---------------|----------------|
| **CPU ↔ System Memory** | DDR4 (JEDEC standard) | CPU hardware + OS kernel | **None** - just makes normal memory writes |
| **Host ↔ FPGA (MMIO)** | PCIe TLPs (PCI-SIG standard) | PCIe hardware | **Defines** register map, command meanings |
| **FPGA ↔ System Memory (DMA)** | PCIe TLPs (PCI-SIG standard) | PCIe hardware + memory controller | **Defines** data structures, sets up transfer |

---

## Key Takeaway

**hs_bridge is application software that:**
1. Defines the data format and command structure
2. Provides a Python API for FPGA communication
3. Orchestrates the workflow of operations

**hs_bridge relies on standard mechanisms:**
1. OS and CPU hardware for memory access
2. PCIe standard protocol for device communication
3. Standard DMA mechanism for efficient data transfer

**Analogy:** hs_bridge is like a chef writing a recipe (defines what ingredients and steps). The kitchen equipment (CPU, memory controller, PCIe hardware) is what actually cooks the food, using standard techniques (protocols).
