# Packet Encoding Reference

This document provides a complete specification of all packet and data structure encodings used throughout the hs_bridge software and FPGA Verilog code. Understanding these formats is essential for debugging, extending the system, or implementing compatible software.

---

## Table of Contents

1. [Host to FPGA Packets](#host-to-fpga-packets)
2. [FPGA to Host Packets](#fpga-to-host-packets)
3. [HBM Memory Structures](#hbm-memory-structures)
4. [BRAM Memory Structures](#bram-memory-structures)
5. [PCIe Layer](#pcie-layer)

---

## Host to FPGA Packets

These are 512-bit command packets created by `fpga_compiler.py` in hs_bridge and sent to the FPGA via DMA.

### **Command Packet Format (512 bits)**

```
┌─────────────────────────────────────────────────────────────┐
│                   512-bit Command Packet                    │
├───────────────┬─────────────────────────────────────────────┤
│ [511:504]     │ Opcode (8 bits)                             │
│ [503:496]     │ Core ID (8 bits)                            │
│ [495:0]       │ Payload (496 bits, opcode-specific)         │
└───────────────┴─────────────────────────────────────────────┘
```

**Field Descriptions:**

- **Opcode [511:504]:** 8-bit operation type identifier
- **Core ID [503:496]:** Which FPGA core to target (0-31, though typically only core 0 is used)
- **Payload [495:0]:** Operation-specific data (format varies by opcode)

---

### **Opcode Definitions**

| Opcode (hex) | Opcode (binary) | Name | Description |
|--------------|-----------------|------|-------------|
| `0x00` | `8'h00` | INPUT_SPIKES | Inject external axon spikes into BRAM |
| `0x01` | `8'h01` | EXECUTE | Run one simulation timestep |
| `0x02` | `8'h02` | HBM_WRITE | Write data to HBM memory |
| `0x03` | `8'h03` | HBM_READ | Read data from HBM memory |
| `0x04` | `8'h04` | URAM_WRITE | Write neuron states to URAM |
| `0x05` | `8'h05` | URAM_READ | Read neuron states from URAM |
| `0x06` | `8'h06` | CONFIG_WRITE | Write configuration registers |
| `0x07` | `8'h07` | CONFIG_READ | Read configuration registers |
| `0xC8` | `8'hC8` | RESET | Reset FPGA state |

---

### **Opcode 0x00: INPUT_SPIKES**

Injects external spike events (axon activations) into BRAM for processing.

**Payload Format:**
```
[495:480] = Axon ID (16 bits) - which axon is spiking
[479:464] = Spike time (16 bits) - future timestep (optional, usually 0 for immediate)
[463:0]   = Reserved (set to 0)
```

**Example:**
```python
# Axon 42 fires at current timestep
opcode = 0x00
core_id = 0x00
axon_id = 42
spike_time = 0

packet = (opcode << 504) | (core_id << 496) | (axon_id << 480) | (spike_time << 464)
```

**What happens:**
1. FPGA's command_interpreter decodes opcode 0x00
2. Extracts axon_id from payload
3. Writes to BRAM address corresponding to axon_id
4. Sets spike mask bit for this axon

---

### **Opcode 0x01: EXECUTE**

Runs one simulation timestep (processes all pending spikes, updates neurons, generates output spikes).

**Payload Format:**
```
[495:480] = Number of timesteps (16 bits) - typically 1
[479:0]   = Reserved (set to 0)
```

**Example:**
```python
# Execute 1 timestep
opcode = 0x01
core_id = 0x00
num_timesteps = 1

packet = (opcode << 504) | (core_id << 496) | (num_timesteps << 480)
```

**What happens:**
1. FPGA triggers execute state machine
2. Processes all axon spikes (Phase 1: external_events_processor)
3. Processes all neuron spikes (Phase 2: internal_events_processor)
4. Increments execRun_ctr (timestep counter)
5. Returns spike packets to host via output FIFO

---

### **Opcode 0x02: HBM_WRITE**

Writes data directly to HBM memory (for initializing network structure).

**Payload Format:**
```
[495:464] = HBM address (32 bits) - byte address in HBM
[463:432] = Length (32 bits) - number of bytes to write
[431:176] = Data (256 bits) - payload data (up to 32 bytes)
[175:0]   = Reserved
```

**Example:**
```python
# Write synapse data to HBM
opcode = 0x02
core_id = 0x00
hbm_addr = 0x00008000  # Synapse region start
length = 32  # 32 bytes (8 synapses)
data = [0x00100064, 0x00110064, ...]  # Synapse entries

packet = (opcode << 504) | (core_id << 496) | (hbm_addr << 464) | (length << 432) | (data << 176)
```

**What happens:**
1. command_interpreter routes to HBM write controller
2. Issues AXI write transaction to HBM
3. Writes data at specified address

---

### **Opcode 0x04: URAM_WRITE**

Writes neuron state (membrane potential) to URAM.

**Payload Format:**
```
[495:480] = Neuron ID (16 bits) - which neuron to write
[479:444] = Voltage (36 bits) - membrane potential value (signed)
[443:0]   = Reserved
```

**Example:**
```python
# Set neuron 100 voltage to 1000
opcode = 0x04
core_id = 0x00
neuron_id = 100
voltage = 1000  # 36-bit signed value

packet = (opcode << 504) | (core_id << 496) | (neuron_id << 480) | (voltage << 444)
```

**What happens:**
1. command_interpreter routes to URAM write controller
2. Calculates URAM bank (neuron_id >> 13) and local address (neuron_id & 0x1FFF)
3. Performs read-modify-write to update only target neuron (2 neurons per URAM word)
4. Writes back updated 72-bit URAM word

---

### **Opcode 0x06: CONFIG_WRITE**

Writes to configuration registers (threshold, leak parameters, etc.).

**Payload Format:**
```
[495:480] = Register address (16 bits)
[479:416] = Value (64 bits) - configuration value
[415:0]   = Reserved
```

**Register Map:**
| Address | Name | Description |
|---------|------|-------------|
| `0x0000` | THRESHOLD | Spike threshold (36 bits) |
| `0x0001` | LEAK_ENABLE | Enable voltage leak (1 bit) |
| `0x0002` | LEAK_SHIFT | Leak divisor (shift amount) |
| `0x0003` | RESET_VOLTAGE | Voltage after spike |

**Example:**
```python
# Set threshold to 2000
opcode = 0x06
core_id = 0x00
reg_addr = 0x0000  # THRESHOLD register
value = 2000

packet = (opcode << 504) | (core_id << 496) | (reg_addr << 480) | (value << 416)
```

---

## FPGA to Host Packets

These are packets sent from FPGA back to the host, retrieved by `fpga_controller.flush_spikes()`.

### **Spike Packet Format (512 bits)**

```
┌─────────────────────────────────────────────────────────────┐
│                   512-bit Spike Packet                      │
├───────────────┬─────────────────────────────────────────────┤
│ [511:496]     │ Tag = 0xEEEE (identifies as spike packet)   │
│ [495:480]     │ Spike count (16 bits) - number of valid     │
│               │ spikes in this packet (0-14)                │
│ [479:32]      │ Spike data: 14 slots × 32 bits each         │
│               │ Each slot: [31:24] = reserved               │
│               │            [23]    = valid bit              │
│               │            [22:6]  = neuron ID (17 bits)    │
│               │            [5:0]   = sub-timestep (6 bits)  │
│ [31:0]        │ Timestep (32 bits) - execRun_ctr value      │
└───────────────┴─────────────────────────────────────────────┘
```

**Field Descriptions:**

- **Tag [511:496]:** Always `0xEEEE` to identify this as a spike packet
- **Spike count [495:480]:** Number of valid spikes in this packet (1-14)
- **Spike slots [479:32]:** Up to 14 spike entries
  - **Valid bit [23]:** 1 = valid spike, 0 = empty slot
  - **Neuron ID [22:6]:** Which neuron spiked (0-131,071)
  - **Sub-timestep [5:0]:** Fine-grained timing within timestep (usually 0)
- **Timestep [31:0]:** When these spikes occurred (execRun_ctr value)

**Example Packet:**
```
Tag: 0xEEEE
Spike count: 3
Spike 0: neuron_id=42,   valid=1, sub_ts=0
Spike 1: neuron_id=1000, valid=1, sub_ts=0
Spike 2: neuron_id=5123, valid=1, sub_ts=0
Spikes 3-13: valid=0 (empty)
Timestep: 1500
```

**Encoded as:**
```
[511:496] = 0xEEEE
[495:480] = 3 (spike count)
[479:448] = 0x00800150  # Spike 0: neuron 42 (0x2A)
[447:416] = 0x00803E80  # Spike 1: neuron 1000 (0x3E8)
[415:384] = 0x00814046  # Spike 2: neuron 5123 (0x1403)
[383:32]  = 0 (empty slots)
[31:0]    = 1500 (timestep)
```

**Python Parsing:**
```python
def parse_spike_packet(packet_512bit):
    tag = (packet_512bit >> 496) & 0xFFFF
    if tag != 0xEEEE:
        return None  # Not a spike packet

    spike_count = (packet_512bit >> 480) & 0xFFFF
    timestep = packet_512bit & 0xFFFFFFFF

    spikes = []
    for i in range(14):
        spike_word = (packet_512bit >> (32 + i*32)) & 0xFFFFFFFF
        valid = (spike_word >> 23) & 0x1
        if valid:
            neuron_id = (spike_word >> 6) & 0x1FFFF
            sub_ts = spike_word & 0x3F
            spikes.append({'neuron_id': neuron_id, 'timestep': timestep, 'sub_ts': sub_ts})

    return spikes
```

---

## HBM Memory Structures

HBM stores the network structure (pointers and synapses). All addresses are byte addresses.

### **Memory Map**

```
┌──────────────────────────────────────────────────────────┐
│ HBM Memory Layout (8 GB total, 2 GB used)               │
├────────────────┬─────────────────────────────────────────┤
│ 0x00000000     │ Region 1: Axon Pointers                 │
│ - 0x00003FFF   │ Size: 16 KB (16,384 bytes)              │
│                │ Format: 32-bit pointers × 512 axons     │
├────────────────┼─────────────────────────────────────────┤
│ 0x00004000     │ Region 2: Neuron Pointers               │
│ - 0x00007FFF   │ Size: 512 KB                            │
│                │ Format: 32-bit pointers × 131,072 neurons│
├────────────────┼─────────────────────────────────────────┤
│ 0x00008000     │ Region 3: Synapses                      │
│ - 0x7FFFFFFF   │ Size: ~2 GB (variable, network-dependent)│
│                │ Format: Variable-length synapse lists   │
└────────────────┴─────────────────────────────────────────┘
```

---

### **Pointer Format (32 bits)**

Pointers are stored in Regions 1 and 2, mapping axon/neuron IDs to their synapse lists.

```
┌───────────────────────────────────────────────────────────┐
│                  32-bit Pointer Entry                     │
├────────────────┬──────────────────────────────────────────┤
│ [31:23]        │ Length (9 bits) - number of synapse rows│
│ [22:0]         │ Start Address (23 bits) - HBM row index │
│                │ (actual byte address = 0x8000 + addr×32)│
└────────────────┴──────────────────────────────────────────┘
```

**Example:**
```
Axon 5 pointer = 0x00201234
  Length = 0x001 (1 row = 8 synapses)
  Start address = 0x01234 (row index)
  Actual HBM address = 0x8000 + (0x1234 × 32) = 0x2A680
```

**Python Encoding:**
```python
def encode_pointer(start_row, num_rows):
    """
    start_row: Row index in synapse region (not byte address)
    num_rows: Number of consecutive rows (each row = 8 synapses)
    """
    length = num_rows & 0x1FF  # 9 bits
    address = start_row & 0x7FFFFF  # 23 bits
    pointer = (length << 23) | address
    return pointer

def decode_pointer(pointer):
    length = (pointer >> 23) & 0x1FF
    start_row = pointer & 0x7FFFFF
    byte_address = 0x8000 + (start_row * 32)
    return {'num_rows': length, 'start_row': start_row, 'byte_address': byte_address}
```

---

### **Synapse Format (32 bits)**

Synapses are stored in Region 3, organized as rows of 8 synapses each (256 bits = 32 bytes per row).

```
┌───────────────────────────────────────────────────────────┐
│                  32-bit Synapse Entry                     │
├────────────────┬──────────────────────────────────────────┤
│ [31:29]        │ OpCode (3 bits)                          │
│                │   000 = Regular synapse                  │
│                │   100 = Output spike (send to host)      │
│                │   101 = Recurrent connection             │
│ [28:16]        │ Target Address (13 bits)                 │
│                │   For synapse: target neuron ID          │
│                │   For output: neuron to monitor          │
│ [15:0]         │ Weight (16 bits, signed fixed-point)     │
│                │   Interpretation: weight / 32768         │
└────────────────┴──────────────────────────────────────────┘
```

**OpCode Details:**

| OpCode | Binary | Meaning | Target Field | Weight Field |
|--------|--------|---------|--------------|--------------|
| 0 | `3'b000` | Regular synapse | Neuron ID (13 bits, 0-8191 within group) | Synaptic weight (signed 16-bit) |
| 4 | `3'b100` | Output spike | Neuron ID to report | Unused (set to 0) |
| 5 | `3'b101` | Recurrent | Global neuron ID (13 bits) | Synaptic weight |

**Weight Encoding:**

Weights are 16-bit signed integers representing fixed-point values:
- **Range:** -32,768 to +32,767
- **Interpretation:** `weight_value / 32768.0`
- **Examples:**
  - `0x7FFF` (32767) → +0.9999... ≈ +1.0
  - `0x4000` (16384) → +0.5
  - `0x0400` (1024) → +0.03125
  - `0x0000` (0) → 0.0
  - `0xFC00` (-1024) → -0.03125
  - `0x8000` (-32768) → -1.0

**Example Synapses:**
```python
# Regular synapse: target neuron 42, weight +1000 (≈0.0305)
synapse_1 = (0b000 << 29) | (42 << 16) | 1000
# = 0x002A03E8

# Output spike: report neuron 100
synapse_2 = (0b100 << 29) | (100 << 16) | 0
# = 0x80640000

# Negative weight synapse: target neuron 10, weight -500 (inhibitory)
synapse_3 = (0b000 << 29) | (10 << 16) | ((-500) & 0xFFFF)
# = 0x000AFE0C
```

**Python Encoding:**
```python
def encode_synapse(opcode, target, weight):
    """
    opcode: 0=regular, 4=output, 5=recurrent
    target: neuron ID (0-8191 for regular, 0-131071 for global)
    weight: signed integer (-32768 to 32767)
    """
    opcode_bits = (opcode & 0x7) << 29
    target_bits = (target & 0x1FFF) << 16
    weight_bits = weight & 0xFFFF
    synapse = opcode_bits | target_bits | weight_bits
    return synapse

def decode_synapse(synapse):
    opcode = (synapse >> 29) & 0x7
    target = (synapse >> 16) & 0x1FFF
    weight = synapse & 0xFFFF
    # Sign extend weight if necessary
    if weight & 0x8000:  # Negative
        weight = weight - 65536
    return {'opcode': opcode, 'target': target, 'weight': weight}
```

**Synapse Row (256 bits = 8 synapses):**
```
Row at HBM address 0x8000:
  [255:224] = Synapse 7
  [223:192] = Synapse 6
  [191:160] = Synapse 5
  [159:128] = Synapse 4
  [127:96]  = Synapse 3
  [95:64]   = Synapse 2
  [63:32]   = Synapse 1
  [31:0]    = Synapse 0
```

---

## BRAM Memory Structures

BRAM stores spike masks for external events (axon spikes).

### **BRAM Organization**

```
┌──────────────────────────────────────────────────────────┐
│ BRAM: 32,768 rows × 256 bits per row = 1 MB             │
├────────────────┬─────────────────────────────────────────┤
│ Address        │ Content                                 │
├────────────────┼─────────────────────────────────────────┤
│ 0x0000         │ Axon/Event 0 spike mask                 │
│ 0x0001         │ Axon/Event 1 spike mask                 │
│ ...            │ ...                                     │
│ 0x7FFF         │ Axon/Event 32,767 spike mask            │
└────────────────┴─────────────────────────────────────────┘
```

---

### **Spike Mask Format (256 bits)**

Each row contains a 256-bit bitmask indicating which neuron groups should receive this spike.

```
┌───────────────────────────────────────────────────────────┐
│              256-bit Spike Mask (one BRAM row)            │
├────────────────┬──────────────────────────────────────────┤
│ [255:240]      │ Group 15 mask (16 bits)                  │
│ [239:224]      │ Group 14 mask (16 bits)                  │
│ ...            │ ...                                      │
│ [31:16]        │ Group 1 mask (16 bits)                   │
│ [15:0]         │ Group 0 mask (16 bits)                   │
└────────────────┴──────────────────────────────────────────┘
```

**Each 16-bit group mask:**
- Bit 0: First neuron in group should receive spike
- Bit 1: Second neuron in group should receive spike
- ...
- Bit 15: 16th neuron in group should receive spike

**Note:** This is a coarse-grained mask. For fine-grained connectivity, the spike is processed further:
1. BRAM mask identifies which groups get the spike
2. For each group, HBM is read to get the full synapse list
3. Synapse list specifies exact target neurons and weights

**Example:**
```
Axon 5 fires, BRAM row 5 contains:
  Group 0 mask: 0x000F (neurons 0-3 in group 0)
  Group 1 mask: 0x0000 (no neurons in group 1)
  Group 2 mask: 0x8000 (neuron 15 in group 2)
  Groups 3-15: 0x0000

This means axon 5 spike should be delivered to:
  - Neurons 0, 1, 2, 3 in group 0
  - Neuron 15 in group 2
```

**Python Encoding:**
```python
def encode_bram_mask(group_masks):
    """
    group_masks: list of 16 integers (16-bit masks for each group)
    Returns: 256-bit value
    """
    mask = 0
    for i, group_mask in enumerate(group_masks):
        mask |= (group_mask & 0xFFFF) << (i * 16)
    return mask

def decode_bram_mask(mask_256bit):
    """
    mask_256bit: 256-bit value
    Returns: list of 16 group masks
    """
    group_masks = []
    for i in range(16):
        group_mask = (mask_256bit >> (i * 16)) & 0xFFFF
        group_masks.append(group_mask)
    return group_masks
```

---

## PCIe Layer

All communication between host and FPGA travels over PCIe using Transaction Layer Packets (TLPs).

### **PCIe TLP Format**

hs_bridge and the FPGA do NOT directly create PCIe TLPs - the PCIe hardware handles this automatically. However, understanding the format is useful for debugging.

**Memory Write TLP (Host → FPGA MMIO):**
```
┌─────────────────────────────────────────────────────────────┐
│                   PCIe Memory Write TLP                     │
├────────────────────┬────────────────────────────────────────┤
│ Header (3-4 DWords)│                                        │
│ [127:125]          │ Fmt = 010 (write with data, 32-bit addr)│
│ [124:120]          │ Type = 00000 (memory write)            │
│ [95:64]            │ Address (32 bits) - FPGA MMIO address  │
│ [9:0]              │ Length (10 bits) - DWords to transfer  │
├────────────────────┼────────────────────────────────────────┤
│ Data (N DWords)    │ Payload data (up to 4096 bytes)        │
└────────────────────┴────────────────────────────────────────┘
```

**Memory Read TLP (FPGA → Host Memory via DMA):**
```
┌─────────────────────────────────────────────────────────────┐
│                   PCIe Memory Read TLP                      │
├────────────────────┬────────────────────────────────────────┤
│ Header (4 DWords)  │                                        │
│ [127:125]          │ Fmt = 001 (read request, 64-bit addr)  │
│ [124:120]          │ Type = 00000 (memory read)             │
│ [95:0]             │ Address (64 bits) - host DDR4 address  │
│ [9:0]              │ Length (10 bits) - DWords requested    │
└────────────────────┴────────────────────────────────────────┘
```

**Completion TLP (Host → FPGA, returning DMA data):**
```
┌─────────────────────────────────────────────────────────────┐
│                   PCIe Completion TLP                       │
├────────────────────┬────────────────────────────────────────┤
│ Header (3 DWords)  │                                        │
│ [127:125]          │ Fmt = 010 (completion with data)       │
│ [124:120]          │ Type = 01010 (completion)              │
│ [9:0]              │ Byte count (10 bits)                   │
├────────────────────┼────────────────────────────────────────┤
│ Data (N DWords)    │ Requested data from host memory        │
└────────────────────┴────────────────────────────────────────┘
```

**Key Points:**
- **DWord:** 32-bit (4-byte) word
- **Addressing:** Can be 32-bit or 64-bit depending on format
- **Maximum payload:** 4096 bytes (4 KB) per TLP
- **Ordering:** Memory writes are posted (no response), reads require completions

**hs_bridge's Role:**
- hs_bridge does NOT create TLPs directly
- When hs_bridge writes to an MMIO address, the OS kernel driver and PCIe hardware create the TLP
- When FPGA does DMA, the FPGA's PCIe hard block creates Memory Read TLPs automatically

---

## Summary: Packet Flow

### **Host to FPGA Flow:**

```
1. Python (hs_bridge):
   packet = create_512bit_command(opcode=0x01, ...)

2. Write to system memory (DDR4):
   dma_buffer[0] = packet

3. Tell FPGA via MMIO (creates PCIe Memory Write TLP):
   fpga.write_register(DMA_ADDR_REG, physical_address)

4. FPGA reads via DMA (creates PCIe Memory Read TLP):
   FPGA → PCIe: "Send me data from address X"

5. Host responds (PCIe Completion TLP):
   Host → FPGA: "Here's the 512-bit packet"

6. FPGA decodes:
   Extracts opcode, routes to appropriate module
```

### **FPGA to Host Flow:**

```
1. Neuron spikes:
   URAM threshold check → spike detected

2. Spike collection:
   Spike FIFO gathers spikes from all neuron groups

3. Packet assembly:
   spike_fifo_controller creates 512-bit spike packet

4. Write to output FIFO:
   Buffered in FPGA FIFO

5. DMA to host (creates PCIe Memory Write TLP):
   FPGA → Host memory: Write spike packet to DMA buffer

6. Host retrieves:
   fpga_controller.flush_spikes() reads from DMA buffer
```

---

## Quick Reference Tables

### **Command Opcodes**
| Code | Name | Payload |
|------|------|---------|
| 0x00 | INPUT_SPIKES | `[495:480]=axon_id` |
| 0x01 | EXECUTE | `[495:480]=num_timesteps` |
| 0x02 | HBM_WRITE | `[495:464]=addr, [463:432]=len, [431:176]=data` |
| 0x04 | URAM_WRITE | `[495:480]=neuron_id, [479:444]=voltage` |
| 0x06 | CONFIG_WRITE | `[495:480]=reg_addr, [479:416]=value` |

### **Synapse OpCodes**
| Code | Binary | Meaning |
|------|--------|---------|
| 0 | `000` | Regular synapse |
| 4 | `100` | Output spike (send to host) |
| 5 | `101` | Recurrent connection |

### **Memory Regions**
| Region | Base Address | Size | Contents |
|--------|--------------|------|----------|
| Axon Ptrs | 0x00000000 | 16 KB | Axon → synapse pointers |
| Neuron Ptrs | 0x00004000 | 512 KB | Neuron → synapse pointers |
| Synapses | 0x00008000 | ~2 GB | Synapse lists |

---

This reference should provide all the information needed to encode/decode packets and data structures used throughout the hs_bridge and FPGA implementation.
