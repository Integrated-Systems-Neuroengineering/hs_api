# Address Encoding Explained

This document explains how addresses and binary numbers work in hardware systems. Understanding address encoding is fundamental to working with memory, packets, and hardware interfaces.

---

## Binary Address Encoding

Addresses in hardware are binary numbers. Each bit position represents a power of 2:

```
Binary:    0b101010
Positions:   543210  (bit positions, numbered right to left)

Value = (1 × 2^5) + (0 × 2^4) + (1 × 2^3) + (0 × 2^2) + (1 × 2^1) + (0 × 2^0)
      = 32 + 0 + 8 + 0 + 2 + 0
      = 42 (decimal)
```

**Key concepts:**
- **Rightmost bit (bit 0)** is the Least Significant Bit (LSB) - contributes 2^0 = 1
- **Leftmost bit** is the Most Significant Bit (MSB) - contributes the largest power of 2
- Each additional bit doubles the range: n bits can represent 0 to (2^n - 1)

---

## Hexadecimal Notation

Hexadecimal (base 16) is commonly used in hardware because each hex digit represents exactly 4 binary bits:

```
Binary    Hex    Decimal
0000      0      0
0001      1      1
0010      2      2
0011      3      3
0100      4      4
0101      5      5
0110      6      6
0111      7      7
1000      8      8
1001      9      9
1010      A      10
1011      B      11
1100      C      12
1101      D      13
1110      E      14
1111      F      15
```

**Example: Converting between representations**
```
Binary:       0b11010110
Group by 4:   1101  0110
Hexadecimal:    D      6   = 0xD6
Decimal:      (13×16) + 6  = 214
```

**Why use hexadecimal?**
- Compact: 32-bit number = 8 hex digits vs 32 binary digits
- Easy conversion: Each hex digit ↔ 4 bits (no complex math)
- Standard in hardware: Datasheets, debuggers, memory dumps all use hex

---

## Memory Address Format

### **32-bit address example:** `0x12345678`

```
Hex:    1    2    3    4    5    6    7    8
Binary: 0001 0010 0011 0100 0101 0110 0111 1000

This represents byte address 305,419,896 in decimal.

Breakdown:
  0x10000000 = 268,435,456
  0x02000000 =  33,554,432
  0x00300000 =   3,145,728
  0x00040000 =     262,144
  0x00005000 =      20,480
  0x00000600 =       1,536
  0x00000070 =         112
  0x00000008 =           8
  ──────────   ───────────
  0x12345678   305,419,896
```

### **64-bit address example:** `0x0000000123456000`

```
Used in: PCIe Memory Read TLPs, host system physical addresses

Upper 32 bits: 0x00000001
Lower 32 bits: 0x23456000

Decimal: 4,883,857,408 (over 4 GB)
```

---

## Byte Addressing

**Why byte addresses?**
- Most systems are byte-addressable: Each address points to 1 byte (8 bits)
- Even if you read/write larger chunks (32-bit words, 512-bit packets), addresses still refer to bytes
- Consecutive bytes have consecutive addresses

**Example: Address sequence**
```
Address     Content (1 byte each)
─────────   ────────────────────
0x1000      0xAA
0x1001      0xBB
0x1002      0xCC
0x1003      0xDD
0x1004      0xEE
...
```

**Reading multi-byte values:**
```
32-bit word at address 0x1000:
  - Byte 0 (0x1000): 0xAA
  - Byte 1 (0x1001): 0xBB
  - Byte 2 (0x1002): 0xCC
  - Byte 3 (0x1003): 0xDD

Little-endian system: 0xDDCCBBAA (least significant byte first)
Big-endian system:    0xAABBCCDD (most significant byte first)
```

---

## Address Alignment

Many systems require addresses to be aligned to the size of the data being accessed.

**Alignment rules:**
- **Byte (8-bit):** Any address (0x1000, 0x1001, 0x1002...)
- **16-bit (2-byte):** Address must be divisible by 2 (0x1000, 0x1002, 0x1004...)
- **32-bit (4-byte) word:** Address must be divisible by 4 (0x1000, 0x1004, 0x1008...)
- **64-bit (8-byte):** Address must be divisible by 8 (0x1000, 0x1008, 0x1010...)
- **512-bit (64-byte) packet:** Address must be divisible by 64 (0x1000, 0x1040, 0x1080...)

**How to check alignment:**
```python
addr = 0x12345678

# Check if 4-byte aligned (for 32-bit words)
is_4byte_aligned = (addr % 4) == 0
# Equivalent: Check if lower 2 bits are zero
is_4byte_aligned = (addr & 0x3) == 0

# Check if 64-byte aligned (for 512-bit packets)
is_64byte_aligned = (addr % 64) == 0
# Equivalent: Check if lower 6 bits are zero
is_64byte_aligned = (addr & 0x3F) == 0
```

**Why alignment matters:**
- **Performance:** Aligned accesses are faster (single memory transaction)
- **Requirements:** Some hardware can ONLY access aligned addresses
- **Atomicity:** Aligned accesses are often atomic (all-or-nothing)

**Example: Aligned vs unaligned**
```
32-bit word read at address 0x1000 (aligned):
  Single memory access: Read bytes [0x1000-0x1003]

32-bit word read at address 0x1001 (unaligned):
  Two memory accesses needed:
    Read bytes [0x1000-0x1003]
    Read bytes [0x1004-0x1007]
  Extract and combine the middle bytes → slower!
```

---

## Bit Field Notation

When documenting packet formats and registers, we use `[MSB:LSB]` notation to specify bit ranges.

### **Notation Format**

```
[31:0]    means bits 31 down to 0 (32 bits total, all bits of a 32-bit value)
[511:504] means bits 511 down to 504 (8 bits = 1 byte)
[15]      means bit 15 only (single bit)
[7:0]     means bits 7 down to 0 (1 byte, lower 8 bits)
```

**Why MSB:LSB order?**
- **MSB (Most Significant Bit)** comes first: Bit 511 has the highest value (2^511)
- **LSB (Least Significant Bit)** comes last: Bit 0 has the lowest value (2^0)
- Reading left-to-right gives you the range in decreasing significance

### **Example: 512-bit Command Packet**

```
┌─────────────────────────────────────────────────────────────┐
│                   512-bit Command Packet                    │
├───────────────┬─────────────────────────────────────────────┤
│ [511:504]     │ Opcode (8 bits)                             │
│ [503:496]     │ Core ID (8 bits)                            │
│ [495:0]       │ Payload (496 bits)                          │
└───────────────┴─────────────────────────────────────────────┘

Total bits: 512
Bit 511 is the highest/leftmost bit
Bit 0 is the lowest/rightmost bit
```

### **Extracting Bit Fields in Python**

```python
# Example: 512-bit packet (represented as Python integer)
packet = 0x0102000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABCD

# Extract opcode from bits [511:504]
# Step 1: Shift right by 504 bits to move [511:504] to [7:0]
# Step 2: Mask with 0xFF (8 bits = 0b11111111) to keep only lower 8 bits
opcode = (packet >> 504) & 0xFF
# Result: 0x01

# Extract core_id from bits [503:496]
core_id = (packet >> 496) & 0xFF
# Result: 0x02

# Extract payload from bits [495:0]
# Mask with all 1s for lower 496 bits: (2^496 - 1)
payload = packet & ((1 << 496) - 1)
# Result: 0x0000...ABCD (496 bits)

# Extract a middle field, e.g., bits [479:464] (16 bits)
field_16bit = (packet >> 464) & 0xFFFF
```

### **Constructing Packets from Bit Fields**

```python
# Build a 512-bit packet from components
opcode = 0x01      # 8 bits
core_id = 0x00     # 8 bits
payload = 0x1234   # 496 bits (upper bits assumed zero)

# Shift each field to its position and combine with OR
packet = (opcode << 504) | (core_id << 496) | payload

# Result:
# Bits [511:504] = 0x01 (opcode)
# Bits [503:496] = 0x00 (core_id)
# Bits [495:0]   = 0x00...001234 (payload)
```

### **Bit Masks**

Masks are used to isolate specific bits:

```python
# Create a mask for N bits (all 1s)
mask_8bit  = 0xFF          # 0b11111111 (8 bits)
mask_16bit = 0xFFFF        # 0b1111111111111111 (16 bits)
mask_32bit = 0xFFFFFFFF    # (32 bits)

# Create a mask for specific bit range [N:M]
def create_mask(high_bit, low_bit):
    num_bits = high_bit - low_bit + 1
    mask = (1 << num_bits) - 1  # 2^num_bits - 1
    return mask << low_bit

# Example: Mask for bits [15:8] (8 bits in the middle)
mask = create_mask(15, 8)  # Returns 0xFF00

# Use mask to extract field
value = 0x12345678
field = (value & mask) >> 8  # Extract bits [15:8] → 0x56
```

---

## Address Arithmetic

### **Adding Offsets**

```python
base_address = 0x12340000
offset = 0x100  # 256 bytes

# Compute new address
new_address = base_address + offset  # 0x12340100

# For array indexing (4-byte elements)
element_size = 4  # bytes
index = 10
element_address = base_address + (index * element_size)  # 0x12340028
```

### **Splitting Addresses into Pages and Offsets**

Many memory systems organize memory into pages:

```python
address = 0x12345678

# 4KB pages (12-bit offset, 20-bit page number for 32-bit address)
page_size = 4096  # 0x1000
page_offset = address & 0xFFF         # Lower 12 bits: 0x678
page_number = address >> 12           # Upper 20 bits: 0x12345

# Reconstruct address
reconstructed = (page_number << 12) | page_offset  # 0x12345678
```

### **Range Checking**

```python
# Check if address is within a region
region_start = 0xD0000000
region_size = 0x1000  # 4KB
region_end = region_start + region_size

address = 0xD0000500

if region_start <= address < region_end:
    print(f"Address 0x{address:08X} is in region")
    offset_in_region = address - region_start  # 0x500
```

---

## Common Address Ranges in Our System

```
┌──────────────────────────────────────────────────────────────┐
│ System Address Map                                           │
├────────────────────────┬─────────────────────────────────────┤
│ 0x00000000-0x7FFFFFFF  │ Host System RAM (DDR4, 2 GB)        │
│ 0xD0000000-0xD0000FFF  │ FPGA Control Registers (4 KB)       │
│ 0x00000000-0x00003FFF  │ HBM Region 1: Axon Pointers (16 KB) │
│ 0x00004000-0x0007FFFF  │ HBM Region 2: Neuron Ptrs (512 KB)  │
│ 0x00080000-0x7FFFFFFF  │ HBM Region 3: Synapses (~2 GB)      │
└────────────────────────┴─────────────────────────────────────┘

Note: HBM addresses are local to the FPGA (not in host address space)
```

---

## Python Helper Functions

```python
def hex_dump(data, bytes_per_line=16):
    """
    Print data in hexadecimal with addresses

    Example output:
    0x0000: 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F 10
    0x0010: 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F 20
    """
    for i in range(0, len(data), bytes_per_line):
        # Format address
        addr = f"0x{i:04X}:"

        # Format hex bytes
        hex_bytes = " ".join(f"{b:02X}" for b in data[i:i+bytes_per_line])

        print(f"{addr} {hex_bytes}")

def extract_bits(value, high_bit, low_bit):
    """Extract bits [high_bit:low_bit] from value"""
    num_bits = high_bit - low_bit + 1
    mask = (1 << num_bits) - 1
    return (value >> low_bit) & mask

def insert_bits(dest, value, high_bit, low_bit):
    """Insert value into bits [high_bit:low_bit] of dest"""
    num_bits = high_bit - low_bit + 1
    mask = (1 << num_bits) - 1
    # Clear the target bits
    dest &= ~(mask << low_bit)
    # Insert the new value
    dest |= (value & mask) << low_bit
    return dest

def is_aligned(address, alignment):
    """Check if address is aligned to 'alignment' bytes"""
    return (address & (alignment - 1)) == 0

# Usage examples
value = 0x12345678
opcode = extract_bits(value, 31, 24)  # Get upper byte → 0x12

packet = 0
packet = insert_bits(packet, 0x01, 511, 504)  # Set opcode
packet = insert_bits(packet, 0x00, 503, 496)  # Set core_id

print(f"Is 0x1004 aligned to 4 bytes? {is_aligned(0x1004, 4)}")  # True
print(f"Is 0x1005 aligned to 4 bytes? {is_aligned(0x1005, 4)}")  # False
```

---

## Summary

**Key Takeaways:**

1. **Binary is fundamental:** Everything in hardware is binary (0s and 1s)
2. **Hexadecimal is convenient:** Each hex digit = 4 bits, compact representation
3. **Byte addressing:** Addresses point to bytes (8 bits), not bits or words
4. **Alignment matters:** Many operations require aligned addresses for correctness/performance
5. **Bit field notation `[MSB:LSB]`:** Standard way to specify bit ranges in documentation
6. **Bit manipulation:** Shift (>>, <<) and mask (&, |) operations extract/insert bit fields

**When working with hardware:**
- Always check address alignment requirements
- Use bit field notation to understand packet/register formats
- Convert between hex, binary, and decimal as needed
- Verify address ranges before accessing memory

This foundation is essential for understanding the packet formats and memory structures described in the rest of the documentation.
