# 4.2 Reading and Writing Synapses

Implementing Reward-modulated Spike-Timing-Dependent Plasticity (RSTDP) requires the ability to dynamically read and modify synaptic weights during network execution. This page explains how the `read_synapse()` and `write_synapse()` functions work, from the high-level Python API down to the FPGA hardware implementation.

---

## Memory Organization Recap

Before diving into the functions, let's recall how synapses are organized in HBM (from Chapter 1):

**HBM Structure:**
```
hbm[0] = (axon_pointers, neuron_pointers, synapse_data)

Region 1: Axon Pointers (Base 0x0000)
  - Each pointer: (start_row, end_row) in Region 3

Region 2: Neuron Pointers (Base 0x4000)
  - Each pointer: (start_row, end_row) in Region 3

Region 3: Synapses (Base 0x8000)
  - Each row: 8 synapses (256 bits total)
  - Each synapse: 32 bits = [31:29]=OpCode | [28:16]=Address | [15:0]=Weight
```

**Key insight:** Each presynaptic source (axon or neuron) gets one or more **contiguous rows** in HBM Region 3. The pointer tells us where those rows are.

---

## Python API: read_synapse()

**Location:** `hs_bridge/hs_bridge/network.py` (lines 148-166)

### Function Signature

```python
def read_synapse(self, preIndex, postIndex, axonFlag=False)
```

### Parameters

- **preIndex** (int): The index of the presynaptic element (axon or neuron)
  - If `axonFlag=True`: This is an axon index (e.g., `a0` = index 0)
  - If `axonFlag=False`: This is a neuron index (e.g., `h0` = index 0)

- **postIndex** (tuple): A 2-element tuple `(neuron_index, position_within_neuron)`
  - `neuron_index`: Which neuron receives the synapse
  - `position_within_neuron`: Which connection slot (since neurons can have multiple synapses)

- **axonFlag** (bool, optional):
  - `True` = Read from axon → neuron connections (HBM Region 1 pointers)
  - `False` = Read from neuron → neuron connections (HBM Region 2 pointers)

### Return Value

Returns a single synapse tuple: `(operation, address, weight)`
- `operation` (int): OpCode (0=internal connection, 4=spike output)
- `address` (int): 13-bit target neuron address
- `weight` (int): 16-bit synaptic weight (signed fixed-point)

### How It Works

Let's trace through an example: **Read the synapse from axon a0 to neuron h0**

```python
synapse = network.read_synapse(
    preIndex=0,        # a0 is axon 0
    postIndex=(0, 0),  # h0 is neuron 0, position 0
    axonFlag=True      # Reading from axon pointers
)
# Returns: (0, 0, 1000) = operation=0, target=h0, weight=1000
```

**Step-by-step execution:**

1. **Select pointer type:**
   ```python
   pntrs = self.hbm[0][0] if axonFlag else self.hbm[0][1]
   # axonFlag=True → use axon_pointers
   ```

2. **Get synapse range for presynaptic element:**
   ```python
   synapseRange = pntrs.flatten()[preIndex]
   # preIndex=0 → synapseRange = (0, 0)  (start row 0, end row 0)
   # This tells us a0's synapses are in row 0 of Region 3
   ```

3. **Extract synapse data from HBM:**
   ```python
   synapses = self.hbm[0][2][synapseRange[0]:synapseRange[1]+1]
   # self.hbm[0][2] is synapse_data
   # [0:1] extracts row 0
   # synapses is now a 2D array of synapse tuples
   ```

4. **Calculate row and column indices:**
   ```python
   # DATA_PER_ROW = 8 (synapses per row)
   rowIdx = (postIndex[0]*2 + 1) if (postIndex[1]//DATA_PER_ROW == 0) else postIndex[0]*2
   # postIndex = (0, 0)
   # postIndex[1]//DATA_PER_ROW = 0//8 = 0
   # rowIdx = 0*2 + 1 = 1

   columnIdx = postIndex[1] % DATA_PER_ROW
   # columnIdx = 0 % 8 = 0
   ```

5. **Return the synapse:**
   ```python
   return synapses[rowIdx][columnIdx]
   # Returns synapse at [1][0]
   ```

### Important Notes

- **Software-only:** `read_synapse()` reads from the Python data structure (`self.hbm`), not from the actual FPGA hardware
- **Fast:** No PCIe communication, just local memory access
- **Use case:** Query what weight is currently programmed (e.g., for debugging or RSTDP calculations)

---

## Python API: write_synapse()

**Location:** `hs_bridge/hs_bridge/network.py` (lines 181-206)

### Function Signature

```python
def write_synapse(self, preIndex, postIndex, weight, axonFlag=False)
```

### Parameters

Same as `read_synapse()`, plus:
- **weight** (int): The new synaptic weight to write (16-bit signed integer)

### Return Value

None (modifies the synapse in-place and updates hardware)

### How It Works

Continuing our example: **Update the weight from a0 to h0 to 2000**

```python
network.write_synapse(
    preIndex=0,
    postIndex=(0, 0),
    weight=2000,
    axonFlag=True
)
```

**Step-by-step execution:**

1. **Steps 1-4 are identical to read_synapse()** - locate the synapse at `[rowIdx][columnIdx]`

2. **Retrieve current synapse:**
   ```python
   oldSynapse = synapses[rowIdx][columnIdx]
   # oldSynapse = (0, 0, 1000)
   row = synapses[rowIdx]
   # row is the entire row containing this synapse
   ```

3. **Update only the weight, preserving operation and address:**
   ```python
   row[columnIdx] = (oldSynapse[0], oldSynapse[1], weight)
   # row[0] = (0, 0, 2000)
   # Only the weight changed: 1000 → 2000
   ```

4. **Write the entire row back to FPGA hardware:**
   ```python
   write_synapse_row(
       synapseRange[0] + rowIdx,  # HBM row address
       row,                        # Modified row data
       simDump=False,
       coreID=self.coreOveride
   )
   ```

This is where the magic happens - `write_synapse_row()` sends the updated data to the FPGA via PCIe.

---

## Python API: write_synapse_row()

**Location:** `hs_bridge/hs_bridge/FPGA_Execution/fpga_controller.py` (lines 1057-1107)

This is the low-level function that actually communicates with the FPGA hardware.

### Function Signature

```python
def write_synapse_row(r, row, simDump=False, coreID=0)
```

### Parameters

- **r** (int): HBM row address in Region 3 (e.g., 0x8000 for row 0)
- **row** (list of tuples): 8 synapse tuples, each `(operation, address, weight)`
- **simDump** (bool): If True, returns hex commands instead of writing
- **coreID** (int): Target core ID

### How It Works

**Step-by-step execution:**

1. **Create command prefix:**
   ```python
   coreBits = np.binary_repr(coreID, 5) + 3*'0'  # 5 bits for core ID
   coreByte = '{:0{width}x}'.format(int(coreBits, 2), width=2)
   HBM_OP_RW = '02' + coreByte + 27 * '00'
   # Command opcode: 0x02 = HBM read/write command
   ```

2. **Encode the HBM row address:**
   ```python
   rowAddress = '1' + np.binary_repr(r + SYN_BASE_ADDR, 23)
   # Leading '1' = write operation
   # SYN_BASE_ADDR = 0x8000 (Region 3 base)
   # Example: r=0 → rowAddress = '1' + binary(0x8000) = write to row 0x8000
   ```

3. **Encode each synapse in the row:**
   ```python
   for synapse in row:
       if synapse[0] == 0:  # Internal connection
           binCmd = (np.binary_repr(0, SYN_OP_BITS) +
                     np.binary_repr(int(synapse[1]), SYN_ADDR_BITS) +
                     np.binary_repr(int(synapse[2]), SYN_WEIGHT_BITS))
           # [31:29]=000, [28:16]=address, [15:0]=weight
       elif synapse[0] == 4:  # External spike output
           binSpike = (np.binary_repr(4, SYN_OP_BITS) +
                       12*'0' +
                       np.binary_repr(synapse[1], 17))
           # [31:29]=100, [28:17]=0, [16:0]=spike target
   ```

4. **Build the full 512-bit command packet:**
   ```
   [511:504] = 0x02 (CMD_HBM_RW opcode)
   [503:280] = core ID and padding
   [279]     = '1' (write flag)
   [278:256] = 23-bit HBM row address
   [255:0]   = 256-bit row data (8 × 32-bit synapses)
   ```

5. **Send to FPGA via PCIe/DMA:**
   ```python
   exitCode = dmadump.dma_dump_write(cmd_array, len(cmd_array), 1, 0, 0, 0, dmadump.DmaMethodNormal)
   ```

The command travels: **Host PC → PCIe → FPGA Command Interpreter → HBM Processor → HBM**

---

## Hardware Implementation: Command Interpreter

**Location:** `command_interpreter.v` (lines 241, 462-469, 720-725)

The command interpreter is the FPGA's "front desk" - it receives commands from the host PC and routes them to the appropriate module.

### Command Opcode

```verilog
localparam [7:0] CMD_HBM_RW = 8'd2;  // Opcode for HBM read/write
```

### Command Packet Format (from host PC)

```verilog
[511:504] = 0x02 (CMD_HBM_RW)
[279]     = R/W flag (0=read, 1=write)
[278:256] = 23-bit HBM address
[255:0]   = 256-bit data payload (for writes)
```

### Routing Logic (RX State Machine)

```verilog
CMD_HBM_RW: begin
    if (~ci2hbm_full) begin       // Check if HBM queue has space
        ci2hbm_wren = 1'b1;        // Send command to HBM processor
        rxFIFO_rden = 1'b1;        // Consume command from PCIe FIFO
        rx_next_state = RX_STATE_IDLE;
    end
end
```

The command interpreter simply forwards the HBM read/write command to the HBM processor via a FIFO queue.

### Response Path (TX State Machine)

```verilog
if (!hbm2ci_empty) begin
    // HBM processor has response data
    txFIFO_din  = {16'hBBBB, 240'd0, hbm2ci_dout};
    // Format: [511:496]=0xBBBB (response opcode), [255:0]=HBM data
    txFIFO_wren = 1'b1;            // Send packet to host
    hbm2ci_rden = 1'b1;            // Consume from HBM response FIFO
end
```

For read operations, the HBM processor returns data with opcode `0xBBBB`, which the command interpreter forwards back to the host PC.

---

## Hardware Implementation: HBM Processor

**Location:** `hbm_processor.v` (lines 573-750, 1046)

The HBM processor manages all communication with the High Bandwidth Memory. It has two modes:

1. **Automatic mode:** During network execution (reading pointers and synapses)
2. **Host access mode:** When the host PC wants to read/write synapses

### Address Mapping

HBM uses 33-bit byte-aligned addresses, but our synapse addresses are 23-bit row addresses. The mapping:

```verilog
// For host access:
hbm_araddr <= {5'd0, ci2hbm_dout[278:256], 5'd0};
//            |  5  |      23-bit addr     |  5  | = 33 bits total
//            padding    from command      32-byte alignment
```

The 5-bit padding at the end provides 32-byte alignment (2^5 = 32 bytes = 256 bits = one HBM row).

### Host Read Operation (TX State Machine)

```verilog
TX_STATE_READ_HBM_ADDR: begin
    hbm_arvalid <= 1'b1;  // Assert read address valid
    if (hbm_arready) begin
        // HBM accepted the read request
        ci2hbm_rden <= 1'b1;  // Consume command from queue
        tx_next_state <= TX_STATE_IDLE;
    end
end
```

**AXI4 Read Address (AR) Channel:**
- `hbm_araddr`: 33-bit read address
- `hbm_arvalid`: Address valid signal (request active)
- `hbm_arready`: HBM ready to accept address

### Host Write Operation (TX State Machine - 3 Phases)

AXI4 writes require three separate handshakes:

**Phase 1: Write Address**
```verilog
TX_STATE_WRITE_HBM_ADDR: begin
    hbm_awvalid <= 1'b1;  // Assert write address valid
    if (hbm_awready) begin
        tx_next_state <= TX_STATE_WRITE_HBM_DATA;
    end
end
```

**Phase 2: Write Data**
```verilog
TX_STATE_WRITE_HBM_DATA: begin
    hbm_wvalid <= 1'b1;   // Assert write data valid
    if (hbm_wready) begin
        tx_next_state <= TX_STATE_WRITE_HBM_RESP;
    end
end
```

**Phase 3: Write Response**
```verilog
TX_STATE_WRITE_HBM_RESP: begin
    hbm_bready <= 1'b1;   // Ready to accept write response
    if (hbm_bvalid) begin
        // Write completed successfully
        ci2hbm_rden <= 1'b1;  // Consume command
        tx_next_state <= TX_STATE_IDLE;
    end
end
```

**Data Routing:**
```verilog
assign hbm_wdata = ci2hbm_dout[255:0];  // Write data from command
assign hbm_awaddr = {5'd0, ci2hbm_dout[278:256], 5'd0};  // Write address
```

### Host Read Response (RX State Machine)

```verilog
RX_STATE_READ_HBM_RESP: begin
    if (hbm_rvalid & ~hbm2ci_full) begin
        // HBM data available and response queue has space
        hbm_rready <= 1'b1;      // Accept data
        hbm2ci_wren <= 1'b1;     // Send to Command Interpreter
        rx_next_state <= RX_STATE_IDLE;
    end
end

// Forward raw HBM data to Command Interpreter
assign hbm2ci_din = hbm_rdata;
```

**AXI4 Read Data (R) Channel:**
- `hbm_rdata`: 256-bit read data
- `hbm_rvalid`: Read data valid
- `hbm_rready`: Processor ready to accept data

---

## AXI4 Protocol Summary

The HBM Processor uses the AXI4 memory-mapped protocol to communicate with HBM. AXI4 has 5 independent channels:

**Read Transaction:**
1. **AR (Address Read):** Master sends read address
   - `hbm_araddr[32:0]`: Address to read
   - `hbm_arvalid`: Address valid
   - `hbm_arready`: HBM ready

2. **R (Read Data):** HBM returns data
   - `hbm_rdata[255:0]`: 256-bit data
   - `hbm_rvalid`: Data valid
   - `hbm_rready`: Master ready

**Write Transaction:**
1. **AW (Address Write):** Master sends write address
   - `hbm_awaddr[32:0]`: Address to write
   - `hbm_awvalid`: Address valid
   - `hbm_awready`: HBM ready

2. **W (Write Data):** Master sends data
   - `hbm_wdata[255:0]`: 256-bit data
   - `hbm_wvalid`: Data valid
   - `hbm_wready`: HBM ready

3. **B (Write Response):** HBM confirms write
   - `hbm_bvalid`: Response valid
   - `hbm_bready`: Master ready

The key insight: **Address and data can be sent independently**, allowing pipelined transactions for higher throughput.

---

## Complete Data Flow

Let's trace a complete write operation from start to finish:

### Writing Synapse: a0 → h0, weight = 2000

**Step 1: Python (host PC)**
```python
network.write_synapse(preIndex=0, postIndex=(0,0), weight=2000, axonFlag=True)
  ↓
write_synapse_row(r=0, row=[(0,0,2000), (0,1,1000), ...])
```

**Step 2: Python builds PCIe command**
```python
# Command packet (512 bits):
[511:504] = 0x02        # CMD_HBM_RW
[279]     = 0x1         # Write flag
[278:256] = 0x8000      # HBM row address (Region 3, row 0)
[255:224] = 0x00000_7D0 # Synapse 7: (0, 0, 2000)
[223:192] = 0x00001_3E8 # Synapse 6: (0, 1, 1000)
...
[31:0]    = 0x00000_7D0 # Synapse 0: (0, 0, 2000) ← our synapse!
```

**Step 3: PCIe DMA transfer**
```
Host PC → PCIe bus → FPGA PCIe endpoint → pcie2fifos → Input FIFO
```

**Step 4: Command Interpreter (Verilog)**
```verilog
// RX state machine receives command
case (rxFIFO_dout[511:504])
    CMD_HBM_RW:  // Detected HBM read/write
        ci2hbm_wren = 1'b1;  // Forward to HBM processor
endcase
```

**Step 5: HBM Processor (Verilog) - TX Path**
```verilog
// State: TX_STATE_WRITE_HBM_ADDR
hbm_awaddr = {5'd0, 23'h8000, 5'd0} = 33'h0000_0100_0000  // Write address
hbm_awvalid = 1'b1

// State: TX_STATE_WRITE_HBM_DATA
hbm_wdata = 256'h...07D0  // All 8 synapses
hbm_wvalid = 1'b1

// State: TX_STATE_WRITE_HBM_RESP
hbm_bready = 1'b1  // Waiting for confirmation
// When hbm_bvalid asserts → Write complete!
```

**Step 6: HBM**
```
Row 0x8000 in HBM is updated with new data
Synapse 0: [31:0] = 0x00000_7D0 = (OpCode=0, Address=0, Weight=2000)
```

**Result:** The synaptic weight from a0 to h0 is now 2000 instead of 1000!

---

## Complete Read Flow

### Reading Synapse: a0 → h0

**Step 1: Python (host PC)**
```python
synapse = network.read_synapse(preIndex=0, postIndex=(0,0), axonFlag=True)
```

**Currently, this only reads from local Python data structure!**

To read from actual hardware, you would need to:

1. Call `Read_synapse_row(r=0)` from `fpga_controller.py`
2. This sends a read command to the FPGA
3. Hardware returns the row data
4. Decode the response and extract the desired synapse

The `Read_synapse_row()` function follows a similar pattern to `write_synapse_row()`:

```python
def Read_synapse_row(r, simDump=False, coreID=0):
    # Build read command
    rowAddress = '0' + np.binary_repr(r + SYN_BASE_ADDR, 23)
    # Leading '0' = read operation

    # Send command via DMA
    dmadump.dma_dump_write(cmd, ...)

    # Read response
    response = dmadump.dma_dump_read(...)

    # Decode response
    decoded = HBM_decode(response)
    return decoded  # Returns 8 synapses from the row
```

---

## Use Cases

### 1. RSTDP Learning

```python
# During learning, update weights based on spike timing
def apply_rstdp_update(network, pre_idx, post_idx, delta_weight):
    # Read current weight
    synapse = network.read_synapse(pre_idx, post_idx, axonFlag=True)
    current_weight = synapse[2]

    # Calculate new weight
    new_weight = current_weight + delta_weight
    new_weight = np.clip(new_weight, -32768, 32767)  # 16-bit signed range

    # Write updated weight
    network.write_synapse(pre_idx, post_idx, new_weight, axonFlag=True)
```

### 2. Network Debugging

```python
# Verify all synapses were programmed correctly
def check_network_weights(network):
    for axon_idx in range(network.num_axons):
        for post_idx in range(network.num_neurons):
            synapse = network.read_synapse(axon_idx, (post_idx, 0), axonFlag=True)
            print(f"Axon {axon_idx} → Neuron {post_idx}: weight = {synapse[2]}")
```

### 3. Dynamic Reconfiguration

```python
# Disable a connection by setting weight to 0
network.write_synapse(pre_idx=5, postIndex=(10, 0), weight=0, axonFlag=True)

# Later, re-enable it
network.write_synapse(pre_idx=5, postIndex=(10, 0), weight=1000, axonFlag=True)
```

---

## Performance Considerations

### Read Performance
- **Software read:** ~1 μs (local memory access)
- **Hardware read:** ~10-100 μs (PCIe round-trip + HBM access)

### Write Performance
- **Row-based writes:** Each `write_synapse()` writes an entire 8-synapse row (~10-100 μs)
- **Bulk updates:** For many weight changes, batch them to minimize PCIe overhead

### RSTDP Implications
For online RSTDP learning with frequent weight updates:
- Consider **buffering weight changes** and applying them in batches
- Use **eligibility traces** to reduce update frequency
- Trade off **learning rate** vs. **update overhead**

---

## Summary

The synapse read/write system provides a powerful mechanism for dynamic network modification:

**Python API:**
- `read_synapse()`: Query weights from local data structure
- `write_synapse()`: Update weights and synchronize with hardware
- `write_synapse_row()`: Low-level PCIe/DMA communication

**Hardware Path:**
```
Host PC → PCIe → command_interpreter.v → hbm_processor.v → HBM
        ←       ←                      ←                 ←
```

**Key Insights:**
1. Synapses are organized in **rows** (8 per row) in HBM Region 3
2. Pointers (Regions 1 & 2) tell us **which rows** belong to each source
3. Writes update **entire rows**, not individual synapses
4. Hardware uses **AXI4 protocol** for HBM access
5. The system supports **real-time weight modification** for learning

This infrastructure makes RSTDP and other online learning algorithms possible on the neuromorphic FPGA hardware!
