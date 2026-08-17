# 2.3 The Code Behind Execution

Now that we understand the hardware flow from Chapter 2.1, let's trace through the actual software and Verilog code that makes it happen.

### Setup: Sending Inputs to BRAM

#### Python Code: network.step()

File: `hs_api/api.py` (lines 470-552)

```python
def step(self, inputs, target="simpleSim", membranePotential=False):
    """Runs a step of the simulation."""

    # Convert symbolic input names to numerical indices
    # inputs = ['a0', 'a1', 'a2']
    formated_inputs = [
        self.connectome.get_neuron_by_key(symbol).get_coreTypeIdx()
        for symbol in inputs
    ]
    # formated_inputs = [0, 1, 2] (axon indices)

    if self.target == "CRI":
        if self.simDump:
            return self.CRI.run_step(formated_inputs)
        else:
            spikeResult = self.CRI.run_step(formated_inputs, membranePotential)
            # ... (parse and return spikes)
```

This calls `hs_bridge.network.run_step()`, which calls `fpga_controller.input_user()`.

#### Python Code: fpga_controller.input_user()

File: `hs_bridge/FPGA_Execution/fpga_controller.py` (lines 899-1031)

```python
def input_user(inputs, numAxons, simDump=False, coreID=0, reserve=True, cont_flag=False):
    """
    Generates the input command for a given timestep

    Parameters:
    - inputs: list of axon indices (e.g., [0, 1, 2])
    - numAxons: total number of axons (e.g., 5)
    """

    currInput = inputs  # [0, 1, 2]
    currInput.sort()    # Ensure sorted order

    coreBits = np.binary_repr(coreID, 5) + 3*'0'  # 5 bits for coreID
    coreByte = int(coreBits, 2)

    commandList = [0]*62 + [coreByte] + [1]  # Init packet: opcode 0x01

    # For networks with <= 256 axons, we use a single 256-bit chunk
    for count in range(math.ceil(numAxons/256)):
        # Create 256-bit one-hot encoding
        one_hot_bin = ["0"] * 256
        inputSegment = [i for i in currInput
                        if (256*count) <= i and i < (256*count+256)]

        for axon in inputSegment:
            one_hot_bin[axon%256] = "1"

        # Convert to bytes (8 bits per byte, little-endian)
        while one_hot_bin:
            curr_byte = one_hot_bin[:8][::-1]  # Reverse for endianness
            curr_byte = "".join(curr_byte)
            commandList = commandList + [int(curr_byte, 2)]
            one_hot_bin = one_hot_bin[8:]

        # Add tail: padding + coreID + opcode
        tail = 30*[0]
        commandList = commandList + tail + [coreByte, 0]  # Opcode 0x00

    # Example for inputs [0, 1, 2]:
    # one_hot_bin = ["1", "1", "1", "0", "0", "0", ..., "0"]
    # curr_byte = one_hot_bin[:8] = ["1", "1", "1", "0", "0", "0", "0", "0"]
    # curr_byte reversed = ["0", "0", "0", "0", "0", "1", "1", "1"]
    # curr_byte string = "00000111"
    # int("00000111", 2) = 7 = 0x07 ✓

    # So commandList will contain:
    # [0]*62 + [coreID] + [1]  ← Init packet (opcode 0x01)
    # [0x07, 0x00, 0x00, ..., 0x00]  ← Data packet (256 bits = 32 bytes)
    # [0]*30 + [coreID] + [0]  ← Tail (opcode 0x00)

    command = np.array(commandList, dtype=np.uint64)

    # Send via DMA
    exitCode = dmadump.dma_dump_write(command, len(command),
                                       1, 0, 0, 0, dmadump.DmaMethodNormal)
```

**What gets sent:**
1. **Init packet (64 bytes):** Opcode 0x01, tells FPGA "input data incoming"
2. **Data packet (64 bytes):** Opcode 0x00, contains 256-bit one-hot mask
   - Byte 0 = 0x07 (bits 0,1,2 set for axons a0,a1,a2)
   - Bytes 1-31 = 0x00
   - Tail: padding + coreID + opcode

#### Verilog: command_interpreter Receives Input

File: `hardware_code/gopa/CRI_proj/command_interpreter.v` (lines 200-280, conceptual)

```verilog
// State machine receives packets from Input FIFO (rxFIFO from pcie2fifos)
always @(posedge aclk) begin
    case (rx_state)
        RX_IDLE: begin
            if (!rxFIFO_empty) begin
                rxFIFO_rd_en <= 1'b1;
                rx_state <= RX_READ_PACKET;
            end
        end

        RX_READ_PACKET: begin
            packet <= rxFIFO_dout[511:0];
            opcode <= rxFIFO_dout[511:504];  // Top 8 bits
            coreID <= rxFIFO_dout[503:496];  // Next 8 bits
            payload <= rxFIFO_dout[495:0];   // Remaining 496 bits
            rx_state <= RX_PROCESS_OPCODE;
        end

        RX_PROCESS_OPCODE: begin
            case (opcode)
                8'h01: begin  // CMD_EEP_W_INIT - Init input packet
                    // Prepare to receive data packet
                    rx_state <= RX_IDLE;  // Wait for next packet
                end

                8'h00: begin  // CMD_EEP_W - Input data packet
                    // Extract 256-bit one-hot mask from payload
                    axonEvent_data <= payload[255:0];

                    // Calculate write address (for BRAM Future buffer)
                    axonEvent_addr <= bram_write_row;

                    // Signal external_events_processor to write to Future buffer
                    axonEvent_we <= 1'b1;

                    rx_state <= RX_IDLE;
                end
            endcase
        end
    endcase
end

// Signals to external_events_processor
// These write to the FUTURE buffer while execution reads PRESENT buffer
assign eep_wr_en = axonEvent_we;
assign eep_wr_addr = axonEvent_addr;
assign eep_wr_data = axonEvent_data;
```

#### Verilog: external_events_processor Double-Buffered BRAM

File: `hardware_code/gopa/CRI_proj/external_events_processor.v` (lines 100-200, conceptual)

```verilog
// Double-buffered BRAM for external events
// Present buffer: Read during execution
// Future buffer: Written by host for next timestep

reg bram_select;  // 0 = BRAM0 is Present, BRAM1 is Future
                  // 1 = BRAM1 is Present, BRAM0 is Future

// BRAM0: 8K entries × 16 bits (or 16K × 8 bits depending on variant)
RAMB36E2 #(
    .ADDR_WIDTH(13),
    .DATA_WIDTH(16)
) bram0 (
    .CLKA(aclk),
    .ADDRA(bram0_addr),
    .DINA(bram0_din),
    .DOUTA(bram0_dout),
    .WEA(bram0_we),
    .ENA(1'b1)
);

// BRAM1: Identical to BRAM0
RAMB36E2 #(
    .ADDR_WIDTH(13),
    .DATA_WIDTH(16)
) bram1 (
    .CLKA(aclk),
    .ADDRA(bram1_addr),
    .DINA(bram1_din),
    .DOUTA(bram1_dout),
    .WEA(bram1_we),
    .ENA(1'b1)
);

// Write logic: Always write to Future buffer
always @(*) begin
    if (bram_select == 0) begin
        // BRAM0 is Present, BRAM1 is Future
        bram1_addr = eep_wr_addr;
        bram1_din = eep_wr_data;
        bram1_we = eep_wr_en;
    end else begin
        // BRAM1 is Present, BRAM0 is Future
        bram0_addr = eep_wr_addr;
        bram0_din = eep_wr_data;
        bram0_we = eep_wr_en;
    end
end

// Read logic: Always read from Present buffer
always @(*) begin
    if (bram_select == 0) begin
        // BRAM0 is Present
        bram0_addr = exec_rd_addr;
        bram0_we = 1'b0;
        exec_rd_data = bram0_dout;
    end else begin
        // BRAM1 is Present
        bram1_addr = exec_rd_addr;
        bram1_we = 1'b0;
        exec_rd_data = bram1_dout;
    end
end

// Buffer role reassignment on execute command
always @(posedge aclk) begin
    if (execute_cmd) begin
        bram_select <= ~bram_select;  // Flip role assignment bit
        // Buffer that was Future is now designated Present
        // Buffer that was Present is now designated Future
        // NO data is copied - just a pointer swap
    end
end
```

**Physical operation:**
- Host writes to Future buffer: `bram_din = 256'h0000...0007` (bits 0,1,2 set)
- On execute command: `bram_select` flips (role reassignment, no data movement)
- Execution reads from the buffer now designated Present
- Host can immediately start writing next timestep to buffer now designated Future

---

### Triggering Execution: execute() Command

After writing inputs, we trigger execution:

```python
# In hs_bridge/network.py
def run_step(self, inputs):
    # Write inputs to Future buffer
    fpga_controller.input_user(inputs, numAxons, coreID)

    # Trigger execution (reassigns buffer roles, starts Phase 1)
    fpga_controller.execute(coreID)

    # Collect results
    spikes = fpga_controller.flush_spikes(coreID)
    return spikes
```

#### Python Code: fpga_controller.execute()

File: `hs_bridge/FPGA_Execution/fpga_controller.py` (lines 872-892)

```python
def execute(simDump=False, coreID=0):
    """Runs a single step of the network"""

    coreBits = np.binary_repr(coreID, 5) + 3*'0'
    command = np.array([0]*62 + [int(coreBits, 2), 6], dtype=np.uint64)
    # Opcode 0x06 = CMD_EXEC (execute timestep)

    exitCode = dmadump.dma_dump_write(command, len(command),
                                       1, 0, 0, 0, dmadump.DmaMethodNormal)
```

#### Verilog: command_interpreter Triggers Execution

File: `hardware_code/gopa/CRI_proj/command_interpreter.v` (lines 300-350, conceptual)

```verilog
always @(posedge aclk) begin
    case (opcode)
        8'h06: begin  // CMD_EXEC - Execute timestep
            // Set execution control signals
            exec_run <= 1'b1;          // Enable execution
            execRun_ctr <= execRun_ctr + 1;  // Increment timestep counter

            // Trigger external_events_processor Phase 1a
            exec_bram_phase1_start <= 1'b1;  // One-cycle pulse

            rx_state <= RX_IDLE;
        end
    endcase
end

// Phase 1a start signal to external_events_processor
assign eep_phase1_start = exec_bram_phase1_start;
```

---

### Phase 1a: External Event Processing (Pointer Collection)

The key insight: **Phase 1a ONLY collects HBM addresses (pointers) of synapse rows to read**. It does NOT read the synapse data itself.

#### Verilog: external_events_processor Phase 1a State Machine

File: `hardware_code/gopa/CRI_proj/external_events_processor.v` (simplified)

```verilog
// Phase 1a: Read BRAM, generate spike mask, trigger HBM pointer reads

reg [3:0] state;
localparam EEP_IDLE = 0;
localparam EEP_FILL_PIPE = 1;
localparam EEP_READ_BRAM = 2;
localparam EEP_GEN_MASK = 3;
localparam EEP_PHASE1_DONE = 4;

reg [12:0] bram_row_counter;
reg [15:0] current_mask;
reg [15:0] exec_bram_spiked_reg;

always @(posedge aclk) begin
    case (state)
        EEP_IDLE: begin
            if (eep_phase1_start) begin
                // Buffer role reassignment already happened
                bram_row_counter <= 13'b0;
                exec_bram_spiked_reg <= 16'b0;
                state <= EEP_FILL_PIPE;
            end
        end

        EEP_FILL_PIPE: begin
            // BRAM has 3-cycle read latency, fill pipeline
            exec_rd_addr <= bram_row_counter;
            pipeline_ctr <= 3'd0;
            state <= EEP_READ_BRAM;
        end

        EEP_READ_BRAM: begin
            // Wait for pipeline to fill
            if (pipeline_ctr == 3'd2) begin
                current_mask <= exec_rd_data[15:0];
                state <= EEP_GEN_MASK;
            end else begin
                pipeline_ctr <= pipeline_ctr + 1;
            end
        end

        EEP_GEN_MASK: begin
            // Generate 16-bit spike mask for neuron groups
            // Each bit represents one neuron group (0-15)
            // For our small network, only bit 0 will be set

            // Check if any axons in this row are active
            if (current_mask != 16'h0000) begin
                // Axons active in row 0 → neuron group 0
                exec_bram_spiked_reg[0] <= 1'b1;
            end

            // For larger networks, would scan more rows
            bram_row_counter <= bram_row_counter + 1;
            if (bram_row_counter == max_rows) begin
                state <= EEP_PHASE1_DONE;
            end else begin
                state <= EEP_READ_BRAM;
            end
        end

        EEP_PHASE1_DONE: begin
            // Output spike mask to pointer_fifo_controller
            exec_bram_spiked <= exec_bram_spiked_reg;
            exec_bram_phase1_done <= 1'b1;  // Signal completion
            state <= EEP_IDLE;
        end
    endcase
end
```

**What this outputs:**
- `exec_bram_spiked[15:0]` = 16-bit mask indicating which neuron groups have active axons
- For our network: `exec_bram_spiked = 16'h0001` (only group 0 active)
- This signal goes to `hbm_processor` and `pointer_fifo_controller`

**Critically: external_events_processor does NOT read HBM or synapse data!**

---

#### Verilog: hbm_processor Phase 1a - Read Axon Pointers

File: `hardware_code/gopa/CRI_proj/hbm_processor.v` (lines 400-500, conceptual)

```verilog
// Phase 1a: Read axon pointers from HBM Region 1
// Uses exec_bram_spiked to determine which axons to read

reg [2:0] phase1_state;
localparam P1_IDLE = 0;
localparam P1_READ_PTRS = 1;
localparam P1_WAIT_PTRS = 2;
localparam P1_SEND_TO_PFC = 3;

reg [3:0] group_index;
reg [22:0] ptr_addr;

always @(posedge aclk) begin
    case (phase1_state)
        P1_IDLE: begin
            if (exec_bram_phase1_done) begin
                group_index <= 4'b0;
                phase1_state <= P1_READ_PTRS;
            end
        end

        P1_READ_PTRS: begin
            // Check if this group has active axons
            if (exec_bram_spiked[group_index]) begin
                // Calculate HBM address for axon pointers
                // Region 1 base address + group offset
                ptr_addr <= {AXN_PTR_BASE, group_index, 5'b00000};

                // Issue AXI4 read to HBM
                hbm_arvalid <= 1'b1;
                hbm_araddr <= {5'd0, ptr_addr, 5'b00000};  // 33-bit address
                hbm_arlen <= 8'd0;   // 1 beat (256 bits = 16 pointers)
                hbm_arsize <= 3'd5;  // 32 bytes

                phase1_state <= P1_WAIT_PTRS;
            end else begin
                // No activity in this group, skip
                group_index <= group_index + 1;
                if (group_index == 4'd15)
                    phase1_state <= P1_IDLE;  // Done with all groups
            end
        end

        P1_WAIT_PTRS: begin
            // Wait for HBM read completion
            if (hbm_rvalid) begin
                // Got 256 bits = 8 × 32-bit pointers (16 pointers with upper bits)
                ptr_data <= hbm_rdata[255:0];
                phase1_state <= P1_SEND_TO_PFC;
            end
        end

        P1_SEND_TO_PFC: begin
            // Forward pointer data to pointer_fifo_controller
            hbm2pfc_data <= {256'b0, ptr_data};  // Pad to 512 bits
            hbm2pfc_valid <= 1'b1;

            // Continue with next group
            group_index <= group_index + 1;
            if (group_index == 4'd15)
                phase1_state <= P1_IDLE;  // Phase 1a complete
            else
                phase1_state <= P1_READ_PTRS;
        end
    endcase
end
```

**Example for our network (group 0 only):**
```
Input: exec_bram_spiked = 16'h0001 (group 0 active)

Cycle N: P1_READ_PTRS
  group_index = 0
  exec_bram_spiked[0] = 1 (active)
  hbm_araddr = {5'd0, 23'h000000, 5'd0} (Region 1, group 0)

Cycle N+100: P1_WAIT_PTRS (after HBM latency)
  hbm_rdata[255:0] contains 8 × 32-bit axon pointers:
    ptr[0] = {9'b000000001, 23'h8000} (axon 0: 1 row at 0x8000)
    ptr[1] = {9'b000000001, 23'h8001} (axon 1: 1 row at 0x8001)
    ptr[2] = {9'b000000001, 23'h8002} (axon 2: 1 row at 0x8002)
    ptr[3-7] = 32'h0 (unused)

Cycle N+101: P1_SEND_TO_PFC
  hbm2pfc_data = 512-bit packet with pointer data
  hbm2pfc_valid = 1
```

**Key point:** We now have ADDRESSES of where synapses are stored, but haven't read the synapses yet!

---

#### Verilog: pointer_fifo_controller Phase 1 - Demultiplex Pointers

File: `hardware_code/gopa/CRI_proj/pointer_fifo_controller.v` (lines 200-300, conceptual)

```verilog
// Phase 1: Demultiplex HBM pointer data to 16 FIFOs
// Uses exec_bram_spiked to determine which FIFOs to write

reg [255:0] ptr_row_data;
reg [31:0] pointer [0:7];  // 8 pointers per HBM row
reg [3:0] target_fifo;

always @(posedge aclk) begin
    if (hbm2pfc_valid) begin
        ptr_row_data <= hbm2pfc_data[255:0];

        // Extract 8 pointers from 256-bit data
        for (i = 0; i < 8; i = i + 1) begin
            pointer[i] = ptr_row_data[(i*32)+:32];

            if (pointer[i] != 32'h0) begin
                // Valid pointer
                // Format: [31:23]=length, [22:0]=base_address

                // Determine which FIFO to write to
                // Based on which group had activity (exec_bram_spiked)
                target_fifo = find_first_set_bit(exec_bram_spiked);

                // Write 32-bit pointer to FIFO
                ptrFIFO_wr_en[target_fifo] <= 1'b1;
                ptrFIFO_din[target_fifo] <= pointer[i];
            end
        end
    end
end

// 16 pointer FIFOs (one per neuron group)
genvar g;
generate
    for (g = 0; g < 16; g = g + 1) begin : gen_ptr_fifos
        FIFO_SYNC #(
            .DATA_WIDTH(32),  // 32-bit HBM address
            .DEPTH(512)
        ) ptr_fifo (
            .WR_CLK(aclk),        // Write at 225 MHz
            .WR_EN(ptrFIFO_wr_en[g]),
            .DIN(ptrFIFO_din[g]),
            .FULL(ptrFIFO_full[g]),

            .RD_CLK(aclk),        // Read at 225 MHz (same domain)
            .RD_EN(ptrFIFO_rd_en[g]),
            .DOUT(ptrFIFO_dout[g]),
            .EMPTY(ptrFIFO_empty[g])
        );
    end
endgenerate
```

**Example for our network:**
```
After Phase 1a completes, ptrFIFO[0] contains:
  Entry 0: 32'h0080_8000 = {9'd1, 23'h8000} (axon 0 pointer)
  Entry 1: 32'h0080_8001 = {9'd1, 23'h8001} (axon 1 pointer)
  Entry 2: 32'h0080_8002 = {9'd1, 23'h8002} (axon 2 pointer)

ptrFIFO[1-15]: Empty (no activity in other groups)
```

**Critical understanding:** These FIFOs contain **HBM addresses** (pointers), NOT synapse data!

---

### Phase 1b: Internal Event Processing (If Neurons Spiked)

If neurons spiked during Phase 2, they trigger Phase 1b to collect their output pointers.

#### Verilog: internal_events_processor Spike Detection

File: `hardware_code/gopa/CRI_proj/internal_events_processor.v` (lines 600-700, conceptual)

```verilog
// During Phase 2 neuron updates, track which neurons spike
// Output exec_uram_spiked mask for Phase 1b

reg [15:0] uram_spiked_mask;

always @(posedge aclk450) begin
    if (neuron_spike_detected) begin
        // Neuron in group X spiked
        neuron_group = spiked_neuron_addr[16:13];  // Top 4 bits
        uram_spiked_mask[neuron_group] <= 1'b1;
    end
end

// At end of Phase 2, output spike mask
assign exec_uram_spiked = uram_spiked_mask;
assign exec_uram_phase1_done = phase2_complete;
```

**Example after hidden neurons spike:**
```
Neurons h0-h4 all spiked (addresses 0-4, all in group 0)
exec_uram_spiked = 16'h0001 (group 0 has spiking neurons)
```

#### Verilog: hbm_processor Phase 1b - Read Neuron Pointers

This is nearly identical to Phase 1a, but reads from HBM Region 2 instead of Region 1:

```verilog
// Phase 1b: Read neuron pointers from HBM Region 2

always @(posedge aclk) begin
    case (phase1b_state)
        P1B_IDLE: begin
            if (exec_uram_phase1_done) begin
                group_index <= 4'b0;
                phase1b_state <= P1B_READ_PTRS;
            end
        end

        P1B_READ_PTRS: begin
            if (exec_uram_spiked[group_index]) begin
                // Read neuron pointers from Region 2
                ptr_addr <= {NRN_PTR_BASE, group_index, 5'b00000};

                hbm_arvalid <= 1'b1;
                hbm_araddr <= {5'd0, ptr_addr, 5'b00000};

                phase1b_state <= P1B_WAIT_PTRS;
            end
            // ... (same pattern as Phase 1a)
        end
    endcase
end
```

**Result:** More pointers added to the same ptrFIFOs, now containing both axon and neuron output pointers.

---

### Phase 2: Synaptic Processing and Neuron Updates

Now we read the actual synapses using the pointers we collected!

#### Verilog: pointer_fifo_controller Round-Robin Read

File: `hardware_code/gopa/CRI_proj/pointer_fifo_controller.v` (lines 400-500, conceptual)

```verilog
// Phase 2: Multiplex (round-robin read) from 16 FIFOs to HBM processor

reg [3:0] rr_counter;  // Round-robin counter (0-15)

always @(posedge aclk) begin
    if (phase2_active) begin
        // Check current FIFO
        if (!ptrFIFO_empty[rr_counter]) begin
            // Read pointer from this FIFO
            ptrFIFO_rd_en[rr_counter] <= 1'b1;
            current_ptr <= ptrFIFO_dout[rr_counter];

            // Send to HBM processor
            pfc2hbm_ptr <= current_ptr;
            pfc2hbm_valid <= 1'b1;
        end

        // Move to next FIFO
        rr_counter <= (rr_counter == 4'd15) ? 4'd0 : rr_counter + 1;
    end
end
```

**Example execution:**
```
Cycle 0: Check ptrFIFO[0] → not empty
         Pop 32'h0080_8000 (axon 0 pointer)
         Send to hbm_processor

Cycle 5: Check ptrFIFO[1] → empty, skip
Cycle 6: Check ptrFIFO[2] → empty, skip
...
Cycle 21: Check ptrFIFO[0] again → not empty
          Pop 32'h0080_8001 (axon 1 pointer)
```

---

#### Verilog: hbm_processor Phase 2 - Read Synapses

File: `hardware_code/gopa/CRI_proj/hbm_processor.v` (lines 600-800, conceptual)

```verilog
// Phase 2: Use pointers to read synapse data from HBM Region 3

reg [2:0] phase2_state;
localparam P2_IDLE = 0;
localparam P2_READ_SYN = 1;
localparam P2_WAIT_SYN = 2;
localparam P2_PARSE_SYN = 3;
localparam P2_FORWARD = 4;

reg [31:0] current_pointer;
reg [22:0] syn_base_addr;
reg [8:0] syn_length;
reg [255:0] synapse_data;

always @(posedge aclk) begin
    case (phase2_state)
        P2_IDLE: begin
            if (pfc2hbm_valid) begin
                current_pointer <= pfc2hbm_ptr;

                // Extract pointer fields
                syn_length <= current_pointer[31:23];
                syn_base_addr <= current_pointer[22:0];

                phase2_state <= P2_READ_SYN;
            end
        end

        P2_READ_SYN: begin
            // Read synapse row from HBM Region 3
            // Address = SYN_BASE + syn_base_addr
            hbm_arvalid <= 1'b1;
            hbm_araddr <= {5'd0, SYN_BASE[22:0] + syn_base_addr, 5'd0};
            hbm_arlen <= 8'd0;  // 1 beat
            hbm_arsize <= 3'd5; // 32 bytes = 256 bits

            phase2_state <= P2_WAIT_SYN;
        end

        P2_WAIT_SYN: begin
            if (hbm_rvalid) begin
                synapse_data <= hbm_rdata[255:0];
                phase2_state <= P2_PARSE_SYN;
            end
        end

        P2_PARSE_SYN: begin
            // Parse 8 × 32-bit synapses from 256-bit data
            for (i = 0; i < 8; i = i + 1) begin
                syn[i] = synapse_data[(i*32)+:32];
                opcode[i] = syn[i][31:29];
                target[i] = syn[i][28:16];  // 13-bit address
                weight[i] = syn[i][15:0];   // 16-bit weight

                // Check synapse type
                if (syn[i][31]) begin
                    // OpCode bit 31 = 1: This is a spike (send to spike FIFO)
                    spkFIFO_wr_en[i % 8] <= 1'b1;
                    spkFIFO_din[i % 8] <= {4'b0, target[i]};
                end
            end

            phase2_state <= P2_FORWARD;
        end

        P2_FORWARD: begin
            // Forward synapse data to internal_events_processor
            exec_hbm_rdata <= {256'b0, synapse_data};
            exec_hbm_valid <= 1'b1;

            // Back to check for more pointers
            phase2_state <= P2_IDLE;
        end
    endcase
end
```

**Example for axon 0's synapses:**
```
Input: pointer = 32'h0080_8000
       syn_length = 9'd1
       syn_base_addr = 23'h8000

Read HBM[0x8000]:
  synapse_data = 256-bit containing:
    syn[0] = {3'b000, 13'd0, 16'd1000}  (target=h0, weight=1000)
    syn[1] = {3'b000, 13'd1, 16'd1000}  (target=h1, weight=1000)
    syn[2] = {3'b000, 13'd2, 16'd1000}  (target=h2, weight=1000)
    syn[3] = {3'b000, 13'd3, 16'd1000}  (target=h3, weight=1000)
    syn[4] = {3'b000, 13'd4, 16'd1000}  (target=h4, weight=1000)
    syn[5-7] = 32'h0 (unused)

Forward to internal_events_processor:
  exec_hbm_rdata[255:0] = synapse_data
  exec_hbm_valid = 1
```

---

#### Verilog: internal_events_processor Phase 2 - Neuron Updates

File: `hardware_code/gopa/CRI_proj/internal_events_processor.v` (lines 800-1000)

```verilog
// Phase 2: Process synapses from exec_hbm_rdata, update neurons
// Runs @ 450 MHz (aclk450) for higher throughput

reg [2:0] state;
localparam IEP_IDLE = 0;
localparam IEP_PARSE_SYN = 1;
localparam IEP_READ_URAM = 2;
localparam IEP_ACCUMULATE = 3;
localparam IEP_APPLY_MODEL = 4;
localparam IEP_WRITE_URAM = 5;
localparam IEP_CHECK_SPIKE = 6;

reg [3:0] syn_index;  // Which synapse in packet (0-7 or 0-15)
reg [31:0] current_syn;
reg [16:0] target_neuron;
reg [15:0] weight;
reg [35:0] V_old, V_new, V_final;
reg spike;

// Neuron model parameters (programmed during init)
reg [35:0] threshold = 36'd2000;
reg [5:0] leak = 6'd63;  // No leak for IF neurons
reg [1:0] neuron_model = 2'b00;  // 00=IF, 10=LIF

// Pipeline hazard tracking (addresses currently in flight)
reg [16:0] pipeline_addr [0:4];

always @(posedge aclk450) begin
    case (state)
        IEP_IDLE: begin
            if (exec_hbm_valid) begin
                syn_data_reg <= exec_hbm_rdata[255:0];
                syn_index <= 4'd0;
                state <= IEP_PARSE_SYN;
            end
        end

        IEP_PARSE_SYN: begin
            // Extract synapse from data
            current_syn = syn_data_reg[(syn_index*32)+:32];

            if (current_syn == 32'h0) begin
                // Unused synapse slot, skip
                syn_index <= syn_index + 1;
                if (syn_index == 4'd7)
                    state <= IEP_IDLE;  // Done with this packet
            end else begin
                // Valid synapse
                target_neuron = current_syn[28:16];  // 13-bit address
                weight = current_syn[15:0];

                state <= IEP_READ_URAM;
            end
        end

        IEP_READ_URAM: begin
            // Check for read-after-write hazard
            hazard = (target_neuron == pipeline_addr[1]) ||
                     (target_neuron == pipeline_addr[2]) ||
                     (target_neuron == pipeline_addr[3]);

            if (hazard) begin
                // Stall: same neuron still in pipeline
                state <= IEP_READ_URAM;
            end else begin
                // Safe to read
                // URAM stores 2 neurons per 72-bit word
                uram_addr <= target_neuron[16:1];  // Word address
                uram_rd_en <= 1'b1;

                pipeline_addr[0] <= target_neuron;
                state <= IEP_ACCUMULATE;
            end
        end

        IEP_ACCUMULATE: begin
            // URAM read latency: 1 cycle @ 450 MHz
            uram_word <= uram_dout[71:0];

            // Select upper or lower neuron in word
            if (target_neuron[0] == 1'b0)
                V_old = uram_word[35:0];    // Lower neuron
            else
                V_old = uram_word[71:36];   // Upper neuron

            // Check for bypass (recent write to same neuron)
            if (target_neuron == pipeline_addr[1])
                V_old = bypass_data[1];
            else if (target_neuron == pipeline_addr[2])
                V_old = bypass_data[2];
            else if (target_neuron == pipeline_addr[3])
                V_old = bypass_data[3];

            // Accumulate synaptic input
            V_new = V_old + $signed(weight);

            pipeline_addr[1] <= pipeline_addr[0];
            bypass_data[1] <= V_new;
            state <= IEP_APPLY_MODEL;
        end

        IEP_APPLY_MODEL: begin
            // Apply neuron model (leak, etc.)
            if (neuron_model == 2'b10) begin  // LIF
                V_new = V_new - (V_new >>> leak);
            end
            // IF model: no leak

            // Threshold check
            spike = (V_new >= $signed(threshold));

            // Reset if spike
            if (spike)
                V_final = 36'sd0;
            else
                V_final = V_new;

            pipeline_addr[2] <= pipeline_addr[1];
            bypass_data[2] <= V_final;
            state <= IEP_WRITE_URAM;
        end

        IEP_WRITE_URAM: begin
            // Reconstruct 72-bit word with updated neuron
            if (target_neuron[0] == 1'b0)
                uram_din = {uram_word[71:36], V_final};  // Update lower
            else
                uram_din = {V_final, uram_word[35:0]};   // Update upper

            // Write back to URAM
            uram_we <= 1'b1;
            uram_addr <= target_neuron[16:1];

            pipeline_addr[3] <= pipeline_addr[2];
            bypass_data[3] <= V_final;
            state <= IEP_CHECK_SPIKE;
        end

        IEP_CHECK_SPIKE: begin
            if (spike) begin
                // Record spike for Phase 1b
                neuron_group = target_neuron[16:13];
                uram_spiked_mask[neuron_group] <= 1'b1;

                // Send to spike_fifo_controller
                spike_out_valid <= 1'b1;
                spike_out_addr <= target_neuron;
            end

            // Move to next synapse in packet
            syn_index <= syn_index + 1;
            if (syn_index == 4'd7)
                state <= IEP_IDLE;  // Done with packet
            else
                state <= IEP_PARSE_SYN;  // Next synapse
        end
    endcase
end
```

**Example trace for neuron h0 receiving 3 inputs:**

```
═══════════════════════════════════════════
Input 1: From axon a0, weight=1000
═══════════════════════════════════════════

Cycle 0: IEP_IDLE
  exec_hbm_rdata contains a0's synapses

Cycle 1: IEP_PARSE_SYN
  syn_index = 0
  current_syn = {3'b000, 13'd0, 16'd1000}
  target_neuron = 17'd0 (h0)
  weight = 16'd1000

Cycle 2: IEP_READ_URAM
  uram_addr = 0 >> 1 = 0
  No hazard (pipeline empty)

Cycle 3: IEP_ACCUMULATE
  uram_word = {neuron_1_data, neuron_0_data}
  V_old = uram_word[35:0] = 36'd0 (zero)
  V_new = 0 + 1000 = 1000

Cycle 4: IEP_APPLY_MODEL
  neuron_model = IF, no leak
  spike = (1000 >= 2000) = 0
  V_final = 1000

Cycle 5: IEP_WRITE_URAM
  uram_din = {upper_data, 36'd1000}
  uram_we = 1

Cycle 6: IEP_CHECK_SPIKE
  spike = 0, no output
  Continue to next synapse...

═══════════════════════════════════════════
Input 2: From axon a1, weight=1000
═══════════════════════════════════════════

Cycle 100: IEP_PARSE_SYN
  target_neuron = 17'd0 (h0 again!)
  weight = 16'd1000

Cycle 101: IEP_READ_URAM
  uram_addr = 0

Cycle 102: IEP_ACCUMULATE
  uram_word[35:0] = 36'd1000 (from previous!)
  V_old = 1000
  V_new = 1000 + 1000 = 2000

Cycle 103: IEP_APPLY_MODEL
  spike = (2000 >= 2000) = 1  ← SPIKE!
  V_final = 0 (reset)

Cycle 104: IEP_WRITE_URAM
  uram_din = {upper_data, 36'd0}
  uram_we = 1

Cycle 105: IEP_CHECK_SPIKE
  spike = 1
  uram_spiked_mask[0] <= 1 (group 0 spiked)
  spike_out_addr = 17'd0 (neuron h0)

═══════════════════════════════════════════
Input 3: From axon a2, weight=1000
═══════════════════════════════════════════

Cycle 200: IEP_ACCUMULATE
  V_old = 0 (was just reset!)
  V_new = 0 + 1000 = 1000

Cycle 201: IEP_APPLY_MODEL
  spike = (1000 >= 2000) = 0
  V_final = 1000

Cycle 202: IEP_WRITE_URAM
  V = 1000

Final state of h0: V = 1000, spiked once
```

---

### Recurrent Processing: Hidden → Output Neurons

When hidden neurons spike, they automatically trigger Phase 1b → Phase 2 again:

1. **internal_events_processor** outputs `exec_uram_spiked = 16'h0001` (group 0 spiked)
2. **hbm_processor** Phase 1b: Reads neuron pointers from HBM Region 2
3. **pointer_fifo_controller**: Adds neuron output pointers to FIFOs
4. **hbm_processor** Phase 2: Reads synapses from HBM Region 3
5. **internal_events_processor**: Updates output neurons o0-o4

**For output neurons:**
```
Each output receives: 5 hidden neurons × 1000 weight = 5000 total
o0: V = 0 + 5000 = 5000 >= 2000 → SPIKE (at V=2000, reset, then +3000 more)
Final: V = 3000

Output neuron synapses may have OpCode=100 (0b100) which marks them
as "output spikes" to send to host.
```

---

### Reading Results: flush_spikes()

#### Python Code: fpga_controller.flush_spikes()

File: `hs_bridge/FPGA_Execution/fpga_controller.py` (lines 273-343)

```python
def flush_spikes(coreID=0):
    """Reads spike packets from FPGA via PCIe"""

    packetNum = 1
    spikeOutput = []
    n = 0

    time.sleep(800/1000000.0)  # Wait 800 µs for processing

    while True:
        exitCode, batchRead = dmadump.dma_dump_read(
            1, 0, 0, 0, dmadump.DmaMethodNormal, 64*packetNum
        )

        splitRead = np.array_split(batchRead, packetNum)
        splitRead.reverse()
        flushed = False

        for currentRead in splitRead:
            # Check packet type by tag (bytes 62-63)
            if (currentRead[62] == 255 and currentRead[63] == 255):
                # FIFO Empty packet (0xFFFF tag)
                n += 1
                if n == 50:
                    flushed = True
                    break
            elif (currentRead[62] == 238 and currentRead[63] == 238):
                # Spike packet (0xEEEE tag)
                executionRun_counter, spikeList = read_spikes(currentRead)
                spikeOutput = spikeOutput + spikeList
                n = 0
            elif (currentRead[62] == 205 and currentRead[63] == 171):
                # Latency packet (0xCDAB tag) - end of execution
                executionRun_counter, spikeList = read_spikes(currentRead)
                spikeOutput = spikeOutput + spikeList
                flushed = True
                break
            else:
                logging.error("Non-spike packet encountered")

        if flushed:
            break

    return (spikeOutput, latency, hbmAcc)
```

#### Verilog: command_interpreter Batches Spikes

File: `hardware_code/gopa/CRI_proj/command_interpreter.v` (lines 800-900, conceptual)

```verilog
// TX state machine: Collect spikes and send to host

reg [3:0] tx_state;
localparam TX_IDLE = 0;
localparam TX_COLLECT_SPIKES = 1;
localparam TX_BATCH_PACKET = 2;
localparam TX_SEND = 3;

reg [511:0] spike_packet;
reg [3:0] spike_count;
reg [16:0] spike_addr [0:13];  // Up to 14 spikes per packet

always @(posedge aclk) begin
    case (tx_state)
        TX_IDLE: begin
            if (!spk2ciFIFO_empty) begin
                spike_count <= 4'd0;
                tx_state <= TX_COLLECT_SPIKES;
            end
        end

        TX_COLLECT_SPIKES: begin
            if (!spk2ciFIFO_empty && spike_count < 14) begin
                // Read spike from aggregated spike FIFO
                spk2ciFIFO_rd_en <= 1'b1;
                spike_addr[spike_count] <= spk2ciFIFO_dout;
                spike_count <= spike_count + 1;
            end else begin
                // Either FIFO empty or batch full
                tx_state <= TX_BATCH_PACKET;
            end
        end

        TX_BATCH_PACKET: begin
            // Build 512-bit spike packet
            // [511:496] = 0xEEEE (spike packet tag)
            // [495:32] = 14 × 32-bit spike entries
            // [31:0] = timestep counter

            spike_packet[511:496] <= 16'hEEEE;
            spike_packet[31:0] <= execRun_ctr;

            for (i = 0; i < 14; i = i + 1) begin
                if (i < spike_count) begin
                    // Valid spike
                    spike_packet[(i*32)+32 +: 32] <=
                        {7'b0, 1'b1, 7'b0, spike_addr[i]};
                        // [31:25]=0, [24]=valid, [23:17]=0, [16:0]=address
                end else begin
                    // Unused slot
                    spike_packet[(i*32)+32 +: 32] <= 32'h0;
                end
            end

            tx_state <= TX_SEND;
        end

        TX_SEND: begin
            // Write to txFIFO (goes to pcie2fifos → host)
            txFIFO_wr_en <= 1'b1;
            txFIFO_din <= spike_packet;

            tx_state <= TX_IDLE;
        end
    endcase
end
```

**Example spike packet for our network:**

```
512-bit packet:
  Bits [511:496] = 16'hEEEE (spike packet identifier)
  Bits [495:480] = Spike 0: {7'b0, 1'b1, 7'b0, 17'd5} (o0, valid)
  Bits [479:464] = Spike 1: {7'b0, 1'b1, 7'b0, 17'd6} (o1, valid)
  Bits [463:448] = Spike 2: {7'b0, 1'b1, 7'b0, 17'd7} (o2, valid)
  Bits [447:432] = Spike 3: {7'b0, 1'b1, 7'b0, 17'd8} (o3, valid)
  Bits [431:416] = Spike 4: {7'b0, 1'b1, 7'b0, 17'd9} (o4, valid)
  Bits [415:32]  = Spikes 5-13: {32'h0} (unused)
  Bits [31:0]    = 32'd0 (timestep counter)

Host receives and parses:
  spikeList = [(0, 5), (0, 6), (0, 7), (0, 8), (0, 9)]
  Converts to: ['o0', 'o1', 'o2', 'o3', 'o4']
```

---

## Conclusion: The Complete Code Journey

We've now traced the complete code path from Python to Verilog and back:

**Setup:**
1. `network.step(['a0', 'a1', 'a2'])` → `input_user()` creates 256-bit mask
2. DMA writes to FPGA → `command_interpreter` receives opcode 0x00
3. Writes to BRAM **Future buffer** (double-buffered)

**Execute:**
4. `execute()` sends opcode 0x06 → `command_interpreter` reassigns buffer roles
5. **Phase 1a:** `external_events_processor` reads Present BRAM → generates `exec_bram_spiked`
6. **Phase 1a:** `hbm_processor` reads axon pointers from HBM Region 1
7. **Phase 1a:** `pointer_fifo_controller` demuxes pointers to 16 FIFOs

**Process:**
8. **Phase 2:** `pointer_fifo_controller` round-robin reads from FIFOs
9. **Phase 2:** `hbm_processor` pops pointers → reads synapses from HBM Region 3
10. **Phase 2:** `internal_events_processor` updates URAM @ 450 MHz
    - h0: V=0 → 1000 → 2000 (spike!) → 0 → 1000
    - Generates `exec_uram_spiked` for neurons that spiked

**Recurrent:**
11. **Phase 1b:** `hbm_processor` reads neuron pointers from HBM Region 2
12. **Phase 1b:** `pointer_fifo_controller` adds neuron pointers to FIFOs
13. **Phase 2:** Process hidden → output synapses (o0-o4 spike)

**Output:**
14. `spike_fifo_controller` aggregates spikes → `command_interpreter`
15. `command_interpreter` batches spikes into 512-bit packets
16. Sends via `txFIFO` → `pcie2fifos` → PCIe → Host
17. `flush_spikes()` reads packets, parses: `['o0', 'o1', 'o2', 'o3', 'o4']`

**Total time: ~2-5 microseconds** from input to output, with neurons updated at 450 MHz and HBM accessed via efficient burst reads.

The beauty of this architecture is the **two-phase separation**: Phase 1 collects *where* to read (pointers), Phase 2 does the actual reading (synapses) and processing (neuron updates). This allows for efficient memory coalescing and high parallelism across 16 neuron groups.
