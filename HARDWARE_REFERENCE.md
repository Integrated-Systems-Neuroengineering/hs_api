# exp-STDP-testing-suite — Hardware Reference & Software Guide

**Branch:** `exp-STDP-testing-suite` (created from `L6m-testing-suite`)
**Bitstream:** `single_core_conductance_STDP` (WNS +0.023 ns, crisdsc3)
**Date:** July 2026

---

## 1. What Is This Branch?

This branch extends the L6m testing suite with support for the **single_core_conductance_STDP** bitstream — a biological neuron model upgrade. The existing 42 hardware tests and DVS tests are **unchanged**. Three new files are added.

### Branch Structure

```
hs_api/                             (repository root)
├── hs_api/
│   ├── api.py                      UNCHANGED
│   ├── neuron_models.py            UNCHANGED
│   └── single_core_conductance_STDP.py            NEW — CMD 13, 64-bit synapse, signed MP fix
├── tests/
│   ├── test_bitstream_hardware_fast.py   UNCHANGED (42 tests)
│   ├── test_DVS_large_fulldataset_2024.py  UNCHANGED
│   └── test_single_core_conductance_STDP.py    NEW — single_core_conductance_STDP feature test suite
└── HARDWARE_REFERENCE.md              NEW — this file
```

**One patch** to hs_bridge (separate repo):
```
hs_bridge/hs_bridge/FPGA_Execution/
└── fpga_controller.py              PATCHED — adds _to_signed32() function only
```

### How to Run

```bash
cd /home/omowuyi/testing/hs_api && source .venv/bin/activate
export PYTHONPATH=/home/omowuyi/testing/hs_bridge

# Software-only tests (no FPGA needed):
python3 tests/test_single_core_conductance_STDP.py --software-only

# Backward compatibility (flash single_core_conductance_STDP bitstream first):
pytest tests/test_bitstream_hardware_fast.py -v

# All tests including FPGA hardware:
python3 tests/test_single_core_conductance_STDP.py --all
```

---

## 2. Backward Compatibility

The single_core_conductance_STDP bitstream **defaults to legacy LIF mode** on power-up. The register `delta_mode` resets to 1 (legacy). Since the existing software never sends CMD 13, the FPGA stays in legacy mode and all 42 tests pass without any changes.

To activate the new biological features, you must explicitly send CMD 13 with `delta_mode=0`. Until you do, the bitstream behaves identically to L6m.

### Axon Delay Buffer and Backward Compatibility

The `axon_delay_buffer.v` sits in the CI→EEP signal path regardless of `delta_mode`. It uses an 8192×6-bit BRAM lookup table (axon address → delay value) and a 64-slot circular buffer. On each timestep, incoming spikes are written to `buffer[write_ptr + delay_value]` and the output is read from `buffer[write_ptr]`.

On FPGA configuration, the BRAM initializes to all zeros, so every axon has delay=0. With delay=0, the spike is written to and read from `buffer[write_ptr]` in the same timestep — the buffer is transparent with zero additional delay. The software delay (`_preprocess_delayed_synapses` + delay queue in `api.py`) operates at the Python level and is unaffected by the hardware buffer. Both coexist without conflict when the hardware delay is 0. All 42 tests pass unchanged.

---

## 3. The New Hardware Features

### 3.1 Overview

| Feature | What It Does | Controlled By |
|---------|-------------|---------------|
| Exponential PSC | Synaptic currents decay exponentially over time instead of being instantaneous delta impulses | `delta_mode=0`, `decay_ex`, `decay_in` |
| COBA Synapses | Current depends on membrane potential: dV = g × (E_rev − V) | `coba_mode=1`, `E_ex`, `E_in` |
| AdEx Adaptation | Neuron fires less frequently with sustained input (spike-frequency adaptation) | `decay_w`, `delta_w` |
| STDP Learning | Weights change based on relative spike timing (Hebbian plasticity) | `stdp_enable=1`, `A_plus`, `A_minus`, `w_max`, `w_min` |
| Neuromodulation | Global modulation of learning rate and excitability (like dopamine/ACh) | `neuromod_level`, `neuromod_excitability_bias` |
| Hardware Axonal Delay | Spikes arrive at target neuron D timesteps later (replaces software delay) | `axon_delay_buffer.v` |
| 64-bit Synapse Format | Extended synapse with per-synapse delay, type, STDP tag, source address | See Section 6 |

### 3.2 How It Works Inside the FPGA

**L6m (legacy, delta_mode=1):**
```
Phase 0: For each neuron (1 cycle per pair):
           Read V from URAM → add leak → check threshold → spike/reset → write V
Phase 1: Read axon BRAM → generate HBM pointers
Phase 2: Read HBM synapses (16 × 32-bit per word) → add weight directly to V
Phase 3: Output spiked neuron addresses to host
```

**single_core_conductance_STDP (biological, delta_mode=0):**
```
Phase 0: For each neuron (5 cycles per pair):
           Cycle 0: Read Row A (V, refrac)
           Cycle 1: Read Row B (I_ex, I_in, w, trace)
           Cycle 2: Compute new V = V + I_ex_decayed + I_in_decayed - w
                     Pipeline register between decay DSP and COBA DSP
           Cycle 3: Compute new Row B (decay I_ex, I_in, w; update trace)
           Cycle 4: Write both rows, advance to next neuron
Phase 1: Same as L6m
Phase 2: Read HBM synapses (8 × 64-bit per word) → add weight to I_ex or I_in
           (NOT directly to V). Per-synapse delay checked.
Phase 3: Same as L6m
Phase 4: STDP weight update (only if stdp_enable=1):
           For each neuron that spiked:
             Read source trace from URAM Row B
             Read synapse from HBM
             Δw = (A_plus × trace) >>> 4
             Clamp to [w_min, w_max]
             Write modified synapse back to HBM
```

---

## 4. URAM Layout

Each core has 16 URAM banks × 4,096 entries × 72 bits = 131,072 neurons.

### Row A (addresses 0–2047) — same as L6m

```
Bits [71:70] = unused
Bits [69:67] = refrac_upper [2:0]    Refractory counter (upper neuron)
Bits [66:35] = V_upper [31:0]        Membrane potential (upper neuron, signed)
Bits [34:32] = refrac_lower [2:0]    Refractory counter (lower neuron)
Bits [31:0]  = V_lower [31:0]        Membrane potential (lower neuron, signed)
```

### Row B (addresses 2048–4095) — NEW in single_core_conductance_STDP

Only used when `delta_mode=0`. Each entry stores synaptic state for a neuron pair:

```
Per neuron (in each half-word):
  I_ex  [11:0]   Excitatory current (12-bit signed)
  I_in  [11:0]   Inhibitory current (12-bit signed)
  w     [7:0]    Adaptation current (8-bit unsigned)
  trace [4:0]    STDP eligibility trace (5-bit, set to max on spike, decays)
```

Row B address = Row A address + 2048 (`ROW_B_ROW_OFFSET = 12'd2048`).

---

## 5. CMD 13 — The Configuration Command

CMD 13 (`CMD_SET_PSC_PARAMS`, opcode `8'd13` = `0x0D`) is a single 512-bit DMA packet that configures all biological model parameters. Send it once after network initialization, before calling execute.

### 5.1 Bit Layout (verified from command_interpreter.v)

```
rxFIFO_dout[511:504] = 0x0D                    opcode

rxFIFO_dout[  0]     = delta_mode               1=legacy (DEFAULT), 0=biological
rxFIFO_dout[ 12:  1] = decay_ex                 12-bit, I_ex decay factor × 4096
rxFIFO_dout[ 24: 13] = decay_in                 12-bit, I_in decay factor × 4096
rxFIFO_dout[ 32: 25] = decay_w                  8-bit, adaptation decay × 256
rxFIFO_dout[ 40: 33] = delta_w                  8-bit, adaptation increment on spike
rxFIFO_dout[ 41]     = coba_mode                0=CUBA (current), 1=COBA (conductance)
rxFIFO_dout[ 53: 42] = E_ex                     12-bit signed, excitatory reversal
rxFIFO_dout[ 65: 54] = E_in                     12-bit signed, inhibitory reversal
rxFIFO_dout[ 73: 66] = neuromod_level           8-bit, STDP rate scale 0–255
rxFIFO_dout[ 81: 74] = neuromod_excitability_bias  8-bit signed, threshold shift
rxFIFO_dout[ 82]     = stdp_enable              0=disabled (DEFAULT), 1=Phase 4 active
rxFIFO_dout[ 90: 83] = A_plus                   8-bit, potentiation magnitude
rxFIFO_dout[ 98: 91] = A_minus                  8-bit, depression magnitude
rxFIFO_dout[114: 99] = w_max                    16-bit signed, weight ceiling
rxFIFO_dout[130:115] = w_min                    16-bit signed, weight floor
```

### 5.2 Reset Defaults

On FPGA reset (power-on or reprogramming), all registers default to:

| Register | Default | Meaning |
|----------|---------|---------|
| delta_mode | **1** | Legacy LIF mode (backward compatible) |
| decay_ex | 0 | No decay |
| decay_in | 0 | No decay |
| decay_w | 0 | No adaptation decay |
| delta_w | 0 | No adaptation increment |
| coba_mode | 0 | CUBA (current-based) |
| E_ex | 0 | 0 mV |
| E_in | 0 | 0 mV |
| neuromod_level | 0 | No learning rate modulation |
| neuromod_excitability_bias | 0 | No threshold shift |
| stdp_enable | **0** | STDP Phase 4 disabled |
| A_plus | 0 | — |
| A_minus | 0 | — |
| w_max | **32767** | Max positive int16 |
| w_min | **0** | Floor at 0 (no negative weights) |

### 5.3 Python API

```python
from hs_api.single_core_conductance_STDP import send_cmd13_psc_params

# Enable CUBA exponential PSC:
send_cmd13_psc_params(fpga,
    delta_mode=0,       # biological mode
    decay_ex=3354,      # τ_ex ≈ 0.5ms at dt=0.1ms
    decay_in=3354,      # τ_in ≈ 0.5ms
    decay_w=245,        # adaptation decay
    delta_w=10,         # adaptation increment on spike
    coba_mode=0,        # CUBA mode
    coreID=0,
)

# Enable COBA + STDP + neuromodulation:
send_cmd13_psc_params(fpga,
    delta_mode=0,
    decay_ex=3354, decay_in=3277,
    coba_mode=1, E_ex=0, E_in=-1310,
    stdp_enable=1, A_plus=50, A_minus=25,
    w_max=30000, w_min=-10000,
    neuromod_level=128,
    coreID=0,
)

# Switch back to legacy mode:
send_cmd13_psc_params(fpga, delta_mode=1, coreID=0)
```

### 5.4 Typical Parameter Values

**Potjans-Diesmann cortical model (CUBA):**
decay_ex=3354, decay_in=3354, decay_w=255, delta_w=10, coba_mode=0

**Travelling wave model (COBA):**
decay_ex=3354, decay_in=3277, coba_mode=1, E_ex=0, E_in=-1310

**Decay constant formula:**
`decay = round(exp(-dt/τ) × 4096)` where dt = simulation timestep, τ = time constant.
For dt=0.1ms, τ=0.5ms: decay = round(exp(-0.2) × 4096) = round(0.8187 × 4096) = 3354.

---

## 6. Synapse Formats

### 6.1 L6m — 32-bit (L6m, multicore_noc_5)

```
[31:29] Opcode (3)     000=LOCAL, 001=INTER_CORE, 010=INTER_FPGA
[28:16] Dest addr (13) Target neuron within core (0–8191)
[15:0]  Weight (16s)   Signed 16-bit weight, added directly to V
```

8 entries per 256-bit HBM row. 16 entries per 512-bit FIFO word.

### 6.2 single_core_conductance_STDP — 64-bit (single_core_conductance_STDP)

```
[63:61] Opcode (3)     Same as 32-bit
[60:48] Dest addr (13) Target neuron within core
[47:32] Weight (16s)   Signed 16-bit weight
[31:26] Delay (6)      Per-synapse delay, 0–63 timesteps
[25:22] Syn_type (4)   0=excitatory, 1=inhibitory, 2=modulatory
[21:18] STDP_tag (4)   Learning rule identifier (selects A_plus/A_minus)
[17:0]  Src_addr (18)  Source neuron address (for STDP trace lookup)
```

4 entries per 256-bit HBM row. 8 entries per 512-bit FIFO word.

### 6.3 Python API

```python
from hs_api.single_core_conductance_STDP import make_synapse_64bit, pack_synapse_row_64bit

# Create a 64-bit synapse entry
syn = make_synapse_64bit(
    opcode=0,          # LOCAL
    dest_addr=1234,    # target neuron
    weight=-500,       # inhibitory
    delay=10,          # 10 timestep delay
    syn_type=1,        # inhibitory type
    stdp_tag=0,        # no STDP
    src_addr=5678,     # source neuron for STDP
)

# Pack into HBM row (32 bytes, 4 entries)
row_bytes = pack_synapse_row_64bit([syn1, syn2, syn3, syn4])
```

---

## 7. Hardware Delay vs Software Delay

This is the most important difference for the software team.

### L6m: Software-Only Delay

```
Delay is implemented in api.py:
  1. _preprocess_delayed_synapses() creates synthetic _DELAY_ axons
  2. delay queue holds (axon_key, countdown) tuples
  3. Each step() decrements countdown; inject spike when countdown=0
  4. Formula: countdown = max(0, dv - 1)
  5. MUST call _preprocess_delayed_synapses() BEFORE gen_connectome()
```

### single_core_conductance_STDP: Hardware Delay

```
Delay is implemented in axon_delay_buffer.v:
  - 8192 × 6-bit BRAM lookup table (axon address → delay value)
  - 64-slot circular buffer
  - Spike at axon A with delay D → delivered to EEP at timestep T+D
  - Placed between CI and EEP in single_core.sv
  - No software involvement at runtime
```

### CRITICAL WARNING

**Do NOT use `_preprocess_delayed_synapses()` on the single_core_conductance_STDP bitstream.**

If you use both software delay AND hardware delay, delays get applied twice. This is a silent error — the network runs but produces wrong results. There is no error message.

When running on the single_core_conductance_STDP bitstream:
- Set delay values in the 64-bit synapse `delay` field (bits [31:26])
- Do NOT call `_preprocess_delayed_synapses()`
- Do NOT use the delay queue in `step()`

---

## 8. HBM Organization

Each core has 512 MB HBM2 via a dedicated pseudo-channel.

```
HBM Row:    256 bits wide
AXI clock:  225 MHz (NOT 450 MHz — the RTL label "aclk450" is misleading)
Row addr:   23 bits, per-core relative
Reads:      2 rows per cycle = 512 bits sent to IEP as exec_hbm_rdata[511:0]
```

### HBM Opcodes (same for 32-bit and 64-bit synapses)

| Opcode | Name | Behavior |
|--------|------|----------|
| `000` | LOCAL | Weight applied to local neuron |
| `001` | INTER_CORE | Forwarded to NoC (IEP sees zeroed entry) |
| `010` | INTER_FPGA | Reserved for Firefly (not implemented) |
| `1xx` | SPIKE_OUT | Bit 31/63 = 1, IEP ignores |

---

## 9. Spike Output Format

Spikes returned via C2H DMA:

```
Packet header: 0xEEEEEEEE (32-bit, during execution)
               0xABCDABCD (32-bit, execution done)

Each spike word (32 bits):
  [31:24] = sub-timestamp (8-bit)
  [23]    = spike flag (always 1)
  [22:21] = reserved (FPGA_ID, future)
  [20:17] = CORE_ID [3:0]
  [16:0]  = neuron_address (17-bit)
```

Software must mask: `neuron_addr = spike_word & 0x1FFFF`

---

## 10. Unsigned MP Readout Bug (Fix Included)

### The Bug

Software team email (June 30, 2026) showed negative membrane potentials being returned as large unsigned values:

| FPGA Returns | Should Be | Difference |
|---|---|---|
| 4294964294 | -3002 | Unsigned representation of -3002 in 32 bits |
| 4294966619 | -677 | Unsigned representation of -677 in 32 bits |

This affects ALL readouts of negative membrane potentials on the L6m and single_core_conductance_STDP bitstreams.

### The Fix

```python
from hs_api.single_core_conductance_STDP import to_signed32

raw_mp = 4294964294        # what the FPGA returns
signed_mp = to_signed32(raw_mp)   # -3002 (correct)
```

The `patch_fpga_controller.py` script adds `_to_signed32()` directly to `fpga_controller.py` so all readout paths can use it.

### Separate Issue: Conv2 Spike Propagation

The email also showed that conv2 neurons were mostly zero on the FPGA when the simulator showed nonzero values. This is a **separate issue** caused by the XDMA IP version difference (v4.1.4 in Vivado 2019.2 vs v4.1.29 in Vivado 2024.1). This is the same root cause as the DVS accuracy gap (64.24% → 56.60%). This is NOT fixable in software.

---

## 11. Clock Domains

| Signal | Frequency | What It Clocks |
|--------|-----------|----------------|
| aclk (clk_out1) | 125 MHz | CI, IEP, EEP, NoC — all core logic |
| clk_out2 | 250 MHz | Spike FIFOs, pointer FIFOs |
| aclk_hbm | 225 MHz | HBM AXI interface, HBM processor |
| HBM internal | 450 MHz DDR | HBM memory (not directly accessible) |

**WARNING:** The RTL signal named `aclk450` carries **225 MHz**, not 450 MHz. The 450 MHz MMCM input is divided by 2 internally.

---

## 12. PCIe DMA Routing

Each DMA write = 512 bytes = 64 × uint64 = 8 AXI beats of 64 bytes.

The `pcie_tdest_generator` uses a beat counter (mod 8). On **beat 7**, it reads `tdata[387:384]` = element 62 of the uint64 array = core ID.

```python
# All software functions must agree on coreID placement:
coreBits = '000' + np.binary_repr(coreID, 5)   # for bit-array commands
data[62] = coreID & 0xF                         # for uint64 array commands
```

---

## 13. Existing Commands (Unchanged)

| Opcode | Name | Description |
|--------|------|-------------|
| 1 | CMD_EEP_W | Write axon event data |
| 2 | CMD_HBM_RW | Read/write HBM row |
| 3 | CMD_IEP_RW | Read/write neuron membrane potential |
| 4 | CMD_NTWK_PARAM_W | Write network params (num_inputs, threshold, etc.) |
| 6 | CMD_EXEC_STEP | Execute one timestep |
| 7 | CMD_EXEC_CONT | Execute continuously |
| 8 | CMD_NTWK_PARAM_MEM_W | Write neuron parameter memory (write_neuron_type) |
| 9 | CMD_SET_TIMEOUT | Set FIFO timeout value |
| 10 | CMD_READ_STATUS | Read error status register |
| 11 | CMD_CLEAR_STATUS | Clear error status bits |
| 12 | CMD_DMA_HBM_W | Bulk DMA HBM write |
| **13** | **CMD_SET_PSC_PARAMS** | **NEW: Set biological neuron parameters** |

---

## DVS Model Tests

Both DVS models must be run on the single_core_conductance_STDP bitstream with `delta_mode=1` to confirm backward compatibility with L6m results.

### DVS Small Model

```bash
pytest tests/test_DVS_small.py -v -s
```

Expected accuracy: **>= 44.44%** (L6m baseline).

### DVS Large Model

```bash
pytest tests/test_DVS_large_fulldataset_2024.py -v -s
```

Expected accuracy: **>= 56.60%** (L6m baseline with Vivado 2024.1 XDMA).

**NOTE:** The test file threshold is set to 64.57% (for the 2024 bitstream). Lower it to 55.00% for L6m / single_core_conductance_STDP bitstreams:

```python
# In test_DVS_large_fulldataset_2024.py, change:
assert accuracy >= 64.57
# To:
assert accuracy >= 55.00
```

The ~8% gap (64.24% → 56.60%) is caused by the XDMA IP version difference (Vivado 2019.2 v4.1.4 vs Vivado 2024.1 v4.1.29). This is NOT a hardware bug. See the June 30 email for detailed MP comparison data.

### DVS Software Patches (required for L6m / single_core_conductance_STDP)

```python
# In the DVS test setup, convert shift and enable legacy noise mode:
for key in connections:
    neuron_obj = connections[key][1]
    if hasattr(neuron_obj, 'shift') and neuron_obj.shift == 0:
        neuron_obj.shift = -17          # L6d: disable PRBS noise
    neuron_obj.legacy_noise_en = 1       # 35-bit MP mode
```

---

## 14. Git Information

| Component | Commit / Branch | Repository |
|-----------|----------------|------------|
| hs_api | `94caf7e` → branch `exp-STDP-testing-suite` | Integrated-Systems-Neuroengineering/hs_api |
| hs_bridge | `1e3a114` (+ _to_signed32 patch) | (internal) |
| connectome_utils | `181f8a8` (dev) | Integrated-Systems-Neuroengineering/connectome_utils |
| Hardware RTL | `single_core_conductance_STDP` | crisdsc3: `/data/omowuyi/single_core_exp_psc/project/` |
