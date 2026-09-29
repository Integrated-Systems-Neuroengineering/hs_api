#!/usr/bin/env python3
"""
single_core_conductance_STDP.py — HiAER-Spike single_core_conductance_STDP (single_core_conductance_STDP) FPGA interface
================================================================================
RTL-verified against command_interpreter.v (July 2026, crisdsc3 build).

This module extends the existing fpga_controller.py with functions for the
biological neuron model features in single_core_conductance_STDP:

  - CMD 13 (CMD_SET_PSC_PARAMS, opcode 0x0D): Configure delta_mode, decay
    constants, COBA mode, neuromodulation, and STDP parameters
  - 64-bit synapse packing for the upgraded HBM format
  - Signed membrane potential conversion (fixes the June 30 bug)

CRITICAL NOTES:
  1. delta_mode is NOT in write_neuron_type. It is set via CMD 13 only.
  2. delta_mode defaults to 1 (legacy) on FPGA reset. Existing tests never
     send CMD 13, so backward compatibility is automatic.
  3. single_core_conductance_STDP uses HARDWARE delay (axon_delay_buffer.v). Do NOT use the
     software delay path (_preprocess_delayed_synapses / delay queue).
  4. A_plus and A_minus are 8-bit (not 16-bit). w_max and w_min are 16-bit signed.

USAGE:
  import single_core_conductance_STDP as p2

  # Fix unsigned MP readout
  signed_mp = p2.to_signed32(raw_fpga_value)

  # Enable biological neuron mode (send ONCE after network init)
  p2.send_cmd13_psc_params(fpga, delta_mode=0, decay_ex=3354, ...)

  # Pack 64-bit synapses for HBM write
  row_bytes = p2.pack_synapse_row_64bit([syn1, syn2, syn3, syn4])
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

# =========================================================================
# CMD 13 Packet Format — Verified from command_interpreter.v
# =========================================================================
# rxFIFO_dout[511:504] = opcode = 8'd13
# rxFIFO_dout[130:0]   = payload (131 bits)
#
# Field                          rxFIFO[H:L]  Width
# ---------------------------------------------------
# delta_mode                     [  0:  0]      1   1=legacy (DEFAULT), 0=exp PSC
# decay_ex                       [ 12:  1]     12   I_ex decay factor (x4096)
# decay_in                       [ 24: 13]     12   I_in decay factor (x4096)
# decay_w                        [ 32: 25]      8   Adaptation decay (x256)
# delta_w_param                  [ 40: 33]      8   Adaptation increment on spike
# coba_mode                      [ 41: 41]      1   0=CUBA, 1=COBA
# E_ex                           [ 53: 42]     12   Excitatory reversal (12-bit signed)
# E_in                           [ 65: 54]     12   Inhibitory reversal (12-bit signed)
# neuromod_level                 [ 73: 66]      8   STDP learning rate scale
# neuromod_excitability_bias     [ 81: 74]      8   Threshold shift (signed)
# stdp_enable                    [ 82: 82]      1   Enable STDP Phase 4
# A_plus                         [ 90: 83]      8   Potentiation magnitude
# A_minus                        [ 98: 91]      8   Depression magnitude
# w_max                          [114: 99]     16   Max weight (signed)
# w_min                          [130:115]     16   Min weight (signed)
#
# Reset defaults: delta_mode=1, stdp_enable=0, w_max=32767, w_min=0
# =========================================================================

CMD_SET_PSC_PARAMS = 13  # opcode 0x0D

# 64-bit synapse opcodes
OPCODE_LOCAL      = 0b000
OPCODE_INTER_CORE = 0b001
OPCODE_INTER_FPGA = 0b010

# Synapse types
SYNTYPE_EXC = 0  # excitatory
SYNTYPE_INH = 1  # inhibitory
SYNTYPE_MOD = 2  # modulatory


# =========================================================================
# Signed Conversion — Fixes the June 30 2026 Bug
# =========================================================================

def to_signed32(val):
    """uint32 → int32. Verified against the software team email data."""
    val = int(val) & 0xFFFFFFFF
    return val - 0x100000000 if val >= 0x80000000 else val


def to_signed16(val):
    """uint16 → int16."""
    val = int(val) & 0xFFFF
    return val - 0x10000 if val >= 0x8000 else val


def to_signed12(val):
    """uint12 → int12."""
    val = int(val) & 0xFFF
    return val - 0x1000 if val >= 0x800 else val


# =========================================================================
# CMD 13: Set PSC Parameters
# =========================================================================

def _set_rxfifo_bits(cmd, high, low, value, width):
    """Set rxFIFO_dout[high:low] in the 512-bit command array.

    cmd is a list of 512 chars ('0'/'1'), cmd[0]=MSB=rxFIFO_dout[511].
    rxFIFO_dout[N] maps to cmd[511-N].
    """
    assert high - low + 1 == width, f"[{high}:{low}] is {high-low+1} bits, expected {width}"
    # Binary representation, MSB-first
    if value < 0:
        # Two's complement for signed fields
        value = value & ((1 << width) - 1)
    bits = format(value & ((1 << width) - 1), f'0{width}b')
    for i in range(width):
        bit_pos = high - i          # rxFIFO_dout bit position (MSB first)
        cmd_idx = 511 - bit_pos     # command array index
        cmd[cmd_idx] = bits[i]


def build_cmd13_packet(delta_mode=1, decay_ex=0, decay_in=0, decay_w=0,
                       delta_w=0, coba_mode=0, E_ex=0, E_in=0,
                       neuromod_level=0, neuromod_excitability_bias=0,
                       stdp_enable=0, A_plus=0, A_minus=0,
                       w_max=32767, w_min=0, syn_64bit_en=0, coreID=0):
    """Build a 512-bit CMD 13 packet as a uint64 numpy array.

    Bit positions are RTL-verified against command_interpreter.v.

    Args:
        delta_mode: 1=legacy LIF (default), 0=exp PSC biological model
        decay_ex:   12-bit, I_ex decay factor (fixed-point x4096, e.g. 3354 for τ=0.5ms)
        decay_in:   12-bit, I_in decay factor (fixed-point x4096)
        decay_w:    8-bit, adaptation current decay (fixed-point x256)
        delta_w:    8-bit, adaptation increment on spike
        coba_mode:  0=CUBA current-based, 1=COBA conductance-based
        E_ex:       12-bit signed, excitatory reversal potential
        E_in:       12-bit signed, inhibitory reversal potential
        neuromod_level: 8-bit, STDP learning rate scale (0-255)
        neuromod_excitability_bias: 8-bit signed, threshold shift
        stdp_enable: 0=Phase 4 disabled (default), 1=STDP active
        A_plus:     8-bit, potentiation magnitude
        A_minus:    8-bit, depression magnitude
        w_max:      16-bit signed, weight upper bound (default 32767)
        w_min:      16-bit signed, weight lower bound (default 0)
        coreID:     target core (0-15)

    Returns:
        numpy array of 64 uint64 elements, ready for dma_dump_write
    """
    cmd = ['0'] * 512

    # Opcode at rxFIFO_dout[511:504]
    opcode_bits = format(CMD_SET_PSC_PARAMS, '08b')
    for i in range(8):
        cmd[i] = opcode_bits[i]

    # coreID at rxFIFO_dout[503:496] (byte 1, bits [499:496] for tdest)
    core_bits = '000' + format(coreID & 0xF, '05b')
    for i in range(8):
        cmd[8 + i] = core_bits[i]

    # Payload fields — each verified against RTL
    _set_rxfifo_bits(cmd,   0,   0, delta_mode & 1, 1)
    _set_rxfifo_bits(cmd,  12,   1, decay_ex & 0xFFF, 12)
    _set_rxfifo_bits(cmd,  24,  13, decay_in & 0xFFF, 12)
    _set_rxfifo_bits(cmd,  32,  25, decay_w & 0xFF, 8)
    _set_rxfifo_bits(cmd,  40,  33, delta_w & 0xFF, 8)
    _set_rxfifo_bits(cmd,  41,  41, coba_mode & 1, 1)
    _set_rxfifo_bits(cmd,  53,  42, E_ex, 12)     # signed
    _set_rxfifo_bits(cmd,  65,  54, E_in, 12)     # signed
    _set_rxfifo_bits(cmd,  73,  66, neuromod_level & 0xFF, 8)
    _set_rxfifo_bits(cmd,  81,  74, neuromod_excitability_bias, 8)  # signed
    _set_rxfifo_bits(cmd,  82,  82, stdp_enable & 1, 1)
    _set_rxfifo_bits(cmd,  90,  83, A_plus & 0xFF, 8)
    _set_rxfifo_bits(cmd,  98,  91, A_minus & 0xFF, 8)
    _set_rxfifo_bits(cmd, 114,  99, w_max, 16)    # signed
    _set_rxfifo_bits(cmd, 130, 115, w_min, 16)    # signed
    _set_rxfifo_bits(cmd, 131, 131, int(syn_64bit_en) & 1, 1)   # 64-bit synapse entries

    return _cmd_to_uint64_array(cmd, coreID)


def send_cmd13_psc_params(fpga_controller, **kwargs):
    """Send CMD 13 to the FPGA. Wraps build_cmd13_packet + dma_dump_write.

    Example:
        send_cmd13_psc_params(fpga,
            delta_mode=0,       # enable biological model
            decay_ex=3354,      # τ_ex ~ 0.5ms at dt=0.1ms
            decay_in=3354,      # τ_in ~ 0.5ms
            coba_mode=0,        # CUBA mode
            stdp_enable=1,      # enable STDP
            A_plus=50,
            A_minus=25,
            w_max=32767,
            w_min=-32768,
        )
    """
    data = build_cmd13_packet(**kwargs)
    fpga_controller.dma_dump_write(data)
    log.info("CMD 13 sent: delta_mode=%d, stdp_enable=%d, coba_mode=%d",
             kwargs.get('delta_mode', 1),
             kwargs.get('stdp_enable', 0),
             kwargs.get('coba_mode', 0))


# =========================================================================
# 64-bit Synapse Format
# =========================================================================
# Bit [63:61] = Opcode (3)
# Bit [60:48] = Dest addr (13)
# Bit [47:32] = Weight (16 signed)
# Bit [31:26] = Delay (6, 0-63 timesteps)
# Bit [25:22] = Syn_type (4)
# Bit [21:18] = STDP_tag (4)
# Bit [17:0]  = Src_addr (18)
# =========================================================================

def make_synapse_64bit(opcode, dest_addr, weight, delay=0,
                       syn_type=0, stdp_tag=0, src_addr=0):
    """Create a single 64-bit synapse entry."""
    weight_u16 = weight & 0xFFFF
    return ((opcode    & 0x7)     << 61) | \
           ((dest_addr & 0x1FFF)  << 48) | \
           ((weight_u16)          << 32) | \
           ((delay     & 0x3F)    << 26) | \
           ((syn_type  & 0xF)     << 22) | \
           ((stdp_tag  & 0xF)     << 18) | \
           ((src_addr  & 0x3FFFF))


def pack_synapse_row_64bit(entries):
    """Pack up to 4 x 64-bit entries into 32-byte HBM row.

    Uses per-entry byte reversal (8 bytes/entry), same convention
    as the existing 32-bit _pack_synapse_row in fpga_compiler.py.
    """
    while len(entries) < 4:
        entries.append(0)
    result = []
    for entry in entries[:4]:
        entry_bytes = [(entry >> (8 * (7 - b))) & 0xFF for b in range(8)]
        result.extend(reversed(entry_bytes))  # per-entry byte reversal
    return result


# =========================================================================
# Helpers
# =========================================================================

def _cmd_to_uint64_array(cmd_bits, coreID):
    """Convert 512-char bit list to 64-element uint64 array for DMA.
    Each element = one byte. element[0] = rxFIFO_dout[7:0], element[63] = rxFIFO_dout[511:504].
    cmd_bits[0] = MSB = rxFIFO_dout[511]. cmd_bits[511] = LSB = rxFIFO_dout[0].
    """
    data = np.zeros(64, dtype=np.uint64)
    for byte_idx in range(64):
        byte_val = 0
        for bit_pos in range(8):
            # rxFIFO_dout[byte_idx*8 + bit_pos] = cmd_bits[511 - (byte_idx*8 + bit_pos)]
            cmd_idx = 511 - (byte_idx * 8 + bit_pos)
            if 0 <= cmd_idx < 512 and cmd_bits[cmd_idx] == '1':
                byte_val |= (1 << bit_pos)
        data[byte_idx] = np.uint64(byte_val)
    return data


def decode_cmd13_packet(data):
    """Decode a uint64[64] DMA packet back to CMD 13 fields.

    Useful for verifying packets before sending to FPGA.
    """
    # Reconstruct 512-bit string
    bits = ''
    for i in range(8):
        bits += format(int(data[i]), '064b')
    bits += '0' * (512 - len(bits))

    def get_field(high, low, width, signed=False):
        val = 0
        for b in range(low, high + 1):
            bit_val = int(bits[511 - b])
            val |= (bit_val << (b - low))
        if signed and val >= (1 << (width - 1)):
            val -= (1 << width)
        return val

    opcode = int(bits[0:8], 2)
    return {
        'opcode': opcode,
        'delta_mode': get_field(0, 0, 1),
        'decay_ex': get_field(12, 1, 12),
        'decay_in': get_field(24, 13, 12),
        'decay_w': get_field(32, 25, 8),
        'delta_w': get_field(40, 33, 8),
        'coba_mode': get_field(41, 41, 1),
        'E_ex': get_field(53, 42, 12, signed=True),
        'E_in': get_field(65, 54, 12, signed=True),
        'neuromod_level': get_field(73, 66, 8),
        'neuromod_excitability_bias': get_field(81, 74, 8, signed=True),
        'stdp_enable': get_field(82, 82, 1),
        'A_plus': get_field(90, 83, 8),
        'A_minus': get_field(98, 91, 8),
        'w_max': get_field(114, 99, 16, signed=True),
        'w_min': get_field(130, 115, 16, signed=True),
    }


# =========================================================================
# Self-Test
# =========================================================================

def self_test():
    """Verify all functions against known values."""
    print("=== single_core_conductance_STDP.py Self-Test ===\n")
    all_pass = True

    # Test 1: Signed conversion (the software team's data)
    print("Test 1: to_signed32 (June 30 test data)")
    vectors = [(4294964294, -3002), (4294967169, -127), (4294966619, -677),
               (117762, 117762), (0, 0)]
    for raw, expected in vectors:
        got = to_signed32(raw)
        ok = got == expected
        if not ok: all_pass = False
        print(f"  {raw:>12d} -> {got:>8d} {'OK' if ok else 'FAIL'}")

    # Test 2: CMD 13 round-trip encode/decode
    print("\nTest 2: CMD 13 encode/decode round-trip")
    params = dict(delta_mode=0, decay_ex=3354, decay_in=3277, decay_w=245,
                  delta_w=10, coba_mode=1, E_ex=0, E_in=-1310,
                  neuromod_level=128, neuromod_excitability_bias=-5,
                  stdp_enable=1, A_plus=50, A_minus=25,
                  w_max=30000, w_min=-10000, coreID=0)
    pkt = build_cmd13_packet(**params)
    decoded = decode_cmd13_packet(pkt)

    for key in ['delta_mode', 'decay_ex', 'decay_in', 'decay_w', 'delta_w',
                'coba_mode', 'E_ex', 'E_in', 'neuromod_level',
                'neuromod_excitability_bias', 'stdp_enable',
                'A_plus', 'A_minus', 'w_max', 'w_min']:
        expected = params[key]
        got = decoded[key]
        ok = got == expected
        if not ok: all_pass = False
        print(f"  {key:<32s} sent={expected:>8d}  decoded={got:>8d}  {'OK' if ok else 'FAIL'}")
    assert decoded['opcode'] == 13, f"opcode={decoded['opcode']}"
    print(f"  {'opcode':<32s} = {decoded['opcode']} OK")

    # Test 3: 64-bit synapse round-trip
    print("\nTest 3: 64-bit synapse encode/decode")
    cases = [(0, 1234, -500, 10, 0, 3, 5678),
             (1, 8191, 32767, 63, 15, 15, 262143),
             (0, 0, 0, 0, 0, 0, 0)]
    for op, d, w, dl, st, tg, sa in cases:
        syn = make_synapse_64bit(op, d, w, dl, st, tg, sa)
        g_op = (syn >> 61) & 7
        g_d  = (syn >> 48) & 0x1FFF
        g_w  = to_signed16((syn >> 32) & 0xFFFF)
        g_dl = (syn >> 26) & 0x3F
        g_st = (syn >> 22) & 0xF
        g_tg = (syn >> 18) & 0xF
        g_sa = syn & 0x3FFFF
        ok = (g_op==op and g_d==d and g_w==w and g_dl==dl and g_st==st and g_tg==tg and g_sa==sa)
        if not ok: all_pass = False
        print(f"  op={op} dest={d} w={w} delay={dl} type={st} tag={tg} src={sa}  {'OK' if ok else 'FAIL'}")

    # Test 4: Row packing
    print("\nTest 4: 64-bit row packing")
    row = pack_synapse_row_64bit([make_synapse_64bit(0, 100, 500)])
    ok = len(row) == 32
    if not ok: all_pass = False
    print(f"  4 entries -> {len(row)} bytes  {'OK' if ok else 'FAIL'}")

    print(f"\n{'=== ALL TESTS PASSED ===' if all_pass else '=== SOME TESTS FAILED ==='}")
    return all_pass


if __name__ == '__main__':
    self_test()
