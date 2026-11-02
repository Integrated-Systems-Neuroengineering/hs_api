"""
plasticity.py -- host side of the HiAER-Spike plasticity controller (stdp_ctrl.v, opcodes 19/20, feature bit 4).

  cfg_packets(pairs, coreID)      opcode 19: up to 10 {addr[15:0], data[31:0]} pairs per packet, pair k at bits
                                  [48k+47:48k], count at [495:492]; byte 62 = core, byte 63 = opcode
  reward_packet(R, coreID)        opcode 20: reward R (signed, Q8.8 weight units) at bits [15:0]
  rule_pairs(slot, rule)          the six configuration words of one rule slot (1..15)
  table_pairs(table, values)      one 4096-entry trace table (Q1.15)
  layout(hbm, n_axons, n_neurons, free_row)
                                  post-pass on a compiled HBM image: walks every outgoing list, fills src on plastic
                                  entries, builds the incoming index, axon-stamp and eligibility-shadow regions, and
                                  returns (new rows, base-register pairs)
Packets are numpy uint64 arrays of 64 byte values, element k = bits [8k+7:8k], as everywhere in hs_bridge.
"""
import math
try:
    import numpy as np
except ImportError:                      # the layout and encoders are pure Python; packets need numpy
    np = None

OPC_PLAST_CFG, OPC_REWARD = 19, 20
FEAT_PLASTICITY = 1 << 4
AXON_PTR_BASE, NRN_PTR_BASE = 0, 16384          # fixed by the HBM processor's pointer addressing
CFG = dict(AXON_PTR_BASE=0x100, NRN_PTR_BASE=0x101, INC_PTR_BASE=0x102, AXON_STAMP_BASE=0x103,
           SHADOW_BASE=0x104, PLASTIC_START=0x105, PLASTIC_END=0x106)
DMAX, OLD = 4095, 4095 - 128
NEVER = (-4095) & 0xFFFFFFFF                   # axon stamp of an axon that has not fired since the clear

# ---------------------------------------------------------------- packets
def _core_byte(coreID):
    return int(format(coreID, '05b') + '000', 2)

def cfg_packets(pairs, coreID=0):
    out = []
    for i in range(0, len(pairs), 10):
        chunk = pairs[i:i + 10]
        v = len(chunk) << 492
        for k, (a, d) in enumerate(chunk):
            if not (0 <= a <= 0xFFFF and 0 <= d <= 0xFFFFFFFF): raise ValueError("cfg pair out of range: %r" % ((a, d),))
            v |= ((a << 32) | d) << (48 * k)
        b = [(v >> (8 * k)) & 0xFF for k in range(62)] + [_core_byte(coreID), OPC_PLAST_CFG]
        out.append(np.array(b, dtype=np.uint64) if np is not None else b)
    return out

def reward_packet(R, coreID=0):
    if not -32768 <= R <= 32767: raise ValueError("reward %d outside signed 16-bit Q8.8" % R)
    r = R & 0xFFFF
    b = [r & 0xFF, r >> 8] + [0] * 60 + [_core_byte(coreID), OPC_REWARD]
    return np.array(b, dtype=np.uint64) if np is not None else b

def packet_hex(pkt):
    """simDump form: 128 hex digits, most significant byte first (as hs_bridge's other simDump paths)."""
    return ''.join('{:02x}'.format(int(x)) for x in reversed(list(pkt)))

# ---------------------------------------------------------------- rules and tables
def rule_pairs(slot, rule):
    """The six 32-bit words of rule slot 1..15, as stdp_ctrl.v decodes them.  `rule` is a stdp_ref.Rule."""
    if not 1 <= slot <= 15: raise ValueError("rule slot must be 1..15")
    sg = rule.s2p | (rule.s3p << 1) | (rule.s2m << 2) | (rule.s3m << 3) | (rule.s0 << 4)
    words = [(rule.A3p << 16) | rule.A2p, (rule.A3m << 16) | rule.A2m,
             rule.A0 | (sg << 16) | (int(rule.soft) << 21) | (int(rule.elig) << 22) | (int(rule.elig_reset) << 23) | (rule.e_shift << 24),
             ((rule.w_max & 0xFFFF) << 16) | (rule.w_min & 0xFFFF), rule.inv_range,
             rule.eta | (rule.lut_p << 16) | (rule.lut_m << 19) | (rule.lut_y << 22) | (rule.lut_x << 25) | (rule.lut_e << 28)]
    return [(16 * slot + k, w) for k, w in enumerate(words)]

def trace_table(tau_steps):
    """Q1.15 table of exp(-d/tau), 0 from OLD on (the usable window of the neuron stamps)."""
    t = [0] * (DMAX + 1)
    for d in range(OLD):
        t[d] = min(32768, int(round(32768 * math.exp(-d / tau_steps))))
    return t

def decay_table(tau_steps, e_shift):
    """Eligibility decay table: entry i = exp(-(i << e_shift)/tau); the last entry is 0 so long gaps decay fully."""
    t = [min(32768, int(round(32768 * math.exp(-(i << e_shift) / tau_steps)))) for i in range(DMAX + 1)]
    t[DMAX] = 0
    return t

def table_pairs(table, values):
    if not 0 <= table <= 7 or len(values) != DMAX + 1: raise ValueError("table 0..7 with 4096 entries")
    return [(0x8000 | (table << 12) | i, v & 0xFFFF) for i, v in enumerate(values)]

# ---------------------------------------------------------------- layout post-pass
def post_group(o, s):
    """Neuron processor's 64-bit Phase-2 decode: entry in row offset o (within a pre's list), slot s -> group.
    Derived from internal_events_processor.v; verified end to end by the plasticity-on project test."""
    return (((o >> 1) & 1) << 3) | ((1 - (o & 1)) << 2) | (3 - s)

def _lanes(row, width):
    return [(row >> (width * k)) & ((1 << width) - 1) for k in range(256 // width)]

def layout(hbm, n_axons, n_neurons, free_row):
    """hbm: {row: 256-bit int} -- the compiled image (pointer tables and synapse rows), modified in place for src.
    Returns (new_rows {row: int}, base_pairs).  Plastic entries are op 0 with stdp_tag != 0."""
    incoming = {}
    plastic_rows = []
    for kind, base, count in (('a', AXON_PTR_BASE, n_axons), ('n', NRN_PTR_BASE, n_neurons)):
        for i in range(count):
            ptr = (hbm.get(base + (i >> 3), 0) >> (32 * (i & 7))) & 0xFFFFFFFF
            addr, ln = ptr & 0x7FFFFF, ptr >> 23
            if addr == 0: continue
            for o in range(ln + 1):
                row = addr + o; r = hbm.get(row, 0)
                for s in range(4):
                    e = (r >> (64 * s)) & ((1 << 64) - 1)
                    if (e >> 61) != 0 or ((e >> 18) & 0xF) == 0: continue
                    src = ((1 << 17) | i) if kind == 'a' else i
                    e = (e & ~((1 << 18) - 1)) | src
                    r = (r & ~(((1 << 64) - 1) << (64 * s))) | (e << (64 * s))
                    post = ((e >> 48) & 0x1FFF) * 16 + post_group(o & 3, s)
                    incoming.setdefault(post, []).append((row, s))
                    plastic_rows.append(row)
                hbm[row] = r
    new = {}
    row = free_row
    INC_PTR = row; row += (n_neurons + 7) // 8
    for n in range(n_neurons):
        lst = incoming.get(n, [])
        if not lst: continue
        nrows = (len(lst) + 7) // 8
        new[INC_PTR + n // 8] = new.get(INC_PTR + n // 8, 0) | ((((nrows - 1) << 23) | row) << (32 * (n % 8)))
        for j, (r_, s_) in enumerate(lst):
            new[row + j // 8] = new.get(row + j // 8, 0) | (((1 << 31) | (s_ << 29) | r_) << (32 * (j % 8)))
        row += nrows
    AXON_STAMP = row
    for a in range(n_axons):
        new[AXON_STAMP + a // 8] = new.get(AXON_STAMP + a // 8, 0) | (NEVER << (32 * (a % 8)))
    row += (n_axons + 7) // 8
    PS = min(plastic_rows) if plastic_rows else row
    PE = (max(plastic_rows) + 1) if plastic_rows else row
    SHADOW = row
    for r_ in range(PE - PS): new[SHADOW + r_] = 0
    row += PE - PS
    if row >= (1 << 23): raise ValueError("plasticity regions do not fit below HBM row 2^23")
    bases = dict(AXON_PTR_BASE=AXON_PTR_BASE, NRN_PTR_BASE=NRN_PTR_BASE, INC_PTR_BASE=INC_PTR, AXON_STAMP_BASE=AXON_STAMP,
                 SHADOW_BASE=SHADOW, PLASTIC_START=PS, PLASTIC_END=PE)
    return new, [(CFG[k], v) for k, v in bases.items()]

# ---------------------------------------------------------------- network load
def load_packets(compiler, n_axons, n_neurons, rules, tables, coreID=0):
    """Every packet plasticity adds to a core's network load, in order: the plasticity rows (incoming index, axon
    stamps, eligibility shadows, and synapse rows whose src the layout filled) as CMD 2 writes, then opcode-19
    configuration for the rules {slot: Rule}, the trace/decay tables {table: 4096 values} and the base registers.
    Send them after the compiled image and before clear(): its opcode 18 restarts plasticity time at 0."""
    row_packets, base_pairs, image, new = compiler.plasticity_rows(n_axons, n_neurons)
    inv = {v: k for k, v in CFG.items()}
    compiler.plasticity_bases = {inv[a]: v for a, v in base_pairs}     # kept for read-back (tests, tools)
    pairs = []
    for slot in sorted(rules): pairs += rule_pairs(slot, rules[slot])
    for t in sorted(tables): pairs += table_pairs(t, tables[t])
    pairs += base_pairs
    return list(row_packets) + cfg_packets(pairs, coreID)

def send(packets):
    """Write packets to the card in batches of 10 (as the compiler does)."""
    import hs_bridge.wrapped_dmadump.dmadump as dmadump
    for i in range(0, len(packets), 10):
        chunk = np.concatenate([np.asarray(p, dtype=np.uint64) for p in packets[i:i + 10]])
        dmadump.dma_dump_write(chunk, len(chunk), 1, 0, 0, 0, dmadump.DmaMethodNormal)
