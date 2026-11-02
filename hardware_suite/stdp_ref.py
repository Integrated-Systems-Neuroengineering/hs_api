#!/usr/bin/env python3
"""
stdp_ref.py (v3) -- bit-exact reference of the HiAER-Spike plasticity rule.  RTL, compiler and tests are
judged against this file.  With default signs, A0 = 0 and eligibility off it is identical to v2.

Rule (per plastic synapse; stdp_tag 1..15 selects a rule slot, 0 = static):
  Pfister & Gerstner (2006) full triplet, nearest-spike, generalised with signed amplitudes:
    post fires at t (LTP event):  p = + x_j * (s2p*A2p + s3p*A3p*y_i)
    pre  fires at t (LTD event):  p = - o_i * (s2m*A2m + s3m*A3m*r_j) + s0*A0
      x_j = exp(-(t - t_j)/tau_p), y_i = exp(-ISI_i/tau_y), o_i = exp(-(t - t_i)/tau_m), r_j = exp(-ISI_j/tau_x)
      s* = +1 or -1 per amplitude (default +1): symmetric, anti-Hebbian and inhibitory (Vogels 2011) windows;
      A0 = constant applied at every presynaptic spike (Vogels' -alpha, Kempter's w_in).
  same step: if pre and post fire in the same step the synapse is not changed in that step (constant included)
  direct rule (eligibility off): w += p, with
      soft bounds (default): a positive change scales by (w_max - w)/(w_max - w_min), a negative one by
      (w - w_min)/(w_max - w_min); additive: unscaled; then neuromod gating * g/128, rounding, clamp.
  eligibility rule (three-factor): e = e * exp(-(t - t_e)/tau_e) + p, t_e = t (no bounds, no gating);
      a reward command with signed R at step t gives  w += eta * R * e(t)  (bounds, rounding, clamp; no g),
      then e = 0 (reset rules) or e(t) (the trace keeps decaying), t_e = t.
  order within a step: LTD pass, LTP pass; a reward command applies after the step.

Fixed point (identical in the RTL).  sgn(v) = -1 if v < 0 else +1;  scale(v, m, k) = sgn(v) * ((|v| * m) >> k)
  traces : tables LUT[d], d = 0..4095, u16 Q1.15; rules pick tau_p, tau_m, tau_x, tau_y, tau_e tables
  amps   : A2p, A3p, A2m, A3m, A0 u16 magnitudes in Q8.8 weight units + one sign bit each
  a      : (s2 ? -A2 : A2) + (s3 ? -((A3*tr) >> 15) : (A3*tr) >> 15)
  LTP    : p = scale(a, x, 15)
  LTD    : p = -scale(a, o, 15) + (s0 ? -A0 : A0)
  apply  : [soft] s = p >= 0 ? max(w_max - w, 0) : max(w - w_min, 0);  f = (s * inv_range) >> 16;  p = scale(p, f, 16)
           inv_range = min(round(2^32 / (w_max - w_min)), 2^32 - 1): a 32-bit reciprocal, exact to 1 in 65536 at
           every range (a 16-bit one was 22% off at a range of 40000)
           p = scale(p, g, 7);  dw = sgn(p) * ((|p| + 128) >> 8);  w = clamp(w + dw, w_min, w_max)
  e      : signed 32-bit, Q8.8 weight units, saturating; D = LUT_e[min((t - t_e) >> e_shift, 4095)]
           decay advances t_e by exactly the time decayed: idx = min((t - t_e) >> e_shift, 4095);
           t_e += idx << e_shift (t_e = t when idx = 4095), so the remainder carries forward and never compounds
           accumulate: e = sat(scale(e, D, 15) + p)
           reward (R signed 16-bit Q8.8, eta u16 Q8.8):  ed = scale(e, D, 15)
                    p = sgn(ed)*sgn(R) * ((|ed| * |R|) >> 8);  p = sat(scale(p, eta, 8), +/-(2^26 - 1));  w = apply(p, g = 128)
Spike times: as v2 -- neurons on chip (12-bit stamp + ISI, scrubbed), axons in HBM (32-bit stamps);
eligibility stamps t_e are 32-bit (HBM shadow entries), so they never wrap.
"""
import math
from stdp_ref_v2 import make_lut, NeuronStamps, AxonStamps, TB, DMAX, SCRUB_PERIOD, OLD

E_MAX, E_MIN = (1 << 31) - 1, -(1 << 31)
P_SAT = (1 << 26) - 1

def sgn(v): return -1 if v < 0 else 1
def scale(v, m, k): return sgn(v) * ((abs(v) * m) >> k)
def sat32(v): return max(E_MIN, min(E_MAX, v))

class Rule:
    def __init__(self, A2p, A3p, A2m, w_min, w_max, soft=True, A3m=0, A0=0,
                 s2p=0, s3p=0, s2m=0, s3m=0, s0=0, elig=False, eta=256, elig_reset=False, e_shift=0,
                 lut_p=0, lut_m=1, lut_y=2, lut_x=3, lut_e=4):
        self.A2p, self.A3p, self.A2m, self.A3m, self.A0 = A2p, A3p, A2m, A3m, A0
        self.s2p, self.s3p, self.s2m, self.s3m, self.s0 = s2p, s3p, s2m, s3m, s0
        self.w_min, self.w_max, self.soft = w_min, w_max, soft
        self.inv_range = min(int(round(2 ** 32 / (w_max - w_min))), 2 ** 32 - 1)
        self.elig, self.eta, self.elig_reset, self.e_shift = elig, eta, elig_reset, e_shift
        self.lut_p, self.lut_m, self.lut_y, self.lut_x, self.lut_e = lut_p, lut_m, lut_y, lut_x, lut_e

def clamp(w, rule): return max(rule.w_min, min(rule.w_max, w))

def amp(A2, s2, A3, s3, tr):
    t3 = (A3 * tr) >> 15
    return (-A2 if s2 else A2) + (-t3 if s3 else t3)

def event_p(rule, luts, ltp, d, isi):
    """Raw signed change of one event in Q8.8, or None when the pair is same-step (no change at all)."""
    if d == 0:
        return None
    if ltp:
        a = amp(rule.A2p, rule.s2p, rule.A3p, rule.s3p, luts[rule.lut_y][isi])
        return scale(a, luts[rule.lut_p][d], 15)
    a = amp(rule.A2m, rule.s2m, rule.A3m, rule.s3m, luts[rule.lut_x][isi])
    return -scale(a, luts[rule.lut_m][d], 15) + (-rule.A0 if rule.s0 else rule.A0)

def apply_p(rule, p, w, g):
    if rule.soft:
        s = max(rule.w_max - w, 0) if p >= 0 else max(w - rule.w_min, 0)
        p = scale(p, (s * rule.inv_range) >> 16, 16)
    p = scale(p, g, 7)
    return clamp(w + sgn(p) * ((abs(p) + 128) >> 8), rule)

def decay(rule, luts, e, t_e, t):
    """Lazy decay: returns (decayed e, new t_e).  t_e advances by exactly the time decayed."""
    idx = min((t - t_e) >> rule.e_shift, DMAX)
    return scale(e, luts[rule.lut_e][idx], 15), (t if idx == DMAX else t_e + (idx << rule.e_shift))

def elig_accumulate(rule, luts, e, t_e, t, p):
    ed, t_e = decay(rule, luts, e, t_e, t)
    return sat32(ed + p), t_e

def reward_apply(rule, luts, e, t_e, t, R, w):
    ed, t_e = decay(rule, luts, e, t_e, t)
    p = sgn(ed) * sgn(R) * ((abs(ed) * abs(R)) >> 8)
    p = max(-P_SAT, min(P_SAT, scale(p, rule.eta, 8)))      # any larger change reaches the bound regardless
    w = apply_p(rule, p, w, 128)
    return (w, 0, t) if rule.elig_reset else (w, sat32(ed), t_e)

def run_synapse(rule, luts, pre_spikes, post_spikes, steps, w0, g=128, pre_is_axon=False, rewards=()):
    """One synapse, quantized. rewards: {step: R}. Returns (weights per step, traces per step)."""
    pre = AxonStamps(1) if pre_is_axon else NeuronStamps(1)
    post = NeuronStamps(1)
    pre.clear(); post.clear()
    w, e, t_e, W, Ev = w0, 0, 0, [], []
    P, Q, Rw = set(pre_spikes), set(post_spikes), dict(rewards)
    for t in range(steps):
        fp, fq = t in P, t in Q
        if fp: pre.spike(0)
        if fq: post.spike(0)
        for ltp, fired, d, isi in ((False, fp, post.d(0), pre.isi[0]), (True, fq, pre.d(0), post.isi[0])):
            if not fired: continue
            p = event_p(rule, luts, ltp, d, isi)
            if p is None: continue
            if rule.elig: e, t_e = elig_accumulate(rule, luts, e, t_e, t, p)
            else: w = apply_p(rule, p, w, g)
        if rule.elig and t in Rw:
            w, e, t_e = reward_apply(rule, luts, e, t_e, t, Rw[t], w)
        W.append(w); Ev.append(e)
        pre.advance(); post.advance()
    return W, Ev

def run_synapse_float(A2p, A3p, A2m, A3m, A0, signs, tp, tm, ty, tx, w_min, w_max, soft, pre_spikes, post_spikes,
                      steps, w0, g=1.0, elig=False, tau_e=1000.0, eta=1.0, reset=False, rewards=()):
    """The same rule in floating point at exact times. signs = (s2p, s3p, s2m, s3m, s0) as 0/1."""
    sg = [(-1.0 if s else 1.0) for s in signs]
    w, e, t_e, lp, lq = float(w0), 0.0, 0, None, None
    P, Q, Rw, out = set(pre_spikes), set(post_spikes), dict(rewards), []
    R_ = w_max - w_min
    def apply(dp, w):
        if soft: dp *= ((w_max - w) if dp >= 0 else (w - w_min)) / R_
        return max(w_min, min(w_max, w + dp))
    for t in range(steps):
        fp, fq = t in P, t in Q
        evs = []
        if fp and not fq:
            o = math.exp(-(t - lq) / tm) if lq is not None else 0.0
            r = math.exp(-(t - lp) / tx) if lp is not None else 0.0
            evs.append(-o * (sg[2] * A2m + sg[3] * A3m * r) + sg[4] * A0)
        if fq and not fp:
            x = math.exp(-(t - lp) / tp) if lp is not None else 0.0
            y = math.exp(-(t - lq) / ty) if lq is not None else 0.0
            evs.append(x * (sg[0] * A2p + sg[1] * A3p * y))
        for p in evs:
            if elig: e = e * math.exp(-(t - t_e) / tau_e) + p; t_e = t
            else: w = apply(p * g, w)
        if elig and t in Rw:
            ed = e * math.exp(-(t - t_e) / tau_e)
            w = apply(eta * Rw[t] * ed, w)
            e, t_e = (0.0 if reset else ed), t
        if fp: lp = t
        if fq: lq = t
        out.append(w)
    return out
