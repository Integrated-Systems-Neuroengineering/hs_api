#!/usr/bin/env python3
"""
stdp_ref.py -- bit-exact reference of the HiAER-Spike STDP rule (RTL, compiler and tests are judged
against this file).

Rule: Pfister & Gerstner (2006) FULL triplet model, nearest-spike interactions, per plastic synapse
(stdp_tag 1..15 selects a rule slot; 0 = static).  The minimal model is the special case A3m = 0
(and A2p = 0 for the visual-cortex fit).
  post neuron i fires at step t  (LTP, every incoming synapse from pre j):
      dw+ = x_j * (A2p + A3p * y_i),  x_j = exp(-(t - t_j)/tau_p),  y_i = exp(-ISI_i/tau_y)
  pre neuron/axon j fires at step t  (LTD, every outgoing synapse to post i):
      dw- = o_i * (A2m + A3m * r_j),  o_i = exp(-(t - t_i)/tau_m),  r_j = exp(-ISI_j/tau_x)
  (t_j, t_i: latest spike of the partner; ISI: interval from the firing neuron's previous spike)
  soft bounds (default): dw+ *= (w_max - w)/(w_max - w_min),  dw- *= (w - w_min)/(w_max - w_min)
  additive (per rule): unscaled, then clamped to [w_min, w_max]
  neuromodulation gating: both * neuromod_level / 128   (0 = learning off, 128 = nominal)
  same step: if pre and post fire in the same step, that synapse is not changed in that step
  order within a step: LTD pass, then LTP pass

Fixed point (identical in the RTL):
  traces       : tables LUT[d], d = 0..4095 steps, u16 Q1.15 (32768 = 1.0); a bank of tables, each rule
                 picks its tau_p, tau_m, tau_x, tau_y table; entries from OLD on must be 0
  amplitudes   : A2p, A3p, A2m, A3m  u16 Q8.8 weight units
  soft bound   : inv_range = round(65536 / (w_max - w_min)), factor = s * inv_range, Q0.16
  neuromod     : g = neuromod_level (u8), 128 = 1.0
  LTP          : a = A2p + ((A3p * y) >> 15);  p = (a * x) >> 15
  LTD          : a = A2m + ((A3m * r) >> 15);  p = (a * o) >> 15
  both         : [soft] p = (p * f) >> 16;  p = (p * g) >> 7;  dw = (p + 128) >> 8
  weight       : 16-bit signed; w' = clamp(w + dw (LTP) or w - dw (LTD), w_min, w_max)
Spike times (identical in the RTL):
  neurons (on chip): 12-bit stamp + 12-bit ISI; d = (t - stamp) mod 4096.  A clear sets every stamp to
    d = OLD and ISI = 4095; a background scrub visits every entry within SCRUB_PERIOD steps and re-ages
    any entry with d >= OLD back to d = OLD.  ISI at a spike = d if d < OLD else 4095.
  axons (in HBM): 32-bit stamp, never wraps; d = min(t - stamp, 4095); a clear sets "never fired";
    ISI at a spike = min(t - previous stamp, 4095) (4095 if never fired).
  Both index the same tables, which are 0 from OLD on, so the two give the same traces.
  Usable window: d < OLD (= 3967 with SCRUB_PERIOD 128).
"""
import math

TB = 12
DMAX = (1 << TB) - 1                # 4095
SCRUB_PERIOD = 128
OLD = DMAX - SCRUB_PERIOD           # 3967

def make_lut(tau, scale=32768):
    """Q1.15 exponential table for one time constant (in steps); 0 from OLD on."""
    lut = [0] * (DMAX + 1)
    for d in range(OLD):
        lut[d] = min(32768, int(round(scale * math.exp(-d / tau))))
    return lut

class Rule:
    def __init__(self, A2p, A3p, A2m, w_min, w_max, soft=True, A3m=0, lut_p=0, lut_m=1, lut_y=2, lut_x=3):
        self.A2p, self.A3p, self.A2m, self.A3m = A2p, A3p, A2m, A3m
        self.w_min, self.w_max, self.soft = w_min, w_max, soft
        self.inv_range = int(round(65536 / (w_max - w_min)))
        self.lut_p, self.lut_m, self.lut_y, self.lut_x = lut_p, lut_m, lut_y, lut_x

def _apply(rule, p, w, g, ltp):
    if rule.soft:
        s = max(rule.w_max - w, 0) if ltp else max(w - rule.w_min, 0)
        p = (p * (s * rule.inv_range)) >> 16
    p = (p * g) >> 7
    return (p + 128) >> 8

def ltp_dw(rule, luts, d_pre, isi_post, w, g):
    if d_pre == 0:
        return 0
    x = luts[rule.lut_p][d_pre]; y = luts[rule.lut_y][isi_post]
    a = rule.A2p + ((rule.A3p * y) >> 15)
    return _apply(rule, (a * x) >> 15, w, g, True)

def ltd_dw(rule, luts, d_post, isi_pre, w, g):
    if d_post == 0:
        return 0
    o = luts[rule.lut_m][d_post]; r = luts[rule.lut_x][isi_pre]
    a = rule.A2m + ((rule.A3m * r) >> 15)
    return _apply(rule, (a * o) >> 15, w, g, False)

def clamp(w, rule):
    return max(rule.w_min, min(rule.w_max, w))

class NeuronStamps:
    """On-chip neuron state: 12-bit stamps + 12-bit ISI, clear, background scrub."""
    def __init__(self, n):
        self.n = n; self.stamp = [0] * n; self.isi = [DMAX] * n; self.t = 0; self.scrub_next = 0
    def clear(self):
        self.stamp = [(self.t - OLD) % (DMAX + 1)] * self.n; self.isi = [DMAX] * self.n
    def d(self, k):
        return (self.t - self.stamp[k]) % (DMAX + 1)
    def spike(self, k):
        dk = self.d(k)
        self.isi[k] = dk if dk < OLD else DMAX
        self.stamp[k] = self.t % (DMAX + 1)
    def advance(self):
        self.t += 1
        for _ in range(-(-self.n // SCRUB_PERIOD)):
            k = self.scrub_next
            if self.d(k) >= OLD:
                self.stamp[k] = (self.t - OLD) % (DMAX + 1)
            self.scrub_next = (k + 1) % self.n

class AxonStamps:
    """HBM axon state: 32-bit stamps, no wrap, no scrub."""
    NEVER = None
    def __init__(self, n):
        self.n = n; self.stamp = [None] * n; self.isi = [DMAX] * n; self.t = 0
    def clear(self):
        self.stamp = [None] * self.n; self.isi = [DMAX] * self.n
    def d(self, k):
        return DMAX if self.stamp[k] is None else min(self.t - self.stamp[k], DMAX)
    def spike(self, k):
        self.isi[k] = self.d(k)
        self.stamp[k] = self.t
    def advance(self):
        self.t += 1

def run_synapse(rule, luts, pre_spikes, post_spikes, steps, w0, g=128, pre_is_axon=False):
    """One synapse, quantized: returns the weight after each step."""
    pre = AxonStamps(1) if pre_is_axon else NeuronStamps(1)
    post = NeuronStamps(1)
    pre.clear(); post.clear()
    w, trace = w0, []
    P, Q = set(pre_spikes), set(post_spikes)
    for t in range(steps):
        fired_pre, fired_post = t in P, t in Q
        if fired_pre: pre.spike(0)
        if fired_post: post.spike(0)
        if fired_pre:                                    # LTD pass
            w = clamp(w - ltd_dw(rule, luts, post.d(0), pre.isi[0], w, g), rule)
        if fired_post:                                   # LTP pass
            w = clamp(w + ltp_dw(rule, luts, pre.d(0), post.isi[0], w, g), rule)
        trace.append(w)
        pre.advance(); post.advance()
    return trace

def run_synapse_float(A2p, A3p, A2m, A3m, tp, tm, ty, tx, w_min, w_max, soft, pre_spikes, post_spikes, steps, w0, g=1.0):
    """Same rule in floating point, exact times: the mathematical model the hardware approximates."""
    w, last_pre, prev_pre, last_post, prev_post = float(w0), None, None, None, None
    P, Q, trace = set(pre_spikes), set(post_spikes), []
    R = w_max - w_min
    for t in range(steps):
        fp, fq = t in P, t in Q
        if fp and not fq and last_post is not None:
            r = math.exp(-(t - last_pre) / tx) if last_pre is not None else 0.0
            dm = math.exp(-(t - last_post) / tm) * (A2m + A3m * r) * g
            if soft: dm *= (w - w_min) / R
            w = max(w_min, min(w_max, w - dm))
        if fq and not fp and last_pre is not None:
            y = math.exp(-(t - last_post) / ty) if last_post is not None else 0.0
            dp = math.exp(-(t - last_pre) / tp) * (A2p + A3p * y) * g
            if soft: dp *= (w_max - w) / R
            w = max(w_min, min(w_max, w + dp))
        if fp: last_pre = t
        if fq: last_post = t
        trace.append(w)
    return trace
