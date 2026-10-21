"""
hw_ref_model.py -- bit-exact reference of ONE HiAER-Spike neuron, written directly from
internal_events_processor.v (HiAER-Spike core RTL).  Used by
tests/test_bio_features_hw.py to predict the exact membrane potential the hardware must return.

Per timestep (IEP phase order):
  Phase 0  neuron update, using the currents left by the PREVIOUS step's Phase 2
  Phase 2  this step's input weights are accumulated
  read_membrane() after step() returns V as left at the end of the step.

delta_mode = 0 (exp-PSC), IEP lines 509-526, 1748-1860, 1950-1985:
  Ix' = wrap12((Ix * decay_ex) >>> 12)     12-bit signed currents
  Ii' = wrap12((Ii * decay_in) >>> 12)
  w'  = ((w * decay_w) & 0xFFFF) >> 8      8-bit adaptation
  if refrac > 0:  refrac -= 1, V unchanged                  (currents/w still decay)
  else:
    CUBA  drive = Ix' (+ Ii')
    COBA  drive = wrap24(Ix' * (E_ex - V[11:0])) >>> 12  (+ same for Ii', E_in)
    Vn = V - leak_term + drive_ex + drive_in - w' + noise  (leak>=32 -> 0, shift=-17 -> 0)
    eff = threshold - neuromod_bias
    if Vn > eff (signed):  spike, refrac = refractory_max,
                           V = (Vn - eff) if soft_reset else 0,  w = (w' + delta_w) & 0xFF
    else: V = Vn
  Phase 2: weight >= 0 -> Ix = wrap12(Ix + weight);  weight < 0 -> Ii = wrap12(Ii + weight)
           (sign-routed, independent of dual_synapse_en; accumulates during refractory)

delta_mode = 1 (legacy) is used only for the control test: Phase 2 adds the weight
straight into V, so with no leak/noise and an unreachable threshold V just accumulates.
"""

def _wrap(v, bits):
    m = 1 << bits
    v &= m - 1
    return v - m if v >= (m >> 1) else v

def wrap12(v): return _wrap(v, 12)
def wrap24(v): return _wrap(v, 24)
def wrap32(v): return _wrap(v, 32)

def simulate(n_steps, inputs, p):
    """
    n_steps : number of steps
    inputs  : dict step -> list of weights arriving at the neuron in that step
    p       : dict with keys delta_mode, threshold, refractory_max, soft_reset,
              decay_ex, decay_in, decay_w, delta_w, coba_mode, E_ex, E_in, neuromod_bias
    returns : list of V after each step, list of bools (spiked in that step)
    """
    V, Ix, Ii, w, refrac = 0, 0, 0, 0, 0
    eff = wrap32(p.get('threshold', 1 << 30) - p.get('neuromod_bias', 0))
    Vs, spikes = [], []
    for t in range(n_steps):
        spiked = False
        if p.get('delta_mode', 1) == 0:
            Ixd = wrap12((Ix * p.get('decay_ex', 0)) >> 12)
            Iid = wrap12((Ii * p.get('decay_in', 0)) >> 12)
            wd = ((w * p.get('decay_w', 0)) & 0xFFFF) >> 8
            if refrac > 0:
                refrac -= 1
                w = wd
            else:
                if p.get('coba_mode', 0):
                    v12 = wrap12(V)
                    dex = wrap24(Ixd * (p.get('E_ex', 0) - v12)) >> 12
                    din = wrap24(Iid * (p.get('E_in', 0) - v12)) >> 12
                else:
                    dex, din = Ixd, Iid
                Vn = wrap32(V + dex + din - wd)
                if Vn > eff:
                    spiked = True
                    refrac = p.get('refractory_max', 0)
                    V = wrap32(Vn - eff) if p.get('soft_reset', False) else 0
                    w = (wd + p.get('delta_w', 0)) & 0xFF
                else:
                    V = Vn
                    w = wd
            Ix, Ii = Ixd, Iid
            for wt in inputs.get(t, []):
                if wt >= 0: Ix = wrap12(Ix + wt)
                else:       Ii = wrap12(Ii + wt)
        else:
            for wt in inputs.get(t, []):
                V = wrap32(V + wt)
        Vs.append(V); spikes.append(spiked)
    return Vs, spikes

if __name__ == "__main__":
    # hand-checked: I=1000, decay 3354 -> 818 (3354000/4096=818.8), 669 (818*3354/4096=669.8)
    Vs, _ = simulate(4, {0: [1000]}, dict(delta_mode=0, decay_ex=3354))
    assert Vs == [0, 818, 818 + 669, 818 + 669 + 547], Vs
    # inhibitory: -500*3354/4096 = -409.4 -> floor -410
    Vs, _ = simulate(2, {0: [-500]}, dict(delta_mode=0, decay_in=3354))
    assert Vs == [0, -410], Vs
    # soft reset subtracts the EFFECTIVE threshold; refractory holds V
    Vs, sp = simulate(3, {0: [800]}, dict(delta_mode=0, decay_ex=4095, threshold=500, soft_reset=True))
    assert sp[1] and Vs[1] == 799 - 500, (Vs, sp)
    print("ref_model self-check OK")
