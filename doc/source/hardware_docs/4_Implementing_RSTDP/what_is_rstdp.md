# 4.1 What is R-STDP?

## Spike-Timing Dependent Plasticity (STDP)

STDP is a biologically plausible synaptic weight update rule based on the relative timing of pre- and post-synaptic spikes. It implements Hebbian learning: "cells that fire together wire together."

**The STDP Rule:**

$$\Delta t = t_{post} - t_{pre}$$

$$W(\Delta t) = \begin{cases}
A_+ e^{-\Delta t/\tau_+} & \text{if } \Delta t \geq 0 \\
-A_- e^{\Delta t/\tau_-} & \text{if } \Delta t < 0
\end{cases}$$

- **If the pre-synaptic neuron fires before the post-synaptic neuron** (Δt > 0): The synapse is **strengthened** (potentiation)
- **If the pre-synaptic neuron fires after the post-synaptic neuron** (Δt < 0): The synapse is **weakened** (depression)

The weight change decays exponentially as the time difference increases, creating a critical window (typically tens of milliseconds) during which timing matters.

---

## The Problem: Delayed Rewards

Standard STDP has a fundamental limitation: it can only associate events that occur within a narrow time window. But in the real world, **rewards often arrive long after the neural activity that caused them**. For example:
- A rat navigates a maze → finds food 10 seconds later
- A robot takes an action → receives feedback 100 timesteps later

By the time the reward arrives, the STDP window has closed. Which synapses should be updated?

---

## The Eligibility Trace: Marking Synapses for Learning

An **eligibility trace** is a decaying memory variable associated with each synapse that tracks recent pre/post-synaptic activity. Think of it as a "tag" that marks synapses saying: *"We recently had correlated activity here."*

**Mathematical form:**

$$\dot{c}(t) = -\frac{c}{\tau_c} + W(\Delta t)\delta(t - s_{pre/post})C_1$$

where:
- $c(t)$ is the eligibility trace
- $\tau_c$ is the decay time constant (typically hundreds of milliseconds)
- $W(\Delta t)$ is the STDP rule
- $\delta(t - s_{pre/post})$ captures spike events
- $C_1$ is a scaling constant

**Key insight:** The eligibility trace has a **much longer time constant** than the STDP window. While the STDP window might be 20-50ms, the eligibility trace can persist for hundreds of milliseconds or even seconds.

When pre and post neurons spike with appropriate timing:
1. The STDP rule computes a weight change magnitude based on spike timing
2. This value is added to the eligibility trace
3. The trace decays slowly over time

**Visual Example (Coincidence Detection):**

```
pre    ║  │  │      │            500 ms
post   ║  │  │      │
       ┊         ┊
       └─────────┴─────────────────────
         eligibility trace, c(t)
```

When both neurons spike together, the eligibility trace increases. It then decays gradually, persisting well beyond the STDP window.

---

## Reward-Modulated STDP (R-STDP)

R-STDP combines eligibility traces with a reward signal (typically dopamine in biological systems) to solve the delayed reward problem:

$$\dot{w}(t) = R(t) \times c(t)$$

where:
- $w(t)$ is the synaptic weight
- $R(t)$ is the reward signal (dopamine)
- $c(t)$ is the eligibility trace

**How it works:**

1. **Neural activity happens:** Pre and post neurons spike, creating an eligibility trace that says "this synapse was recently active with appropriate timing"

2. **Eligibility trace persists:** The trace decays slowly (τ_c ~ 100-1000 ms), maintaining a memory of which synapses had recent STDP-eligible activity

3. **Reward arrives (delayed):** When reward $R(t)$ arrives—potentially seconds later—it modulates learning across ALL synapses with non-zero eligibility traces

4. **Weight updates occur:** Only synapses that (a) had recent correlated activity AND (b) receive the reward signal actually change their weights

**The role of dopamine (in biological systems):**
- **Positive reward prediction error** → dopamine burst → strengthen eligible synapses
- **Negative reward prediction error** → dopamine dip → weaken eligible synapses
- **Zero reward prediction error** → no dopamine change → no learning

---

## Why This Solves the Delayed Reward Problem

The eligibility trace acts as a **bridge across time**:

```
Time:     0ms         50ms        100ms       500ms       1000ms
          │           │           │           │           │
Activity: [Pre→Post]  [STDP       [eligibility trace      │
          spike       window      still elevated...]       │
          timing      closes]                              │
                                                           [REWARD!]
                                                           └─> Updates all
                                                               eligible synapses
```

Without eligibility traces, the reward at 1000ms would have no way to know which synapses (out of potentially millions) were involved in the action that led to the reward.

With eligibility traces, the system automatically "remembers" which synapses had recent activity patterns consistent with STDP, and **only those synapses are eligible for modification** when reward arrives.

---

## Summary

**R-STDP = STDP + Eligibility Trace + Reward Signal**

- **STDP** provides the local plasticity rule based on spike timing
- **Eligibility traces** extend the memory of synaptic activity beyond the STDP window
- **Reward signals** (dopamine) determine whether eligible synapses are actually modified

This three-component system allows neural networks to:
1. Learn from delayed rewards
2. Credit assignment: figure out which synapses were responsible for outcomes
3. Implement reinforcement learning in a biologically plausible way

