#!/usr/bin/env python3
"""
Complete api.py patch for axonal delay support.
Applies to CLEAN fb811b4 api.py (after git checkout).

Changes:
  1. Validation: accept 3-element synapse tuples
  2. gen_connectome: skip delayed synapses, store separately
  3. __init__: add delay initialization + setup after gen_connectome
  4. Add _setup_delay_axons method (creates synthetic axons, rebuilds connectome)
  5. step(): inject delayed activations, track countdowns, filter delay axon spikes
"""

IEP_PATH = "/home/omowuyi/testing/hs_api/hs_api/api.py"

with open(IEP_PATH, "r") as f:
    content = f.read()

changes = 0

# ============================================================
# 1. Validation: accept 3-element tuples (may already be done)
# ============================================================
if "len(values) == 2)" in content:
    content = content.replace(
        "len(values) == 2)",
        "len(values) in (2, 3))"
    )
    changes += 1
    print(f"  [{changes}] Validation: accept 3-element tuples")
else:
    print(f"  [skip] Validation already accepts 3-element tuples")

# ============================================================
# 2. gen_connectome axon synapse loop: skip delayed, store separately
# ============================================================
old_axon_loop = """        for axonKey in self.userAxons:
            synapses = self.userAxons[axonKey]
            for axonSynapse in synapses:
                weight = axonSynapse[1]
                postsynapticNeuron = self.connectome.connectomeDict[axonSynapse[0]]
                self.connectome.connectomeDict[axonKey].addSynapse(
                    postsynapticNeuron, weight
                )"""

new_axon_loop = """        for axonKey in self.userAxons:
            synapses = self.userAxons[axonKey]
            for axonSynapse in synapses:
                weight = axonSynapse[1]
                delayed = axonSynapse[2] if len(axonSynapse) == 3 else False
                postsynapticNeuron = self.connectome.connectomeDict[axonSynapse[0]]
                if delayed and hasattr(self, '_delayed_synapses'):
                    self._delayed_synapses.append((axonKey, axonSynapse[0], weight))
                else:
                    self.connectome.connectomeDict[axonKey].addSynapse(
                        postsynapticNeuron, weight
                    )"""

if old_axon_loop in content:
    content = content.replace(old_axon_loop, new_axon_loop)
    changes += 1
    print(f"  [{changes}] gen_connectome axon loop: skip delayed synapses")
else:
    print("  ERROR: axon synapse loop not found!")
    import sys; sys.exit(1)

# ============================================================
# 3. gen_connectome neuron synapse loop: skip delayed, store separately
# ============================================================
old_neuron_loop = """        for neuronKey in self.userConnections:
            # breakpoint()
            synapses = self.userConnections[neuronKey][synapseIdx]
            # breakpoint()
            for neuronSynapse in synapses:
                weight = neuronSynapse[1]
                postsynapticNeuron = self.connectome.connectomeDict[neuronSynapse[0]]
                self.connectome.connectomeDict[neuronKey].addSynapse(
                    postsynapticNeuron, weight
                )"""

new_neuron_loop = """        for neuronKey in self.userConnections:
            # breakpoint()
            synapses = self.userConnections[neuronKey][synapseIdx]
            # breakpoint()
            for neuronSynapse in synapses:
                weight = neuronSynapse[1]
                delayed = neuronSynapse[2] if len(neuronSynapse) == 3 else False
                postsynapticNeuron = self.connectome.connectomeDict[neuronSynapse[0]]
                if delayed and hasattr(self, '_delayed_synapses'):
                    self._delayed_synapses.append((neuronKey, neuronSynapse[0], weight))
                else:
                    self.connectome.connectomeDict[neuronKey].addSynapse(
                        postsynapticNeuron, weight
                    )"""

if old_neuron_loop in content:
    content = content.replace(old_neuron_loop, new_neuron_loop)
    changes += 1
    print(f"  [{changes}] gen_connectome neuron loop: skip delayed synapses")
else:
    print("  ERROR: neuron synapse loop not found!")
    import sys; sys.exit(1)

# ============================================================
# 4. __init__: add delay initialization before gen_connectome
#    and _setup_delay_axons() after it
# ============================================================
old_init = """        self.connectome = None
        self.gen_connectome()"""

new_init = """        self.connectome = None
        self._delayed_synapses = []
        self._delay_queue = []
        self._delay_map = {}
        self.gen_connectome()
        self._setup_delay_axons()"""

if old_init in content:
    content = content.replace(old_init, new_init, 1)
    changes += 1
    print(f"  [{changes}] __init__: delay initialization + setup call")
else:
    print("  ERROR: __init__ gen_connectome pattern not found!")
    import sys; sys.exit(1)

# ============================================================
# 5. Add _setup_delay_axons method before __format_input
# ============================================================
old_format = """    def __format_input(self, axons, connections):"""

new_format = """    def _setup_delay_axons(self):
        \"\"\"Create synthetic axons for delayed (Group B) synapse delivery.
        When a neuron with delayed synapses spikes, the software delay queue
        schedules the synthetic axon activation after delay_value timesteps.\"\"\"
        if not self._delayed_synapses:
            return
        # For each delayed synapse, create a synthetic axon in userAxons
        for source_key, target_key, weight in self._delayed_synapses:
            axon_key = "_DELAY_" + source_key + "_" + target_key
            self.userAxons[axon_key] = [(target_key, weight)]
            # Track delay mapping: source neuron -> list of (axon_key, delay_value)
            delay_value = self.userConnections[source_key][1].get_delay_value()
            if source_key not in self._delay_map:
                self._delay_map[source_key] = []
            self._delay_map[source_key].append((axon_key, delay_value))
        # Rebuild connectome with synthetic axons included
        self._delayed_synapses = []  # clear to prevent re-adding during rebuild
        self.gen_connectome()

    def __format_input(self, axons, connections):"""

if old_format in content:
    content = content.replace(old_format, new_format, 1)
    changes += 1
    print(f"  [{changes}] Added _setup_delay_axons method")
else:
    print("  ERROR: __format_input not found!")
    import sys; sys.exit(1)

# ============================================================
# 6. step() CRI non-membranePotential path: inject delayed activations
#    and track countdowns after spike results
# ============================================================
old_step_cri = """                else:
                    # breakpoint()
                    spikeResult = self.CRI.run_step(formated_inputs, membranePotential)
                    # breakpoint()
                    spikeList = spikeResult[0]
                    spikeList = [
                        self.connectome.get_neuron_by_hbmIdx(spike[1]).get_user_key()
                        for spike in spikeList
                    ]
                    return (spikeList, spikeResult[1], spikeResult[2])"""

new_step_cri = """                else:
                    # Inject delayed axon activations whose countdown reached 0
                    if hasattr(self, '_delay_queue') and self._delay_queue:
                        due_axons = [akey for akey, ctr in self._delay_queue if ctr <= 0]
                        for akey in due_axons:
                            axon_obj = self.connectome.connectomeDict.get(akey)
                            if axon_obj:
                                formated_inputs.append(axon_obj.coreTypeIdx)
                        self._delay_queue = [(a, c) for a, c in self._delay_queue if c > 0]
                    spikeResult = self.CRI.run_step(formated_inputs, membranePotential)
                    spikeList = spikeResult[0]
                    spikeList_filtered = []
                    for spike in spikeList:
                        try:
                            key = self.connectome.get_neuron_by_hbmIdx(spike[1]).get_user_key()
                            if not key.startswith("_DELAY_"):
                                spikeList_filtered.append(key)
                        except (KeyError, IndexError):
                            pass
                    # Track delayed spikes: if a spiked neuron has delayed synapses, start countdown
                    if hasattr(self, '_delay_map'):
                        for spiked_key in spikeList_filtered:
                            if spiked_key in self._delay_map:
                                for akey, dval in self._delay_map[spiked_key]:
                                    self._delay_queue.append((akey, dval))
                        # Decrement all countdowns
                        self._delay_queue = [(a, c - 1) for a, c in self._delay_queue]
                    return (spikeList_filtered, spikeResult[1], spikeResult[2])"""

if old_step_cri in content:
    content = content.replace(old_step_cri, new_step_cri)
    changes += 1
    print(f"  [{changes}] step() CRI path: delay injection + tracking + filtering")
else:
    print("  ERROR: step() CRI path not found!")
    import sys; sys.exit(1)

# ============================================================
# Write result
# ============================================================
with open(IEP_PATH, "w") as f:
    f.write(content)

print(f"\nAll {changes} changes applied to api.py successfully.")
print("Axonal delay: synthetic axon approach (no shadow neurons needed)")
