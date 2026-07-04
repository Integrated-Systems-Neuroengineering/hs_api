#!/usr/bin/env python3
"""
Software Patch for crisdsc0 — CORRECTED
=========================================
Patches 5 files:
1. neuron_models.py - Add get_shadow_uram_offset + get_legacy_noise_en to ANN_neuron
2. fpga_controller.py - Add legacy_noise_en parameter, set bit [7]
3. network.py - Pass legacy_noise_en to write_neuron_type
4. test_DVS_small.py - Set legacy_noise_en=1 on loaded neuron models
5. test_DVS_large_fulldataset.py - Same
"""
import sys

# ============================================================
# FILE 1: neuron_models.py
# ============================================================
NM = "/home/omowuyi/testing/hs_api/hs_api/neuron_models.py"
with open(NM, "r") as f:
    nm = f.read()

# 1a. Add legacy_noise_en and shadow_uram_offset to LIF_neuron.__init__
old = "    def __init__(self, threshold, shift, leak, refractory_max=0, dual_synapse_en=False, delay_value=0):"
new = "    def __init__(self, threshold, shift, leak, refractory_max=0, dual_synapse_en=False, delay_value=0, legacy_noise_en=0, shadow_uram_offset=0):"
assert old in nm, f"FAIL 1a: LIF_neuron.__init__ signature not found"
nm = nm.replace(old, new)

old = """        self.delay_value = delay_value

    def get_threshold(self):
        return self.threshold

    def get_shift(self):
        return self.shift

    def get_leak(self):
        return self.leak

    def get_neuronModel(self):
        return 2"""
new_r = """        self.delay_value = delay_value
        self.legacy_noise_en = legacy_noise_en
        self.shadow_uram_offset = shadow_uram_offset

    def get_threshold(self):
        return self.threshold

    def get_shift(self):
        return self.shift

    def get_leak(self):
        return self.leak

    def get_neuronModel(self):
        return 2"""
assert old in nm, f"FAIL 1b: LIF_neuron body not found"
nm = nm.replace(old, new_r)

# 1b. Add get_legacy_noise_en to LIF_neuron (after get_shadow_uram_offset)
old = """    def get_shadow_uram_offset(self):
        return getattr(self, 'shadow_uram_offset', 0)


class ANN_neuron"""
new_r = """    def get_shadow_uram_offset(self):
        return getattr(self, 'shadow_uram_offset', 0)

    def get_legacy_noise_en(self):
        return getattr(self, 'legacy_noise_en', 0)


class ANN_neuron"""
assert old in nm, f"FAIL 1c: LIF get_shadow_uram_offset block not found"
nm = nm.replace(old, new_r)

# 1c. Add get_shadow_uram_offset and get_legacy_noise_en to ANN_neuron
# ANN_neuron has duplicate get_delay_value - find the LAST occurrence
old = """    def get_delay_value(self):
        return getattr(self, 'delay_value', 0)"""
idx = nm.rfind(old)
assert idx != -1, f"FAIL 1d: ANN get_delay_value not found"
replacement = """    def get_delay_value(self):
        return getattr(self, 'delay_value', 0)

    def get_shadow_uram_offset(self):
        return getattr(self, 'shadow_uram_offset', 0)

    def get_legacy_noise_en(self):
        return getattr(self, 'legacy_noise_en', 0)"""
nm = nm[:idx] + replacement + nm[idx + len(old):]

with open(NM, "w") as f:
    f.write(nm)
print("[1/5] neuron_models.py — OK")

# ============================================================
# FILE 2: fpga_controller.py
# ============================================================
FC = "/home/omowuyi/testing/hs_bridge/hs_bridge/FPGA_Execution/fpga_controller.py"
with open(FC, "r") as f:
    fc = f.read()

old = "def write_neuron_type(stopAddr, Threshold, neuronModel, shift, leak, refractory_max=0, dual_synapse_en=0, delay_value=0, shadow_uram_offset=0, coreID=0, simDump=False):"
new = "def write_neuron_type(stopAddr, Threshold, neuronModel, shift, leak, refractory_max=0, dual_synapse_en=0, delay_value=0, shadow_uram_offset=0, legacy_noise_en=0, coreID=0, simDump=False):"
assert old in fc, f"FAIL 2a: write_neuron_type signature not found"
fc = fc.replace(old, new)

old = 'print(f"[DEBUG write_neuron_type] stopAddr={stopAddr} boundary={(stopAddr+15)//16} Threshold={Threshold} model={neuronModel} shift={shift} leak={leak} refr={refractory_max} delay={delay_value} dual_syn={dual_synapse_en} shadow_off={shadow_uram_offset}")'
new = 'print(f"[DEBUG write_neuron_type] stopAddr={stopAddr} boundary={(stopAddr+15)//16} Threshold={Threshold} model={neuronModel} shift={shift} leak={leak} refr={refractory_max} delay={delay_value} dual_syn={dual_synapse_en} shadow_off={shadow_uram_offset} legacy_noise={legacy_noise_en}")'
assert old in fc, f"FAIL 2b: debug print not found"
fc = fc.replace(old, new)

old = "    command[-13:-9] = list(np.binary_repr(shadow_uram_offset, 4)) #12-9: shadow_uram_offset"
new = "    command[-13:-9] = list(np.binary_repr(shadow_uram_offset, 4)) #12-9: shadow_uram_offset\n    command[-8:-7] = list(np.binary_repr(int(legacy_noise_en), 1)) #7: legacy_noise_en (1=2024 noise+35bit MP)"
assert old in fc, f"FAIL 2c: shadow_uram_offset command not found"
fc = fc.replace(old, new)

with open(FC, "w") as f:
    f.write(fc)
print("[2/5] fpga_controller.py — OK")

# ============================================================
# FILE 3: network.py
# ============================================================
NW = "/home/omowuyi/testing/hs_bridge/hs_bridge/network.py"
with open(NW, "r") as f:
    nw = f.read()

old = "shadow_uram_offset=neuronModel.get_shadow_uram_offset(), coreID=self.coreOveride, simDump = True)"
new = "shadow_uram_offset=neuronModel.get_shadow_uram_offset(), legacy_noise_en=neuronModel.get_legacy_noise_en(), coreID=self.coreOveride, simDump = True)"
assert old in nw, f"FAIL 3a: simDump=True call not found"
nw = nw.replace(old, new)

old = "shadow_uram_offset=neuronModel.get_shadow_uram_offset(), coreID=self.coreOveride)"
new = "shadow_uram_offset=neuronModel.get_shadow_uram_offset(), legacy_noise_en=neuronModel.get_legacy_noise_en(), coreID=self.coreOveride)"
assert old in nw, f"FAIL 3b: hardware call not found"
nw = nw.replace(old, new)

with open(NW, "w") as f:
    f.write(nw)
print("[3/5] network.py — OK")

# ============================================================
# FILE 4: test_DVS_small.py  (modelIdx=1 per api.py line 13)
# ============================================================
DVS_SMALL = "/home/omowuyi/testing/hs_api/tests/test_DVS_small.py"
with open(DVS_SMALL, "r") as f:
    dvs = f.read()

old = """        # Create network
        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="CRI"
        )"""
new = """        # Enable 2024 noise mode for DVS model (shift=-17 needs legacy noise behavior)
        for key in connections:
            neuron_obj = connections[key][1]  # modelIdx=1 per api.py line 13
            neuron_obj.legacy_noise_en = 1

        # Create network
        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="CRI"
        )"""
assert old in dvs, f"FAIL 4: test_DVS_small.py CRI_network block not found"
dvs = dvs.replace(old, new)

with open(DVS_SMALL, "w") as f:
    f.write(dvs)
print("[4/5] test_DVS_small.py — OK")

# ============================================================
# FILE 5: test_DVS_large_fulldataset.py
# ============================================================
DVS_LARGE = "/home/omowuyi/testing/hs_api/tests/test_DVS_large_fulldataset.py"
with open(DVS_LARGE, "r") as f:
    dvsl = f.read()

target = "        network = CRI_network("
assert target in dvsl, f"FAIL 5: test_DVS_large_fulldataset.py CRI_network not found"

insert = """        # Enable 2024 noise mode for DVS model (shift=-17 needs legacy noise behavior)
        for key in connections:
            neuron_obj = connections[key][1]  # modelIdx=1 per api.py line 13
            neuron_obj.legacy_noise_en = 1

"""
dvsl = dvsl.replace(target, insert + target, 1)

with open(DVS_LARGE, "w") as f:
    f.write(dvsl)
print("[5/5] test_DVS_large_fulldataset.py — OK")

print("\n=== ALL 5 FILES PATCHED ===")
print("Run: find /home/omowuyi/testing -name '__pycache__' -exec rm -rf {} + 2>/dev/null")
