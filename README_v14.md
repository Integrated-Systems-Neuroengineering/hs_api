# hs_api for the v14 bitstream

Use this branch with `hs_bridge` `v14-multicore` and `connectome_utils` `v12b_tested`; the `hs_bridge` README explains the setup and the bitstream.

## What changed

`CRI_network` takes three more arguments: `membership` and `n_cores` place a network on several cores, and `plasticity` passes plasticity rules to the hardware. Spikes read from the card are reported with the network's own core numbering.

## Hardware tests

Every test in `tests/` whose name ends in `_hw.py` runs on the board with the v14 bitstream loaded. `HIAER_CORE_ID` chooses the core:

```
HIAER_CORE_ID=3 pytest -s tests/test_fanin_9000_hw.py
```

The hardware suite runs these tests on every core, with a fresh flash for each: `hardware_suite/hw_v12_sweep.sh` on the `v12b_tested` branch, whose default bitstream is v14. It finishes with `ALL PASS ON EVERY CORE`.

The suite's tests are `test_bio_features_hw.py`, `test_bitstream_hardware_fast.py`, `test_delay_overflow_hw.py`, `test_fanin_9000_hw.py`, `test_heavy_spike_psc_hw.py`, `test_plasticity_protocol_hw.py`, `test_syn64_isolated_hw.py` and `test_syn64_long_delay_hw.py`; `rec29_common.py` is shared by two of them. Also here: `test_microphase_hw.py`, `test_delay_buffers_hw.py`, `test_cmd2_readback_hw.py`, `test_relay_net_hw.py`, `test_bio_noflush_hw.py` and `test_DVS_ch10_nomp.py`.

Christopher Deng's DVS spike tests (`test_DVS_ch=10_conv=2_spikes.py` and `test_DVS_ch=10_conv=2_spikes_indepth.py`) are on his `L6m-testing-suite` branch.

## Verification

On the board: V14 CARD ACCEPTANCE: PASS on 2026-11-13: Christopher Deng's two DVS spike tests, equivalence across 1, 4 and 16 cores, bio and plasticity, concentrated spikes (R48b, R48c), count packets (R57), the single-core suite on all 16 cores, and 12 large networks up to 122,891 neurons including 10,000 neurons on one core all firing, every spike and membrane exact
