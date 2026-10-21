# Hardware tests

How to run the hardware test suites in `tests/` on the FPGA card in crisdsc0: the regression suite on any core, the biological neuron model tests, the per-core sweep, and the NoC tests.

| File | What it tests |
|---|---|
| `test_bitstream_hardware_fast.py` | Regression suite, 42 tests, on the core given by `HIAER_CORE_ID` |
| `test_bio_features_hw.py` | Biological neuron model, 13 tests, compared step by step with `hw_ref_model.py` |
| `hw_ref_model.py` | Exact model of one neuron's update, used by `test_bio_features_hw.py` (not a test file) |
| `run_percore_sweep.sh` | Both suites above on a list of cores, with one log per core and suite |
| `test_noc_hw.py` | Spike delivery between cores: same cluster, across clusters, multicast, ring |

All commands below are run from the `hs_api` checkout, in the environment from section 1.

## 1. Repositories and environment

The tests need four repositories side by side, as `pyproject.toml` expects:

| Repository | Used for |
|---|---|
| `hs_api` | network API, CMD 13 interface (`hs_api/single_core_conductance_STDP.py`), these tests |
| `hs_bridge` | FPGA command layer, DMA extension, `noc_routing.py` |
| `connectome_utils` | network partitioning across cores |
| `fxpmath` | fixed-point arithmetic used by `hs_api` |

### Option A: the shared tree on crisdsc0

The four repositories are already in `/home/omowuyi/testing`, with the DMA extension built and a virtual environment in `hs_api/.venv`. Each session:

```
cd /home/omowuyi/testing/hs_api
```

```
source .venv/bin/activate
```

```
export PYTHONPATH=/home/omowuyi/testing/hs_bridge
```

`PYTHONPATH` puts the `hs_bridge` checkout itself on the path, which is where `noc_routing.py` lives.

To see which branch of each repository you are testing:

```
for r in hs_api hs_bridge connectome_utils; do echo "$r: $(git -C /home/omowuyi/testing/$r branch --show-current) $(git -C /home/omowuyi/testing/$r log --oneline -1)"; done
```

### Option B: your own checkout on crisdsc0

Clone `hs_api`, `hs_bridge`, `connectome_utils` and `fxpmath` into one directory, check out the same branches as the shared tree (command above), then from `hs_api`:

```
poetry install --with fpga
```

Build the DMA extension once, from `hs_bridge/hs_bridge/wrapped_dmadump`:

```
python setup.py build_ext --inplace
```

Each session, from `hs_api`:

```
poetry shell
```

```
export PYTHONPATH=$(realpath ../hs_bridge)
```

### Card access

- Flashing needs `sudo` on crisdsc0.
- Running tests needs read and write access to `/dev/adxdma0*`, which belong to the `crisdsc` group. Check with `id`; if `crisdsc` is missing, ask for it to be added.

## 2. Flashing

Bitstream for the regression and biological tests: `/bitstreams/sixteen_core_noc_only_v7.bit`. With this bitstream, test cores **1 to 15**.

```
sudo /bitstreams/scripts/flash.sh --pcie-addr 0000:81:00.0 /bitstreams/sixteen_core_noc_only_v7.bit
```

The last line must be `Bitstream programming completed successfully`.

The NoC tests use the NoC test bitstream instead (section 7).

## 3. Checks without the card

The CMD 13 interface checks its own encoding:

```
python -c "from hs_api import single_core_conductance_STDP as p2; p2.self_test()"
```

Expected last line: `=== ALL TESTS PASSED ===`

pytest can list the hardware tests without touching the card:

```
python -m pytest tests/test_bio_features_hw.py tests/test_noc_hw.py --collect-only -q
```

Expected: 13 biological tests and 4 NoC tests listed.

## 4. Regression suite on one core

`HIAER_CORE_ID` selects the core. Without it, everything runs on core 0.

```
HIAER_CORE_ID=1 timeout 600 python -m pytest tests/test_bitstream_hardware_fast.py -q
```

Expected: `42 passed` in about one minute.

## 5. Biological neuron model on one core

```
HIAER_CORE_ID=1 timeout 900 python -m pytest tests/test_bio_features_hw.py -v
```

Expected (about two minutes):
```
tests/test_bio_features_hw.py::test_F00_control_delta_mode PASSED
tests/test_bio_features_hw.py::test_feature[F01_exp_psc_excitatory] PASSED
...
tests/test_bio_features_hw.py::test_feature[F12_syn64_delay_2] PASSED
13 passed
```

To see the hardware and reference traces of every test, add `-s`:

```
HIAER_CORE_ID=1 timeout 900 python -m pytest tests/test_bio_features_hw.py -v -s
```

To save all traces to a file:

```
HIAER_CORE_ID=1 HIAER_BIO_TRACES=tests/bio_traces_core1.json timeout 900 python -m pytest tests/test_bio_features_hw.py -q
```

To run selected tests, use `-k` with the test IDs (F00 always runs first, because it measures the read timing the other tests use):

```
HIAER_CORE_ID=1 timeout 900 python -m pytest tests/test_bio_features_hw.py -v -k "F01 or F06 or F10"
```

## 6. All cores: the sweep

Runs the regression suite and the biological tests on every listed core, about three minutes per core. The first argument is a tag used in the log names. Run it under `nohup` so it keeps going if the SSH session drops:

```
nohup bash tests/run_percore_sweep.sh v7 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 > tests/sweep_console.txt 2>&1 &
```

Check progress at any time:

```
cat tests/pytest_v7_sweep_summary.log
```

Each core adds:
```
===== core 3 =====
  regression  : 42 passed, 1 warning in 59.8s
  bio features: 13 passed in 118.2s
```

Per core N, `tests/` then holds `pytest_v7_core<N>.log`, `pytest_v7_bio_core<N>.log` and `bio_traces_v7_core<N>.json`.

## 7. NoC tests

Flash the NoC test bitstream (its path comes with it), then:

```
HIAER_NOC_TESTS=1 timeout 1200 python -m pytest tests/test_noc_hw.py -v -s
```

Without `HIAER_NOC_TESTS=1` the four tests are reported as skipped.

Default cores: source 0, same-cluster destination 1, other-cluster destination 12, multicast to 1, 2 and 3, a ring through all 16 cores, and a control neuron on core 5. Change them with environment variables, for example:

```
HIAER_NOC_TESTS=1 HIAER_NOC_SRC=1 HIAER_NOC_L1=2 HIAER_NOC_L2=13 HIAER_NOC_MCAST=2,3,5 HIAER_NOC_RING=1-8 timeout 1200 python -m pytest tests/test_noc_hw.py -v -s
```

| Variable | Default | Meaning |
|---|---|---|
| `HIAER_NOC_SRC` | 0 | core of the source neuron |
| `HIAER_NOC_L1` | 1 | destination in the same 4-core cluster |
| `HIAER_NOC_L2` | 12 | destination in another cluster |
| `HIAER_NOC_MCAST` | 1,2,3 | multicast destinations |
| `HIAER_NOC_RING` | 0-15 | ring cores, in order (list or range) |
| `HIAER_NOC_CTRL` | 5 | core of the control neuron, which must receive nothing |
| `HIAER_NOC_STEPS` | 10 | steps per delivery test |

How each test decides:
- **Same cluster, across clusters, multicast:** a source neuron is driven every step and fires every step; it connects with weight 50 to each destination neuron, which only accumulates. Each destination must end with V = 50 × k, with k at least 1 and within 2 of the source's spike count. The control neuron must read 0.
- **Ring:** one input starts a spike that passes from each ring neuron to the next. Every ring neuron must fire, in ring order.

With `-s`, each test prints the spike count and every destination's V.

## 8. What each biological test sets

Each test builds one neuron `N0` driven by axon `A0` (and `A1` in F04), configures the core with CMD 13, runs the steps, reads N0's membrane potential after every step, and compares every value with `hw_ref_model.py`. **A test passes only if every step matches exactly.**

| Test | Feature | Settings |
|---|---|---|
| F00 | control, legacy delta mode | weight 1000 once; measures the read timing |
| F01 | exponential PSC, excitatory | decay_ex 3354, weight 1000 once |
| F02 | decay constant | decay_ex 2048 (compare with F01) |
| F03 | inhibitory PSC | decay_in 3354, weight -600 once |
| F04 | dual time constant | +800 with decay_ex 3800 and -400 with decay_in 1500 |
| F05 | COBA conductance | coba_mode 1, E_ex 1500, E_in -1500 |
| F06 | AdEx adaptation | threshold 600, decay_w 128, delta_w 40, weight 300 every step |
| F07 | neuromodulation, bias +100 | threshold 1000, effective threshold 900 |
| F08 | neuromodulation, bias -100 | threshold 1001, effective threshold 1101 |
| F09 | refractory period | refractory_max 3 |
| F10 | soft reset | after a spike V keeps V minus the effective threshold |
| F11 | 64-bit synapse format | must reproduce F01 exactly |
| F12 | per-synapse delay | delay 2: F01's response two steps later |

Neuron rules the reference follows: an input weight enters the synaptic current in the step it arrives and reaches V from the next step; synaptic currents are 12-bit signed, so weights must stay below 2048; a neuron in its refractory period holds V; a spike resets V to 0, or to V minus the effective threshold with soft reset; effective threshold = threshold minus the neuromodulation bias. Each test uses its own threshold value, which makes the core clear all neuron state before the test's first step, so nothing carries over between tests.

## 9. Reading results

- **PASSED:** every step matched the reference.
- **FAILED:** the message gives the first step that differs, the hardware value, the expected value, and both full traces, for example:
  ```
  Failed: decay_constant on core 1: first difference at step 2: hardware 501, expected 500
    hw    = [0, 0, 501, ...]
    model = [0, 500, 750, ...]
  ```
- If **F00** fails, every other test on that core errors out, because nothing can be judged without the control. Report the core number and try another core.
- The JSON trace file holds, per test, the status, the measured read timing, and the hardware and reference traces.

## 10. CMD 13 from your own code

`hs_api.single_core_conductance_STDP` configures the biological model of any core:

```
python -c "from hs_api import single_core_conductance_STDP as p2; p2.send_cmd13_psc_params(coreID=1, delta_mode=0, decay_ex=3354, decay_in=3354)"
```

Calling it with only `coreID` returns that core to the reset defaults (`delta_mode=1`):

```
python -c "from hs_api import single_core_conductance_STDP as p2; p2.send_cmd13_psc_params(coreID=1)"
```

`build_cmd13_packet(...)` returns the packet without sending it, and `decode_cmd13_packet(packet)` reads every field back, including `coreID`.

## 11. When a run stops

Every command above is wrapped in `timeout`, so a run that stops responding is ended. Flash again before the next run:

```
sudo /bitstreams/scripts/flash.sh --pcie-addr 0000:81:00.0 /bitstreams/sixteen_core_noc_only_v7.bit
```

If the flash reports that the driver module is in use, reboot crisdsc0 and flash again:

```
sudo reboot
```

Do not let the computer you connect from suspend or hibernate while a flash or a test is running. Losing the SSH session in the middle of a flash leaves the card unusable until crisdsc0 is rebooted. Use `nohup` (section 6) for anything long.

## 12. What to send back

- `tests/pytest_<tag>_sweep_summary.log`, `tests/pytest_<tag>_core*.log`, `tests/pytest_<tag>_bio_core*.log`
- `tests/bio_traces_<tag>_core*.json`
- the console output of any single run
