# Hardware Test Results

This table records the results of running `tests/test_bitstream_hardware.py` against physical FPGA hardware for each bitstream release. Each row corresponds to a specific bitstream tested against a snapshot of the software dependencies, identified by commit hash. Test results are recorded as `p` (pass) or `f` (fail) for each test in the suite.

```
| Bitstream | hs_api commit hash | hs_bridge commit hash | connectome_utils commit hash | fxpmath commit hash | test_max_number_axons | test_number_axons_multiple_256 | test_number_axons_not_multiple_256 | test_2layers_no_input | test_LIF_neuron_negative_input | test_spike_readout | test_LIF_neuron_no_noise | test_LIF_neuron_with_noise | test_network_reset | test_max_axonal_fanout | test_max_axonal_fan_in | test_max_neuronal_fanout | test_neuronal_fan_in |
|-----------|-------------------|----------------------|------------------------------|---------------------|----------------------|-------------------------------|-----------------------------------|----------------------|-------------------------------|-------------------|-------------------------|--------------------------|-------------------|----------------------|----------------------|------------------------|---------------------|
| Example_release | a1b2c3d | 9f8e7d6 | 3c4d5e6 | 7a8b9c0 | p | f | f | p | f | p | f | f | p | f | p | f | p |
```
