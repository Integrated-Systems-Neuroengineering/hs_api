#!/usr/bin/env python3

from abc import ABC, abstractmethod


class neuron_model(ABC):
    @abstractmethod
    def get_threshold(self):
        pass

    @abstractmethod
    def get_neuronModel(self):
        pass

    @abstractmethod
    def get_shift(self):
        pass

    @abstractmethod
    def get_leak(self):
        pass

    def __hash__(self):
        return hash(
            (
                self.get_threshold(),
                self.get_neuronModel,
                self.get_shift(),
                self.get_leak(),
            )
        )

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __le__(self, other):
        return hash(self) <= hash(other)


class LIF_neuron(neuron_model):
    """
    LIF neuron model.

    shift: int
        noise pertubation magnitude

    """

    def __init__(self, threshold, shift = -17, leak = 63, refractory_max=0, dual_synapse_en=False, delay_value=0, shadow_uram_offset=0, legacy_noise_en=0):
        self.threshold = threshold
        self.shift = shift
        self.leak = leak
        self.refractory_max = refractory_max
        self.dual_synapse_en = 1 if dual_synapse_en else 0
        self.delay_value = delay_value
        self.shadow_uram_offset = shadow_uram_offset
        self.legacy_noise_en = legacy_noise_en

    def get_threshold(self):
        return self.threshold

    def get_shift(self):
        return self.shift

    def get_leak(self):
        return self.leak

    def get_neuronModel(self):
        return 2

    def get_refractory_max(self):
        return getattr(self, 'refractory_max', 0)

    def get_dual_synapse_en(self):
        return getattr(self, 'dual_synapse_en', 0)

    def get_delay_value(self):
        return getattr(self, 'delay_value', 0)

    def get_shadow_uram_offset(self):
        return getattr(self, 'shadow_uram_offset', 0)

    def get_legacy_noise_en(self):
        return getattr(self, 'legacy_noise_en', 0)


class ANN_neuron(neuron_model):
    """
    Memory-less neuron model.

    leak : int
        set to 0 for memory-less neuron

    """

    def __init__(self, threshold, shift = -17, leak=0, refractory_max=0, dual_synapse_en=False, delay_value=0, shadow_uram_offset=0, legacy_noise_en=0):
        self.threshold = threshold
        self.shift = shift
        self.leak = leak
        self.refractory_max = refractory_max
        self.dual_synapse_en = 1 if dual_synapse_en else 0
        self.delay_value = delay_value
        self.shadow_uram_offset = shadow_uram_offset
        self.legacy_noise_en = legacy_noise_en

    def get_threshold(self):
        return self.threshold

    def get_shift(self):
        return self.shift

    def get_leak(self):
        return self.leak

    def get_neuronModel(self):
        return 0

    def get_refractory_max(self):
        return getattr(self, 'refractory_max', 0)

    def get_dual_synapse_en(self):
        return getattr(self, 'dual_synapse_en', 0)

    def get_delay_value(self):
        return getattr(self, 'delay_value', 0)

    def get_shadow_uram_offset(self):
        return getattr(self, 'shadow_uram_offset', 0)

    def get_legacy_noise_en(self):
        return getattr(self, 'legacy_noise_en', 0)

class IF_neuron(neuron_model):
    """
    IF (Integrate-and-Fire) neuron model.

    True non-leaky integrate-and-fire. MP only changes via synaptic weight
    accumulation. No leak, no noise. Use with soft_reset=True to approximate
    ReLU activation (residual MP preserved after spike).

    legacy_noise_en defaults to 1 (legacy 2024 mode: 35-bit MP, no refractory
    counter, unsigned noise/leak) since this is the mode DVS inference with
    IF neurons was designed and validated against.
    """

    def __init__(self, threshold, shift=-17, refractory_max=0, delay_value=0,
                 dual_synapse_en=False, soft_reset=False, legacy_noise_en=0):
        self.threshold = threshold
        self.shift = shift
        self.refractory_max = refractory_max
        self.delay_value = delay_value
        self.dual_synapse_en = 1 if dual_synapse_en else 0
        self.soft_reset = soft_reset
        self.legacy_noise_en = legacy_noise_en

    def get_threshold(self):
        return self.threshold

    def get_shift(self):
        return getattr(self, 'shift', 0)

    def get_leak(self):
        return 0

    def get_refractory_max(self):
        return getattr(self, 'refractory_max', 0)

    def get_delay_value(self):
        return getattr(self, 'delay_value', 0)

    def get_dual_synapse_en(self):
        return getattr(self, 'dual_synapse_en', 0)

    def get_soft_reset(self):
        return int(self.soft_reset)

    def get_legacy_noise_en(self):
        return getattr(self, 'legacy_noise_en', 0)

    def get_neuronModel(self):
        return 3
