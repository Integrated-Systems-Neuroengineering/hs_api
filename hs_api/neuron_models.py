#!/usr/bin/env python3

from abc import ABC, abstractmethod


class neuron_model(ABC):
    @abstractmethod
    def get_theta(self):
        pass

    @abstractmethod
    def get_neuronModel(self):
        pass

    @abstractmethod
    def get_nu(self):
        pass

    @abstractmethod
    def get_Lambda(self):
        pass

    def __hash__(self):
        return hash(
            (
                self.get_theta(),
                self.get_neuronModel,
                self.get_nu(),
                self.get_Lambda(),
            )
        )

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __le__(self, other):
        return hash(self) <= hash(other)


class LIF_neuron(neuron_model):
    """
    LIF neuron model.

    nu: int
        noise perturbation magnitude

    """

    def __init__(self, theta, nu = -17, Lambda = 63, refractory_max=0, dual_synapse_en=False, delay_value=0, shadow_uram_offset=0, legacy_noise_en=0):
        self.theta = theta
        self.nu = nu
        self.Lambda = Lambda
        self.refractory_max = refractory_max
        self.dual_synapse_en = 1 if dual_synapse_en else 0
        self.delay_value = delay_value
        self.shadow_uram_offset = shadow_uram_offset
        self.legacy_noise_en = legacy_noise_en

    def get_theta(self):
        return self.theta

    def get_nu(self):
        return self.nu

    def get_Lambda(self):
        return self.Lambda

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

    Lambda : int
        set to 0 for memory-less neuron

    """

    def __init__(self, theta, nu = -17, Lambda=0, refractory_max=0, dual_synapse_en=False, delay_value=0, shadow_uram_offset=0, legacy_noise_en=0):
        self.theta = theta
        self.nu = nu
        self.Lambda = Lambda
        self.refractory_max = refractory_max
        self.dual_synapse_en = 1 if dual_synapse_en else 0
        self.delay_value = delay_value
        self.shadow_uram_offset = shadow_uram_offset
        self.legacy_noise_en = legacy_noise_en

    def get_theta(self):
        return self.theta

    def get_nu(self):
        return self.nu

    def get_Lambda(self):
        return self.Lambda

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
    """

    def __init__(self, theta, nu=-17, refractory_max=0, delay_value=0,
                 dual_synapse_en=False, soft_reset=False, legacy_noise_en=0):
        self.theta = theta
        self.nu = nu
        self.refractory_max = refractory_max
        self.delay_value = delay_value
        self.dual_synapse_en = 1 if dual_synapse_en else 0
        self.soft_reset = soft_reset
        self.legacy_noise_en = legacy_noise_en

    def get_theta(self):
        return self.theta

    def get_nu(self):
        return getattr(self, 'nu', -17)

    def get_Lambda(self):
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