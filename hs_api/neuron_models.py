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

    @abstractmethod
    def get_refractory_max(self):
        pass

    @abstractmethod
    def get_delay_value(self):
        pass

    @abstractmethod
    def get_dual_synapse_en(self):
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

    def __init__(self, threshold, shift, leak, refractory_max=0, delay_value=0, dual_synapse_en=False):
        self.threshold = threshold
        self.shift = shift
        self.leak = leak
        self.refractory_max = refractory_max
        self.delay_value = delay_value
        self.dual_synapse_en = dual_synapse_en

    def get_threshold(self):
        return self.threshold

    def get_shift(self):
        return self.shift

    def get_leak(self):
        return self.leak

    def get_refractory_max(self):
        return self.refractory_max

    def get_delay_value(self):
        return self.delay_value

    def get_dual_synapse_en(self):
        return int(self.dual_synapse_en)

    def get_neuronModel(self):
        return 2


class ANN_neuron(neuron_model):
    """
    Memory-less neuron model.

    leak : int
        set to 0 for memory-less neuron

    """

    def __init__(self, threshold, shift, leak=0, refractory_max=0, delay_value=0, dual_synapse_en=False):
        self.threshold = threshold
        # TODO: to be determined
        self.shift = shift
        self.leak = leak
        self.refractory_max = refractory_max
        self.delay_value = delay_value
        self.dual_synapse_en = dual_synapse_en

    def get_threshold(self):
        return self.threshold

    def get_shift(self):
        return self.shift

    def get_leak(self):
        return self.leak

    def get_refractory_max(self):
        return self.refractory_max

    def get_delay_value(self):
        return self.delay_value

    def get_dual_synapse_en(self):
        return int(self.dual_synapse_en)

    def get_neuronModel(self):
        return 0
