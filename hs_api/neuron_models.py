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
        return hash( (self.get_threshold(), self.get_neuronModel, self.get_shift(), self.get_leak() ))

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
    def __init__(self, threshold, shift, leak):
        self.threshold = threshold
        self.shift = shift
        self.leak = leak

    def get_threshold(self):
        return self.threshold

    def get_shift(self):
        return self.shift

    def get_leak(self):
        return self.leak

    def get_neuronModel(self):
        return 2

class ANN_neuron(neuron_model):
    """
        Memory-less neuron model.
        
        leak : int
            set to 0 for memory-less neuron
        
    """
    def __init__(self, threshold, shift, leak = 0):
        self.threshold = threshold
        #TODO: to be determined 
        self.shift = shift
        self.leak = leak
    
    def get_threshold(self):
        return self.threshold

    def get_shift(self):
        return self.shift

    def get_leak(self):
        return self.leak

    def get_neuronModel(self):
        return 0
