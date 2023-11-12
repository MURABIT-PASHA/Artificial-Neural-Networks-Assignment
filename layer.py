from typing import List

from layer_types import LayerTypes
from neuron import Neuron


class Layer:
    def __init__(self, neurons: List[Neuron], layer_type: LayerTypes):
        self.neurons = neurons
        self.layer_type = layer_type
