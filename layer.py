from typing import List

from layer_types import LayerTypes
from neuron import Neuron


class Layer:
    """
    Bu sınıf yapay sinir ağındaki katmanları temsil eder
    """
    def __init__(self, neurons: List[Neuron], layer_type: LayerTypes):
        """
        :param neurons: Nöronlardan oluşan bir liste almaktadır
        :param layer_type: :class:`LayerTypes` sınıfından türetilmiş bir tip değişkeni almaktadır.
        Bu katmanın tipini belirtmek için kullanılır
        """
        self.neurons = neurons
        self.layer_type = layer_type
