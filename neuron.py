import math
from typing import List


class Edge:
    def __init__(self, weight: float, input_neuron: 'Neuron', output_neuron: 'Neuron' = None):
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.weight = weight


class Neuron:
    def __init__(self,
                 input_edges: List[Edge] = None,
                 output_edges: List[Edge] = None,
                 value: float = None,
                 error: float = None):
        self.input_edges = input_edges
        self.value = value
        self.output_edges = output_edges
        self.error = error

    def calculate_input_value(self):
        """
        Bu fonksiyon gelen kenarlara göre **Sigmoid** fonksiyonu kullanarak
        nöronun değerini ayarlar.
        :return:
        """
        if self.input_edges:
            result = 0
            for input_edge in self.input_edges:
                if input_edge is not None:
                    result += (input_edge.weight * input_edge.input_neuron.value)

            self.value = 1 / (1 + (pow(math.e, -result)))

    def set_output_edges(self, edges: List[Edge]):
        """
        Bu fonksiyon çıkış kenarlarını ayarlamak için kullanılır.
        Parametre olarak :class:`Edge` sınıfından oluşan bir liste alır.
        Bu liste bir nöronun sinapslarını temsil eder.
        :param edges:
        :return:
        """
        self.output_edges = edges
