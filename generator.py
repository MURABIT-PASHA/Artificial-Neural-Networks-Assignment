import random
from typing import List

from artificial_neural_network import ArtificialNeuralNetwork
from neuron import Edge
from layer import Layer
from layer_types import LayerTypes
from neuron import Neuron
import pandas as pd


class Generator:

    def __init__(self, count: List[int], file_path: str):
        self.hidden_layers_counts = count
        self.file = pd.read_csv(file_path)

    def start(self):
        artificial_networks = []
        for row in range(len(self.file)):
            neurons = []
            input_layer = None
            intermediate_layer = None
            output_layer = None

            for key in self.file.iloc[row].keys():
                if key.__contains__("x"):
                    weights = [random.randint(1, 10) / 10 for _ in range(self.hidden_layers_counts[0])]
                    value = float(self.file.loc[row, key])
                    print(value)
                    neuron = Neuron(
                        value=value,
                    )
                    edges = []
                    for j in range(self.hidden_layers_counts[0]):
                        edge = Edge(
                            weight=weights[j],
                            input_neuron=neuron
                        )
                        edges.append(edge)
                    neuron.set_output_edges(edges=edges)
                    neurons.append(neuron)
            if neurons:
                input_layer = Layer(
                    layer_type=LayerTypes.input_layer,
                    neurons=neurons,
                )

            if input_layer:
                neurons = []
                for j in range(self.hidden_layers_counts[0]):
                    weights = [random.randint(1, 10) / 10 for _ in
                               range(sum("target" in column for column in self.file.columns))]
                    input_edges = []
                    output_edges = []
                    for input_neuron in input_layer.neurons:
                        input_edges.append(input_neuron.output_edges[j])
                    neuron = Neuron(
                        input_edges=input_edges,
                    )
                    neuron.calculate_input_value()
                    for k in range(sum("target" in column for column in self.file.columns)):
                        edge = Edge(
                            input_neuron=neuron,
                            weight=weights[k]
                        )
                        output_edges.append(edge)
                    neuron.set_output_edges(output_edges)
                    neurons.append(neuron)

                if neurons:
                    for _neuron in input_layer.neurons:
                        for j, edge in enumerate(_neuron.output_edges):
                            edge.output_neuron = neurons[j]

                    intermediate_layer = Layer(
                        layer_type=LayerTypes.hidden_layer,
                        neurons=neurons
                    )

            if intermediate_layer:
                neurons = []
                for j in range(sum("target" in column for column in self.file.columns)):
                    input_edges = []
                    for hidden_neuron in intermediate_layer.neurons:
                        input_edges.append(hidden_neuron.output_edges[j])
                    neuron = Neuron(
                        input_edges=input_edges,
                    )
                    neuron.calculate_input_value()
                    neurons.append(neuron)
                if neurons:
                    for _neuron in intermediate_layer.neurons:
                        for j, edge in enumerate(_neuron.output_edges):
                            edge.output_neuron = neurons[j]

                    output_layer = Layer(
                        layer_type=LayerTypes.output_layer,
                        neurons=neurons
                    )

            if output_layer:
                artificial_neural_network = ArtificialNeuralNetwork(
                    input_layer=input_layer,
                    hidden_layers=intermediate_layer,
                    output_layer=output_layer,
                    targets=[float(self.file.loc[row, key]) for key in self.file.iloc[row].keys() if "target" in key]
                )
                artificial_networks.append(artificial_neural_network)
        if artificial_networks:
            for artificial_network in artificial_networks:
                artificial_network.create_image()
