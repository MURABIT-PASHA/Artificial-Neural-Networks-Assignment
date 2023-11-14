from typing import List
from matplotlib.widgets import Button

from layer import Layer
import networkx as nx
import matplotlib.pyplot as plt

LEARNING_FACTOR = 0.1


class ArtificialNeuralNetwork:
    def __init__(self, first_layer: Layer, hidden_layers: Layer, output_layer: Layer, targets: List[float]):
        self.first_layer = first_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.targets = targets
        self.graph = nx.DiGraph()

    def draw_diagram(self):
        for i, first in enumerate(self.first_layer.neurons):
            self.graph.add_node(f'x{i}', label=f"{first.value}", pos=([0, -i]))

        for i, hidden in enumerate(self.hidden_layers.neurons):
            self.graph.add_node(f'h{i}', label=f"{hidden.value}", pos=([20, -i]))

        for i, output in enumerate(self.output_layer.neurons):
            self.graph.add_node(f'o{i}', label=f"{output.value}", pos=([40, -i]))

        for i, first in enumerate(self.first_layer.neurons):
            for j, hidden in enumerate(self.hidden_layers.neurons):
                weight = self.first_layer.neurons[i].output_edges[j].weight
                self.graph.add_edge(f'x{i}', f'h{j}', weight=weight)

        for i, hidden in enumerate(self.hidden_layers.neurons):
            for j, output in enumerate(self.output_layer.neurons):
                weight = self.hidden_layers.neurons[i].output_edges[j].weight
                self.graph.add_edge(f'h{i}', f'o{j}', weight=weight)
        pos = nx.get_node_attributes(self.graph, 'pos')
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        node_labels = nx.get_node_attributes(self.graph, 'label')
        print(node_labels)
        nx.draw(self.graph, pos, with_labels=True, node_size=500, node_color='skyblue', labels=node_labels)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

    def create_image(self):
        self.draw_diagram()

        def on_button_click(event):
            plt.clf()
            self.calculate_tolerance()
            self.draw_diagram()
            button_pos = plt.axes([0.8, 0.05, 0.2, 0.05])
            b1 = Button(button_pos, 'Devam Et')
            b1.on_clicked(on_button_click)
            plt.draw()

        button_pos = plt.axes([0.8, 0.05, 0.2, 0.05])
        b1 = Button(button_pos, 'Devam Et')
        b1.on_clicked(on_button_click)
        plt.show()

    def calculate_tolerance(self):
        has_error = False
        for index, output_neuron in enumerate(self.output_layer.neurons):
            if self.targets[index] != output_neuron.value:
                has_error = True
                s = output_neuron.value * (1 - output_neuron.value) * (
                        self.targets[index] - output_neuron.value)
                self.output_layer.neurons[index].error = s
        print("Hata hesaplaması yapılıyor...")
        for output in self.output_layer.neurons:
            print(f"{output.value} değerli çıkış için hata hesaplaması: {output.error}")
        if has_error:
            # Bu kısım hata varsa ağırlıkları değiştirir ve değerleri yeniden ayarlar
            # Ağırlıkların ayarlanması
            for hidden_neuron in self.hidden_layers.neurons:
                result = 0
                for hidden_output_edge in hidden_neuron.output_edges:
                    result += hidden_output_edge.weight * hidden_output_edge.output_neuron.error
                hidden_neuron.error = hidden_neuron.value * (1 - hidden_neuron.value) * result
                for hidden_input_edge in hidden_neuron.input_edges:
                    hidden_input_edge.weight = hidden_input_edge.weight + hidden_input_edge.input_neuron.value * LEARNING_FACTOR * hidden_neuron.error

            for hidden_neuron in self.hidden_layers.neurons:
                for hidden_output_edge in hidden_neuron.output_edges:
                    hidden_output_edge.weight = hidden_output_edge.weight + hidden_output_edge.output_neuron.value * LEARNING_FACTOR * hidden_output_edge.output_neuron.error

            # Değerlerin yeniden ayarlanması
            for hidden_neuron in self.hidden_layers.neurons:
                hidden_value = 0
                for input_edge in hidden_neuron.input_edges:
                    hidden_value += input_edge.weight * input_edge.input_neuron.value
                hidden_neuron.value = hidden_value

            for output_neuron in self.output_layer.neurons:
                output_value = 0
                for input_edge in output_neuron.input_edges:
                    output_value += input_edge.weight * input_edge.input_neuron.value
                output_neuron.value = output_value
