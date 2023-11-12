from typing import List
from diagrams import Diagram, Edge, Node
from layer import Layer


class ArtificialNeuralNetwork:
    def __init__(self, first_layer: Layer, hidden_layers: Layer, output_layer: Layer, targets: List[float]):
        self.first_layer = first_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.targets = targets

    def create_diagram(self):
        pass

    def create_image(self):
        with Diagram("YSA", show=False, direction="LR"):
            first_nodes = []
            hidden_nodes = []
            output_nodes = []
            for first in self.first_layer.neurons:
                first_nodes.append(Node(label=f"{first.value}", shape="circle", height=".25"))
            for hidden in self.hidden_layers.neurons:
                hidden_nodes.append(Node(label=f"{hidden.value}", shape="circle", height=".25"))
            for output in self.output_layer.neurons:
                output_nodes.append(Node(label=f"{output.value}", shape="circle", height=".25"))

            for first in range(len(first_nodes)):
                for hidden in range(len(hidden_nodes)):
                    first_nodes[first].connect(hidden_nodes[hidden], edge=Edge(color="red",
                                                                               label=f"\t{self.first_layer.neurons[first].output_edges[hidden].weight}\t\n",
                                                                               forward=True, fontsize="15"))
            for hidden in range(len(hidden_nodes)):
                for output in range(len(output_nodes)):
                    hidden_nodes[hidden].connect(output_nodes[output], edge=Edge(color="red",
                                                                                 label=f"\t{self.hidden_layers.neurons[hidden].output_edges[output].weight}\t\n",
                                                                                 forward=True, fontsize="15"))

    def calculate_tolerance(self):
        self.create_image()
        has_error = False
        for index in range(len(self.targets)):
            if self.targets[index] != self.output_layer.neurons[index].value:
                has_error = True
                s = self.output_layer.neurons[index].value * (1 - self.output_layer.neurons[index].value) * (
                        self.targets[index] - self.output_layer.neurons[index].value)
                self.output_layer.neurons[index].error = s
        print("Hata hesaplaması yapılıyor...")
        for output in self.output_layer.neurons:
            print(f"{output.value} değerli çıkış için hata hesaplaması: {output.error}")
        if has_error:
            # Bu kısımda ağırlıkları değiştireceğim
            hidden_layer_results = []
            for hidden_neuron in self.hidden_layers.neurons:
                result = 0
                for hidden_output_edge in hidden_neuron.output_edges:
                    result += hidden_output_edge.weight * hidden_output_edge.output_neuron.error
                hidden_layer_results.append(result)
            print(hidden_layer_results)



