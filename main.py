import random

from artificial_neural_network import ArtificialNeuralNetwork
from neuron import Edge
from layer import Layer
from layer_types import LayerTypes
from neuron import Neuron


table = [
    {
        "x0": 1,
        "x1": 0,
        "x2": 0,
        "x3": 0,
        "target1": 0,
        "target2": 0
    },
    {
        "x0": 0,
        "x1": 1,
        "x2": 0,
        "x3": 0,
        "target1": 0,
        "target2": 1
    },
    {
        "x0": 0,
        "x1": 0,
        "x2": 1,
        "x3": 0,
        "target1": 1,
        "target2": 0
    },
    {
        "x0": 0,
        "x1": 0,
        "x2": 0,
        "x3": 1,
        "target1": 1,
        "target2": 1
    },
]

try:
    hidden_layer = int(input("Kaç adet gizli katman var?"))
except ValueError:
    print("Oops! Sayı girmelisin")
else:
    artificial_networks = []
    for i in table:
        neurons = []
        input_layer = None
        intermediate_layer = None
        output_layer = None

        for key in i.keys():
            # Giriş katmanının oluşturulduğu yer.
            # Buradaki nöronların giriş kenarları yoktur.
            # Sadece değerleri ve çıkış kenarları vardır.
            if key.__contains__("x"):
                weights = [random.randint(1, 10) / 10 for _ in range(hidden_layer)]
                value = float(i.get(key))
                neuron = Neuron(
                    value=value,
                )
                edges = []
                for j in range(hidden_layer):
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
        # Giriş katmanını ekleme işlemi bitti

        # Eğer giriş katmanı oluşmuşsa gizli katman oluşturulacak
        if input_layer:
            neurons = []
            for j in range(hidden_layer):
                weights = [random.randint(1, 10) / 10 for _ in range(2)]
                input_edges = []
                output_edges = []
                for input_neuron in input_layer.neurons:
                    input_edges.append(input_neuron.output_edges[j])
                neuron = Neuron(
                    input_edges=input_edges,
                )
                neuron.calculate_input_value()
                for k in range(2):
                    edge = Edge(
                        input_neuron=neuron,
                        weight=weights[k]
                    )
                    output_edges.append(edge)
                neuron.set_output_edges(output_edges)
                neurons.append(neuron)

            # Gizli katman oluşturuldu
            if neurons:
                # Burası ilk katman ile gizli katman arası bağlantıyı sağlayan yer, çok önemli
                # Eğer burası olmazsa bağlı liste özelliği olmaz
                for _neuron in input_layer.neurons:
                    for j, edge in enumerate(_neuron.output_edges):
                        edge.output_neuron = neurons[j]

                intermediate_layer = Layer(
                    layer_type=LayerTypes.hidden_layer,
                    neurons=neurons
                )

        # Eğer ara (gizli) katman varsa son katmanımız oluşturulacak
        if intermediate_layer:
            neurons = []
            for j in range(2):
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
                first_layer=input_layer,
                hidden_layers=intermediate_layer,
                output_layer=output_layer,
                targets=[float(i[key]) for key in i.keys() if "target" in key]
            )
            artificial_networks.append(artificial_neural_network)
    if artificial_networks:
        for artificial_network in artificial_networks:
            artificial_network.create_image()
