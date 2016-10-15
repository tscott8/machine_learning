import numpy as np
from sklearn import datasets

learn_speed = 1.0  # the update frequency


class Neuron(object):
    def __str__(self, neuron_id):
        ret = "|N" + str(neuron_id) + "| => w:("
        for i in range(len(self.weights)):
            ret += str(round(self.weights[i], 2)) + ","
        ret += ")\n"
        return ret


    def __init__(self, num_inputs, weights=[], bias=-1):
        """ The Neuron recieves a collection of inputs and based
        on the weight for each of those inputs it determines
        whether or not it will fire."""
        self.num_inputs = num_inputs  # collection of incoming values
        self.weights = weights
        self.get_weights(weights, num_inputs)  # the weights for the inputs; managed by neuron
        self.bias = bias

    def get_weights(self, weights, num_inputs):
        if not self.weights:
            self.weights = np.random.random(num_inputs+1)
        else:
            self.update_weights(weights)

    def update_weights(self, weights):
        return weights

    def compute_output(self, inputs):
        output = 0
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        # add the bias node because there should be one more weight unaccounted for
        output += self.bias * self.weights[i+1]
        print(output)

        if output > 0:
            return 1
        else:
            return 0


class Layer(object):
    """Each Layer contains multiple neurons"""
    def __str__(self, layer_id):
        ret = "|L" + str(layer_id) + "| => \n"
        for neuron in range(len(self.neurons)):
                ret += "\t" + self.neurons[neuron].__str__(neuron)
        return ret

    def __init__(self, num_neurons, inputs_per_neuron):
        self.num_neurons = num_neurons
        self.inputs_per_neuron = inputs_per_neuron
        self.neurons = self.build_neurons(num_neurons, inputs_per_neuron)  # an array aka network of layers of neurons.

    def build_neurons(self, num_neurons, inputs_per_neuron):
        """ Calls the neuron class and adds it to an np.array based on the number specified in params"""
        neurons = []
        for i in range(num_neurons):
            neurons.append(Neuron(inputs_per_neuron))
        return neurons
        # return np.asarray(neurons)


class Neural_Network(object):
    """The Network is a collection of layers that learns and classifies targets"""
    def __str__(self):
        ret = "|Neural Network|\n"
        for layer in range(len(self.layers)):
                ret += self.layers[layer].__str__(layer) + "\n"
        return ret

    def print(self):
        print(self)

    def __init__(self, data=[], targets=[], target_names=[], layer_params=[]):
        self.data = data
        self.targets = targets
        self.target_names = target_names
        self.num_layers = len(layer_params)
        self.layers = self.build_layers(layer_params)  # an array aka network of layers of neurons.

    def build_layers(self, layer_params):
        """ Calls the layer class and adds it to an list based on the params"""
        layers = []
        for i in range(len(layer_params)):
            layers.append(Layer(layer_params[i], len(self.data[0])))
        return layers
        # return np.asarray(layers)

    def train(self, data, target):
        for row in range(len(data)):
            for layer in range(len(self.layers)):
                # print(self.layers[layer].__str__(layer))
                for neuron in range(len(self.layers[layer].neurons)):
                    # print(self.layers[layer].neurons[neuron].__str__(neuron))
                    print(self.layers[layer].neurons[neuron].compute_output(self.data[row]))

    def predict(self, data):
        pass

    def accuracy(self, predicted, actual):
        denominator = len(actual)
        numerator = 0
        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                numerator += 1
        percent = float((numerator/denominator)*100)
        return percent

iris = datasets.load_iris()
nn = Neural_Network(iris.data, iris.target, iris.target_names, [3,4,5,4,3])
# print(nn)
nn.train(iris.data, iris.target)
