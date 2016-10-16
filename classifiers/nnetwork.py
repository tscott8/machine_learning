import numpy as np
from sklearn import datasets

learn_speed = 1.0  # the update frequency


class Neuron(object):
    def __str__(self, neuron_id):
        ret = "|N" + str(neuron_id) + "| => w:("
        for i in range(len(self.weights)):
            ret += str(round(self.weights[i], 2)) + ","
        ret += ") \t num_i:(" + str(self.num_inputs) + ")\n"
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

    def update_weights(self, expected):
        for i in range(len(self.weights)):
            self.weights[i] -= learn_speed * (output - expected)
        return weights

    def compute_output(self, inputs, learn=-1):
        output = 0
        # print('preappend:',inputs)
        inputs = np.append(inputs, [self.bias])
        # print('postappend:',inputs)

        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        # add the bias node because there should be one more weight unaccounted for
        # output += self.bias * self.weights[len(self.weights)-1]
        output = 1 if output > 0 else 0
        if learn is not -1 and output != learn:
            if output is not learn:
                self.update_weights(learn, output)
        return output

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
            neurons.append(Neuron(num_inputs=inputs_per_neuron, bias=-1))
        return neurons
        # return np.asarray(neurons)

    def collect_output(self, inputs):
        output = []
        for neuron in self.neurons:
            output.append(neuron.compute_output(inputs))
        return output

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
            inputs_per_neuron =  len(self.data[0]) if i == 0 else layer_params[i-1]
            layers.append(Layer(layer_params[i], inputs_per_neuron))
        return layers
        # return np.asarray(layers)
    def process_data(self, inputs):
        output = 0
        for layer in self.layers:
            output = layer.collect_output(inputs)
            inputs = output
        return output

    def train(self, data, targets):
        # print(self.process_data(data[0]))
        training_output = []
        for row in range(len(data)):
            fire = self.process_data(data[row])
            fire = fire[0]
            if int(fire) == int(targets[row]):
                training_output.append(1)
            else:
                training_output.append(0)
        print(training_output)
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
nn = Neural_Network(iris.data, iris.target, iris.target_names, [5,4,3,2,1])
print(nn)
nn.train(iris.data, iris.target)
