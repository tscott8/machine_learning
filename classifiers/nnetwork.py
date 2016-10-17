import numpy as np
from sklearn import datasets
from sklearn.preprocessing import normalize

learn_speed = 0.1  # the update frequency


class Neuron(object):
    def __str__(self, neuron_id):
        ret = "|N" + str(neuron_id) + "| => w:( "
        for i in range(len(self.weights)):
            ret += str(round(self.weights[i], 3)) + " "
        ret += ") \t num_i:(" + str(self.num_inputs) + ") \t"
        ret += "err:(" + str(self.error_margin) + ") \n"
        return ret


    def __init__(self, num_inputs, bias=-1):
        """ The Neuron recieves a collection of inputs and based
        on the weight for each of those inputs it determines
        whether or not it will fire."""
        self.num_inputs = num_inputs  # collection of incoming values
        self.weights = np.random.uniform(-0.5, 0.5, (num_inputs + 1)) # the weights for the inputs; managed by neuron
        self.bias = bias
        self.error_margin = 0

    def update_weights(self, output, expected_output):
        for i in range(len(self.weights)):
            self.weights[i] -= learn_speed * (output - expected_output)

    def compute_output(self, inputs, expected_output=-1):
        output = 0
        # add the bias node because there should be one more weight unaccounted for
        inputs = np.append(inputs, [self.bias])
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        # output += self.bias * self.weights[len(self.weights)-1]
        if expected_output is not -1 and output != expected_output:
            self.update_weights(output, expected_output)
        return cleanup_float(output)

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

    def collect_output(self, inputs, error_margin):
        output = []
        for neuron in self.neurons:
            output.append(neuron.compute_output(inputs, error_margin))
        return output

    def back_propogate(self, errors):
        pass

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

    def process_data(self, inputs, expected_output):
        output = 0
        for layer in self.layers:
            output = layer.collect_output(inputs, expected_output)
            inputs = output
        print('output: ', output)
        print('max: ', max(output))
        return output.index(max(output))

    def compare_target(self, expected_target, actual_target):
        if expected_target == actual_target:
            return True
        else:
            return False


    def train(self, data, targets):
        training_output = []
        for row in range(len(data)):
            # processed_output = self.process_data(data[row], targets[row])
            processed_output = undiscretize(self.process_data(data[row], targets[row]))
            # while targets[row] != processed_output:
            #     processed_output = undiscretize(self.process_data(data[row], targets[row]))
            training_output.append(undiscretize(processed_output))
            # print(processed_output)
        # print(training_output)
            #while the output doesnt match expected stick with row until it does get the correct output

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

def cleanup_float(num):
    return float(str(round(num, 3)))

def discretize(item):
    if item == 0:
        return [1, 0, 0]
    if item == 1:
        return [0, 1, 0]
    if item == 2:
        return [0, 0, 1]

def undiscretize(item):
    if item == [1, 0, 0]:
        return int(0)
    if item == [0, 1, 0]:
        return int(1)
    if item == [0, 0, 1]:
        return int(2)
    return int(0)

iris = datasets.load_iris()
arr_targets = []
for i in iris.target:
    arr_targets.append(discretize(i))
nn = Neural_Network(iris.data, iris.target, iris.target_names, [4,5,4,len(iris.target_names)])
print(nn)
nn.train(normalize(iris.data), iris.target)
print(nn)
