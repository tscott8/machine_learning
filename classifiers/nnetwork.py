import numpy as np
from sklearn import datasets
from sklearn.preprocessing import normalize

learn_speed = 0.1  # the update frequency


class Neuron(object):
    def __str__(self, neuron_id):
        ret = "|N" + str(neuron_id) + "| => w:( "
        for i in range(len(self.weights)):
            ret += str(round(self.weights[i], 3)) + " "
        ret += ") \t num_i:(" + str(self.num_inputs) + ") \n"
        # ret += "err:(" + str(self.error) + ") \n"
        return ret


    def __init__(self, num_inputs, output_fn, error_fn):
        """ The Neuron recieves a collection of inputs and based
        on the weight for each of those inputs it determines
        whether or not it will fire."""
        self.num_inputs = num_inputs  # collection of incoming values
        self.weights = np.random.uniform(-0.5, 0.5, (num_inputs + 1)) # the weights for the inputs; managed by neuron
        self.output = output_fn
        self.error = error_fn

    def update_weights(self, output, expected_output, next_weights):
        current_weights = self.weights[:]
        error = self.error(output, expected_output, next_weights)
        for i in range(len(self.weights)):
            self.weights[i] -= learn_speed * output * error
        return current_weights, error

    def compute_output(self, inputs):
        output = 0
        # add the bias node as an input
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        return output


class Layer(object):
    """Each Layer contains multiple neurons"""
    def __str__(self, layer_id):
        ret = "|L" + str(layer_id) + "| => \n"
        # ret += ("Discontinuous" if self.fn == 0 else "Sigmoid") + "\n"
        for neuron in range(len(self.neurons)):
            ret += "\t" + self.neurons[neuron].__str__(neuron)
        return ret

    def __init__(self, num_neurons, inputs_per_neuron, bias=-1, fn=0):
        self.num_neurons = num_neurons
        self.inputs_per_neuron = inputs_per_neuron
        self.bias = bias
        self.neurons = self.build_neurons(num_neurons, inputs_per_neuron, fn)  # an array aka network of layers of neurons.
        self.fn = fn

    def build_neurons(self, num_neurons, inputs_per_neuron, fn):
        """ Calls the neuron class and adds it to an np.array based on the number specified in params"""
        neurons = []
        for i in range(num_neurons):
            if fn == 0:
                neurons.append(Neuron(num_inputs=inputs_per_neuron, output_fn=self.sig_output, error_fn=self.end_error))
            else:
                neurons.append(Neuron(num_inputs=inputs_per_neuron, output_fn=self.sig_output, error_fn=self.hidden_error))
        return neurons

    @staticmethod
    def discontinuous_output(x):
        return 1 if x > 0  else 0

    @staticmethod
    def sig_output(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def end_error(a, errors, weights):
        return a + (1 - a) * (a - errors)

    @staticmethod
    def hidden_error(a, errors, weights):
        sum_errors = 0
        assert len(weights) == len(errors)
        for i in range(len(weights)):
            sum_errors += weights[i] * errors[i]
        return a * (1 - a) * sum_errors

    def collect_output(self, inputs):
        output = []
        inputs = list(inputs)
        inputs.append(self.bias)
        for neuron in self.neurons:
            output.append(neuron.compute_output(inputs))
        return output

    def back_propogate(self, output, errors, weights):
        layer_weights = []
        layer_errors = []
        print(output, errors, weights)
        for i, neuron in enumerate(self.neurons):
            attached_weights = []
            for j in range(len(weights)):
                attached_weights.append(weights[j][i])
            layer_weight, layer_error = neuron.update_weights(output[i], errors[i], attached_weights)
            print(layer_errors)
            layer_weights.append(layer_weight)
            layer_errors.append(layer_error)
        return layer_weights, layer_errors

class Neural_Network(object):
    """The Network is a collection of layers that learns and classifies targets"""
    def __str__(self):
        ret = "|Neural Network|\n"
        for layer in range(len(self.layers)):
                ret += self.layers[layer].__str__(layer) + "\n"
        return ret

    def print(self):
        print(self)

    def __init__(self, layer_params=[4,5,4,3]):
        self.data = []
        self.targets = []
        self.num_layers = len(layer_params)
        self.layers = self.build_layers(layer_params)  # an array aka network of layers of neurons.

    def build_layers(self, layer_params):
        """ Calls the layer class and adds it to an list based on the params"""
        layers = []
        for i in range(len(layer_params)):
            inputs_per_neuron =  layer_params[i-1] if i > 0 else layer_params[i]
            layers.append(Layer(layer_params[i], inputs_per_neuron,
                                0 if len(layer_params)-1 == i else 1))
        return layers

    def train(self, data, targets):
        self.data = data
        self.targets = targets
        training_output = []
        for row in range(len(data)):
            layer_inputs = data[row]
            layer_output = []
            for layer in range(len(self.layers)):
                layer_output.append([])
                collected_output = self.layers[layer].collect_output(layer_inputs)
                layer_output[layer] = collected_output
                layer_inputs = collected_output
            next_weights = []
            next_errors = []
            
            for layer in reversed(range(len(self.layers))):
                next_weights, next_errors = self.layers[layer].back_propogate(
                    output=layer_output[layer],
                    errors=targets[row] if layer == len(self.layers)-1 else next_errors,
                    weights=next_weights)

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
nn = Neural_Network([4, 5, 4, len(iris.target_names)])
print(nn)
data = list(iris.data)
target = list(iris.target)
print(data, target)
trained = nn.train(iris.data, target)
print(nn.accuracy(iris.target, trained))
print(nn)
