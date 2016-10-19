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


    def __init__(self, num_inputs, bias, output_fn, error_fn):
        """ The Neuron recieves a collection of inputs and based
        on the weight for each of those inputs it determines
        whether or not it will fire."""
        self.num_inputs = num_inputs  # collection of incoming values
        # self.weights = 2*np.random.random(num_inputs+1) - 1
        self.weights = np.random.uniform(-0.5, 0.5, (num_inputs + 1)) # the weights for the inputs; managed by neuron
        self.bias = bias
        self.output = output_fn
        self.error = error_fn

    def update_weights(self, output, expected_output):
        for i in range(len(self.weights)):
            self.weights[i] -= learn_speed * self.error(output, self.weights, expected_output)

    def compute_output(self, inputs):
        output = 0
        # add the bias node because there should be one more weight unaccounted for
        inputs = np.append(inputs, [self.bias])
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        # output += self.bias * self.weights[len(self.weights)-1]
        # if learning is not -1 and output != learning:
        #     self.update_weights(output, learning)
        return cleanup_float(output)


class Layer(object):
    """Each Layer contains multiple neurons"""
    def __str__(self, layer_id):
        ret = "|L" + str(layer_id) + "| => "
        ret += ("Discontinuous" if self.fn == 0 else "Sigmoid") + "\n"
        for neuron in range(len(self.neurons)):
            ret += "\t" + self.neurons[neuron].__str__(neuron)
        return ret

    def __init__(self, num_neurons, inputs_per_neuron, fn=0):
        self.num_neurons = num_neurons
        self.inputs_per_neuron = inputs_per_neuron
        self.neurons = self.build_neurons(num_neurons, inputs_per_neuron, fn)  # an array aka network of layers of neurons.
        self.fn = fn

    def build_neurons(self, num_neurons, inputs_per_neuron, fn):
        """ Calls the neuron class and adds it to an np.array based on the number specified in params"""
        neurons = []
        for i in range(num_neurons):
            if fn == 0:
                neurons.append(Neuron(num_inputs=inputs_per_neuron, bias=-1,
                                      output_fn=self.discontinuous_output, error_fn=self.end_error))
            else:
                neurons.append(Neuron(num_inputs=inputs_per_neuron, bias=-1,
                                      output_fn=self.sig_output, error_fn=self.hidden_error))
        return neurons

    @staticmethod
    def discontinuous_output(x):
        return 1 if x > 0  else 0

    @staticmethod
    def sig_output(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def end_error(a, weights, targets):
        return a + (1 - a)*(a - targets)

    @staticmethod
    def hidden_error(a, weights, errors):
        sum_errors = 0
        assert len(weights) == len(errors)
        for i in range(len(weights)):
            sum_errors += weights[i] * errors[i]
        return a * (1 - a) * sum_errors

    def collect_output(self, inputs, expected_output):
        output = []
        for neuron in self.neurons:
            output.append(neuron.compute_output(inputs))
            neuron.update_weights(output[-1], expected_output)
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
        print(layer_params)
        for i in range(len(layer_params)):
            inputs_per_neuron =  len(self.data[0]) if i == 0 else layer_params[i-1]
            layers.append(Layer(layer_params[i], inputs_per_neuron,
                                0 if len(layer_params)-1 == i else 1))
        return layers

    def process_data(self, inputs, expected_output):
        output = 0
        for layer in self.layers:
            output = layer.collect_output(inputs, expected_output)
            inputs = output
        # print('output: ', output)
        # print('max: ', max(output))
        return output.index(max(output))


    def train(self, data, targets):
        training_output = []
        for row in range(len(data)):
            processed_output = self.process_data(data[row], targets[row])
            training_output.append(processed_output)

        return training_output

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
nn = Neural_Network(iris.data, iris.target, iris.target_names, [4, 5, 4, len(iris.target_names)])
print(nn)
trained = nn.train(normalize(iris.data), iris.target)
print(nn.accuracy(iris.target, trained))
print(nn)
