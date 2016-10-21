import numpy as np
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn import cross_validation
from sklearn.datasets.base import Bunch

learn_speed = 0.3  # the update frequency


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
        self.num_inputs = num_inputs
        self.weights = np.random.uniform(-0.5, 0.5, (num_inputs))
        self.output = output_fn
        self.error = error_fn

    def update_weights(self, output, expected_output, next_weights):
        current_weights = self.weights[:]
        # print('IN UPDATE WEIGHTS args:', output, expected_output, next_weights)
        error = self.error(output, expected_output, next_weights)
        for i in range(len(self.weights)):
            self.weights[i] -= learn_speed * output * error
        return current_weights, error

    def compute_output(self, inputs):
        output = 0
        # print(inputs)
        # print(self.weights)
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        return self.output(output)


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
                neurons.append(Neuron(num_inputs=inputs_per_neuron,
                                      output_fn=self.sigmoid_output,
                                      error_fn=self.end_error))
            else:
                neurons.append(Neuron(num_inputs=inputs_per_neuron,
                                      output_fn=self.sigmoid_output,
                                      error_fn=self.hidden_error))
        return neurons

    @staticmethod
    def discontinuous_output(x):
        return 1 if x > 0  else 0

    @staticmethod
    def sigmoid_output(x):
        # print('IN sigmoid_output', x)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def end_error(a, target, weights):
        # print('IN END_ERROR args', a, target, weights)
        return a * (1 - a) * (a - target)

    @staticmethod
    def hidden_error(a, error, weights):
        sum_errors = 0
        # print('in hidden_error args:', a, error, weights)
        # assert len(weights) == len(errors)
        for i in range(len(weights)):
            sum_errors += weights[i] * error
        return a * (1 - a) * sum_errors

    def collect_output(self, inputs):
        output = []
        inputs.append(self.bias)
        for neuron in self.neurons:
            output.append(neuron.compute_output(inputs))
        return output

    def back_propogate(self, output, errors, weights):
        layer_weights = []
        layer_errors = []
        # print('back_propogate args', output, errors, weights)
        for i, neuron in enumerate(self.neurons):
            attached_weights = []
            for j in range(len(weights)):
                attached_weights.append(weights[i][j])
            # print('ERRORS[I]',errors)
            layer_weight, layer_error = neuron.update_weights(output[i], errors[i], attached_weights)
            # print('LAYER WEIGHT:', layer_weight)
            layer_weights.append(layer_weight)
            layer_errors.append(layer_error)
            # print('LAYER WEIGHTS!: ', layer_weights[0])
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
        self.layers = self.build_layers(layer_params)  # an array aka network of layers of neurons.

    def build_layers(self, layer_params):
        """ Calls the layer class and adds it to an list based on the params"""
        layers = []
        for i in range(len(layer_params)):
            layers.append(Layer(num_neurons=layer_params[i],
                                inputs_per_neuron=(layer_params[i-1] + 1) if i != 0 else (layer_params[i] + 1) ,
                                bias=-1,
                                fn=0 if len(layer_params)-1 == i else 1))
        return layers

    def train(self, data, targets):
        data = normalize(data)
        # print('DATA', data)
        # print('TARGETS', targets)
        for k in range(1):
            trained=[]
            for row in range(len(data)):
                layer_inputs = []
                layer_output = []
                for i, layer in enumerate(self.layers):
                    layer_inputs = data[row].tolist() if i == 0 else layer_output[i-1]
                    # print('INPUTS HERE!: ', layer_inputs)
                    collected_output = layer.collect_output(layer_inputs)
                    layer_output.append(collected_output)
                    # print('COLLECTED: ', collected_output)
                # print('LAYER OUTPUT!: ', layer_output)
                next_weights = []
                next_errors = []
                for i in reversed(range(len(self.layers))):
                    # print('LAYER_ID', i)
                    # print('LAYER ON REVERSE', self.layers[i].__str__(i))
                    # print('next_weights', next_weights)
                    # print('next_errors', next_errors)
                    errors = discretize(targets[row]) if i == (len(self.layers)-1) else next_errors
                    # print('ERRORS', errors)
                    next_weights, next_errors = self.layers[i].back_propogate(
                        output=layer_output[i],
                        #errors=targets[row] if layer == len(self.layers)-1 else next_errors,
                        errors = errors,
                        weights = next_weights)
                    # print('NEXT WEIGHTS!: ', next_weights)
                trained.append(undiscretize(layer_output[len(layer_output)-1]))
            print(self.accuracy(trained, targets))

    def predict(self, data):
        predicted = []
        for row in range(len(data)):
            layer_inputs = []
            layer_output = []
            for i, layer in enumerate(self.layers):
                layer_inputs = data[row].tolist() if i == 0 else layer_output
                layer_output = layer.collect_output(layer_inputs)
                # print('layer_output', layer_output)
            predicted.append(undiscretize(layer_output))
        return np.array(predicted)

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
        return [0, 0, 0, 1]
    if item == 1:
        return [0, 0, 1, 0]
    if item == 2:
        return [0, 1, 0, 0]

def undiscretize(item):
    max_index = item.index(max(item))
    item[max_index] = 1
    for i in range(len(item)):
        if item[i] != 1:
            item[i] = int(0)

    if item == [0,0,0,1]:
        # print(item)
        return 0
    if item == [0,0,1,0]:
        # print(item)
        return 1
    if item == [0,1,0,0]:
        # print(item)
        return 2

nn = Neural_Network([4]*8)
iris = datasets.load_iris()
X = normalize(iris.data)
y = iris.target
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

train_permutations = []
test_permutations = []
for i in range(100):
    train = Bunch()
    test = Bunch()
    train['data'], test['data'], train['target'], test['target'] = cross_validation.train_test_split(X, y, test_size=0.3)
    train_permutations.append(train)
    test_permutations.append(test)
for j in range(len(train_permutations)):
    nn.train(train_permutations[j].data, train_permutations[j].target)
    # nn.predict(test_permutations[j].data,test_permutations[j])


# nn.train(X_train, y_train)

predicted = nn.predict(X_test)
print(predicted)
print(y_test)
accuracy = nn.accuracy(predicted, y_test)
print(accuracy)
