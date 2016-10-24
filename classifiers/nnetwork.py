import numpy as np
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn import cross_validation
from sklearn.datasets.base import Bunch
from copy import deepcopy

learn_speed = 0.1  # the update frequency


class Neuron(object):
    """ A single neuron """
    def __str__(self, neuron_id):
        """
        Prints the neuron out nice.
        :param neuron_id: the position of the neuron in the layer
        :type neuron_id: int
        :return: string
        """
        ret = "|N" + str(neuron_id) + "| => w:( "
        for i in range(len(self.weights)):
            ret += str(round(self.weights[i], 3)) + " "
        ret += ") \t num_i:(" + str(self.num_inputs) + ") \n"
        return ret


    def __init__(self, num_inputs, error_fn):
        """
        Constructor for the neuron. Each neuron will manage its own weights
        :param num_inputs: representsthe quantity of inputs that will be processed in the neuron.
        :param error_fn: specifies the type of neuron errors based on if its in a hidden or end layer.
        :type num_inputs: int
        :type error_fn: function
        :return: None
        """
        self.num_inputs = num_inputs
        self.weights = np.random.uniform(-0.5, 0.5, (num_inputs))
        self.error = error_fn

    def update_weights(self, output, expected_output, attached_weights, attached_inputs):
        """
        Update the weights for the neurons inputs based on the amount of error given.
        first it will save the current_weights to be passed to the next layer of neurons.
        then using the error it calculates with the error function (specified in constructor),
        it will loop through updating each weight accordingly
        :param output: what the neuron fired
        :param expected_output: what the neuron should have fired
        :param attached_weights: the weights to be updated
        :param attached_inputs: the inputs to be updated
        :type output: float
        :type expected_output: float
        :type attached_weights: list
        :type attached_inputs: list
        :return: list, float
        """
        current_weights = self.weights[:]
        error = self.error(output, expected_output, attached_weights)
        for i in range(len(attached_inputs)):
            self.weights[i] -= learn_speed * attached_inputs[i] * error
        return current_weights, error

    def compute_output(self, inputs):
        """
        Process the inputs to see if the neuron fires or not using the sigmoid function.
        Alternatively, learn from the expected value if an expected value is provided.
        :param inputs: list of inputs. Inputs must be floats
        :type inputs: list
        :return: float
        """
        output = 0
        for i in range(len(self.weights)):
            output += inputs[i] * self.weights[i]
        output = (1 / (1 + np.exp(-output)))
        return output


class Layer(object):
    """ Each Layer contains multiple neurons """
    def __str__(self, layer_id):
        """
        Prints the layer out nice.
        :param layer_id: the position of the layer in the network
        :type layer_id: int
        :return: string
        """
        ret = "|L" + str(layer_id) + "| => \n"
        for neuron in range(len(self.neurons)):
            ret += "\t" + self.neurons[neuron].__str__(neuron)
        return ret

    def __init__(self, num_neurons, inputs_per_neuron, bias=-1.0, layer_type=0):
        """
        Constructor for the Layer.
        :param num_neurons: number of neurons in the layer
        :param inputs_per_neuron: number of inputs for each neuron in the layer
        :param bias: the bias to be added to the list of inputs for each neuron
        :param layer_type: specifies whether its a hidden or output layer
        :type num_neurons: int
        :type inputs_per_neuron: int
        :type bias: int
        :type layer_type: int (1 or 0)
        :return: None
        """
        self.num_neurons = num_neurons
        self.inputs_per_neuron = inputs_per_neuron
        self.bias = bias
        self.layer_type = layer_type
        neurons = []
        for i in range(num_neurons):
            if layer_type == 0:
                neurons+=[Neuron(num_inputs=inputs_per_neuron + 1,
                                      error_fn=self.output_layer_error)]
            else:
                neurons+=[Neuron(num_inputs=inputs_per_neuron + 1,
                                      error_fn=self.hidden_layer_error)]
        self.neurons = neurons # an array aka network of layers of neurons.


    @staticmethod
    def output_layer_error(a, target, weights):
        """
        error for the output layer.
        :param a: output from the neuron
        :param target: target output to compare
        :param weights: weights for that neuron's inputs
        :type a: float
        :type target: float or int
        :type weights: list (UNUSED)
        :return: float
        """
        return a * (1 - a) * (a - target)

    @staticmethod
    def hidden_layer_error(a, errors, weights):
        """
        error for the hidden layer.
        :param a: output from the neuron
        :param error: error from another layer output to compare
        :param weights: weights for that neuron's inputs
        :type a: float
        :type error: float
        :type weights: list
        :return: float
        """
        sum_errors = 0
        assert len(weights) == len(errors)
        for i in range(len(weights)):
            sum_errors += weights[i] * errors[i]
        return a * (1 - a) * sum_errors

    def collect_output(self, inputs):
        """
        calculates the output for each neuron in the layer
        :param inputs: inputs to be processed by the neurons
        :type inputs: list of floats
        :return: list
        """
        output = []
        # inputs.append(self.bias)
        inputs += [self.bias]
        for neuron in self.neurons:
            output+=[neuron.compute_output(inputs)]
        return output

    def back_propogate(self, layer_output, forwarded_errors, forwarded_weights, forwarded_inputs):
        """
        passes back the errors through each layer, updating the weights at each neuron
        :param layer_output: outputs of the this layer
        :param forwarded_errors: errors that have been forwarded from the previous layer (or target if at end)
        :param forwarded_weights: weights that have been forwarded from the previous layer
        :param forwarded_inputs: inputs that have been forwarded from the previous layer
        :type layer_output: list of floats
        :type forwarded_errors: list of floats
        :type forwarded_weights: list of lists
        :type forwarded_inputs: list of floats
        :return: list, list
        """
        layer_weights = [] # this layers weights
        layer_errors = [] # this layers errors
        for i, neuron in enumerate(self.neurons):
            attached_weights = [] #  weights for this neuron forwarded from the previous layer
            for j in range(len(forwarded_weights)):
                attached_weights+=[forwarded_weights[j][i]] # add the weight[j] for neuron[i]
            neuron_weights, neuron_error = neuron.update_weights(layer_output[i],
                                                                    forwarded_errors[i] if self.layer_type == 0 else forwarded_errors,
                                                                    attached_weights, forwarded_inputs)
                # update the weights for this neuron with the error for this neuron forwarded from previous layer,
                # and the weights for this neuron pulled out of the list of weights forwarded from previous layer
            layer_weights+=[neuron_weights]
            layer_errors+=[neuron_error]
        return layer_weights, layer_errors


class Neural_Network(object):
    """The Network is a collection of layers that learns and classifies targets"""
    def __str__(self):
        """
        Prints the Network out nice.
        :return: string
        """
        ret = "|Neural Network|\n"
        for layer in range(len(self.layers)):
                ret += self.layers[layer].__str__(layer) + "\n"
        return ret

    def print(self):
        print(self)

    def __init__(self, layer_params=[4,5,4,3]):
        """
        Constructor for the Neural Network
        :param layer_params: specifies the architecture of the network
        :type layer_params: list of ints
        :return: None
        """
        layers = []
        for i, layer_height in enumerate(layer_params):
            layers+=[Layer(num_neurons=layer_height,
                                inputs_per_neuron=layer_params[i-1] if i > 0 else layer_params[i],
                                bias=-1.0,
                                layer_type=0 if len(layer_params) - 1 == i else 1)]
        self.layers = layers

    def train(self, data, targets):
        """
        Trains the network
        :param data: data to be process training
        :param targets: targets to compare training against
        :type data: list of lists (or ndarray)
        :type targets: list of ints
        :return: list
        """
        data = normalize(data)
        trained=[]
        for row in range(len(data)):
            data_row_copy = deepcopy(data[row]).tolist()
            layer_inputs = []
            layer_outputs = []
            for i in range(len(self.layers)):
                layer_outputs.append([])
                layer_outputs[i] = self.layers[i].collect_output(layer_outputs[i-1] if i > 0 else data[row].tolist())
            forward_weights = []
            forward_errors = []
            for i in reversed(range(len(self.layers))):
                forward_weights, forward_errors = self.layers[i].back_propogate(
                    layer_output=layer_outputs[i],
                    forwarded_errors=targets[row] if i == len(self.layers) - 1 else forward_errors,
                    forwarded_weights=forward_weights,
                    forwarded_inputs=layer_outputs[i-1] if i != 0 else data[row].tolist())
            trained+=[layer_outputs[len(layer_outputs)-1]]
        return trained

    def predict(self, data):
        """
        Makes predictions using the trained network
        :param data: data to be used in predictions
        :type data: list of lists (or ndarray)
        :return: list
        """
        predicted = []
        for row in range(len(data)):
            layer_inputs = []
            layer_output = []
            for i, layer in enumerate(self.layers):
                layer_inputs = data[row].tolist() if i == 0 else layer_output
                layer_output = layer.collect_output(layer_inputs)
            predicted+=[layer_output]
        return predicted

    def accuracy(self, predicted, actual):
        """
        Calculates the accuracy of the predictions
        :param predicted: predicted values
        :param actual: target values
        :type predicted: list
        :type actual: list
        :return: float
        """
        denominator = len(actual)
        numerator = 0
        for i in range(len(predicted)):
            count = len(predicted[i])
            for j in range(len(predicted[i])):
                if predicted[i][j] != actual[i][j]:
                    count -= 1
            if count is len(actual[i]):
                numerator += 1
        percent = round((numerator/denominator)*100, 2)
        return percent

# def discretize(item):
#     if item == 0:
#         return [0, 0, 1]
#     if item == 1:
#         return [0, 1, 0]
#     if item == 2:
#         return [1, 0, 0]
#
# def undiscretize(item):
#     max_index = item.index(max(item))
#     item[max_index] = 1
#     for i in range(len(item)):
#         if item[i] != 1:
#             item[i] = int(0)
#     return item
    # if item == [0,0,1]:
    #     # print(item)
    #     return 0
    # if item == [0,1,0]:
    #     # print(item)
    #     return 1
    # if item == [1,0,0]:
    #     # print(item)
    #     return 2

# def discretize_targets(dataset):
#     target_names = []
#     for i in range(len(np.unique(dataset.target))):
#         new_target_name = [0]*len(np.unique(dataset.target))
#         new_target_name[i] = 1
#         target_names += [new_target_name]
#     target_names = np.array(target_names)
#     # print('discretize_targets_names:',target_names)
#     target = []
#     for i in range(len(dataset.target)):
#         new_target = target_names[dataset.target[i]]
#         target += [new_target]
#     return target

# nn = Neural_Network([4,3])
# print(nn)
# iris = datasets.load_iris()
# X = iris.data
# iris['target'] = discretize_targets(iris)
# y = iris.target
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
# #
# # train_bunch = Bunch()
# # train_bunch['data'], train_bunch['target'] = X_train, y_train
# # train_permutations = []
# # test_permutations = []
# # for i in range(500):
# #     temp_train = Bunch()
# #     temp_test = Bunch()
# #     temp_train['data'], temp_test['data'], temp_train['target'], temp_test['target'] = cross_validation.train_test_split(train_bunch.data, train_bunch.target, test_size=0.5)
# #     train_permutations.append(temp_train)
# #     test_permutations.append(temp_test)
# # for epoch in range(len(train_permutations)):
# #     trained = nn.train(train_permutations[epoch].data, train_permutations[epoch].target)
# #     #print('epoch:', epoch, 'accuracy:', nn.accuracy(trained, train_permutations[epoch].target))
# #     # nn.predict(test_permutations[j].data,test_permutations[j])
# #
# #
#
# nn.train(X_train, y_train)
#
# #
# predicted = nn.predict(X_test)
# print(predicted)
# print(y_test)
# accuracy = nn.accuracy(predicted, y_test)
# print(accuracy)
# print(nn)
