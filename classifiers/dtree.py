import numpy as np


class ID3_Decision_Tree:

    def __init__(self):
        self.data = []
        self.targets = []

    def train(self, data, targets):
        self.data = data
        self.targets = targets

    def predict(self, inputs):
        nInputs = np.shape(inputs)[0]
        closest = np.zeros(nInputs)
        for n in range(nInputs):
            # Compute distances
            distances = np.sum((self.data-inputs[n, :])**2, axis=1)
            # Identify the nearest neighbours
            indices = np.argsort(distances, axis=0)
            classes = np.unique(self.targets[indices[:self.k]])
            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(self.k):
                    counts[self.targets[indices[i]]] += 1
                    closest[n] = np.max(counts)
        return closest

    def accuracy(self, predicted, actual):
        denominator = len(actual)
        numerator = 0
        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                numerator += 1
        percent = float((numerator/denominator)*100)
        return percent
