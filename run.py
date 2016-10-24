# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:55:00 2016

@author: Tyler Scott
"""
import sys
import numpy as np
from pprint import pprint
from load import Loader
from sklearn import datasets

from classifiers.hardcoded import Hard_Coded
from classifiers.knn import k_Nearest_Neighbor
from classifiers.dtree import ID3_Decision_Tree
from classifiers.nnetwork import Neural_Network

import warnings
warnings.filterwarnings("ignore")

class Run:

    def __init__(self):
        self.location = 'prima'
        self.split_amount = float(0.7)
        self.classifier = {}
        self.classifier['type'] = 'nn'
        self.classifier['k'] = 1
        self.classifier['lp'] = [4]

    def getInput(self):
        self.location = input("Enter the dataset name (iris, boston, "
                              "diabetes, digits, linnerud, car, "
                              "lenses, votes, prima): ") or 'iris'
        self.split_amount = float(input("Enter split percentage as decimal "
                                        "(default = 0.7): ") or 0.7)
        self.classifier['type'] = input("Enter classification mode "
                                        "(hardcoded, knn, id3, nn): ") or 'nn'
        if 'knn' in self.classifier['type']:
            self.classifier['k'] = int(input("Enter a value for k "
                                             "(default = 1): ") or 1)
        if 'nn' in self.classifier['type']:
            lp = [int(s) for s in input("Enter layer parameters (i.e. 3 4 3 1): ").split()] or [3]
            self.classifier['lp'] = lp
        return self.location, self.split_amount, self.classifier

    def process_data(self, training_dataset, testing_dataset, classifier):
        accuracy = 0

        if 'nn' is classifier['type']:
            dl = Loader()
            training_dataset['target'] = dl.discretize_targets(training_dataset)
            testing_dataset['target'] = dl.discretize_targets(testing_dataset)
            nn = Neural_Network(layer_params=[len(training_dataset.data[0])] + classifier['lp'] + [len(training_dataset.target_names)])
            train_permutations = []
            for i in range(1000):
                train_permutations.append(dl.split_dataset(training_dataset, 0.5)[0])
            for epoch in range(len(train_permutations)):
                trained = nn.train(train_permutations[epoch].data, train_permutations[epoch].target)
                for i in range(len(trained)):
                    trained[i] = np.array(dl.undiscretize_output(trained[i]))
                if epoch % 100 is 0:
                    print('Epoch ', epoch, ' accuracy: ', nn.accuracy(trained, train_permutations[epoch].target),'%')
            predicted = nn.predict(testing_dataset.data)
            for j in range(len(predicted)):
                    predicted[j] = np.array(dl.undiscretize_output(predicted[j]))
            accuracy = nn.accuracy(predicted, testing_dataset.target)

        if 'id3' in classifier['type']:
            id3 = ID3_Decision_Tree()
            id3.train(training_dataset.data, training_dataset.target)
            # predicted = id3.predict(testing_dataset.data)
            result = []
            for i in range(len(testing_dataset.data)):
                result.append(id3.predict(testing_dataset.data[i]))
            accuracy = id3.accuracy(result, testing_dataset.target)

        if 'knn' in classifier['type']:
            knn = k_Nearest_Neighbor(classifier['k'])
            knn.train(training_dataset.data, training_dataset.target)
            accuracy = knn.accuracy(knn.predict(testing_dataset.data),
                                    testing_dataset.target)

        if 'hardcoded' in classifier['type']:
            hc = Hard_Coded()
            hc.train(training_dataset.data, training_dataset.target)
            accuracy = hc.predict(testing_dataset.data, testing_dataset.target)

        print("Method Accuracy = {0}%".format(float(accuracy)))

    def console_messages(self, dataset, training_dataset, testing_dataset):
        print('original dataset', dataset)

        print("dataset length:", len(dataset))
        print("training_dataset length:", len(training_dataset))
        print("testing_dataset length:", len(testing_dataset))

        print("training_dataset.data:", training_dataset.data)
        print("training_dataset.target:", training_dataset.target)
        print("training_dataset.target_names:", training_dataset.target_names)

        print("testing_dataset.data:", testing_dataset.data)
        print("testing_dataset.target:", testing_dataset.target)
        print("testing_dataset.target_names:", testing_dataset.target_names)

        print("train_dataset.data length:", len(training_dataset.data))
        print("train_dataset.target length:", len(training_dataset.target))
        print("testing_dataset.data length:", len(testing_dataset.data))
        print("testing_dataset.target length:", len(testing_dataset.target))

    def main(self, args):
        dl = Loader()
        # self.getInput()
        dataset = dl.load_dataset(self.location)
        training_dataset, testing_dataset = dl.split_dataset(dataset, self.split_amount)
        # self.console_messages(dataset, training_dataset, testing_dataset)
        self.process_data(training_dataset, testing_dataset, self.classifier)

if __name__ == "__main__":
    run = Run()
    run.main(sys.argv)
