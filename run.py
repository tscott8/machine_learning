# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:55:00 2016

@author: Tyler Scott
"""
import sys
from pprint import pprint
from load import Loader
from sklearn import datasets

from classifiers.hardcoded import Hard_Coded
from classifiers.knn import k_Nearest_Neighbor
from classifiers.dtree import ID3_Decision_Tree
from classifiers.nnetwork import Neural_Network


class Run:

    def __init__(self):
        self.loader = int(2)
        self.location = 'iris'
        self.split_amount = float(0.7)
        self.classifier = {}
        self.classifier['type'] = 'nn'
        # self.classifier['k'] = 1
        self.classifier['lp'] = [4,4,4]

    def getInput(self):
        self.location = input("Enter the dataset name (iris, boston, "
                              "diabetes, digits, linnerud, car, "
                              "lenses, votes): ") or 'iris'
        self.split_amount = float(input("Enter split percentage as decimal "
                                        "(default = 0.7): ") or 0.7)
        self.classifier['type'] = input("Enter classification mode "
                                        "(hardcoded, knn, id3, nn): ") or 'nn'
        if 'knn' in self.classifier['type']:
            self.classifier['k'] = int(input("Enter a value for k "
                                             "(default = 1): ") or 1)
        if 'nn' in self.classifier['type']:
            lp = [int(s) for s in input("Enter layer parameters (i.e. 3 4 3 1): ").split()] or [4,4,4]
            self.classifier['lp'] = lp
        return self.location, self.split_amount, self.classifier

    def process_data(self, training_dataset, testing_dataset, classifier):
        accuracy = 0

        if 'nn' in classifier['type']:
            lp = classifier['lp']
            lp.append(len(training_dataset.target_names))
            nn = Neural_Network()
            dl = Loader()
            permutations = []
            for i in range(10):
                permutation, junk = dl.split_dataset(training_dataset, .1)
                permutations.append(permutation)
            print(permutations)
            for j in range(len(permutations)):
                nn.train(permutations[j].data, permutations[j].target)
            accuracy = nn.accuracy(nn.predict(testing_dataset.data), testing_dataset.target)

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

        print("Method Accuracy = {0}%".format(int(accuracy)))

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
        self.console_messages(dataset, training_dataset, testing_dataset)
        self.process_data(training_dataset, testing_dataset, self.classifier)

if __name__ == "__main__":
    run = Run()
    run.main(sys.argv)
