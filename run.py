# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:55:00 2016

@author: Tyler Scott
"""
import sys
from pprint import pprint
from load import Loader
from classifiers.hardcoded import Hard_Coded
from classifiers.knn import k_Nearest_Neighbor
from classifiers.dtree import ID3_Decision_Tree


class Run:

    def __init__(self):
        self.loader = int(1)
        self.location = 'iris'
        self.split_amount = float(0.7)
        self.classifier = {}

    def getInput(self):
        self.loader = int(input("Select Loader ([1] Load sklearn dataset, "
                                "[2] Load .csv file): ") or 1)
        if self.loader == 1:
            self.location = input("Enter the dataset name (iris, boston, "
                                  "diabetes, digits, linnerud): ") or 'iris'
        else:
            self.location = input("Enter the filename: ") or 'car.csv'

        self.split_amount = float(input("Enter split percentage as decimal "
                                        "(default = 0.7): ") or 0.7)
        self.classifier['type'] = input("Enter classification mode "
                                        "(hardcoded, knn, id3): ") or 'knn'
        if 'knn' in self.classifier['type']:
            self.classifier['k'] = int(input("Enter a value for k "
                                             "(default = 1): ") or 1)
        # if 'id3' in classifier['type']:
        # return location, split_amount, classifier

    def process_data(self, training_dataset, testing_dataset, classifier):
        accuracy = 0

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
        self.getInput()
        dataset = dl.load_dataset(self.location)
        training_dataset, testing_dataset = dl.split_dataset(dataset,
                                                             self.split_amount)
        self.process_data(training_dataset, testing_dataset, self.classifier)
        # self.console_messages(dataset, training_dataset, testing_dataset)

if __name__ == "__main__":
    run = Run()
    run.main(sys.argv)
