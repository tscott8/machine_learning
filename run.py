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


def getInput():
    location = input("If you are loading a csv please input the filename, "
                     "otherwise input the name of the dataset: ") or 'iris'
    split_amount = float(input("Enter split percentage as decimal "
                               "(default = 0.7): ") or 0.7)
    classifier = {}
    classifier['type'] = input("Enter classification mode "
                               "(modes = hardcoded, knn): ") or 'knn'
    if 'knn' in classifier['type']:
        classifier['k'] = int(input("Enter a value for k "
                                    "(default = 1): ") or 1)
    return location, split_amount, classifier


def process_data(training_dataset, testing_dataset, classifier):
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


def console_messages(dataset, training_dataset, testing_dataset):
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


def printAll(dataset):
    table = [['sep_len', 'sep_wid', 'pet_len', 'pet_wid', 'class']]
    for i in range(len(dataset.data)):
        row = []
        for j in range(len(dataset.data[i])):
            row.append(str(dataset.data[i][j]))
        row.append(dataset.target_names[dataset.target[i]])
        table.append(row)
    pprint(table)


def main(args):
    dl = Loader()
    location, split_amount, classifier = getInput()
    dataset = dl.load_dataset(location)
    training_dataset, testing_dataset = dl.split_dataset(dataset, split_amount)
    process_data(training_dataset, testing_dataset, classifier)
    # console_messages(dataset, training_dataset, testing_dataset)

if __name__ == "__main__":
    main(sys.argv)
