# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:55:00 2016

@author: Tyler Scott
"""
import sys
from pprint import pprint
from load import Dataset_Loader as dl
from classifiers.hardcoded import Hard_Coded as hardcoded
from classifiers.knn import k_Nearest_Neighbor as knn

def getInput():
    location = input("If you are loading a csv please input the filename, otherwise input the name of the dataset: ") or 'iris'
    opts = input("Enter loading options separated by commas (options available = shuffled, split): ") or 'shuffled'
    split_amount = 0.7
    if 'split' in opts:
        split_amount = float(input("Enter the split percentage as a decimal (default is 0.7): ") or 0.7)
    print("loading dataset @ " + location + " with " + opts +" enabled...")
    classifier = {}
    classifier['type'] = input("Enter classification mode (moes available = hardcoded, knn): ") or 'hardcoded'
    if 'knn' in classifier['type']:
        classifier['k'] = input("Enter a value for k (default is 1): ") or 1
    return location, opts, split_amount, classifier

def process_data(datas, classifier):
    if 'knn' in classifier['type']:
        knn.train(datas[0].data, datas[0].target)
        knn.predict(datas[1].data, datas[1].target)
    else:
        hardcoded.train(datas[0].data, datas[0].target)
        hardcoded.predict(datas[1].data, datas[1].target)

def console_messages(dataset, training_dataset, testing_dataset):
    print("dataset length:", len(dataset))
    print("training_dataset length:", len(training_dataset))
    print("testing_dataset length:", len(testing_dataset))

    print("training_dataset.data:", training_dataset.data)
    print("training_dataset.target:", training_dataset.target)
    print("training_dataset.target_names:", training_dataset.target_names)
    printAll(training_dataset)

    print("testing_dataset.data:", testing_dataset.data)
    print("testing_dataset.target:", testing_dataset.target)
    print("testing_dataset.target_names:", testing_dataset.target_names)
    printAll(testing_dataset)

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
    location, opts, split_amount, classifier = getInput()
    dataset = dl.load_dataset(location, opts)
    training_dataset, testing_dataset = dl.split_dataset(dataset, split_amount)
    # console_messages(dataset, training_dataset, testing_dataset)
    process_data([training_dataset, testing_dataset], classifier)


if __name__ == "__main__":
    main(sys.argv)
