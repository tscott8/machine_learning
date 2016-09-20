# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:55:00 2016

@author: Tyler Scott
"""
import sys
from load import Dataset_Loader as dl
from classifiers.hardcoded import Hard_Coded as hc


def getInput():
    location = input("If you are loading a csv please input the filename, otherwise input the name of the dataset: ") or 'iris'
    opts = input("Enter loading options separated by commas (options available = shuffled, split): ") or 'shuffled'
    split_amount = 0.0
    if 'split' in opts:
        split_amount = input("Enter the split percentage as a decimal (default is 0.7): ") or 0.7
    print("loading dataset @ " + location + " with " + opts +" enabled...")
    return location, opts, split_amount

def process_data(datas):
    # for i in range(len(datasets)):
    #     for j in range(len(datasets[i].target)):
    #         print("j ", datasets[i].target[j])0
    hc.train(datas[0].data, datas[0].target)
    hc.predict(datas[1].data, datas[1].target)

def main(args):
    location, opts, split_amount = getInput()
    print("SPLIT AMT:", split_amount)

    dataset = dl.load_dataset(location, opts)
    training_dataset, testing_dataset = dl.split_dataset(dataset, split_amount)

    # print("dataset length:", len(dataset))
    # print("training_dataset length:", len(training_dataset))
    # print("testing_dataset length:", len(testing_dataset))
    #
    # print("training_dataset.data:", training_dataset.data)
    # print("training_dataset.target:", training_dataset.target)
    # print("training_dataset.target_names:", training_dataset.target_names)
    #
    # print("testing_dataset.data:", testing_dataset.data)
    # print("testing_dataset.target:", testing_dataset.target)
    # print("testing_dataset.target_names:", testing_dataset.target_names)
    #
    # print("train_dataset.data length:", len(training_dataset.data))
    # print("train_dataset.target length:", len(training_dataset.target))
    # print("testing_dataset.data length:", len(testing_dataset.data))
    # print("testing_dataset.target length:", len(testing_dataset.target))
    process_data([training_dataset, testing_dataset])
    hc.printAll(dataset)

if __name__ == "__main__":
    main(sys.argv)
