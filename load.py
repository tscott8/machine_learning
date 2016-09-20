
from sklearn import datasets
from random import shuffle
from numpy.random import permutation
from copy import deepcopy as dc
import csv

class Dataset_Loader:

    def load_csv(location):
        dataset = csv.reader(location, dialect='excel')
        return dataset

    def shuffle_dataset(dataset):
        data = dataset.data
        target = dataset.target
        zipData = list(zip(data, target))
        shuffle(zipData)
        data[:], target[:] = zip(*zipData)
        dataset.data = data
        dataset.target = target
        return dataset

    def split_dataset(dataset, split_amount):
        split_index = split_amount * len(dataset.data)
        split_index = int(split_index)
        indices = permutation(len(dataset.data))
        train = dc(dataset)
        test = dc(dataset)
        train.data, train.target = dataset.data[indices[:split_index]], dataset.target[indices[:split_index]]
        test.data, test.target = dataset.data[indices[split_index:]], dataset.target[indices[split_index:]]
        return train, test

    def load_dataset(location, opts):
        dataset = []
        if '.csv' in location:
            dataset = Dataset_Loader.load_csv("./datasets/"+location)
        else:
            method= 'load_'+ location
            dataset = getattr(datasets, method)()
        if 'shuffled' in opts:
            dataset = Dataset_Loader.shuffle_dataset(dataset)
        return dataset
