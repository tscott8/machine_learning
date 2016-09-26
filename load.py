from sklearn import datasets
from random import shuffle
from numpy.random import permutation
from copy import deepcopy as dc
import csv
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.datasets.base import Bunch


class Loader:

    def load_car_dataset(self, location):
        df = pd.read_csv(location)
        replaces = ('vhigh', 4), ('high', 3), ('med', 2), ('low', 1),\
            ('5more', 6), ('more', 5), ('small', 1), ('big', 3), ('unacc', 0),\
            ('acc', 1), ('good', 2), ('vgood', 3), ('1', 1), ('2', 2),\
            ('3', 3), ('4', 4), ('5', 5), ('6', 6)
        df = reduce(lambda a, kv: a.replace(*kv), replaces, df)
        car = df.values
        dataset = Bunch()
        dataset['data'], dataset['target'] = car[:, :6], car[:, 6]
        dataset['target_names'] = ['unacc', 'acc', 'good', 'vgood']
        return dataset

    def load_csv(self, location):
        dataset = csv.reader(open(location, "rb"), skipinitialspace=False)
        if 'car.csv' in location:
            dataset = self.load_car_dataset(location)
        return dataset

    def shuffle_dataset(self, dataset):
        # unneccessary with the split_dataset implemented thanks to permutation
        data = dataset.data
        target = dataset.target
        zipData = list(zip(data, target))
        shuffle(zipData)
        data[:], target[:] = zip(*zipData)
        dataset.data = data
        dataset.target = target
        return dataset

    def split_dataset(self, dataset, split_amount):
        split_index = split_amount * len(dataset.data)
        split_index = int(split_index)
        indices = permutation(len(dataset.data))
        train = dc(dataset)
        test = dc(dataset)
        train.data, train.target = dataset.data[indices[:split_index]],\
            dataset.target[indices[:split_index]]
        test.data, test.target = dataset.data[indices[split_index:]],\
            dataset.target[indices[split_index:]]
        return train, test

    def load_dataset(self, location):
        dataset = []
        if '.csv' in location:
            dataset = self.load_csv("./datasets/"+location)
        else:
            method = 'load_{0}'.format(location)
            dataset = getattr(datasets, method)()
        return dataset

# dl = Loader()
# dl.load_car_dataset('./datasets/car.csv')
