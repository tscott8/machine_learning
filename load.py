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

    def load_car(self):
        df = pd.read_csv('datasets/car.csv')
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

    def load_loans(self):
        df = pd.read_csv('datasets/loans.csv')
        loans = df.values
        dataset = Bunch()
        dataset['data'], dataset['target'] = loans[:, :4], loans[:, 4]
        dataset['target_names'] = np.unique(dataset['target'])
        return dataset

    def load_lenses(self):
        df = pd.read_csv('datasets/lenses.csv', header=None)
        lenses = df.values
        dataset = Bunch()
        dataset['data'], dataset['target'] = lenses[:, :4], lenses[:, 4]
        dataset['target_names'] = np.unique(dataset['target'])
        return dataset

    def load_votes(self):
        df = pd.read_csv('datasets/votes.csv', header=None)
        # replaces = ('?', 2), ('y', 1), ('n', 0),\
        # ('republican', 1), ('democrat', 0)
        # df = reduce(lambda a, kv: a.replace(*kv), replaces, df)
        votes = df.values
        dataset = Bunch()
        dataset['data'], dataset['target'] = votes[:, 1:], votes[:, 0]
        dataset['target_names'] = np.unique(dataset['target'])
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
        if ('iris' or 'boston' or 'diabetes' or
                'digits' or 'linnerud') in location:
            method = 'load_{0}'.format(location)
            dataset = getattr(datasets, method)()
        else:
            method = 'load_{0}'.format(location)
            dataset = getattr(self, method)()
        return dataset
