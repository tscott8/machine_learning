from sklearn import datasets
from random import shuffle
from numpy.random import permutation
from copy import deepcopy as dc
import csv
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import normalize
from sklearn import cross_validation

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
        dataset['target_names'] = np.unique(dataset['target'])
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

    def load_prima(self):
        df = pd.read_csv('datasets/prima.csv', header=None)
        # replaces = ('?', 2), ('y', 1), ('n', 0),\
        # ('republican', 1), ('democrat', 0)
        # df = reduce(lambda a, kv: a.replace(*kv), replaces, df)
        votes = df.values
        dataset = Bunch()
        dataset['data'], dataset['target'] = votes[:, :8], votes[:, 8]
        dataset['target_names'] = np.unique(dataset['target'])
        return dataset

    def split_dataset(self, dataset, split_amount):
        # old way:
        # split_index = split_amount * len(dataset.data)
        # split_index = int(split_index)
        # indices = permutation(len(dataset.data))
        # train = dc(dataset)
        # test = dc(dataset)
        # train.data, train.target = dataset.data[indices[:split_index]],\
        #     dataset.target[indices[:split_index]]
        # test.data, test.target = dataset.data[indices[split_index:]],\
        #     dataset.target[indices[split_index:]]

        # new way:
        train = Bunch()
        test = Bunch()
        split_amount = 1 - split_amount
        train['data'], test['data'], train['target'], test['target'] = cross_validation.train_test_split(dataset.data, dataset.target, test_size=split_amount)
        train['target_names'] = test['target_names'] = dataset.target_names
        return train, test

    def load_dataset(self, location):
        dataset = []
        if str(location) in ('iris', 'boston', 'diabetes', 'digits', 'linnerud'):
            method = 'load_{0}'.format(location)
            dataset = getattr(datasets, method)()
        else:
            method = 'load_{0}'.format(location)
            dataset = getattr(self, method)()
        return dataset

    def discretize_targets(self, dataset):
        target_names = []
        for i in range(len(np.unique(dataset.target))):
            new_target_name = [0]*len(np.unique(dataset.target))
            new_target_name[i] = 1
            target_names += [new_target_name]
        target_names = np.array(target_names)
        target = []
        for i in range(len(dataset.target)):
            if 0 in dataset.target:
                new_target = target_names[dataset.target[i]]
            else:
                new_target = target_names[dataset.target_names.tolist().index(dataset.target[i])]
            target += [new_target]
        return np.array(target)

    def undiscretize_output(self, item):
        max_index = item.index(max(item))
        item[max_index] = 1
        for i in range(len(item)):
            if item[i] != 1:
                item[i] = int(0)
        return item

# dl = Loader()
# dataset = dl.load_dataset('prima')
# dataset['target'] = dl.discretize_targets(dataset)
# print(dataset.data)
# print(dataset.target)
# print(dataset.target_names)
