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

    def load_car(self, df):
        # df = pd.read_csv(location)
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

    def load_votes(self, df):
        # replaces = ('?', 2), ('y', 1), ('n', 0), ('republican', 1), ('democrat', 0)
        # df = reduce(lambda a, kv: a.replace(*kv), replaces, df)
        votes = df.values
        dataset = Bunch()
        dataset['data'], dataset['target'] = votes[:, 1:], votes[:, 0]
        dataset['target_names'] = np.unique(dataset['target'])
        return dataset

    def load_csv(self, location):
        df = pd.read_csv(location, header=None)
        # traits = []
        # for i in df.columns:
        #     traits.append(np.unique(df[i].values.ravel()))
        if 'car.csv' in location:
            dataset = self.load_car(df)
        if 'votes.csv' in location:
            dataset = self.load_votes(df)
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
#
# dl = Loader()
# print(dl.load_csv('./datasets/votes.csv'))
