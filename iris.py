# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:55:00 2016

@author: Tyler Scott
"""


from sklearn import datasets
from random import shuffle
from numpy.random import permutation
from numpy import split
from copy import deepcopy as dc

iris = datasets.load_iris()
def shuffle_dataset(dataset):
    data = dataset.data
    target = dataset.target
    zipData = list(zip(data, target))
    shuffle(zipData)
    data[:], target[:] = zip(*zipData)
    dataset.data = data
    dataset.target = target
    return dataset    
    
def split_dataset(dataset):
    data = dataset.data
    target = dataset.target
    train = dc(dataset)
    predict = dc(dataset)
    predict.data, train.data=data[:len(data)//3],data[len(data)//3:]
    predict.target, train.target=target[:len(target)//3],target[len(target)//3:]
#    print("trainData length:", len(train.data))
#    print("trainTarget length:", len(train.target))
#    print("predictData length:", len(predict.data))
#    print("predictTarget length:", len(predict.target))
    return train, predict

def train(dataset):
    if dataset: 
        return True
    else:
        return False

def predict(dataset):
    if dataset:
        return True
    else:
        return False

shuffled_dataset = shuffle_dataset(iris)
training_dataset, predicting_dataset = split_dataset(shuffled_dataset)


print("iris length:", len(iris))
print("shuffled_dataset length:", len(shuffled_dataset))
print("training_dataset length:", len(training_dataset))
print("predicting_dataset length:", len(predicting_dataset))

print("training_dataset.data:\n", training_dataset.data)
print("training_dataset.target:\n", training_dataset.target)
print("training_dataset.target_names:\n", training_dataset.target_names)

print("predicting_dataset.data:\n", predicting_dataset.data)
print("predicting_dataset.target:\n", predicting_dataset.target)
print("predicting_dataset.target_names:\n", predicting_dataset.target_names)

print("train_dataset.data length:", len(training_dataset.data))
print("train_dataset.target length:", len(training_dataset.target))
print("predicting_dataset.data length:", len(predicting_dataset.data))
print("predicting_dataset.target length:", len(predicting_dataset.target))

print(train(training_dataset))
print(predict(predicting_dataset))

def parse_dataset(dataset):
    newData = {}
    
    for i in range(len(dataset.data)):
        newData.append({
            "index" : i,
            "sepal_length": dataset.data[i][0],
            "sepal_width": dataset.data[i][1],
            "petal_length": dataset.data[i][2],
            "petal_width": dataset.data[i][3],    
        })
    return newData