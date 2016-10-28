# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:55:00 2016

@author: Tyler Scott
"""
import sys
import numpy as np
from pprint import pprint
from load import Loader
from sklearn import datasets
from matplotlib import pyplot as plt

from classifiers.hardcoded import Hard_Coded
from classifiers.knn import k_Nearest_Neighbor
from classifiers.dtree import ID3_Decision_Tree
from classifiers.nnetwork import Neural_Network

import warnings
warnings.filterwarnings("ignore")

class Run:

    def __init__(self):
        self.location = 'prima'
        self.split_amount = float(0.7)
        self.classifier = {}
        self.classifier['type'] = 'nn'
        self.classifier['k'] = 1
        self.classifier['lp'] = [5]
        self.dataset = None

    def getInput(self):
        self.location = input("Enter the dataset name (iris, boston, "
                              "diabetes, digits, linnerud, car, "
                              "lenses, votes, prima): ") or 'iris'
        self.split_amount = float(input("Enter split percentage as decimal "
                                        "(default = 0.7): ") or 0.7)
        self.classifier['type'] = input("Enter classification mode "
                                        "(hardcoded, knn, id3, nn): ") or 'nn'
        if 'knn' in self.classifier['type']:
            self.classifier['k'] = int(input("Enter a value for k "
                                             "(default = 1): ") or 1)
        if 'nn' in self.classifier['type']:
            lp = [int(s) for s in input("Enter layer parameters (i.e. 3 4 3 1): ").split()] or [3]
            self.classifier['lp'] = lp
        return self.location, self.split_amount, self.classifier

    def process_data(self, training_dataset, testing_dataset, classifier):
        accuracy = 0

        if 'nn' is classifier['type']:
            dl = Loader()
            training_dataset['target'] = dl.discretize_targets(training_dataset)
            testing_dataset['target'] = dl.discretize_targets(testing_dataset)
            nn = Neural_Network(layer_params=[len(training_dataset.data[0])] + classifier['lp'] + [len(training_dataset.target_names)])
            train_permutations = []
            predict_permutations = []
            training_acc = []
            predicting_acc = []
            perfect_count = 0
            for i in range(2000):
                train_permutations+=[dl.split_dataset(training_dataset, 0.7)[0]]
                # predict_permutations+=[dl.split_dataset(testing_dataset, 0.7)[0]]

            for epoch in range(len(train_permutations)):
                trained = nn.train(train_permutations[epoch].data, train_permutations[epoch].target)
                # predicted = nn.predict(predict_permutations[epoch].data)
                predicted = nn.predict(testing_dataset.data)

                for i in range(len(trained)):
                    trained[i] = np.array(dl.undiscretize_output(trained[i]))
                for j in range(len(predicted)):
                    predicted[j] = np.array(dl.undiscretize_output(predicted[j]))
                if epoch % 200 is 0 and epoch is not 0:
                    print('Epoch '+str(epoch)+': ',' training accuracy: '+str(round(nn.accuracy(trained, train_permutations[epoch].target), 3))+'% ',
                            ' prediction accuracy: '+str(round(nn.accuracy(predicted, testing_dataset.target),3))+'%')
                training_acc +=[nn.accuracy(trained, train_permutations[epoch].target)]
                # predicting_acc +=[nn.accuracy(predicted, predict_permutations[epoch].target)]
                predicting_acc +=[nn.accuracy(predicted, testing_dataset.target)]
                break_flag = False

                if len(training_acc) >= 4:
                    last_num_train, last_num_predict = len(training_acc)-1, len(predicting_acc)-1
                    last_nums_train = last_nums_predict = []
                    average_train = 0
                    average_predict = 0
                    for num in range(4):
                        last_nums_train += [training_acc[last_num_train-num]]
                        last_nums_predict += [predicting_acc[last_num_predict-num]]
                        average_train = sum(last_nums_train)/len(last_nums_train)
                        average_predict = sum(last_nums_predict)/len(last_nums_predict)
                        if average_predict >= 99:
                            break_flag = True
                            break
                        if average_train >= 99 and average_predict > 90:
                            # print(training_acc[num], training_acc[num - 1], training_acc[num - 2], training_acc[num - 3], training_acc[num - 4])
                            print('HIT PERFECT')
                            perfect_count += 1
                        if perfect_count == 5:
                            break_flag = True
                            break
                if break_flag is True:
                    break

            final_prediction = nn.predict(testing_dataset.data)
            for k in range(len(final_prediction)):
                final_prediction[k] = np.array(dl.undiscretize_output(final_prediction[k]))
            final_accuracy = nn.accuracy(final_prediction, testing_dataset.target)

            plt.plot(training_acc, lw=2.0)
            plt.plot(predicting_acc, lw=2.0)
            plt.title('Network Accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')

            accuracy = final_accuracy

        if 'id3' in classifier['type']:
            id3 = ID3_Decision_Tree()
            id3.train(training_dataset.data, training_dataset.target)
            # predicted = id3.predict(testing_dataset.data)
            result = []
            for i in range(len(testing_dataset.data)):
                result.append(id3.predict(testing_dataset.data[i]))
            accuracy = id3.accuracy(result, testing_dataset.target)

        if 'knn' in classifier['type']:
            knn = k_Nearest_Neighbor(classifier['k'])
            knn.train(training_dataset.data, training_dataset.target)
            accuracy = knn.accuracy(knn.predict(testing_dataset.data),
                                    testing_dataset.target)

        if 'hardcoded' in classifier['type']:
            hc = Hard_Coded()
            hc.train(training_dataset.data, training_dataset.target)
            accuracy = hc.predict(testing_dataset.data, testing_dataset.target)

        print("Method Accuracy = {0}%".format(round(accuracy, 2)))
        plt.show()

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
        # self.getInput()
        dataset = dl.load_dataset(self.location)
        self.dataset = dataset
        training_dataset, testing_dataset = dl.split_dataset(dataset, self.split_amount)
        # self.console_messages(dataset, training_dataset, testing_dataset)
        self.process_data(training_dataset, testing_dataset, self.classifier)

if __name__ == "__main__":
    run = Run()
    run.main(sys.argv)
