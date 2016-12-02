from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import neural_network as mlp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split as tts
from sklearn import datasets
import pandas as pd


def load_dataset(s):
    return s.data, s.target


def load_file(file):
    """
    Will split the dataset into the data and the targets if being read from a csv file.
    :param file: the name of the file to be read in
    :return: the data and the targets of the set
    """

    df = pd.read_csv(file)

    data = df.ix[:, df.columns != "className"]
    targets = df.ix[:, df.columns == "className"]

    # names = df.columns
    # n = names[:-1]

    return data.values, targets.values


def get_accuracy(res, tt):
    num_cor = 0

    for r, t in zip(res, tt):
        if r == t:
            num_cor += 1

    # print("Number of correct predictions:", num_cor, " of ", tt.size)
    print("Accuracy rate is {0:.2f}%\n".format((num_cor / tt.size) * 100))


def data_processing(d_data, d_target, classifier):
    # user input for how much should be test and random state being used
    ts = .3
    # while ts < .1 or ts > .5:
    #     ts = float(input("Percentage of data for testing (Enter value between .1 and .5): "))

    rs = 12
    # while rs <= 0:
    #     rs = int(input("Random state for shuffling (Enter positive integer): "))

    # split the data into test and training sets after it shuffles the data
    train_data, test_data, train_target, test_target = tts(d_data, d_target, test_size=ts, random_state=rs)

    ans = None

    # Neural Network
    if classifier == 'n' or classifier == 'N':
        c = mlp.MLPClassifier()
        c.fit(train_data, train_target)
        ans = c.predict(test_data)
    # K Nearest Neighbors
    elif classifier == 'k' or classifier == 'K':
        k = KNeighborsClassifier(5)
        k.fit(train_data, train_target)
        ans = k.predict(test_data)
    # SVM
    elif classifier == 's' or classifier == 'S':
        s = svm.SVC()
        s.fit(train_data, train_target)
        ans = s.predict(test_data)
    # Decision Tree
    elif classifier == 't' or classifier == 'T':
        t = DecisionTreeClassifier()
        t.fit(train_data, train_target)
        ans = t.predict(test_data)
    # ensemble
    elif classifier == 'e' or classifier == 'E':
        ensemble = input("Which ensemble learning:\nA: Adaboosting\nB: Bagging\nF: Random Forest\n>> ")
        if ensemble == 'f' or ensemble == 'F':
            forest = RandomForestClassifier()
            forest.fit(train_data, train_target)
            ans = forest.predict(test_data)
        elif ensemble == 'a' or ensemble == 'A':
            ada = AdaBoostClassifier()
            ada.fit(train_data, train_target)
            ans = ada.predict(test_data)
        elif ensemble == 'b' or ensemble == 'B':
            bag = BaggingClassifier()
            bag.fit(train_data, train_target)
            ans = bag.predict(test_data)
        else:
            print("Invalid command")
    elif classifier == 'a' or classifier == 'A':
        print("Neural Network")
        c = mlp.MLPClassifier()
        c.fit(train_data, train_target)
        ans = c.predict(test_data)
        get_accuracy(ans, test_target)

        print("K nearest neighbors")
        k = KNeighborsClassifier(5)
        k.fit(train_data, train_target)
        ans = k.predict(test_data)
        get_accuracy(ans, test_target)

        print("Support Vector Machine")
        s = svm.SVC()
        s.fit(train_data, train_target)
        ans = s.predict(test_data)
        get_accuracy(ans, test_target)

        print("Decision Tree")
        t = DecisionTreeClassifier()
        t.fit(train_data, train_target)
        ans = t.predict(test_data)
        get_accuracy(ans, test_target)

        print("Adaboosting")
        ada = AdaBoostClassifier()
        ada.fit(train_data, train_target)
        ans = ada.predict(test_data)
        get_accuracy(ans, test_target)

        print("Bagging")
        bag = BaggingClassifier()
        bag.fit(train_data, train_target)
        ans = bag.predict(test_data)
        get_accuracy(ans, test_target)

        print("Random Forrest")
        forest = RandomForestClassifier()
        forest.fit(train_data, train_target)
        ans = forest.predict(test_data)
        get_accuracy(ans, test_target)

    else:
        print("Invalid command\n")

    # get the accuracy
    if classifier != 'a' and classifier != 'A':
        get_accuracy(ans, test_target)


def main():
    # load the data from the database - choose which data set you want to use

    data, targets = load_dataset(datasets.load_iris())

    # data, targets = load_file('../datasets/letters.csv')

    # data, targets = load_file('../datasets/abalone2.csv')

    # data, targets = load_file('../datasets/customer.csv')

    classifier = input("Which classifier would you like to run?\nN: neural network\n" +
                       "K: K nearest neighbors\nT: decision tree\nS: support vector machine\n" +
                       "T: Decision Tree\nE: ensemble learning\nA: Run all\n>> ")

    data_processing(data, targets, classifier)


if __name__ == "__main__":
    main()
