import numpy as np
import scipy.stats as stats
from sklearn.datasets.base import Bunch


class Tree(object):
    def __init__(self):
        self.data = None
        self.branches = None


class ID3_Decision_Tree(object):

    def __init__(self):
        self.data = None
        self.target = None
        self.target_names = None
        self.traits = Bunch()
        self.tree = Tree()
        self.prettyTree = []


    def train(self, data, target):
        self.data = data
        self.target = target
        self.target_names = np.unique(target)
        self.get_traits(data)
        columns = list(range(0, len(self.data[0])))
        self.build_tree(self.tree, self.data, self.target, columns, self.traits)

    def predict(self, data):
        targets = []
        # targets.append(self.traverse(self.tree, data[0]))
        for row in data:
            targets.append(self.traverse(self.tree, row))
            # print('columm: ', col)
            # for item in col:
            #     print(self.tree.branches)
            # subNode = self.tree.branches[item]
            #     print(subNode)
            #     print(node)
                # print('item: ', item)
                # targets.append(self.tree.branches[item].data)
                # targets.append(self.traverse(node.branches[subNode[node.data]], subNode)
                # targets.append(self.traverse(node, subNode))


            # for j in range(len(data[i])):
            #     print(data[i][j])
            #     print(self.tree.branches[data[i][j]])
            #     print(self.tree.branches[data[i][j]].data)
            # targets.append(self.traverse(self.tree.branches, self.tree.branches))
        print(targets)
        print(self.prettyTree)
        return targets

    def accuracy(self, predicted, actual):
        denominator = len(actual)
        numerator = 0
        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                numerator += 1
        percent = float((numerator/denominator)*100)
        return percent

###############################################################
    def get_traits(self, data):
        for i in range(len(data[0])):
            self.traits[i] = np.unique(data[:, i])

    def entropy(self, val):
        return (-val * np.log2(val)) if val != 0 else 0

    def calc_probability(self, data, target, column, traits):
        probability = Bunch()
        weight = Bunch()
        for t in traits:
            probability[t] = Bunch()
            weight[t] = 0
            for tn in self.target_names:
                probability[t][tn] = 0
        for i in range(len(data)):
            probability[data[i, column]][target[i]] += 1
            weight[data[i, column]] += 1
        for t in traits:
            for tn in self.target_names:
                probability[t][tn] /= weight[t]
        return probability, weight

    def calc_entropy(self, data, target, column, traits):
        probability, weight = self.calc_probability(data, target, column, traits)
        # print(probability)
        entropy = 0
        for trait in probability:
            trait_entropy = 0
            for target_name in probability[trait]:
                if probability[trait][target_name] != 0.0:
                    # trait_entropy += entropy(probability[trait][target_name])
                    trait_entropy -= probability[trait][target_name] * np.log2(probability[trait][target_name])
            # entropy = stats.entropy(probability, base=2)
            entropy += weight[trait] / len(data) * trait_entropy
        return entropy

    def calc_entropies(self, data, target, columns, traits):
        entropies = Bunch()
        for column in columns:
            entropies[column] = self.calc_entropy(data, target, column, traits[column])
        return entropies

    def build_tree(self, node, data, target, columns, traits):
        # print(columns)
        if len(self.target_names) == 1:
            node.data = target[0]
            return node
        elif len(columns) < 2:
            node.data = columns[0]
            if node.data not in self.prettyTree:
                self.prettyTree.append(node.data)
            return node
        else:
            entropies = self.calc_entropies(data, target, columns, traits)
            # print(entropies)
            column, best = -1, 0
            for col in entropies:
                if entropies[col] > best:
                    best = entropies[col]
                    column = col

            columns.remove(column)
            node.data = column
            node.branches = Bunch()
            if node.data not in self.prettyTree:
                self.prettyTree.append(node.data)

        print('tree', node.data)
        # self.prettyTree.append(node.data)
        for trait in traits[col]:
            node.branches[trait] = Tree()
            self.build_tree(node.branches[trait], data, target, columns, traits)

    def traverse(self, node, subNode):
        if node.branches is not None:
            return self.traverse(node.branches[subNode[node.data]], subNode)
        else:
            return node.data



# dt = ID3_Decision_Tree()
# dataset = [0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1]
# print(dt.choose_best(dataset))
# print(dt.calc_total_entropy([1, 0, 1, 0, 1]))
# print(stats.entropy([3/5, 2/5], base=2))
