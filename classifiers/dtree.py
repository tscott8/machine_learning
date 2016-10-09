import numpy as np
import scipy.stats as stats
import pandas
from sklearn.datasets.base import Bunch


class Node(object):
    def __init__(self):
        self.data = None
        self.branches = None

    def __str__(self, level=0):
        ret = "|---" * level + repr(self.data) + "\n"
        if self.branches is not None:
            for branch in self.branches:
                ret += self.branches[branch].__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node>'


class ID3_Decision_Tree(object):

    def __init__(self):
        self.df = pandas.DataFrame()
        # self.data = None
        # self.target = None
        self.target_names = None
        self.tree = Node()
        self.traits = Bunch()

    def train(self, data, target):
        columns = list(range(0, len(data[0])))
        self.df = pandas.DataFrame(data=data, columns=columns)
        self.df['target'] = target
        self.target_names = np.unique(target)
        self.get_traits(data)
        self.build_tree(self.tree, self.df, columns, self.traits)

    def predict(self, data):
        return self.traverse(self.tree, data)

    def accuracy(self, predicted, actual):
        denominator = len(actual)
        numerator = 0
        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                numerator += 1
        percent = float((numerator/denominator)*100)
        return percent

###############################################################
# HELPER FUNCTIONS
###############################################################

    def get_traits(self, data):
        for i in range(len(data[0])):
            self.traits[i] = np.unique(data[:, i])

    # def entropy(self, val):
    #     return (-val * np.log2(val)) if val != 0 else 0

    def split_df(self, df, columns, column, trait):
        split_df = df[df[column] == trait]
        del split_df[column]
        columns.remove(column)
        return split_df, columns

    def pick_best(self, entropies):
        column, best = -1, 1.1
        for col in entropies:
            if entropies[col] < best:
                best = entropies[col]
                column = col
        return column

    def pick_leaf(self, df, target):
        target_frequencies = Bunch()
        for t in target:
            target_frequencies[t] = df[df["target"] == t].size
        best_target, best_count = '', -1
        for t in target_frequencies:
            if target_frequencies[t] > best_count:
                best_count = target_frequencies[t]
                best_target = t
        return best_target

    def calc_probability(self, df, column, traits):
        probability = Bunch()
        weight = Bunch()
        for trait in traits:
            probability[trait] = Bunch()
            weight[trait] = 0
            for target_name in self.target_names:
                probability[trait][target_name] = 0

        for row in df.iterrows():
            probability[row[1][column]][row[1]['target']] += 1
            weight[row[1][column]] += 1

        for trait in traits:
            for target_name in self.target_names:
                if weight[trait] != 0:
                    probability[trait][target_name] /= weight[trait]
                else:
                    probability[trait][target_name] = 0.0
        return probability, weight

    def calc_entropy(self, df, column, traits):
        probability, weight = self.calc_probability(df, column, traits)
        entropy = 1
        if len(df) != 0:
            for trait in probability:
                trait_entropy = 0
                for target_name in probability[trait]:
                    if probability[trait][target_name] != 0.0:
                        trait_entropy -= probability[trait][target_name] * np.log2(probability[trait][target_name])
            # entropy = stats.entropy(probability, base=2)
                entropy -= weight[trait] / len(df) * trait_entropy
        return entropy

    def calc_entropies(self, df, columns, traits):
        entropies = Bunch()
        for column in columns:
            entropies[column] = self.calc_entropy(df, column, traits[column])
        return entropies

    def build_tree(self, node, df, columns, traits):
        if len(np.unique(df["target"])) == 1:
            node.data = np.unique(df["target"])[0]
        elif len(columns) == 1:
            node.data = self.pick_leaf(df, df["target"])
        else:
            entropies = self.calc_entropies(df, columns, traits)
            best = self.pick_best(entropies)
            node.data = best
            node.branches = Bunch()
            for trait in traits[best]:
                split_df, split_columns = self.split_df(df[:], columns[:], best, trait)
                node.branches[trait] = Node()
                if not split_df.empty:
                    self.build_tree(node.branches[trait],  split_df, split_columns, traits)
                else:
                    node.branches[trait].data = 'undecided'

    def traverse(self, node, subNode):
        if node.branches is not None:
            return self.traverse(node.branches[subNode[node.data]], subNode)
        else:
            return node.data

    def print(self):
        print(self.tree)
