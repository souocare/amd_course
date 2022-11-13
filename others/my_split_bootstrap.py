#!/usr/bin/python
# -*- coding: utf-8 -*-

# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v03
# ===============================


# _______________
# library import
# _______________
from sklearn.utils import resample
from pandas import DataFrame
from numpy import array
# abc = Abstract Base Classes
from abc import ABC, abstractmethod


# ______________________________________________________________________________
# Classes
# abstract class that defines the general "split" behaviour
# ____
class MyBootstrap(ABC):
    @abstractmethod
    def get_seed(): pass

    def __init__(self, seed=None):
        self.seed = seed
        self.reset_tt_split_indexes()

    def reset_tt_split_indexes(self):
        self.tt_split_indexes = None

    # train|test split
    def split(self, X, y=None):
        # if train|test split already exists, then return
        if self.tt_split_indexes != None:
            return self.tt_split_indexes

        # _____________________________
        # build a new train|test split
        dim_dataset = len(X)
        indexes = list(range(dim_dataset))

        # training set is created from resamples (samples with reposition)
        seed = self.get_seed()
        #print( "seed = ", seed )
        train_indexes = resample(indexes, n_samples=None, random_state=seed)

        # testing set is created from individuals, i, not in training set
        test_indexes = [i for i in indexes if i not in train_indexes]

        self.tt_split_indexes = [(train_indexes, test_indexes)]
        return self.tt_split_indexes


# ____
class MyBootstrapSplitOnce(MyBootstrap):
    def get_seed(self):
        return self.seed


# ____
class MyBootstrapSplitRepeated(MyBootstrap):
    def __init__(self, n_repeat, seed=None):
        super().__init__(seed)
        self.n_repeat = n_repeat
        self.tt_split_repeated_indexes = None

    def get_seed(self):
        seed_current = self.seed
        if self.seed != None:
            self.seed = self.seed + 1
        return seed_current

    def split(self, X, y=None):
        # if train|test split-repeated already exists, then return
        if self.tt_split_repeated_indexes != None:
            return self.tt_split_repeated_indexes

        # ______________________________________
        # build a new train|test split-repeated
        self.tt_split_repeated_indexes = list()
        for i in range(self.n_repeat):
            self.reset_tt_split_indexes()
            self.tt_split_repeated_indexes = self.tt_split_repeated_indexes + \
                super().split(X)
        return self.tt_split_repeated_indexes


# ______________________________________________________________________________
# for testing purposes
# ______________________________________________________________________________
def simple_dataset():
    data = [[11, 22, 0],
            [12, 23, 1],
            [13, 24, 1],
            [14, 25, 0],
            [15, 26, 0],
            [16, 27, 1]]
    data = DataFrame(data)
    data.columns = ["x1", "x2", "y"]
    #print( data )
    #print( type( data ) )
    return data


# ______________________________________________________________________________
# ______________________________________________________________________________
# the "main" for testing purposes
def main():
    D = simple_dataset()
    X = D.values[:, 0:-1]
    y = D.values[:, -1]
    # <your-code-here>
    seed = 5  # 55 #None

    print(2*"\n" + "<<< ----- >>>")
    bs = MyBootstrapSplitOnce(seed)
    tt_split_indexes = bs.split(X)
    print("tt_split_indexes | once\n", tt_split_indexes)

    # 5 c)
    n_repeat = 5
    print(2*"\n" + "<<< ----- >>>")
    bsR = MyBootstrapSplitRepeated(n_repeat, seed)
    tt_split_indexes = bsR.split(X)
    print("tt_split_indexes | repeated\n", tt_split_indexes)


# ______________________________________________________________________________
# The "main" of this module (in case it was not loaded from another module)
if __name__ == "__main__":
    main()
