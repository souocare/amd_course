# %%
#!/usr/bin/python
# -*- coding: utf-8 -*-

# ===============================
# author: Paulo Trigo Silva (PTS)
# version: v02
# ===============================


# ______________________________________________________________________________
# Use the virtual environment but not its python interpreter
# _PATH_virtualEnv = "/ptrigo/_MyPython/__VirtualEnv"
# _activate_this = _PATH_virtualEnv + "/bin/activate_this.py"
# exec( open( _activate_this ).read(), dict(__file__=_activate_this) )


# _______________
# library import
# _______________
from u01_util import my_print
import Orange as DM

from pandas import read_csv, DataFrame
from numpy import array, set_printoptions

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, \
    KFold, StratifiedKFold, \
    RepeatedKFold, RepeatedStratifiedKFold, \
    LeaveOneOut, LeavePOut, \
    cross_val_score
from my_split_bootstrap import MyBootstrap, \
    MyBootstrapSplitOnce, MyBootstrapSplitRepeated

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OrdinalEncoder

from sklearn.exceptions import UndefinedMetricWarning
import warnings
# NOTE: use "ignore" (instead of "always") if you want to avoid warning due to:
# - "no predicted samples" (thrown by score_precision and f_score metrics)
# - "no true samples" (thrown by score_recall and f_score metrics)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)  # use "always" or "ignore"

# utility to check the version of each library
# import pandas, numpy, sklearn
# print('pandas: {}'.format(pandas.__version__))
# print('numpy: {}'.format(numpy.__version__))
# print('sklearn: {}'.format(sklearn.__version__))


# Utility Functions
# __________________
# summarize data
def show_data(data, numFirstRows=10):
    firstRows = data[0:numFirstRows]
    set_printoptions(precision=3)
    print(">> summarized data (max = {n:d} instances)".format(n=numFirstRows),
          firstRows, sep="\n")


def show_train_test_split(X, y, tt_split_indexes, numFirstRows=10):
    set_printoptions(precision=3)
    for (train_index, test_index) in tt_split_indexes.split(X, y):
        print("summarized data (max = {n:d} instances)".format(n=numFirstRows))
        train_index = train_index[0:numFirstRows]
        test_index = test_index[0:numFirstRows]
        print("\n> train-indexes, test-indexes")
        print("train-indexes:", train_index)
        print(" test-indexes:", test_index)
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index],  y[test_index]
        print("\nX_train, y_train")
        print(X_train, y_train, sep="\n")
        print("\nX_test, y_test")
        print(X_test, y_test, sep="\n")


def show_score(score_all):
    if not score_all:
        return
    print("::all-evaluated-datasets::")

    all_evaluated_datasets = [i*100.0 for i in score_all]
    for v in all_evaluated_datasets:
        print(" %.2f%% " % (v), end="|")
    print()

    if isinstance(score_all, list):
        score_all = array(score_all)
    print("%.2f%% (+/- %.2f%%)" %
          (score_all.mean()*100.0, score_all.std()*100.0))


def show_function_name(tag, func):
    print()
    print("_" * len(tag))
    print(tag + " ", end="")
    print(func.__name__)


# _____________________
# for testing purposes
def simple_dataset():
    data = [[11, 22, 0],
            [12, 23, 1],
            [13, 24, 1],
            [14, 25, 0],
            [15, 26, 0],
            [16, 27, 1]]
    data = DataFrame(data)
    data.columns = ["x1", "x2", "y"]
    # print( data )
    # print( type( data ) )
    return data

# ____________________________________
# Data-Split & Score Recipe Functions
# ____________________________________
# load dataset
# either from the fileName or by executing the func_datasetLoader function


def load_dataset(fileName, featureName=None, func_datasetLoader=None):
    # see if is to load from a function (by executing func_datasetLoader)
    if func_datasetLoader:
        return func_datasetLoader()

    # otherwise, load from "fileName"
    D = read_csv(fileName, names=featureName, dtype=str)
    return D


# split feature, X, from class, y, attributes
# (assume that class is the "last column")
def split_dataset_Xy(D):
    # all rows and all columns except last column
    X = D.values[:, 0:-1]

    # all rows and just the last column
    y = D.values[:, -1]
    return (X, y)


# _____________________________
# (some) Data-Split techniques
# _____________________________
def holdout(test_size, seed=None):
    # yields indices to random split once the data into:
    # - one train dataset and one test dataset
    tt_split_indexes = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return tt_split_indexes


def stratified_holdout(test_size, seed=None):
    # yields indices to random split of data into:
    # - one train dataset and one test dataset
    # - datasets preserve the percentage of samples for each class
    tt_split_indexes = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return tt_split_indexes


def repeated_holdout(test_size, n_repeat, seed=None):
    # yields indices to random split of data into:
    # - a given number (n_repeat) of training and test datasets
    tt_split_indexes = ShuffleSplit(n_splits=n_repeat, test_size=test_size, random_state=seed)
    return tt_split_indexes


def repeated_stratified_holdout(test_size, n_repeat, seed=None):
    # yields indices to random split of data into:
    # - a given number (n_repeat) of training and test datasets
    # - datasets preserve the percentage of samples for each class
    tt_split_indexes = StratifiedShuffleSplit(n_splits=n_repeat, test_size=test_size, random_state=seed)
    return tt_split_indexes


def fold_split(k_folds, seed=None):
    # yields indices to random split of data into:
    # - k consecutive folds (k_folds parameter)
    # - each fold used once as validation; the k-1 remaining folds as training dataset
    # - shuffle=True, so dataset is shuffled (random split) before building the folds
    tt_split_indexes = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    return tt_split_indexes


def stratified_fold_split(k_folds, seed=None):
    # yields indices to random split of data into:
    # - k consecutive folds (k_folds parameter)
    # - each fold used once as validation; the k-1 remaining folds as training dataset
    # - shuffle=True, so dataset is shuffled (random split) before building the folds
    # - datasets preserve the percentage of samples for each class
    tt_split_indexes = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    return tt_split_indexes


def repeated_fold_split(k_folds, n_repeat, seed=None):
    # yields indices to random split of data into:
    # - k consecutive folds (k_folds parameter)
    # - each fold used once as validation; the k-1 remaining folds as training dataset
    # - repeats (n_repeat times) KFold with different randomization in each repetition
    tt_split_indexes = RepeatedKFold(n_splits=k_folds, n_repeats=n_repeat, random_state=seed)
    return tt_split_indexes


def repeated_stratified_fold_split(k_folds, n_repeat, seed=None):
    # yields indices to random split of data into:
    # - k consecutive folds (k_folds parameter)
    # - each fold used once as validation; the k-1 remaining folds as training dataset
    # - repeats (n_repeat times) StratifiedKFold with different randomization in each repetition
    tt_split_indexes = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=n_repeat, random_state=seed)
    return tt_split_indexes


def leave_one_out():
    # yields indices to split of data so that:
    # - each sample is used once as a test set (singleton), and
    # - the remaining samples form the training set
    # - same as KFold(n_splits=n), where n is number of dataset samples
    tt_split_indexes = LeaveOneOut()
    return tt_split_indexes


def leave_p_out(p):
    # yields indices to split of data so that:
    # - testing on all distinct samples of size p, and
    # - the remaining n-p samples form the training set in each iteration
    # - NOT same as KFold(n_splits=n_samples//p), which creates non-overlapping test sets
    tt_split_indexes = LeavePOut(p=p)
    return tt_split_indexes


# different implementation because:
# sklearn does not directly implements bootstrap, so
# - it must use "resample" which does not return indexes
def bootstrap_split_once(seed=None):
    tt_split_indexes = MyBootstrapSplitOnce(seed=seed)
    return tt_split_indexes


# different implementation because:
# sklearn does not directly implements bootstrap, so
# - it must use "resample" which does not return indexes
def bootstrap_split_repeated(n_repeat, seed=None):
    tt_split_indexes = MyBootstrapSplitRepeated(n_repeat=n_repeat, seed=seed)
    return tt_split_indexes


# ____________________________________
# the general Train-Test-Split Recipe
# ____________________________________
def train_test_split_recipe(D, func_tt_split, *args_tt_split):
    # split the dataset into features, X, and class, y, attributes
    (X, y) = split_dataset_Xy(D)
    # get train and test (tt) split indexes
    tt_split_indexes = func_tt_split(*args_tt_split)
    return (X, y, tt_split_indexes)


# _________________________
# the general Score Recipe
# _________________________
def score_recipe(classifier, X, y, tt_split_indexes, f_score, **keyword_args_score):
    score_all_list = list()

    for (train_index, test_index) in tt_split_indexes.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test,  y_test = X[test_index],  y[test_index]

        # fit (build) model using classifier, X_train and y_train (training dataset)
        classifier.fit(X_train, y_train)

        # predict using the model and X_test (testing dataset)
        y_predict = classifier.predict(X_test)

        # score the model using y_test (expected) and y_predict (predicted by model)
        score = f_score(y_test, y_predict, **keyword_args_score)

        score_all_list.append(score)

    return score_all_list


'''
# ______________________________________________________________________________
# lists to define the:
# - train|test split methods
# - classification techniques
# - score metrics
# ______________________________________________________________________________
seed = 5  # used by random generator (value is integer or None)

# _________________________
# train|test split methods
list_func_tt_split = \
    [
        # (holdout, (1.0/3.0, seed)),
        # (stratified_holdout, (1.0/3.0, seed)),
        # (repeated_holdout, (1.0/3.0, 2, seed)),
        (repeated_stratified_holdout, (1.0/3.0, 10, seed)),
        # (fold_split, (3, seed)),
        # (stratified_fold_split, (3, seed)),
        # (repeated_fold_split, (3, 2, seed)),
        # (repeated_stratified_fold_split, (3, 2, seed)),
        # (leave_one_out, ()),
        # (leave_p_out, (2, )),
        # (bootstrap_split_once, (seed, )),
        # (bootstrap_split_repeated, (2, seed))
    ]


# __________________________
# classification techniques
list_func_classifier = \
    [
        # (GaussianNB, ()),  # NB
        # (DecisionTreeClassifier, ())  # ID3
    ]


# ______________
# score metrics
list_score_metric = \
    [
        (accuracy_score, {}),
        # (precision_score, {"average": "weighted"}),  # macro #micro #weighted
        # (recall_score, {"average": "weighted"}),  # macro #micro #weighted
        # (f1_score, {"average": "weighted"}),  # macro #micro #weighted
        # (cohen_kappa_score, {}),
    ]

# ACCURACY:
# - the number of predictions that exactly match the expected class value

# PRECISION:
# - the ratio tp / (tp + fp)
# where tp is the number of true positives and fp the number of false positives
# - intuitively the ability not to label as positive a sample that is negative

# RECALL:
# - the ratio tp / (tp + fn)
# where tp is the number of true positives and fn the number of false negatives
# - intuitively the ability of the classifier to find all the positive samples

# F1:
# - the ratio 2 * (precision * recall) / (precision + recall)
#  weighted average of the precision and recall
# F1 score reaches its best value at 1 and worst score at 0
# F1 score is also known as balanced F-score or F-measure

# KAPPA:
# - Cohen kappa expresses the level of agreement between two annotators
# on a classification problem. It is defined as,
# k = (p_o - p_e) / (1 - p_e)
# where p_o is empirical probability of agreement (the observed agreement ratio),
# and p_e is expected agreement when both annotators assign labels randomly
# - the kappa statistic is a number between -1 and 1
# - the maximum value means complete agreement
# - zero or lower means chance agreement


# _____________________________________________________
# - the file name of dataset and list of feature names
# and
# - a function that returns a dataset (or None)
# _____________________________________________________
fileName = "./datasets/fpa_dataset.csv"
featureName = ['age', 'tearRate', 'isMyope', 'isAstigmatic', 'isHypermetrope', 'prescribedLenses']

func_datasetLoader = None  # None (if we want to load the "fileName) # simple_dataset

# ______________________________________________________________________________
# ______________________________________________________________________________


def main():
    D = load_dataset(fileName, featureName=featureName, func_datasetLoader=func_datasetLoader)
    show_data(D)

    for (f_tt_split, args_tt_split) in list_func_tt_split:
        (X, y, tt_split_indexes) = train_test_split_recipe(D, f_tt_split, *args_tt_split)
        show_function_name("train_test_split:", f_tt_split)
        show_train_test_split(X, y, tt_split_indexes, numFirstRows=10)

        for (f_classifier, args_classifier) in list_func_classifier:
            classifier = f_classifier(*args_classifier)
            show_function_name("classifier:", f_classifier)

            for (f_score, keyword_args_score) in list_score_metric:
                score_all = score_recipe(classifier, X, y, tt_split_indexes, f_score, **keyword_args_score)
                show_function_name("score_method:", f_score)
                show_score(score_all)

        print(2*"\n" + "<<< ----- >>>" + 2*"\n")


# ______________________________________________________________________________
# The "main" of this module (in case it was not loaded from another module)
if __name__ == "__main__":
    main()
'''
