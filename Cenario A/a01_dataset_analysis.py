
# _______________________________________________________________________________
# Modules to Evaluate
import sys
from u01_util import my_print
import Orange as DM
import numpy as np
from sklearn import metrics
import sklearn


# _______________________________________________________________________________
# Auxiliary Functions
# _______________________________________________________________________________
# load the dataset
def load(fileName):
    try:
        dataset = DM.data.Table(fileName)
    except:
        my_print("--->>> error - can not open the file: %s" % fileName)
        exit()
    return dataset


# _______________________________________________________________________________
# get the variable dataset-structure given a string with its name
def get_variableFrom_str(dataset, str_name):
    variable_list = dataset.domain.variables
    for variable in variable_list:
        if(variable.name == str_name):
            return variable
    my_print(">>error>> \"{}\" is not a variable name in dataset!".format(str_name))
    return None


# _______________________________________________________________________________
# Data Analysis Functions
# _______________________________________________________________________________
# percentage of missing values of each variable
def get_missingValuePercentage(dataset):
    variable_list = dataset.domain.variables
    var_len = len(variable_list)
    missingValue_list = [0.0] * var_len
    for instance in dataset:
        for var_index in range(var_len):
            if np.isnan(instance[var_index]):
                missingValue_list[var_index] += 1
    missingValue_list = list(map(lambda x, dim=len(dataset): x / dim*100.0, missingValue_list))
    return (variable_list, missingValue_list)


# _______________________________________________________________________________
# contingencyMatrix; i.e., the joint-frequency table
# this implementation does not account for missing-values
# (i.e., missing-values are not included in the variables-domain)
# M(row, column)
def get_contingencyMatrix(dataset, rowVar, colVar):
    if(isinstance(rowVar, str)):
        rowVar = get_variableFrom_str(dataset, rowVar)
    if(isinstance(colVar, str)):
        colVar = get_variableFrom_str(dataset, colVar)
    if(not (rowVar and colVar)):
        return ([], [], None)
    if(not (rowVar.is_discrete and colVar.is_discrete)):
        my_print(">>error>> variables are expected to be discrete")
        return ([], [], None)

    rowDomain, colDomain = rowVar.values, colVar.values
    len_rowDomain, len_colDomain = len(rowDomain), len(colDomain)
    contingencyMatrix = np.zeros((len_rowDomain, len_colDomain))
    for instance in dataset:
        rowValue, colValue = instance[rowVar], instance[colVar]
        if(np.isnan(rowValue) or np.isnan(colValue)):
            continue

        rowIndex, colIndex = rowDomain.index(rowValue), colDomain.index(colValue)
        contingencyMatrix[rowIndex, colIndex] += 1
    return (rowDomain, colDomain, contingencyMatrix)


# _______________________________________________________________________________
# P( H | E )
# H means Hypothesis, E means Evidence
# a frequency approach
def get_conditionalProbability(dataset, H, E):
    if(isinstance(H, str)):
        H = get_variableFrom_str(dataset, H)
    if(isinstance(E, str)):
        E = get_variableFrom_str(dataset, E)
    if(not (H and E)):
        return ([], [], None)
    (rowDomain, colDomain, cMatrix) = get_contingencyMatrix(dataset, H, E)

    len_rowDomain, len_colDomain = len(rowDomain), len(colDomain)
    E_marginal = np.zeros(len_colDomain)
    for col in range(len_colDomain):
        E_marginal[col] = sum(cMatrix[:, col])

    for row in range(len_rowDomain):
        for col in range(len_colDomain):
            cMatrix[row, col] = cMatrix[row, col] / E_marginal[col]
    return (rowDomain, colDomain, cMatrix)


# _______________________________________________________________________________
# 1R Related Functions
# _______________________________________________________________________________
# error matrix for a given feature and considering the datatset class
def get_errorMatrix(dataset, feature):
    if(isinstance(feature, str)):
        feature = get_variableFrom_str(dataset, feature)
    the_class = dataset.domain.class_var
    (rowDomain, colDomain, cMatrix) = get_conditionalProbability(dataset, the_class, feature)
    if(not (rowDomain or colDomain)):
        return ([], [], None)

    errorMatrix = 1 - cMatrix
    return (rowDomain, colDomain, errorMatrix)


# _______________________________________________________________________________
# Data Presentation Functions
# _______________________________________________________________________________
# show contingency matrix
def show_contingencyMatrix(dataset, rowVar, colVar):
    my_print("{} & {} :".format(rowVar.name, colVar.name))
    (rowDomain, colDomain, cMatrix) = get_contingencyMatrix(dataset, rowVar, colVar)
    print(cMatrix)
    for row in rowDomain:
        showStr = "<" + row + "> "
        for col in colDomain:
            showStr += col + ": " + str(cMatrix[rowDomain.index(row), colDomain.index(col)]) + " | "
        print(showStr)


# _______________________________________________________________________________
# show all contingency matrix
# - for each feature and the class
def showAll_contingencyMatrix(dataset):
    feature_list, the_class = dataset.domain.attributes, dataset.domain.class_var
    for feature in feature_list:
        show_contingencyMatrix(dataset, feature, the_class)
        print()


# _______________________________________________________________________________
# P( H | E )
# show matriz and textual description
def show_conditionalProbability(dataset, H, E):
    (rowDomain, colDomain, cMatrix) = get_conditionalProbability(dataset, H, E)
    print(cMatrix)
    print()

    for h in rowDomain:
        for e in colDomain:
            rowIndex, colIndex = rowDomain.index(h), colDomain.index(e)
            P_h_e = cMatrix[rowIndex, colIndex]
            print("  P({} | {}) = {:.3f}".format(h, e, P_h_e))


def getLowestErrorFeature(dataset, saveFile=False):
    text = '1R\n( attr, valueAttr, valueTarget ) : (error, total)\n\n'

    featureErrors = {}
    feature_list, the_class = dataset.domain.attributes, dataset.domain.class_var
    for feature in feature_list:
        (rowDomain, colDomain, cMatrix) = get_contingencyMatrix(dataset, feature, the_class)

        valueError = 0
        featureErrors[feature.name] = 0
        for row in rowDomain:
            freqs = cMatrix[rowDomain.index(row), :]
            rowTotal = sum(freqs)
            errors = list(rowTotal - freqs)
            valueError += min(errors)
            featureErrors[feature.name] += rowTotal

            error = min(errors)
            text += f'( {feature.name}, {row}, {colDomain[errors.index(error)]} ) : ({error}, {rowTotal})\n'

        featureErrors[feature.name] = valueError / featureErrors[feature.name]

    if (saveFile):
        with open('oneR_OUTPUT.txt', 'w') as f:
            f.write(text)

    for feature in featureErrors:
        if(featureErrors[feature] == min(featureErrors.values())):
            return feature


# _______________________________________________________________________________
# implementation of some test cases
def test():
    # fileName = "./datasets/lenses_fromLecture"
    fileName = "lenses_fromLecture"
    dataset = load(fileName)

    the_feature = getLowestErrorFeature(dataset)

    print()
    H = "lenses"
    E = the_feature
    aStr = ">> P( %s | %s ) <<" % (H, E)
    my_print(aStr)
    show_conditionalProbability(dataset, H, E)

    print()
    aStr = "(1R-approach) >>Error Matrix>> %s & %s <<" % (the_feature, dataset.domain.class_var)
    my_print(aStr)
    (classDomain, featureDomain, errorMatrix) = get_errorMatrix(dataset, the_feature)
    if(not (classDomain or featureDomain)):
        return
    print(classDomain)
    print(featureDomain)
    print(errorMatrix)
    print()
    print("1R for the '{}' feature are:".format(the_feature))
    for feature in range(len(featureDomain)):
        errorFeature = errorMatrix[:, feature]
        errorMin = min(errorFeature)
        errorMinIndex = errorFeature.tolist().index(errorMin)
        featureValue = featureDomain[feature]
        classValue = classDomain[errorMinIndex]
        showStr = "(" + the_feature + ", " + featureValue + ", " + classValue + ") : "
        print(showStr + "{:.3f}".format(errorMin))





# _______________________________________________________________________________
# the main of this module (in case this module is imported from another module)
if __name__ == "__main__":
    test()
