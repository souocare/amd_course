from my_split_and_eval import *
import joblib

# Get method to use
# ______________


def getMethod():
    validInput = False
    while not validInput:
        method = input('Method (1r | id3 | nb): ')
        if method != '1r' and method != 'id3' and method != 'nb':
            print('Invalid method')
            validInput = False
        else:
            validInput = True

    return method


# Get patient information
# _______________
def getPatient():
    validInput = False
    while not validInput:
        age = input('Patient age (young | presbyopic | pre-presbyopic): ')
        if age != 'young' and age != 'presbyopic' and age != 'pre-presbyopic':
            print('Invalid age')
            validInput = False
        else:
            validInput = True

    validInput = False
    while not validInput:
        tearRate = input('Patient tear-rate (normal | reduced): ')
        if tearRate != 'normal' and tearRate != 'reduced':
            print('Invalid tearRate')
            validInput = False
        else:
            validInput = True

    validInput = False
    while not validInput:
        isMyope = input('Patient isMyope (true | false): ')
        if isMyope != 'true' and isMyope != 'false':
            print('Invalid isMyope')
            validInput = False
        else:
            validInput = True

    validInput = False
    while not validInput:
        isAstigmatic = input('Patient isAstigmatic (true | false): ')
        if isAstigmatic != 'true' and isAstigmatic != 'false':
            print('Invalid isAstigmatic')
            validInput = False
        else:
            validInput = True

    validInput = False
    while not validInput:
        isHypermetrope = input('Patient isHypermetrope (true | false): ')
        if isHypermetrope != 'true' and isHypermetrope != 'false':
            print('Invalid isHypermetrope')
            validInput = False
        else:
            validInput = True

    print()
    return age, tearRate, isMyope, isAstigmatic, isHypermetrope


# Load dataset
# ________________
def loadDataset():
    fileName = "fpa_dataset.csv"
    featureName = ['age', 'tearRate', 'isMyope', 'isAstigmatic', 'isHypermetrope', 'prescribedLenses']
    func_datasetLoader = None

    dataset = load_dataset(fileName, featureName=featureName, func_datasetLoader=func_datasetLoader)
    return dataset


# Print result
# ___________________________
def printLenses(lenses=None):
    if lenses == 'none':
        print('The patient does not need to wear lenses')
    else:
        print(f'The patient needs to wear \'{lenses}\' lenses')


# _________
# ID3 or NB
# _________

# ___________________________
# lists to define the:
# - train|test split methods
# - classification techniques
# - score metrics
# ___________________________
seed = 5

# ________________________
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
        # (repeated_stratified_fold_split, (3, 10, seed)),
        # (leave_one_out, ()),
        # (leave_p_out, (2, )),
        # (bootstrap_split_once, (seed, )),
        # (bootstrap_split_repeated, (10, seed))
    ]

# _____________
# score metrics
list_score_metric = \
    [
        (accuracy_score, {}),
        (precision_score, {"average": "weighted"}),  # macro #micro #weighted
        (recall_score, {"average": "weighted"}),  # macro #micro #weighted
        (f1_score, {"average": "weighted"}),  # macro #micro #weighted
        (cohen_kappa_score, {}),
    ]


def fitClassifier(classifier=None):
    '''
    Fit the dataset
    Return the classifier to predict the lenses needed for the patient
    Return the encoder to transform the patient data
    '''

    classifierName = classifier.__name__
    classifier = classifier()

    D = loadDataset()
    # show_data(D)

    for (f_tt_split, args_tt_split) in list_func_tt_split:
        (X, y, tt_split_indexes) = train_test_split_recipe(D, f_tt_split, *args_tt_split)

        # show_function_name("train_test_split:", f_tt_split)
        # show_train_test_split(X, y, tt_split_indexes, numFirstRows=10)

        encoder = OrdinalEncoder()
        encoder.fit(X)
        X = encoder.transform(X)

        for (f_score, keyword_args_score) in list_score_metric:
            score_all = score_recipe(classifier, X, y, tt_split_indexes, f_score, **keyword_args_score)
            show_function_name("score_method:", f_score)
            show_score(score_all)

    print(classifierName)
    joblib.dump(classifier, f'./models/{classifierName}/classifier')
    joblib.dump(encoder, f'./models/{classifierName}/encoder')

    print()
    return classifier, encoder


def modelClassifier(age=None, tearRate=None, isMyope=None, isAstigmatic=None, isHypermetrope=None, fittedClassifier=None, encoder=None):
    '''
    Predict the lenses needed for the patient based on the fitted classifier and transforming the data
    '''

    patient = [[age, tearRate, isMyope, isAstigmatic, isHypermetrope]]
    patient = encoder.transform(patient)
    lenses = fittedClassifier.predict(patient)
    printLenses(lenses[0])


# __
# 1R
# __

# _________________________________________________________________________________________
def model1R(age=None, tearRate=None, isMyope=None, isAstigmatic=None, isHypermetrope=None):
    '''
    Using the file a01_dataset_analysis.py,
    the 1R result says that the feature to be considered is 'isAstigmatic'
    '''

    lenses = 'none' if isAstigmatic == 'false' else 'hard'
    printLenses(lenses)


# ____________
def score1R():
    D = loadDataset()

    y_test = list(D['prescribedLenses'])
    y_predict = ['none' if value == 'false' else 'hard' for value in D['isAstigmatic']]

    for (f_score, keyword_args_score) in list_score_metric:
        score = f_score(y_test, y_predict, **keyword_args_score)
        show_function_name("score_method:", f_score)
        show_score([score])

    print()


# _________
def main():
    method = getMethod()

    print()
    if method == '1r':
        print('1R')
        score1R()
    else:
        if method == 'id3':
            classifier = DecisionTreeClassifier  # GaussianNB() or DecisionTreeClassifier()
            print(classifier.__name__)
        elif method == 'nb':
            classifier = GaussianNB  # GaussianNB() or DecisionTreeClassifier()
            print(classifier.__name__)

        # Comment next line to load only, uncomment to fit and save
        fittedClassifier, encoder = fitClassifier(classifier)  # Train classifier and encoder

        # Load classifier and encoder
        fittedClassifier, encoder = joblib.load(f'./models/{classifier.__name__}/classifier'), joblib.load(f'./models/{classifier.__name__}/encoder')

    age, tearRate, isMyope, isAstigmatic, isHypermetrope = getPatient()

    if method == '1r':
        model1R(age, tearRate, isMyope, isAstigmatic, isHypermetrope)
    elif method == 'id3':
        modelClassifier(age, tearRate, isMyope, isAstigmatic, isHypermetrope, fittedClassifier, encoder)
    elif method == 'nb':
        modelClassifier(age, tearRate, isMyope, isAstigmatic, isHypermetrope, fittedClassifier, encoder)


if __name__ == '__main__':
    main()
