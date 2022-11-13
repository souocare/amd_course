import sys
import numpy as np
import pandas as pd
import Orange as ON
import sklearn

filename = "datasets/fpa_dataset.tab"
dataset = ON.data.Table(filename)

final_data = pd.DataFrame(dataset)
print(final_data)
X = final_data.drop("Cultivator", axis=1) #retirar a coluna "cultivator" pois esta é a coluan que contém a classe
y = final_data["Cultivator"] #guardar as classes em 1 variavel única

X_train, X_test, y_train, y_test = sklearn.train_test_split(X, y) #fazer a separação dos dados em valores de treino e teste aleatóriamente usando os valores existentes inicialmente

print(len(X_train))
print(len(y_train))