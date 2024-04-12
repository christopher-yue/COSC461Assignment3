import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import IsolationForest

df = pd.read_csv('salarydata.csv')
df_10 = df.head(10)

x_train = None
x_test = None
y_train = None
y_test = None

clf = SVC()
param_grids = [{'kernel': ['rbf'], 'C': [15,14,13,12,11,10,1,0.1,0.001]}]
grid = GridSearchCV(clf, param_grids, cv=20)
grid.fit(x_train, y_train)
# print(grid.best_score_, grid.best_params_)

# ======------------======

best_C = grid.best_params_['C']

print('RBF Kernel with gamma as scale')

rbf = SVC(gamma='scale')
param_grids = [{'kernel': ['rbf'], 'C': [best_C]}]
grid = GridSearchCV(rbf, param_grids, cv=20)
grid.fit(x_train, y_train)
print(grid.best_score_, grid.best_params_)

print('RBF Kernel with gamma as auto')

rbf = SVC(gamma='auto')
param_grids = [{'kernel': ['rbf'], 'C': [best_C]}]
grid = GridSearchCV(rbf, param_grids, cv=20)
grid.fit(x_train, y_train)
print(grid.best_score_, grid.best_params_)

print('Grid search for the kernel of Linear')

lin = SVC()
param_grids = [{'kernel': ['linear'], 'C': [best_C]}]
grid = GridSearchCV(lin, param_grids, cv=20)
grid.fit(x_train, y_train)
print(grid.best_score_, grid.best_params_)

print('Grid search for the kernel of Polynomial')

poly = SVC()
param_grids = [{'kernel': ['poly'], 'C': [best_C]}]
grid = GridSearchCV(poly, param_grids, cv=20)
grid.fit(x_train, y_train)
print(grid.best_score_, grid.best_params_)


