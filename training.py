import pandas as pd
from function_1 import featue_engineering
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from algoritm import alg_func
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = featue_engineering(df)

X = pd.get_dummies(df.drop('категория', axis=1), drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

model_mlp = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(10, 30, 10), learning_rate='constant', solver='adam')
rfc = RandomForestClassifier(n_estimators=128, max_features=3)
param_grid_rfc = {'n_estimators': [64, 100, 128, 200],
              'max_features': [2, 3, 4]}
grid_rfc = GridSearchCV(rfc, param_grid_rfc)
"""This one is for gridsearch"""
grid_rfc.fit(X_train, y_train)
#print(alg_func(grid_rfc, X_train, y_train, X_test, y_test))

"""This one is for random forest algorithm"""
#print(alg_func(rfc, X_train, y_train, X_val, y_val))
#print(alg_func(rfc, X_train, y_train, X_test, y_test))

ada_model = AdaBoostClassifier(n_estimators=59)
#print(alg_func(model_mlp, X_train, y_train, X_val, y_val))
#print(alg_func(model_mlp, X_train, y_train, X_test, y_test))

svc = SVC()
param_grid_1 = {'C': [0.001, 0.01, 0.1, 0.5, 1],
              'kernel': ['linear', 'rbf', 'poly'],
              'gamma': ['scale', 'auto'],
              'degree': [2, 3, 4]}
grid = GridSearchCV(svc, param_grid_1)
grid.fit(X_train, y_train)
#print(alg_func(grid, X_train, y_train, X_test, y_test))
print(grid.best_params_)