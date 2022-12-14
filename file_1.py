import pandas as pd
from function_1 import featue_engineering
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from algoritm import random_forest

df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df.dropna(axis=0, inplace=True)

df = featue_engineering(df)

X = pd.get_dummies(df.drop('категория', axis=1), drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)


rfc = RandomForestClassifier(n_estimators=128, max_features=3)
n_estimators = [64, 100, 128, 200]
max_features = [2, 3, 4]
param_grid = {'n_estimators': n_estimators,
              'max_features': max_features}
grid = GridSearchCV(rfc, param_grid)
"""This one is for gridsearch"""
#grid.fit(X_train, y_train)
#print(random_forest(grid, X_train, y_train, X_test, y_test))

"""This one is for random forest algorithm"""
#print(random_forest(rfc, X_train, y_train, X_val, y_val))
#print(random_forest(rfc, X_train, y_train, X_test, y_test))

ada_model = AdaBoostClassifier(n_estimators=59)
print(random_forest(ada_model, X_train, y_train, X_val, y_val))
print(random_forest(ada_model, X_train, y_train, X_test, y_test))