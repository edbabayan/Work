import pandas as pd
from data_remaker import featue_engineering
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = featue_engineering(df)

X = pd.get_dummies(df.drop('категория', axis=1), drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

model_mlp = MLPClassifier()
param_grid = {'hidden_layer_sizes': [(10, 30, 10), (20,)],
              'activation': ['tanh', 'relu'],
              'solver': ['sgd', 'adam'],
              'alpha': [0.0001, 0.05],
              'learning_rate': ['constant', 'adaptive']}

grid = GridSearchCV(model_mlp, param_grid)
grid.fit(X_train, y_train)
print(grid.best_params_)


svc = SVC()
param_grid_1 = {'C': [0.001, 0.01, 0.1, 0.5, 1],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3, 4]}
