import pandas as pd
from data_remaker import feature_engineering
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = feature_engineering(df)

X = df.drop('категория', axis=1)
X = pd.get_dummies(data=X, columns=['first_prod', 'gender', 'flg_currency'], drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

svc = SVC()
param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 1],
              'gamma': ['scale', 'auto'],
              'degree': [2, 3, 4]}

grid = GridSearchCV(svc, param_grid)
grid.fit(X_train, y_train)
print(grid.best_params_)
