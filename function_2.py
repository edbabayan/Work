import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

def svc(data):
    X = data.drop(columns=data.columns[0], axis=1)
    y = data.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=30, stratify=y_train)

    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_val = scaler.fit(X_val)
    scaled_X_test =  scaler.fit(X_test)

    svr = SVR()
    param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 1],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['scale', 'auto'],
                  'degree': [2, 3, 4],
                  'epsilon': [0, 0.01, 0.1, 0.5, 1, 2]}

    grid = GridSearchCV(svr, param_grid)
    grid.fit(scaled_X_train, y_train)
    grid_pred_val = grid.predict(scaled_X_val)
    val_mean = mean_absolute_error(y_val, grid_pred_val)
    print(f'This is the mean absolute error of val {val_mean}')
    print('\n')

    grid_pred = grid.predict(scaled_X_test)
    test_mean = mean_absolute_error(y_test, grid_pred)
    print(f'This is the mean absolute error of test {test_mean}')