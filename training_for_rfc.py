import pandas as pd
from data_remaker import featue_engineering
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = featue_engineering(df)

X = pd.get_dummies(df.drop('категория', axis=1), drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

model = RandomForestClassifier()
param_grid_rfc = {'n_estimators': [64, 100, 128, 200],
                  'max_features': [2, 3, 4]}

grid = GridSearchCV(model, param_grid_rfc)
grid.fit(X_train, y_train)
print(grid.best_params_)
