import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from data_remaker import featue_engineering
from precision_counter import precision_func
from recall_counter import recall_func

df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = featue_engineering(df)

X = pd.get_dummies(df.drop('категория', axis=1), drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

model_rfc = RandomForestClassifier(n_estimators=128, max_features=3)
model_rfc.fit(X_train, y_train)
df_1 = pd.DataFrame(data=model_rfc.feature_importances_, index=X.columns)
print(np.sum(df_1 == 0))

columns = []

for i in df_1.index:
    if df_1.loc[i, 0] == 0:
        columns.append(i)
    else:
        pass

X.drop(columns, axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
scaled_X_val = scaler.transform(X_val)

presicion_val = precision_func(model_rfc, scaled_X_train, y_train, scaled_X_val, y_val)
print(f"Precision for validation is {presicion_val}")
recall_val = recall_func(model_rfc, scaled_X_train, y_train, scaled_X_val, y_val)
print(f"Recall for validation is {recall_val}")
precision_test = precision_func(model_rfc, scaled_X_train, y_train, scaled_X_test, y_test)
print(f"Precision for test is {precision_test}")
recall_test = recall_func(model_rfc, scaled_X_train, y_train, scaled_X_test, y_test)
print(f"Recall for test is {recall_test}")
