import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from data_remaker import feature_engineering
from precision_counter import precision_func
from recall_counter import recall_func
from sklearn.cluster import DBSCAN

df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = feature_engineering(df)

X = df.drop('категория', axis=1)
X = pd.get_dummies(data=X, columns=['first_prod', 'gender', 'flg_currency'], drop_first=True)
X_cluster = df.drop('категория', axis=1)
X_cluster = pd.get_dummies(data=X_cluster, columns=['first_prod', 'gender', 'flg_currency'], drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

model_rfc = RandomForestClassifier(n_estimators=128, max_features=3)
model_rfc.fit(X_train, y_train)
df_1 = pd.DataFrame(data=model_rfc.feature_importances_, index=X.columns)
df_1.to_excel('feature_importance.xlsx')

X_new = pd.Series(model_rfc.feature_importances_, index=X.columns)
X_new.sort_values(inplace=True)

X_cluster.drop(X_new.iloc[:-5].index, axis=1, inplace=True)
X_cluster['Category'] = y

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X_cluster)

number_of_outliers = []

for eps in np.linspace(0.001, 10, 100):
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(scaled_X)
    number_of_outliers.append(np.sum(dbscan.labels_ == -1))

sns.lineplot(x=np.linspace(0.001, 10, 100), y=number_of_outliers)
plt.savefig('dbscan_outliers.jpg')

model_dbscan = DBSCAN(eps=0.5)
cluster = model_dbscan.fit_predict(scaled_X)

X_cluster['CLuster'] = cluster
X_cluster.drop('Category', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_cluster, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

print(recall_func(model_rfc, X_train, y_train, X_test, y_test))
print(precision_func(model_rfc, X_train, y_train, X_test, y_test))
print(recall_func(model_rfc, X_train, y_train, X_val, y_val))
print(precision_func(model_rfc, X_train, y_train, X_val, y_val))
