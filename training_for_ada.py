import pandas as pd
import matplotlib.pyplot as plt
from data_remaker import featue_engineering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = featue_engineering(df)

X = pd.get_dummies(df.drop('категория', axis=1), drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

error_rates = []
for n in range(1, 100):
    model = AdaBoostClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    err = 1 - accuracy_score(y_test, prediction)
    error_rates.append(err)

plt.plot(range(1, 100), error_rates)
plt.ylabel('Error rates')
plt.savefig('ada_results.jpg')
