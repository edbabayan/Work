import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from data_remaker import feature_engineering
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier


df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = feature_engineering(df)

X = df.drop('категория', axis=1)
X = pd.get_dummies(data=X, columns=['first_prod', 'gender', 'flg_currency'], drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

model_mlp = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(10, 30, 10),
                          learning_rate='constant', solver='adam', max_iter=1000)
model_rfc = RandomForestClassifier(n_estimators=128, max_features=3)
model_ada = AdaBoostClassifier(n_estimators=59)
alg_models = [model_mlp, model_rfc, model_ada]

accuracy_higher_90 = []
accuracy_lower_90 = []

for i in X.columns:
    X_new = X.drop(i, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.15, random_state=42, stratify=y)
    model_mlp.fit(X_train, y_train)
    prediction = recall_score(y_test, model_mlp.predict(X_test))
    print(prediction)
    if prediction > 0.9:
        accuracy_higher_90.append(i)
    else:
        accuracy_lower_90.append(i)

print(f"Columns with recall score lower 90: {accuracy_lower_90}")
print(f"Columns with recall score higher 90: {accuracy_higher_90}")


