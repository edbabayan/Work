import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from precision_counter import precision_func
from recall_counter import recall_func
from data_remaker import feature_engineering

df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = feature_engineering(df)

X = df.drop('категория', axis=1)
X = pd.get_dummies(data=X, columns=['first_prod', 'gender', 'flg_currency'], drop_first=True)
df = pd.get_dummies(df['категория'], drop_first=True)
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model_mlp = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(10, 30, 10),
                          learning_rate='constant', solver='adam', max_iter=1000)
model_rfc = RandomForestClassifier(n_estimators=128, max_features=3)
model_ada = AdaBoostClassifier(n_estimators=59)
model_svc = SVC(C=1, degree=2, gamma='scale')
alg_models = [model_mlp, model_rfc, model_ada, model_svc]

recall_metric_test = []
precision_metric_test = []

for i in range(len(alg_models)):
    rec = recall_func(alg_models[i], X_train, y_train, X_test, y_test)
    recall_metric_test.append(rec)
    precision = precision_func(alg_models[i], X_train, y_train, X_test, y_test)
    precision_metric_test.append(precision)

print("Recall for test")
print(f"MLP: {recall_metric_test[0]}")
print(f"RFC: {recall_metric_test[1]}")
print(f"AdaBoost: {recall_metric_test[2]}")
print(f"SVC: {recall_metric_test[3]}")
print('\n')
print("Precision for test")
print(f"MLP: {precision_metric_test[0]}")
print(f"RFC: {precision_metric_test[1]}")
print(f"AdaBoost: {precision_metric_test[2]}")
print(f"SVC: {precision_metric_test[3]}")
print('\n')

recall_metric_val = []
precision_metric_val = []

for i in range(len(alg_models)):
    rec = recall_func(alg_models[i], X_train, y_train, X_val, y_val)
    recall_metric_val.append(rec)
    precision = precision_func(alg_models[i], X_train, y_train, X_val, y_val)
    precision_metric_val.append(precision)

print("Recall for validation")
print(f"MLP: {recall_metric_val[0]}")
print(f"RFC: {recall_metric_val[1]}")
print(f"AdaBoost: {recall_metric_val[2]}")
print(f"SVC: {recall_metric_val[3]}")
print('\n')
print("Precision for validation")
print(f"MLP: {precision_metric_val[0]}")
print(f"RFC: {precision_metric_val[1]}")
print(f"AdaBoost: {precision_metric_val[2]}")
print(f"SVC: {precision_metric_val[3]}")

alg_names = [model.__class__.__name__ for model in alg_models]

fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=200)
axs[0, 0].plot(alg_names, recall_metric_val, '-gD', markevery=0.3, label='line with markers')
axs[0, 0].set_title('Recall for validation')
axs[0, 0].set_ylabel('Recall for validation')
axs[0, 0].set_ylim([0.9, 1])
axs[0, 1].plot(alg_names, precision_metric_val, '-gD', markevery=0.3, label='line with markers')
axs[0, 1].set_title('Precision for validation')
axs[0, 1].set_ylabel('Precision for validation')
axs[0, 1].set_ylim([0.9, 1])
axs[1, 0].plot(alg_names, recall_metric_test, '-bD', markevery=0.3, label='line with markers')
axs[1, 0].set_title('Recall for test')
axs[1, 0].set_ylabel("Recall for test")
axs[1, 0].set_ylim([0.9, 1])
axs[1, 1].plot(alg_names, precision_metric_test, '-bD', markevery=0.3, label='line with markers')
axs[1, 1].set_title('Precision for test')
axs[1, 1].set_ylabel('Precision for test')
axs[1, 1].set_ylim([0.9, 1])

plt.savefig('results.jpg')
