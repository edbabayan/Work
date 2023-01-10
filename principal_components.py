import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_remaker import feature_engineering

df = pd.read_csv('Cross sale k-drivers_v2_1.csv')
df = feature_engineering(df)

X = pd.get_dummies(data=df, columns=['first_prod', 'gender', 'flg_currency', 'категория'], drop_first=True)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

explained_variance = []

for n in range(1, 34):
    pca = PCA(n_components=n)
    pca.fit(scaled_X)
    explained_variance.append(np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(12, 8), dpi=200)
plt.plot(range(1, 34), explained_variance, 'o--')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.savefig('component_variance.jpg')
