import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
from catboost import CatBoostRegressor, Pool
import numpy as np

df = pd.read_csv('../newDataset/min_max_filled_knn.csv')
X = df.drop(['weight', 'height'], axis=1)

correlation_matrix = X.corr()
plt.figure(figsize=(55,40))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Macierz Korelacji')
plt.savefig('correlation_matrix.png')



X = df.drop(['weight', 'height', 'optime'], axis=1)
y = df['optime']

model = CatBoostRegressor()
model.load_model('../models/std_filled_knn.csv.cbm')



data_pool = Pool(X, y)


feature_importance = model.get_feature_importance(data_pool, type="PredictionValuesChange")
feature_names = X.columns
sorted_idx = feature_importance.argsort()

plt.figure(figsize=(18, 14))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
plt.xlabel("Istotność zmiennych")
plt.title("Istotność zmiennych dla modelu std_filled_knn")
plt.savefig('feature_importance_1.png')

