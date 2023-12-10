import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
import numpy as np

files = [
    'min_max_filled_knn.csv',
    'std_filled_knn.csv'
]

for file in files:
    df = pd.read_csv(f'../newDataset/{file}')
    X = df.drop(['weight', 'height'], axis=1)
    correlation_matrix = X.corr()
    plt.figure(figsize=(55, 40))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title(f'Macierz Korelacji {file}')
    plt.savefig(f'{file}_correlation_matrix.png')

    plt.figure()
    plt.boxplot(correlation_matrix['optime'])
    plt.title(f'Boxplot wskaźników korelacji zmiennych dla {file}')
    plt.savefig(f'{file}_relevant_threshold_boxplot.png')

    relevant_threshold = 0.05
    optime_correlations = correlation_matrix.columns[abs(correlation_matrix['optime']) >= relevant_threshold]
    relevant_features = optime_correlations.tolist()

    print(relevant_features)
    print(f'Number of relevant features: {len(relevant_features)}')

    df[relevant_features].to_csv(f'./selected_{file}', index=False)


for file in files:
    df = pd.read_csv(f'../newDataset/{file}')
    X = df.drop(['weight', 'height', 'optime'], axis=1)
    y = df['optime']

    model = CatBoostRegressor()
    model.load_model(f'../models/{file}.cbm')

    data_pool = Pool(X, y)

    feature_importance = model.get_feature_importance(data_pool, type='PredictionValuesChange')
    feature_names = X.columns
    sorted_idx = feature_importance.argsort()

    plt.figure(figsize=(18, 14))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.xlabel('Istotność zmiennych')
    plt.title(f'Istotność zmiennych dla modelu {file}')
    plt.savefig(f'{file}_feature_importance.png')

