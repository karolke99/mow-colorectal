import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from utils import cross_validation, calculate_original_metrics

df = pd.read_csv('./best_std_filled_knn.csv')
original_df_path = '../newDataset/filled_knn.csv'

X = df.drop(['optime'], axis=1)
y = df['optime']

# Split datasets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

X_init = pd.concat([X_train, X_eval], ignore_index=True)
y_init = pd.concat([y_train, y_eval], ignore_index=True)
# Cross validation
model = CatBoostRegressor(random_state=1, verbose=0)
cv_results = cross_validation(model, X_init, y_init, 5, original_df_path)
print(f'CV results: {cv_results}')

model = CatBoostRegressor(random_state=1, verbose=0)
# GridSearch
param_grid = {
    'iterations': [i * 50 for i in range(1, 7)],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [30, 50, 60],
    'loss_function': ['RMSE', 'MAE', 'Poisson', 'MAPE'],
    'bagging_temperature':  [0.5, 1, 2],
    'random_strength': [0, 0.5, 0.7],
    'depth': [2, 4, 8],
    'od_type': ['IncToDec', 'Iter']
}
#
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=10)
# grid_search = RandomizedSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=True)
grid_search.fit(X_init, y_init)
print(grid_search.best_params_)
print(grid_search.best_score_)

best_model = CatBoostRegressor(**grid_search.best_params_, random_state=1, verbose=0)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
rmse, r2 = calculate_original_metrics(original_df_path, y_pred, y_test)
print(f'Metrics: {rmse}, R2_score: {r2}')

best_model_cv = CatBoostRegressor(**grid_search.best_params_, random_state=1, verbose=0)
cv_results = cross_validation(best_model_cv, X_init, y_init, 5, original_df_path)
print(f'CV results: {cv_results}')
