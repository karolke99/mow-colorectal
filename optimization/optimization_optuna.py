import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from utils import cross_validation
import optuna
from utils import calculate_original_metrics

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


def objective(trial):
    param_grid = {
        'iterations': trial.suggest_int('iterations', 1, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 50, 1),
        'loss_function': trial.suggest_categorical('loss_function', ['RMSE', 'MAE', 'Poisson', 'MAPE']),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'depth': trial.suggest_int('depth', 1, 10),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
    }

    cb_model = CatBoostRegressor(**param_grid, random_state=1, verbose=0)
    # cb_model.fit(X_train, y_train)
    # y_pred = cb_model.predict(X_eval)

    # rmse, r2 = calculate_original_metrics(original_df_path, y_pred, y_eval)

    cv_results = cross_validation(cb_model, X_init, y_init, 5, original_df_path)
    rmse = cv_results[0]
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_jobs=-1, n_trials=1000)
trial = study.best_params
print(trial)

print(f'Best: params: {study.best_params}')
print(f'Best value: {study.best_value}')
print(f'Best trial: {study.best_trial}')

best_params = study.best_params
best_params.update({'random_state': 1, 'verbose': 0})

best_model = CatBoostRegressor(**best_params)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
rmse, r2 = calculate_original_metrics('../newDataset/filled_knn.csv', y_pred, y_test)

print(f'Metrics: {rmse}, R2_score: {r2}')


best_model_cv = CatBoostRegressor(**best_params)
cv_results = cross_validation(best_model_cv, X_init, y_init, 5, original_df_path)
print(f'CV results: {cv_results}')
