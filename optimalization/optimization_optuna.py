import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def cross_validation(model, X, y, k=5):
    n = len(X.columns) - 1
    fold_size = n // k

    mse_scores = []

    for i in range(k):
        test_indices = list(range(i * fold_size, (i+1) * fold_size))
        train_indices = list(set(range(n)) - set(test_indices))

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mse = mean_squared_error(y_test, pred)
        mse_scores.append(mse)

    average_mse = np.mean(mse_scores)
    min_mse = np.min(mse_scores)

    return average_mse, min_mse, mse_scores


# df = pd.read_csv('../dataset/min_max_filled_knn.csv')
df = pd.read_csv('../optimalization/selected_min_max_filled_knn.csv')
df2 = pd.read_csv('../newDataset/filled_knn.csv')

max = df2['optime'].max()
min = df2['optime'].min()
y_original = df2['optime']

# X = df.drop(["weight", "height", "optime"], axis=1)
X = df.drop(["optime"], axis=1)

X_train, X_temp, y_train, y_temp = train_test_split(X, df['optime'], test_size=0.3, random_state=1)
X_eval, X_test, y_eval, t_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

print(X_train.shape, X_eval.shape, X_test.shape)

cb_model = CatBoostRegressor(random_state=1, verbose=0)
cb_model.fit(X_train, y_train)
cb_pred = cb_model.predict(X_eval)

cb_pred = cb_pred * (max - min) + min
y = y_eval * (y_original.max() - y_original.min()) + y_original.min()
cb_mse = np.sqrt(mean_squared_error(y, cb_pred))

print("Catboost: ", cb_mse)
print("Cross validation", cross_validation(cb_model, X, df['optime']))

def objective(trial):
    param_grid = {
        'iterations': trial.suggest_int('iterations', 1, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 50, 1),
        'loss_function': trial.suggest_categorical('loss_function', ['RMSE', 'MAE', 'Poisson', 'MAPE']),
        # 'boostrap_type': ['MVS', 'Bayesian', 'Bernoulli'],
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        # 'subsample': [0, 1],
        # 'sampling_frequency': ['PerTree', 'PerTreeLevel'],
        # 'sampling_unit': ['Object', 'Group'],
        # 'mvs_reg': [0, 0.5, 1],
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        # 'use_best_model': trial.suggest_categorical('use_best_model', [True, False]),
        # 'best_model_min_tree': [0, 1, 2],

        'depth': trial.suggest_int('depth', 1, 10),
        # 'grow_policy': ['SymmetricTree','Lossguide','Depthwise'],
        # 'min_data_in_leaf': [1, 2, 3],
        # 'max_leaves': [None, 10, 20],
        # 'rsm': [0.01, 0.1, 0.2, 0.5, 0.9, 1],
        # 'leaf_estimation_method': ['Newton', 'Gradient', 'Exact'],
        # 'fold_len_multiplier': [1.5, 2, 2.5, 3, 3.5],
        # 'approx_on_full_history': [True, False],
        # 'boosting_type': ['Ordered', 'Plain'],
        # 'posterior_sampling': [True, False],
        # 'score_function': ['Cosine', 'L2', 'NewtonCosine', 'NewtonL2'],
        # 'monotone_constraints': [1, 0, -1],
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        # 'od_pval': trial.suggest_float('od_pval', 1e-20, 1e-5),
        # 'od_wait': trial.suggest_int('od_wait', 10, 30, 5),
    }

    cb_model = CatBoostRegressor(**param_grid)
    cb_model.fit(X_train, y_train)

    y_pred = cb_model.predict(X_eval)

    y_pred = y_pred * (max - min) + min
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    # mse = mean_squared_error(y_eval, y_pred)
    return rmse



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_jobs=-1, n_trials=2000)
trial = study.best_params
print(trial)

print(f'Best: params: {study.best_params}')
print(f'Best value: {study.best_value}')
print(f'Best trial: {study.best_trial}')
