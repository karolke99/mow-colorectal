import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


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


files = [
    ('std_filled_average.csv', 'filled_average.csv'),
    ('std_filled_knn.csv', 'filled_knn.csv'),
    ('std_filled_median.csv', 'filled_median.csv'),
    ('min_max_filled_average.csv', 'filled_average.csv'),
    ('min_max_filled_knn.csv', 'filled_knn.csv'),
    ('min_max_filled_median.csv', 'filled_median.csv')
]

for file, file2 in files:
    print('\nFile: ', file)
    df = pd.read_csv('../dataset/' + file)
    df2 = pd.read_csv('../dataset/' + file2)
    X = df.drop(["weight", "height", "optime"], axis=1)

    mean = df2['optime'].mean()
    std = df2['optime'].std()
    X_train, X_test, y_train, y_test = train_test_split(X, df['optime'], test_size=0.2, random_state=1)

    # Random Forest
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    # predicted_values_original_scale = rf_pred * std + mean
    # y_test_original_scale = y_test * std + mean
    # rf_mse = mean_squared_error(y_test_original_scale, predicted_values_original_scale)
    rf_mse = mean_squared_error(y_test, rf_pred)
    print("Random Forest: ", rf_mse)

    print("Cross validation", cross_validation(rf_model, X, df['optime']))


    # SVC
    # SVR
    svr_model = SVR()
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(X_test)
    svr_mse = mean_squared_error(y_test, svr_pred)
    print("SVR: ", svr_mse)
    print("Cross validation", cross_validation(svr_model, X, df['optime']))

    # NuSVR
    nusvr_model = NuSVR()
    nusvr_model.fit(X_train, y_train)
    nusvr_pred = nusvr_model.predict(X_test)
    nusvr_mse = mean_squared_error(y_test, svr_pred)
    print("NuSVR: ", nusvr_mse)
    print("Cross validation", cross_validation(nusvr_model, X, df['optime']))

    #CatBoost
    cb_model = CatBoostRegressor(random_state=1, verbose=0)
    cb_model.fit(X_train, y_train)
    cb_pred = cb_model.predict(X_test)
    cb_mse = mean_squared_error(y_test, cb_pred)
    print("Catboost: ", rf_mse)
    print("Cross validation", cross_validation(cb_model, X, df['optime']))

    # XGBBoost
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    print("XGBoost: ", xgb_mse)
    print("Cross validation", cross_validation(xgb_model, X, df['optime']))
