import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score


files = [
    ('std_filled_average.csv', 'filled_average.csv'),
    # ('std_filled_knn.csv', 'filled_knn.csv'),
    # ('std_filled_median.csv', 'filled_median.csv'),
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
    predicted_values_original_scale = rf_pred * std + mean
    y_test_original_scale = y_test * std + mean
    rf_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))
    rf_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("Random Forest: ", rf_mse, rf_r2_score)

    # SVC
    # SVR
    svr_model = SVR()
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(X_test)

    predicted_values_original_scale = svr_pred * std + mean
    svr_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))

    svr_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("SVR: ", svr_mse, svr_r2_score)

    # NuSVR
    nusvr_model = NuSVR()
    nusvr_model.fit(X_train, y_train)
    nusvr_pred = nusvr_model.predict(X_test)
    predicted_values_original_scale = nusvr_pred * std + mean
    nusvr_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))

    nusvr_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("NuSVR: ", nusvr_mse, nusvr_r2_score)

    #CatBoost
    cb_model = CatBoostRegressor(random_state=1, verbose=0)
    cb_model.fit(X_train, y_train)
    cb_pred = cb_model.predict(X_test)
    predicted_values_original_scale = cb_pred * std + mean
    cb_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))

    cb_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("Catboost: ", cb_mse, cb_r2_score)

    # XGBBoost
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    predicted_values_original_scale = xgb_pred * std + mean
    xgb_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))

    xgb_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("XGBoost: ", xgb_mse, xgb_r2_score)


files = [
    ('min_max_filled_average.csv', 'filled_average.csv'),
    ('min_max_filled_knn.csv', 'filled_knn.csv'),
    ('min_max_filled_median.csv', 'filled_median.csv')
]

for file, file2 in files:
    print('\nFile: ', file)
    df = pd.read_csv('../dataset/' + file)
    df2 = pd.read_csv('../dataset/' + file2)
    X = df.drop(["weight", "height", "optime"], axis=1)

    # original_min = np.min(df['optime'], axis=0)
    # original_max = np.max(df['optime'], axis=0)
    original_min = df2['optime'].min()
    original_max = df2['optime'].max()
    X_train, X_test, y_train, y_test = train_test_split(X, df['optime'], test_size=0.2, random_state=1)
    y_test_original_scale = y_test * (original_max - original_min) + original_min

    # Random Forest
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    predicted_values_original_scale = rf_pred * (original_max - original_min) + original_min
    rf_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))
    rf_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("Random Forest: ", rf_mse, rf_r2_score)

    # SVC
    # SVR
    svr_model = SVR()
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(X_test)

    predicted_values_original_scale = svr_pred * (original_max - original_min) + original_min
    svr_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))

    svr_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("SVR: ", svr_mse, svr_r2_score)

    # NuSVR
    nusvr_model = NuSVR()
    nusvr_model.fit(X_train, y_train)
    nusvr_pred = nusvr_model.predict(X_test)
    predicted_values_original_scale = nusvr_pred * (original_max - original_min) + original_min
    nusvr_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))

    nusvr_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("NuSVR: ", nusvr_mse, nusvr_r2_score)

    #CatBoost
    cb_model = CatBoostRegressor(random_state=1, verbose=0)
    cb_model.fit(X_train, y_train)
    cb_pred = cb_model.predict(X_test)
    predicted_values_original_scale = cb_pred * (original_max - original_min) + original_min
    cb_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))

    cb_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("Catboost: ", cb_mse, cb_r2_score)

    # XGBBoost
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    predicted_values_original_scale = xgb_pred * (original_max - original_min) + original_min
    xgb_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))

    xgb_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("XGBoost: ", xgb_mse, xgb_r2_score)


# predicted_values = model.predict(scaled_features)  # Zakodowane przez model wartości
# original_min = np.min(original_features, axis=0)
# original_max = np.max(original_features, axis=0)
#
# # Przywracanie pierwotnych wartości
# predicted_values_original_scale = predicted_values * (original_max - original_min) + original_min