import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred, model_label):
    plt.figure()
    plt.plot(range(len(y_true)), y_true, label='Actual')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted')
    plt.title('Actual vs Predicted: ' + model_label)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    # plt.show()
    plt.savefig(f'trained_plots/{model_label}.png')
    plt.close()


files = [
    ('std_filled_average.csv', 'filled_average.csv'),
    ('std_filled_knn.csv', 'filled_knn.csv'),
    ('std_filled_median.csv', 'filled_median.csv'),
]

rsme_list = []

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
    # plot_predictions(y_test, rf_pred, 'Random Forest std')
    predicted_values_original_scale = rf_pred * std + mean
    y_test_original_scale = y_test * std + mean

    rf_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))
    rf_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)

    plot_predictions(y_test_original_scale, predicted_values_original_scale, f'Random Forest Std Scaler {file}')
    print("Random Forest: ", rf_mse, rf_r2_score)

    # SVC
    # SVR
    svr_model = SVR()
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(X_test)
    predicted_values_original_scale = svr_pred * std + mean
    svr_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))
    svr_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    plot_predictions(y_test_original_scale, predicted_values_original_scale, f'SVR Std Scaler {file}')
    print("SVR: ", svr_mse, svr_r2_score)

    # NuSVR
    nusvr_model = NuSVR()
    nusvr_model.fit(X_train, y_train)
    nusvr_pred = nusvr_model.predict(X_test)
    predicted_values_original_scale = nusvr_pred * std + mean
    nusvr_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))
    plot_predictions(y_test_original_scale, predicted_values_original_scale, f'NuSVR Std Scaler {file}')
    nusvr_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("NuSVR: ", nusvr_mse, nusvr_r2_score)

    # CatBoost
    cb_model = CatBoostRegressor(random_state=1, verbose=0)
    cb_model.fit(X_train, y_train)
    cb_pred = cb_model.predict(X_test)
    predicted_values_original_scale = cb_pred * std + mean
    cb_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))
    plot_predictions(y_test_original_scale, predicted_values_original_scale, f'CatBoost Std Scaler {file}')
    cb_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("Catboost: ", cb_mse, cb_r2_score)

    # XGBBoost
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    predicted_values_original_scale = xgb_pred * std + mean
    xgb_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))
    plot_predictions(y_test_original_scale, predicted_values_original_scale, f'XGBoost Std Scaler {file}')
    xgb_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("XGBoost: ", xgb_mse, xgb_r2_score)

    plt.figure()
    plt.bar(['RF', 'SVR', 'NUSVR', 'CB', 'XGB'], [rf_mse, svr_mse, nusvr_mse, cb_mse, xgb_mse], color='b')
    plt.title('Root Mean Squared Error (RMSE) Plot for Std scaled datasets')
    plt.grid()
    plt.xlabel(file)
    plt.ylabel('RMSE Value')
    # plt.show()
    plt.savefig(f'trained_plots/{file}.png')
    plt.close()

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

    original_min = df2['optime'].min()
    original_max = df2['optime'].max()
    X_train, X_test, y_train, y_test = train_test_split(X, df['optime'], test_size=0.2, random_state=1)
    y_test_original_scale = y_test * (original_max - original_min) + original_min

    # Random Forest
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    predicted_values_original_scale = rf_pred * (original_max - original_min) + original_min
    plot_predictions(y_test_original_scale, predicted_values_original_scale, f'Random Forest Min-Max Scaler {file}')
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
    plot_predictions(y_test_original_scale, predicted_values_original_scale, f'SVR Min-Max Scaler {file}')
    svr_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("SVR: ", svr_mse, svr_r2_score)

    # NuSVR
    nusvr_model = NuSVR()
    nusvr_model.fit(X_train, y_train)
    nusvr_pred = nusvr_model.predict(X_test)
    predicted_values_original_scale = nusvr_pred * (original_max - original_min) + original_min
    nusvr_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))
    plot_predictions(y_test_original_scale, predicted_values_original_scale, f'NuSVR Min-Max Scaler {file}')
    nusvr_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("NuSVR: ", nusvr_mse, nusvr_r2_score)

    # CatBoost
    cb_model = CatBoostRegressor(random_state=1, verbose=0)
    cb_model.fit(X_train, y_train)
    cb_pred = cb_model.predict(X_test)
    predicted_values_original_scale = cb_pred * (original_max - original_min) + original_min
    cb_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))
    plot_predictions(y_test_original_scale, predicted_values_original_scale, f'CatBoost Min-Max Scaler {file}')
    cb_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("Catboost: ", cb_mse, cb_r2_score)

    # XGBBoost
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    predicted_values_original_scale = xgb_pred * (original_max - original_min) + original_min
    xgb_mse = np.sqrt(mean_squared_error(y_test_original_scale, predicted_values_original_scale))
    plot_predictions(y_test_original_scale, predicted_values_original_scale, f'XGBoost Min-Max Scaler {file}')
    xgb_r2_score = r2_score(y_test_original_scale, predicted_values_original_scale)
    print("XGBoost: ", xgb_mse, xgb_r2_score)

    plt.figure()
    plt.bar(['RF', 'SVR', 'NUSVR', 'CB', 'XGB'], [rf_mse, svr_mse, nusvr_mse, cb_mse, xgb_mse], color='b')
    plt.title('Root Mean Squared Error (RMSE) Plot for Min-Max Scaled Datasets')
    plt.grid()
    plt.xlabel(file)
    plt.ylabel('RMSE Value')
    # plt.show()
    plt.savefig(f'trained_plots/{file}.png')
    plt.close()

    rmse_dict = {}
    rmse_dict['RF'] = rf_mse
    rmse_dict['SVR'] = svr_mse
    rmse_dict['NUSVR'] = nusvr_mse
    rmse_dict['CB'] = cb_mse
    rmse_dict['XGB'] = xgb_mse
    rsme_list.append(rmse_dict)
