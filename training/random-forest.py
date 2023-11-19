import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


files = [
    'std_filled_average.csv',
    'std_filled_knn.csv',
    'std_filled_median.csv',
    'min_max_filled_average.csv',
    'min_max_filled_knn.csv',
    'min_max_filled_median.csv'
]

for file in files:
    print('\nFile: ', file)
    df = pd.read_csv('../dataset/' + file)
    X = df.drop(["weight", "height", "optime"], axis=1)
    mean = df['optime'].mean()
    X_train, X_test, y_train, y_test = train_test_split(X, df['optime'], test_size=0.2, random_state=1)

    # Random Forest
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    print("Random Forest: ", rf_mse)

    # SVC
    # SVR
    svr_model = SVR()
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(X_test)
    svr_mse = mean_squared_error(y_test, svr_pred)
    print("SVR: ", svr_mse)

    # NuSVR
    nusvr_model = NuSVR()
    nusvr_model.fit(X_train, y_train)
    nusvr_pred = nusvr_model.predict(X_test)
    nusvr_mse = mean_squared_error(y_test, svr_pred)
    print("NuSVR: ", nusvr_mse)

    #CatBoost
    cb_model = CatBoostRegressor(random_state=1, verbose=0)
    cb_model.fit(X_train, y_train)
    cb_pred = cb_model.predict(X_test)
    cb_mse = mean_squared_error(y_test, cb_pred)
    print("Catboost: ", rf_mse)

    # XGBBoost
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    print("XGBoost: ", xgb_mse)
    
    






