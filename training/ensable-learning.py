import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


def ensemble_learning(model1, model2, model3, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)

    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    pred3 = model3.predict(X_test)

    mse = mean_squared_error(y_test, (pred1 + pred2 + pred3) / 3)
    return mse


files = [
    ('std_filled_average.csv', 'filled_average.csv'),
    # ('std_filled_knn.csv', 'filled_knn.csv'),
    # ('std_filled_median.csv', 'filled_median.csv'),
    # ('min_max_filled_average.csv', 'filled_average.csv'),
    # ('min_max_filled_knn.csv', 'filled_knn.csv'),
    # ('min_max_filled_median.csv', 'filled_median.csv')
]

for file, file2 in files:
    print('\nFile: ', file)
    df = pd.read_csv('../dataset/' + file)
    df2 = pd.read_csv('../dataset/' + file2)
    X = df.drop(["weight", "height", "optime"], axis=1)


    # Random Forest
    rf_model = RandomForestRegressor(random_state=1)
    svr_model = SVR()
    nusvr_model = NuSVR()
    cb_model = CatBoostRegressor(random_state=1, verbose=0)
    xgb_model = XGBRegressor()

    print("RandomForestRegressor, CatBoostRegressor, XGBRegressor: ", ensemble_learning(rf_model, cb_model, xgb_model, X, df['optime']))
    print("RandomForestRegressor, SVR, NuSVR: ", ensemble_learning(rf_model, svr_model, nusvr_model, X, df['optime']))






