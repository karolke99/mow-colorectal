import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

def ensemble_learning(model1, model2, model3, X, y, mean, std):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)

    pred1 = model1.predict(X_test) * std + mean
    pred2 = model2.predict(X_test) * std + mean
    pred3 = model3.predict(X_test) * std + mean

    y_test_original_scale = y_test * std + mean

    mse = mean_squared_error(y_test_original_scale, (pred1 + pred2 + pred3) / 3)
    return np.sqrt(mse)


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
    mean = df2['optime'].mean()
    std = df2['optime'].std()

    # Random Forest
    rf_model = RandomForestRegressor(random_state=1)
    svr_model = SVR()
    nusvr_model = NuSVR()
    cb_model = CatBoostRegressor(random_state=1, verbose=0)
    xgb_model = XGBRegressor(random_state=1)
    bayes_model = BayesianRidge()
    tree_model = DecisionTreeRegressor()
    lasso_model = Lasso(random_state=1)

    print("RandomForestRegressor, CatBoostRegressor, XGBRegressor: ", ensemble_learning(rf_model, cb_model, xgb_model, X, df['optime'], mean, std))
    print("RandomForestRegressor, SVR, NuSVR: ", ensemble_learning(rf_model, svr_model, nusvr_model, X, df['optime'], mean, std))
    print("BayesianRidge, DecisionTreeRegressor, Lasso: ", ensemble_learning(bayes_model, lasso_model, tree_model, X, df['optime'], mean, std))







