import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def calculate_original_metrics(original_df_path, predicted_values, y_test):
    original_df = pd.read_csv(original_df_path)
    mean = original_df['optime'].mean()
    std = original_df['optime'].std()

    pred_original_scale = predicted_values * std + mean
    y_test_original_scale = y_test * std + mean
    rmse = np.sqrt(mean_squared_error(y_test_original_scale, pred_original_scale))
    r2 = r2_score(y_test_original_scale, pred_original_scale)

    return rmse, r2




def cross_validation(model, X, y, k=5, original_df_path="./newDataset/filled_knn"):
    n = len(X.columns) - 1
    fold_size = n // k

    rmse_scores = []
    r2_scores = []

    for i in range(k):
        test_indices = list(range(i * fold_size, (i+1) * fold_size))
        train_indices = list(set(range(n)) - set(test_indices))

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse, r2 = calculate_original_metrics(original_df_path, pred, y_test)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

    average_rmse = np.mean(rmse_scores)
    average_r2 = np.mean(r2_scores)
    min_rmse = np.min(rmse_scores)

    return average_rmse, min_rmse, rmse_scores, average_r2