import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from utils import calculate_original_metrics
from catboost import CatBoostRegressor


df = pd.read_csv('../newDataset/std_filled_knn.csv')

X = df.drop(['weight', 'height', 'optime'], axis=1)
y = df['optime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
cb_model = CatBoostRegressor(random_state=1, verbose=0)
cb_model.fit(X_train, y_train)
cb_pred = cb_model.predict(X_test)
rmse, r2 = calculate_original_metrics('../newDataset/filled_knn.csv', cb_pred, y_test)
print(f'All features RMSE: {rmse}, R2_score: {r2}')


selector = SelectKBest(score_func=f_regression, k=40)
X_train_selected = selector.fit_transform(X_train, y_train)
cb_model = CatBoostRegressor(random_state=1, verbose=0)
cb_model.fit(X_train_selected, y_train)
X_test_selected = selector.transform(X_test)
y_pred = cb_model.predict(X_test_selected)
rmse, r2 = calculate_original_metrics('../newDataset/filled_knn.csv', y_pred, y_test)
print(f'Selected features RMSE: {rmse}, R2_score: {r2}')

best_feature_indices = selector.get_support(indices=True)
best_feature_names = X.columns[best_feature_indices]
print(f'Selected features: {best_feature_names}')

selected_feature_df = X[best_feature_names]
selected_feature_df.insert(0, 'optime', y)
selected_feature_df.to_csv('./best_std_filled_knn.csv', index=False)
