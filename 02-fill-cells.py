import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


# ------------------------------------- Median -------------------------------------
df = pd.read_csv('colorectal_refactored.csv')
medians = df.median()
df_filled = df.fillna(medians)
df_filled.to_csv('dataset/filled_median.csv', index=False)




# ------------------------------------- Average -------------------------------------
df = pd.read_csv('colorectal_refactored.csv')
columns= df.columns
binary_columns = []

def is_integer(value):
    if np.isnan(value):
        return True
    return value.is_integer()

for column in columns:
    unique_values = df[column].unique()
    is_binary = len(unique_values) == 3 and (0. in unique_values) and (1. in unique_values)
    is_int = df[column].apply(is_integer).all()
    if is_binary or is_int:
        binary_columns.append(column)
# print(binary_columns)

averages = df.mean()
df_filled = df.fillna(averages).round(1)

for column in binary_columns:
    df_filled[column] = df_filled[column].round()

df_filled.to_csv('dataset/filled_average.csv', index=False)



# ------------------------------------- KNN -------------------------------------
df = pd.read_csv('colorectal_refactored.csv')
columns = df.columns
binary_columns = []


def is_integer(value):
    if np.isnan(value):
        return True
    return value.is_integer()


for column in columns:
    unique_values = df[column].unique()
    is_binary = len(unique_values) == 3 and (0. in unique_values) and (1. in unique_values)
    is_int = df[column].apply(is_integer).all()
    if is_binary or is_int:
        binary_columns.append(column)

imputer = KNNImputer(n_neighbors=5)

df = pd.DataFrame(imputer.fit_transform(df), columns=columns).round(1)

for column in binary_columns:
    df[column] = df[column].round()

df.to_csv('dataset/filled_knn.csv', index=False)