import pandas as pd
import numpy as np

files = [
    'filled_average.csv',
    'filled_median.csv',
    'filled_knn.csv'
]


# ------------ min-max scaler -----------------------
def min_max_scaling(df):
    min_vals = df.min()
    max_vals = df.max()
    scaled_df = (df - min_vals) / (max_vals - min_vals)

    return scaled_df

for file in files:
    df = pd.read_csv(file)
    df = min_max_scaling(df)
    df.to_csv('dataset/min_max_' + file, index=False)


# ------------ standard scaler -----------------------
for file in files:
    df = pd.read_csv(file)
    means = df.mean()
    stds = df.std()

    standarized_df = (df - means) / stds
    standarized_df.to_csv('dataset/std_' + file, index=False)