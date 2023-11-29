import numpy as np
import pandas as pd

df = pd.read_csv('colorectal2.csv')

df = df.drop(["optime.1"], axis=1)

def create_new_columns(new_columns, initial_column):
    for value in new_columns:
        if type(value) is float:
            continue
        df[initial_column + "_" + str(value)] = np.where(df[initial_column] == value, 1, 0)
        df[initial_column + "_" + str(value)] = np.where(df[initial_column].isna(), np.nan, df[initial_column + "_" + str(value)])

def refactor_multi_value():
    columns = ['approach', 'position', 'preop_pft', 'airway', 'iv1', 'iv2', 'tubesize']
    for column in columns:
        new_columns = df[column].unique()
        create_new_columns(new_columns, column)
        df.drop(column, axis=1, inplace=True)

refactor_multi_value()
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
df['ane_type'] = df['ane_type'].map({'General': 0, 'Spinal': 1})

df.to_csv('colorectal_refactored.csv', index=False)