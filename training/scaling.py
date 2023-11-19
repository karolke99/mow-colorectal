import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

files = [
    'filled_average.csv',
    'filled_knn.csv',
    'filled_median.csv'
]

for file in files:
    print('\nFile: ', file)
    df = pd.read_csv('../dataset/' + file)
    scaler = StandardScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns)
    df.to_csv('scaled_' + file, index=False)

