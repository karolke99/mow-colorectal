import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../dataset/min_max_filled_knn.csv')
X = df.drop(['weight', 'height', 'optime'], axis=1)

correlation_matrix = df.corr()
plt.figure()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Macierz Korelacji')
plt.show()
