import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Replace with file location of .csv
data = pd.read_csv('/Users/chandlerstevenson/Downloads/onehundredkrandpoints.csv')

# Generate some x-values for the scatter plot
x_values = np.arange(len(data))

# create a density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data['Absolute Difference'], color='black', fill=False)
plt.title('Density Plot of Absolute Differences')
plt.xlabel('Absolute Difference')
plt.ylabel('Density')
plt.grid(False)  # removed gridlines
plt.show()

# create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, data['Absolute Difference'], color='black', s=1, alpha=0.5)
plt.title('Scatter Plot of Absolute Differences')
plt.xlabel('Index')
plt.ylabel('Absolute Difference')
plt.grid(False)  # removed gridlines
plt.show()
