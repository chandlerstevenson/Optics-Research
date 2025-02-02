"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script reads a CSV file containing absolute difference values, generates a density plot,
and creates a scatter plot to visualize data distribution and trends.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV file containing absolute difference values
data = pd.read_csv('/Users/chandlerstevenson/Downloads/onehundredkrandpoints.csv')

# Generate x-values for scatter plot based on data length
x_values = np.arange(len(data))

# Create a density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data['Absolute Difference'], color='black', fill=False)
plt.title('Density Plot of Absolute Differences')
plt.xlabel('Absolute Difference')
plt.ylabel('Density')
plt.grid(False)  # Remove gridlines
plt.show()

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, data['Absolute Difference'], color='black', s=1, alpha=0.5)
plt.title('Scatter Plot of Absolute Differences')
plt.xlabel('Index')
plt.ylabel('Absolute Difference')
plt.grid(False)  # Remove gridlines
plt.show()
