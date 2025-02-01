"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script partitions a time-series dataset and visualizes each partition separately.
The purpose of partitioning is to analyze variations in the signal over time,
which can help in detecting trends, anomalies, or periodic components in the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def k_partition_data(file_name, k_partitions): 
    """
    Partitions time-series data and plots each partition separately.
    
    Parameters:
    file_name (str): Path to the CSV file containing time-series data.
    k_partitions (int): Number of partitions for the dataset.
    
    Returns:
    matplotlib.figure.Figure: A figure containing the time-series plots of each partition.
    """
    data = pd.read_csv(file_name)
    fps = 34.8  # Frames per second 
    data_step = 1 / fps
    num_partitions = k_partitions 

    f_noise = data.iloc[:, 1].values  # Extract signal data
    n = len(f_noise)
    t = np.arange(start=0, stop=n * data_step, step=data_step)
    num_rows = (num_partitions + 1) // 2  # Calculate the number of rows 
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 8))

    for i in range(num_partitions):
        data_number = int(len(data) / num_partitions)
        partition_start = i * data_number
        partition_end = partition_start + data_number
        f_partition = f_noise[partition_start:partition_end]  # Extract partitioned data
        t_partition = t[partition_start:partition_end]  # Extract corresponding time values

        row = i // 2  # Determine subplot row index
        col = i % 2   # Determine subplot column index

        axs[row, col].plot(t_partition, f_partition)
        axs[row, col].set_title(f'Partition {i+1}') 

    plt.tight_layout()
    return fig

file_name_use = 'chandler_bessel_0_Time.csv'
k_partition_data(file_name_use, 10)
plt.show()
