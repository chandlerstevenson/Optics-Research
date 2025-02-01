"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script partitions a time-series dataset and performs the Fourier Transform on each partition.
The motivation behind this approach is that as data partitions become smaller, the Fourier Transform,
which reveals the fundamental frequency (e.g., heart rate), should remain largely consistent.
This helps in analyzing frequency stability over different partitions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq

def normalize(arr, t_min, t_max):
    """
    Normalizes an array to a specified range [t_min, t_max].
    """
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr  

def k_partition_data(file_name, k_partitions): 
    """
    Partitions time-series data and applies the Fourier Transform to each partition.
    
    Parameters:
    file_name (str): Path to the CSV file containing time-series data.
    k_partitions (int): Number of partitions for the dataset.
    
    Returns:
    matplotlib.figure.Figure: A figure containing the Fourier Transforms of each partition.
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

        yf = fft(f_partition)  # Perform Fourier Transform
        xf = fftfreq(data_number, data_step)

        threshold_val = 4.8  # Define a threshold for filtering noise
        yf_abs = np.abs(yf) 

        indices = np.logical_and(xf < threshold_val, xf > 0.6)  # Retain values within frequency range
        yf_clean = np.zeros_like(yf)
        yf_clean[indices] = yf[indices]

        normalized_y = normalize(yf_abs[xf > 0.6], 0, 1)  # Normalize transformed data

        row = i // 2  # Determine subplot row index
        col = i % 2   # Determine subplot column index

        axs[row, col].plot(xf[xf > 0.6], normalized_y)
        axs[row, col].set_title(f'Fourier Transform, Partition {i+1}') 

    plt.tight_layout()
    return fig

file_name_use = 'chandler_bessel_0_Time.csv'
k_partition_data(file_name_use, 10)
plt.show()
