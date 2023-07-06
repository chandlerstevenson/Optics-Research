# This file partitions data and takes the Fourier Transform of each partitions 
#       * The motivation behind this is that as one creates smaller and smaller partitions, the Fourier Transform, which illustrates 
#             the fundamental frequency (heart rate) should remain for the most part, constant. 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq

# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr  


file_name_use = 'chandler_bessel_0_Time.csv' 

def k_partition_data(file_name, k_partitions): 
    # Load the data from the CSV file
    data = pd.read_csv(file_name)
    fps = 34.8  # Frames per second 

    # Calculate the time step based on frames per second
    data_step = 1 / fps

    num_partitions = k_partitions 

    # Extract the signal data from the CSV (assuming it's in the second column)
    f_noise = data.iloc[:, 1].values

    # Generate the time array
    n = len(f_noise)
    t = np.arange(start=0, stop=n * data_step, step=data_step)

    num_rows = (num_partitions + 1) // 2  # Calculate the number of rows 

    # Create a grid of subplots
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 8))

    for i in range(num_partitions):
        data_number = int(len(data) / num_partitions)
        partition_start = i * data_number
        partition_end = partition_start + data_number

        # Extract the partition of the signal data
        f_partition = f_noise[partition_start:partition_end]

        # Perform the Fourier transform
        yf = fft(f_partition)
        xf = fftfreq(data_number, data_step)

        # Define a threshold value for the coefficients:  
        threshold_val = 4.8

        # Apply a frequency filter to remove noise
        yf_abs = np.abs(yf) 

        # Create an array of indices that dictate 
        indices = np.logical_and(xf < threshold_val, xf > 0.6)  # Filter out values under 300 and less than 0.6

        yf_clean = np.zeros_like(yf)

        yf_clean[indices] = yf[indices]  # Noise frequencies will be set to 0 

        # Normalize the Data (0, 1) 
        normalized_y = normalize(yf_abs[xf > 0.6], 0, 1)

        # Determine the row and column index of the subplot
        row = i // 2  
        col = i % 2   

        # Plot the original Fourier transform
        axs[row, col].plot(xf[xf > 0.6], normalized_y)
        axs[row, col].set_title(f'Original Fourier Transform, Partition {i+1}') 

 
    plt.tight_layout()
    return fig

k_partition_data(file_name_use, 10)
plt.show()
