"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script partitions a time-series dataset and performs Fourier Transform analysis on each partition.
The objective is to determine the optimal window size and overlap for partitioning the dataset.
By computing correlations between partitions and analyzing variance, the script finds the most stable
window sizes for frequency-domain analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq 
import ButterWorth_Smooth

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

def partition_data(file_name, window_size, overlap): 
    """
    Partitions a time-series dataset and performs Fourier Transform analysis.
    
    Parameters:
    file_name (str): Path to the CSV file containing the signal data.
    window_size (int): Size of each partitioned window.
    overlap (float): Fraction of overlap between consecutive windows.
    
    Returns:
    numpy.array: Correlation values between the first partition and subsequent partitions.
    """
    data = pd.read_csv(file_name)
    fps = 34.8  # Frames per second
    data_step = 1 / fps

    f_noise_not = data.iloc[19:, 1].values.astype(float)
    f_noise = ButterWorth_Smooth.butter_smooth(f_noise_not, 2, 0.2)
    
    stride = int(window_size * (1 - overlap))
    f_noise_partitions = [f_noise[i : i + window_size] for i in range(0, len(f_noise) - window_size + 1, stride)]
    num_partitions = len(f_noise_partitions)
    normalized_arrays = []

    for f_partition in f_noise_partitions:
        yf = fft(f_partition)
        xf = fftfreq(len(f_partition), data_step)
        yf_abs = np.abs(yf)
        normalized_y = normalize(yf_abs[xf > 0.6], 0, 1)
        normalized_arrays.append(normalized_y)
    
    correlations = np.zeros(num_partitions - 1)
    for i in range(1, num_partitions):
        correlations[i-1] = np.corrcoef(normalized_arrays[0], normalized_arrays[i])[0,1]
    
    return correlations

def find_optimal_window_size(data_name, max_frac_partition, starting_win_size, step_arg, max_overlap=0.5, overlap_step=0.05):  
    """
    Determines the optimal window size and overlap for partitioning time-series data.
    
    Parameters:
    data_name (str): CSV file containing the time-series data.
    max_frac_partition (float): Maximum allowable window size as a fraction of total data length.
    starting_win_size (int): Initial window size to begin analysis.
    step_arg (int): Step increment for testing different window sizes.
    max_overlap (float): Maximum fraction of overlap allowed between partitions.
    overlap_step (float): Step increment for testing different overlap values.
    
    Returns:
    list: A list containing tuples of optimal (variance, window_size, overlap) values.
    """
    data = pd.read_csv(data_name)
    use_data = data.iloc[19:, 1].values.astype(float)
    
    variance_vals, win_size_vals, overlap_vals = [], [], []
    
    for overlap in np.arange(0, max_overlap, overlap_step):
        win_size = starting_win_size  
        while win_size <= (len(use_data) / max_frac_partition):    
            correlation_array = np.array(partition_data(data_name, win_size, overlap))   
            inst_var_val = np.var(correlation_array)
            variance_vals.append(inst_var_val)
            win_size_vals.append(win_size)
            overlap_vals.append(overlap)
            win_size += step_arg
    
    sorted_indices = np.argsort(variance_vals)
    top_three_indices = sorted_indices[:3]  # Get indices for the 3 smallest variances
    optimal_values = [(variance_vals[i], win_size_vals[i], overlap_vals[i]) for i in top_three_indices]
    
    return optimal_values

optimal = find_optimal_window_size('TEK00011.csv', 7, 50, 10) 
print(optimal)
