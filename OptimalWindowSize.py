import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq 
import ButterWorth_Smooth

# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr



def partition_data(file_name, window_size, overlap): 
    # Load the data from the CSV file
    data = pd.read_csv(file_name)
    fps = 34.8  # Frames per second 

    # Calculate the time step based on frames per second
    data_step = 1 / fps

    # Extract the signal data from the CSV (assuming it's in the second column)
    f_noise_not = data.iloc[19:, 1].values.astype(float)  # Convert to numerical array 
    time = data.iloc[19:, 0].values.astype(float)

    f_noise = ButterWorth_Smooth.butter_smooth(f_noise_not, 2, 0.2)

    # Generate the time array
    n = len(f_noise)
    t = np.arange(start=0, stop=n * data_step, step=data_step)

    normalized_arrays = []  # A list to store normalized Fourier Transform arrays

    # Calculate the stride (step size) for the sliding window
    stride = int(window_size * (1 - overlap))

    # Create partitions using a sliding window
    f_noise_partitions = [f_noise[i : i + window_size] for i in range(0, len(f_noise) - window_size + 1, stride)]
    num_partitions = len(f_noise_partitions)

    for i, f_partition in enumerate(f_noise_partitions):

        data_number = len(f_partition)  # Use the actual length of the partition

        # Perform the Fourier transform
        yf = fft(f_partition)
        xf = fftfreq(data_number, data_step)

        # Apply a frequency filter to remove noise
        yf_abs = np.abs(yf)

        # Normalize the Data (0, 1) 
        normalized_y = normalize(yf_abs[xf > 0.6], 0, 1)
        normalized_arrays.append(normalized_y)  # Add the normalized Fourier Transform to the list

    correlations = np.zeros(num_partitions - 1)  # Initialize array to store correlation values
    # Calculate correlation between the first partition and the others
    for i in range(1, num_partitions):
        correlations[i-1] = np.corrcoef(normalized_arrays[0], normalized_arrays[i])[0,1]  # [0,1] to get the off-diagonal element which is the correlation

    return correlations







def find_optimal_window_size(data_name, max_frac_partition, starting_win_size, step_arg, max_overlap=0.5, overlap_step=0.05):  
    # data_name: file name of .csv  
    # max_frac_partition: a fraction of the size of the file that designates 
    #                     the maximum size of a window.  
    # starting_win_size: starting window size to consider:   
    # step_arg: the amount to be tested by in each step

    data = pd.read_csv(data_name)
    # Extract the signal data from the CSV (assuming it's in the second column)
    use_data = data.iloc[19:, 1].values.astype(float)  # Convert to numerical array    

    variance_vals = [] 
    win_size_vals = []
    overlap_vals = []

    for overlap in np.arange(0, max_overlap, overlap_step):
        win_size = starting_win_size  
        while win_size <= (len(use_data) / max_frac_partition):    
            correlation_array = np.array(partition_data(data_name, win_size, overlap))   
            inst_var_val = np.var(correlation_array)
            variance_vals.append(inst_var_val)  
            win_size_vals.append(win_size)
            overlap_vals.append(overlap)
            win_size = win_size + step_arg   

    sorted_indices = np.argsort(variance_vals)
    top_three_indices = sorted_indices[:3]  # Get indices for the 3 smallest variances

    optimal_values = []
    for index in top_three_indices:
        optimal_variance = variance_vals[index]
        optimal_win_size = win_size_vals[index]
        optimal_overlap = overlap_vals[index]
        optimal_values.append((optimal_variance, optimal_win_size, optimal_overlap))

    return optimal_values


optimal = find_optimal_window_size('TEK00011.csv', 7, 50, 10) 

print(optimal)



