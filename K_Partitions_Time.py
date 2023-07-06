import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        t_partition = t[partition_start:partition_end]

        # Determine the row and column index of the subplot
        row = i // 2  
        col = i % 2   

        # Plot the original signal for this partition
        axs[row, col].plot(t_partition, f_partition)
        axs[row, col].set_title(f'Partition {i+1}') 

    plt.tight_layout()
    return fig

k_partition_data(file_name_use, 10)
plt.show()
