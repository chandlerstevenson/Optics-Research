# This is a function that finds the peaks and troughs of a pulsatile, periodic data set:  
#      Assumed use cases:  
#                           * By finding the peaks, we may be able to say something about 
#                                the level noise in the system.  
#                           * Finding the peaks in tandem with the peaks tells you the amplitude  
#                                 which is useful for mapping the intensity relative to the location  
#                                 of the PPG on the graph.  
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def find_peaks(file_name, skip_val, max_distance, peak_trough_distance):
    ppg_data = pd.read_csv(file_name)
    ppg_signal = ppg_data.iloc[:, 1].values

    comparator = np.greater
    peak_indices = argrelextrema(np.array(ppg_signal), comparator)

    comparator = np.less
    trough_indices = argrelextrema(np.array(ppg_signal), comparator)

    # Initialize new peaks and troughs lists
    new_peak_indices = []
    new_trough_indices = []

    # Sort and merge the peak and trough indices
    all_indices = sorted([(i, 'peak') for i in peak_indices[0]] + [(i, 'trough') for i in trough_indices[0]])

    # Initialize the previous index and type (peak or trough)
    prev_index, prev_type = all_indices[0]

    # Iterate over each index from the second one
    for curr_index, curr_type in all_indices[1:]:
        if curr_type != prev_type and abs(curr_index - prev_index) <= peak_trough_distance:
            # If current index is a peak or a trough and is too close to the previous one, skip it
            prev_index, prev_type = curr_index, curr_type
            continue
        else:
            # If current index is not too close to the previous one, add the previous index to the corresponding list
            if prev_type == 'peak':
                new_peak_indices.append(prev_index)
            else:
                new_trough_indices.append(prev_index)

        prev_index, prev_type = curr_index, curr_type

    # Add the last index to the corresponding list
    if prev_type == 'peak':
        new_peak_indices.append(prev_index)
    else:
        new_trough_indices.append(prev_index)

    # Continue as before from here
    peak_values = [ppg_signal[i] for i in new_peak_indices]
    trough_values = [ppg_signal[i] for i in new_trough_indices]

    tolerance_peaks = []
    tolerance_troughs = []

    tolerance_peak = 0
    tolerance_trough = np.inf

    peak_counter = 0
    skip_peak = skip_val

    for peak in new_peak_indices:  # use new_peak_indices here
        if peak_counter % skip_peak == 0:
            tolerance_peak = ppg_signal[peak]
        tolerance_peaks.append(tolerance_peak)
        peak_counter += 1

    trough_counter = 0
    skip_trough = skip_peak
    for trough in new_trough_indices:  # use new_trough_indices here
        if trough_counter % skip_trough == 0:
            tolerance_trough = ppg_signal[trough]
        tolerance_troughs.append(tolerance_trough)
        trough_counter += 1

# Compute the average period from the first 4 peaks
    if len(new_peak_indices) >= 4:
        time_values = ppg_data.iloc[:, 0].values  # Assuming time data is in the first column
        peak_times = time_values[new_peak_indices]
        average_period = np.mean(np.diff(peak_times[:4]))
        frequency = 1 / average_period if average_period != 0 else 0  # Avoid division by zero
    else:
        print("Not enough peaks for frequency calculation")
        frequency = None

    return new_peak_indices, new_trough_indices, tolerance_peaks, tolerance_troughs, frequency


file_name_test = 'FILENAME.csv'
ppg = pd.read_csv(file_name_test) 
ppg_signal = ppg.iloc[:, 1].values

peak_indices, trough_indices, tol_peaks, tol_troughs, freq = find_peaks(file_name_test, 1, 10, 2) 

print(peak_indices)

# Frame rate in frames per second
frame_rate = 34.8 

# Create a time vector for the signal
time_vector = np.arange(len(ppg_signal)) / frame_rate

# Adjust your plotting function to include time:
plt.figure(figsize=(12,8)) 
plt.title(f'HR: {freq*60}')
plt.plot(time_vector, ppg_signal, label=f'PPG Signal') 
plt.plot(time_vector[peak_indices], ppg_signal[peak_indices], "x", label='Peaks', color='r')  
# plt.plot(time_vector[trough_indices], ppg_signal[trough_indices], "x", label='Troughs', color='g') 
# plt.plot(time_vector[peak_indices], tol_peaks, label='Peak Tolerance', color='r')
# plt.plot(time_vector[trough_indices], tol_troughs, label='Trough Tolerance', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fourier_transform(file_name, framerate):
    # Read the data
    data = pd.read_csv(file_name)

    # Get the signal values
    signal = data.iloc[:, 1].values

    # Perform the Fourier Transform
    fft_result = np.fft.fft(signal)

    # Calculate the absolute values of the FFT results to get the spectrum
    spectrum = np.abs(fft_result)

    # Calculate the frequencies for the spectrum
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/framerate)

    # Return the frequencies and the spectrum
    return freqs, spectrum

# Call the function to get frequencies and spectrum
freqs, spectrum = fourier_transform(file_name_test, 34.8)

# Get only the frequencies and spectrum components for freqs > 0.67
filtered_freqs = freqs[freqs > 0.67]
filtered_spectrum = spectrum[freqs > 0.67]



# Find the frequency with maximum power
max_power_freq = filtered_freqs[np.argmax(filtered_spectrum)]
print(f"The frequency with maximum power is {max_power_freq} Hz")
# Create a new figure
plt.figure() 
plt.title(f'HR: {max_power_freq*60}')
# Plot the frequency spectrum
plt.plot(filtered_freqs, filtered_spectrum) 

# Label the axes
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# Show the plot
plt.show()


