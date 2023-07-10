# This is a function that finds the peaks and troughs of a pulsatile, periodic data set:  
#      Assumed use cases:  
#                           * By finding the peaks, we may be able to say something about 
#                                the level noise in the system.  
#                           * Finding the peaks in tandem with the peaks tells you the amplitude  
#                                 which is useful for mapping the intensity relative to the location  
#                                 of the PPG on the graph.  
import pandas as pd 
import ButterWorth_Smooth 
import FloorData
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

def find_peaks(ppg_data, skip_val, max_distance):
    ppg_signal1 = ppg_data.iloc[:, 1].values
    ppg_signal_filt = ButterWorth_Smooth.butter_smooth(ppg_signal1, 2, .65) 
    ppg_signal = FloorData.process_csv_and_floor_signal(ppg_signal_filt)
    comparator = np.greater
    peak_indices = argrelextrema(np.array(ppg_signal), comparator)

    # Initialize new peaks list
    new_peak_indices = []

    # Initialize the previous index
    prev_index = peak_indices[0][0]

    time_values = ppg_data.iloc[:, 0].values  # Assuming time data is in the first column

    # Iterate over each index from the second one
    for curr_index in peak_indices[0][1:]:
        time_difference = time_values[curr_index] - time_values[prev_index]
        if time_difference <= max_distance:
            # If the current index is too close in time to the previous one and the value is higher, update the previous index
            if ppg_signal[curr_index] > ppg_signal[prev_index]:
                prev_index = curr_index
            continue
        else:
            # If current index is not too close in time to the previous one, add the previous index to the list
            new_peak_indices.append(prev_index)

        prev_index = curr_index

    # Add the last index to the list
    new_peak_indices.append(prev_index)

    # Continue as before from here
    peak_values = [ppg_signal[i] for i in new_peak_indices]

    tolerance_peaks = []
    tolerance_peak = 0
    peak_counter = 0
    skip_peak = skip_val

    for peak in new_peak_indices:
        if peak_counter % skip_peak == 0:
            tolerance_peak = ppg_signal[peak]
        tolerance_peaks.append(tolerance_peak)
        peak_counter += 1

    # Compute the average period from all peaks
    if len(new_peak_indices) > 0:
        peak_times = time_values[new_peak_indices]
        average_period = np.mean(np.diff(peak_times))
        frequency = 1 / average_period if average_period != 0 else 0  # Avoid division by zero
    else:
        print("Not enough peaks for frequency calculation")
        frequency = None

    average_amplitude = np.mean([abs(peak) for peak in peak_values])
    print(f'Average amplitude: {average_amplitude}')

    return new_peak_indices, tolerance_peaks, frequency, average_amplitude



file_name_test = '700_Joshr_0_Time.csv'
ppg = pd.read_csv(file_name_test) 
ppg_signal_unfiltered = ppg.iloc[:, 1].values  
# ppg_signal = ppg_signal_unfiltered
ppg_signal_filt = ButterWorth_Smooth.butter_smooth(ppg_signal_unfiltered, 2, .5)
ppg_signal = FloorData.process_csv_and_floor_signal(ppg_signal_filt) 

peak_indices, trough_indices, freq, average_amp = find_peaks(ppg, 1, .5) 

print(peak_indices)

# Frame rate in frames per second
frame_rate = 34.8 

# Create a time vector for the signal
time_vector = np.arange(len(ppg_signal)) / frame_rate

# Adjust your plotting function to include time:
plt.figure(figsize=(12,8)) 
plt.title(f'HR: {freq*60}, Average Amplitude: {average_amp}')
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


