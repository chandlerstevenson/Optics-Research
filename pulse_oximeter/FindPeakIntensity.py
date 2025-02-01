"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script identifies peaks and troughs in a pulsatile, periodic dataset.
It processes photoplethysmography (PPG) signals, smooths them using a Butterworth filter,
and determines the peaks and their corresponding amplitudes and frequencies.
Additionally, it performs a Fourier Transform to analyze the frequency components of the signal.
"""

import pandas as pd 
import ButterWorth_Smooth 
import FloorData
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def find_peaks(ppg_data, skip_val, max_distance):
    """
    Identifies peaks in the given PPG dataset.
    
    Parameters:
    ppg_data (DataFrame): PPG data with time and signal values.
    skip_val (int): Value to determine peak tolerance.
    max_distance (float): Maximum allowable distance between consecutive peaks.
    
    Returns:
    tuple: Indices of peaks, peak tolerance values, signal frequency, and average amplitude.
    """
    ppg_signal1 = ppg_data.iloc[:, 1].values
    ppg_signal_filt = ButterWorth_Smooth.butter_smooth(ppg_signal1, 2, .65) 
    ppg_signal = FloorData.process_csv_and_floor_signal(ppg_signal_filt)
    
    comparator = np.greater
    peak_indices = argrelextrema(np.array(ppg_signal), comparator)
    
    new_peak_indices = []
    prev_index = peak_indices[0][0]
    time_values = ppg_data.iloc[:, 0].values  # Extract time values
    
    for curr_index in peak_indices[0][1:]:
        time_difference = time_values[curr_index] - time_values[prev_index]
        if time_difference <= max_distance:
            if ppg_signal[curr_index] > ppg_signal[prev_index]:
                prev_index = curr_index
            continue
        else:
            new_peak_indices.append(prev_index)
        prev_index = curr_index
    
    new_peak_indices.append(prev_index)
    peak_values = [ppg_signal[i] for i in new_peak_indices]
    
    tolerance_peaks = []
    tolerance_peak = 0
    peak_counter = 0
    
    for peak in new_peak_indices:
        if peak_counter % skip_val == 0:
            tolerance_peak = ppg_signal[peak]
        tolerance_peaks.append(tolerance_peak)
        peak_counter += 1
    
    if len(new_peak_indices) > 0:
        peak_times = time_values[new_peak_indices]
        average_period = np.mean(np.diff(peak_times))
        frequency = 1 / average_period if average_period != 0 else 0
    else:
        print("Not enough peaks for frequency calculation")
        frequency = None
    
    average_amplitude = np.mean([abs(peak) for peak in peak_values])
    print(f'Average amplitude: {average_amplitude}')
    
    return new_peak_indices, tolerance_peaks, frequency, average_amplitude

# Load test data
file_name_test = '700_Joshr_0_Time.csv'
ppg = pd.read_csv(file_name_test) 
ppg_signal_unfiltered = ppg.iloc[:, 1].values  
ppg_signal_filt = ButterWorth_Smooth.butter_smooth(ppg_signal_unfiltered, 2, .5)
ppg_signal = FloorData.process_csv_and_floor_signal(ppg_signal_filt) 

peak_indices, trough_indices, freq, average_amp = find_peaks(ppg, 1, .5) 

print(peak_indices)

# Frame rate and time vector
frame_rate = 34.8 
time_vector = np.arange(len(ppg_signal)) / frame_rate

# Plot results
plt.figure(figsize=(12,8)) 
plt.title(f'HR: {freq*60}, Average Amplitude: {average_amp}')
plt.plot(time_vector, ppg_signal, label='PPG Signal') 
plt.plot(time_vector[peak_indices], ppg_signal[peak_indices], "x", label='Peaks', color='r')  
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

def fourier_transform(file_name, framerate):
    """
    Performs Fourier Transform on PPG data to analyze frequency components.
    
    Parameters:
    file_name (str): Path to CSV file containing PPG signal data.
    framerate (float): Sampling rate in frames per second.
    
    Returns:
    tuple: Frequencies and magnitude spectrum of the signal.
    """
    data = pd.read_csv(file_name)
    signal = data.iloc[:, 1].values
    fft_result = np.fft.fft(signal)
    spectrum = np.abs(fft_result)
    
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/framerate)
    
    return freqs, spectrum

freqs, spectrum = fourier_transform(file_name_test, 34.8)
filtered_freqs = freqs[freqs > 0.67]
filtered_spectrum = spectrum[freqs > 0.67]

max_power_freq = filtered_freqs[np.argmax(filtered_spectrum)]
print(f"The frequency with maximum power is {max_power_freq} Hz")

plt.figure() 
plt.title(f'HR: {max_power_freq*60}')
plt.plot(filtered_freqs, filtered_spectrum) 
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()
