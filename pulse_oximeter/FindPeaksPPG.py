"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script identifies peaks and troughs in a pulsatile, periodic dataset.
It processes photoplethysmography (PPG) signals and determines the peaks
and their corresponding amplitudes and frequencies. Additionally, it performs a Fourier Transform
to analyze the frequency components of the signal.
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def find_peaks(file_name, skip_val, max_distance, peak_trough_distance):
    """
    Identifies peaks and troughs in the given PPG dataset.
    
    Parameters:
    file_name (str): Path to the CSV file containing PPG signal data.
    skip_val (int): Value to determine peak tolerance.
    max_distance (float): Maximum allowable distance between consecutive peaks.
    peak_trough_distance (float): Minimum separation between a peak and a trough.
    
    Returns:
    tuple: Indices of peaks, indices of troughs, peak tolerance values,
           trough tolerance values, and calculated frequency.
    """
    ppg_data = pd.read_csv(file_name)
    ppg_signal = ppg_data.iloc[:, 1].values

    comparator = np.greater
    peak_indices = argrelextrema(np.array(ppg_signal), comparator)

    comparator = np.less
    trough_indices = argrelextrema(np.array(ppg_signal), comparator)

    new_peak_indices, new_trough_indices = [], []
    all_indices = sorted([(i, 'peak') for i in peak_indices[0]] + [(i, 'trough') for i in trough_indices[0]])

    prev_index, prev_type = all_indices[0]
    for curr_index, curr_type in all_indices[1:]:
        if curr_type != prev_type and abs(curr_index - prev_index) <= peak_trough_distance:
            prev_index, prev_type = curr_index, curr_type
            continue
        else:
            if prev_type == 'peak':
                new_peak_indices.append(prev_index)
            else:
                new_trough_indices.append(prev_index)
        prev_index, prev_type = curr_index, curr_type

    if prev_type == 'peak':
        new_peak_indices.append(prev_index)
    else:
        new_trough_indices.append(prev_index)

    tolerance_peaks, tolerance_troughs = [], []
    tolerance_peak, tolerance_trough = 0, np.inf
    peak_counter, trough_counter = 0, 0

    for peak in new_peak_indices:
        if peak_counter % skip_val == 0:
            tolerance_peak = ppg_signal[peak]
        tolerance_peaks.append(tolerance_peak)
        peak_counter += 1

    for trough in new_trough_indices:
        if trough_counter % skip_val == 0:
            tolerance_trough = ppg_signal[trough]
        tolerance_troughs.append(tolerance_trough)
        trough_counter += 1

    if len(new_peak_indices) >= 4:
        time_values = ppg_data.iloc[:, 0].values
        peak_times = time_values[new_peak_indices]
        average_period = np.mean(np.diff(peak_times[:4]))
        frequency = 1 / average_period if average_period != 0 else 0
    else:
        print("Not enough peaks for frequency calculation")
        frequency = None

    return new_peak_indices, new_trough_indices, tolerance_peaks, tolerance_troughs, frequency

file_name_test = 'FILENAME.csv'
ppg = pd.read_csv(file_name_test)
ppg_signal = ppg.iloc[:, 1].values

peak_indices, trough_indices, tol_peaks, tol_troughs, freq = find_peaks(file_name_test, 1, 10, 2)

print(peak_indices)

frame_rate = 34.8 
time_vector = np.arange(len(ppg_signal)) / frame_rate

plt.figure(figsize=(12,8)) 
plt.title(f'HR: {freq*60}')
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
