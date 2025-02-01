"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script processes a signal from a CSV file using a Butterworth filter
and detects peaks in the signal. The processed signal is then normalized
and floored to better visualize signal characteristics.
"""

import os  # Use as necessary 
import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
from scipy.signal import butter, filtfilt

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

def butterworth_filter(data, cutoff, fs, order=4):
    """
    Applies a Butterworth low-pass filter to the given data.
    """
    nyq = 0.4 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def find_peaks(signal, height, distance):
    """
    Identifies peaks in the signal based on a given height threshold and minimum distance.
    """
    peaks = []
    last_peak_index = -1
    for i in range(1, len(signal) - 1):
        if signal[i] > height and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if last_peak_index == -1 or i - last_peak_index >= distance:
                peaks.append(i)
                last_peak_index = i
    return peaks

def process_signal(filename, height, distance):
    """
    Reads a signal from a CSV file, applies filtering, and detects peaks.
    Then, floors the signal for better visualization.
    
    Parameters:
    filename (str): CSV file containing the signal data.
    height (float): Minimum peak height for detection.
    distance (int): Minimum distance between peaks.
    
    Returns:
    tuple: Time step values and floored signal.
    """
    df = pd.read_csv(filename)
    time_step = df.iloc[:, 0]  # Assuming the first column is time
    signals = df.iloc[:, 1]  # Assuming the second column is the signal

    # Apply Butterworth filter before peak detection
    filtered_signals = butterworth_filter(signals, cutoff=3.0, fs=100.0)  # Assuming sampling rate = 100Hz
    peaks = find_peaks(filtered_signals, height, distance)

    floored_signal = np.copy(filtered_signals)
    for i in range(len(peaks)):
        start_index = peaks[i - 1] if i > 0 else 0
        end_index = peaks[i]
        segment = filtered_signals[start_index:end_index + 1]
        floored_segment = segment - np.min(segment)
        floored_signal[start_index:end_index + 1] = floored_segment

    if peaks:
        start_index = peaks[-1]
        end_index = len(filtered_signals)
        segment = filtered_signals[start_index:end_index + 1]
        floored_segment = segment - np.min(segment)
        floored_signal[start_index:end_index + 1] = floored_segment

    return time_step, floored_signal

# Process and visualize the signal
time_step, floored_signal = process_signal('FILENAME.csv', 1, 10)

plt.figure()
plt.plot(time_step, floored_signal, label='Floored Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Processed Signal')
plt.show()
