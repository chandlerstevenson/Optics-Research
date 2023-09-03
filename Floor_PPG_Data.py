import os  # Use as necessary 
import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
from scipy.signal import butter, filtfilt

# Use as necessary 
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr  

def butterworth_filter(data, cutoff, fs, order=4):
    nyq = 0.4 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def find_peaks(signal, height, distance):
    peaks = []
    last_peak_index = -1
    for i in range(1, len(signal) - 1):
        if signal[i] > height and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if last_peak_index == -1 or i - last_peak_index >= distance:
                peaks.append(i)
                last_peak_index = i
    return peaks

def process_signal(filename, height, distance):
    df = pd.read_csv(filename)
    time_step = df.iloc[:, 0]  # assuming that the first column is time step
    signals = df.iloc[:, 1]  # assuming that the second column is the signal

    # Applying Butterworth filter before peak detection
    filtered_signals = butterworth_filter(signals, cutoff=3.0, fs=100.0)  # Assuming sampling rate = 100Hz, modify as needed
    peaks = find_peaks(filtered_signals, height, distance)

    floored_signal = np.copy(filtered_signals)
    for i in range(len(peaks)):
        start_index = peaks[i - 1] if i > 0 else 0
        end_index = peaks[i]
        segment = filtered_signals[start_index : end_index + 1]
        floored_segment = segment - np.min(segment)
        floored_signal[start_index : end_index + 1] = floored_segment

    if peaks:
        start_index = peaks[-1]
        end_index = len(filtered_signals)
        segment = filtered_signals[start_index : end_index + 1]
        floored_segment = segment - np.min(segment)
        floored_signal[start_index : end_index + 1] = floored_segment

    return time_step, floored_signal

time_step, floored_signal = process_signal('FILENAME.csv', 1, 10)

# Plot the results
plt.figure() 
plt.plot(time_step, floored_signal, label='floored') 
plt.legend()
plt.show()
