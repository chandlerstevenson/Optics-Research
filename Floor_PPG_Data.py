import os # Use as necessary 
import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

#Use as necessary 
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr  


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

    peaks = find_peaks(signals, height, distance)

    floored_signal = np.copy(signals)
    for i in range(len(peaks)):
        start_index = peaks[i - 1] if i > 0 else 0
        end_index = peaks[i]
        segment = signals[start_index : end_index + 1]
        floored_segment = segment - np.min(segment)
        floored_signal[start_index : end_index + 1] = floored_segment

    if peaks:
        start_index = peaks[-1]
        end_index = len(signals)
        segment = signals[start_index : end_index + 1]
        floored_segment = segment - np.min(segment)
        floored_signal[start_index : end_index + 1] = floored_segment

    return time_step, floored_signal

time_step, floored_signal = process_signal('bessel_seokee_4.7mw_0_Time.csv', 1, 10)

# Plot the results
plt.figure()
plt.plot(time_step, floored_signal, label='floored')
plt.legend()
plt.show()


