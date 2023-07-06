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

def find_peaks(file_name, skip_val): 
    ppg_data = pd.read_csv(file_name) 
    ppg_signal = ppg_data.iloc[:, 1].values

    comparator = np.greater
    peak_indices = argrelextrema(np.array(ppg_signal), comparator)

    comparator = np.less
    trough_indices = argrelextrema(np.array(ppg_signal), comparator)

    peak_values = []
    trough_values = []

    tolerance_peaks = []
    tolerance_troughs = []

    tolerance_peak = 0
    tolerance_trough = np.inf

    peak_counter = 0 
    skip_peak = skip_val

    for peak in peak_indices[0]:
        peak_values.append(ppg_signal[peak])
        if peak_counter % skip_peak == 0:
            tolerance_peak = ppg_signal[peak]
        tolerance_peaks.append(tolerance_peak)
        peak_counter += 1

    trough_counter = 0
    skip_trough = skip_peak  # added skip_trough variable
    for trough in trough_indices[0]:
        trough_values.append(ppg_signal[trough])
        if trough_counter % skip_trough == 0:  # change 2 to skip_trough
            tolerance_trough = ppg_signal[trough]
        tolerance_troughs.append(tolerance_trough)
        trough_counter += 1

    # return peak and trough indices instead of values
    return peak_indices[0], trough_indices[0], tolerance_peaks, tolerance_troughs

file_name_test = '1000_josh_Time.csv'
ppg = pd.read_csv(file_name_test) 
ppg_signal = ppg.iloc[:, 1].values

peak_indices, trough_indices, tol_peaks, tol_troughs = find_peaks(file_name_test, 1) 

print(peak_indices)

plt.figure(figsize=(12,8))
plt.plot(ppg_signal, label='PPG Signal') 
plt.plot(peak_indices,ppg_signal[peak_indices], "x",label='Peaks', color='r')  
plt.plot(trough_indices, ppg_signal[trough_indices], "x", label='Troughs', color='g') 
plt.plot(peak_indices, tol_peaks, label='Peak Tolerance', color='r')
plt.plot(trough_indices, tol_troughs, label='Peak Tolerance', color='g')
plt.legend()
plt.show()


