"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script performs real-time photoplethysmography (PPG) signal analysis.
It processes the signal using smoothing filters, peak detection, Fourier analysis,
and visualizations to extract heart rate and other signal properties.
It includes interactive buttons for simulation and user control.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, argrelextrema
import matplotlib.widgets as widgets 

def center_time(time_data):
    """Centers the time data to start from zero."""
    time_data = np.array(time_data)
    return time_data - time_data[0]

def find_time_peaks(ppg_signal, skip_val, max_distance, peak_trough_distance):
    """Identifies peaks and troughs in the PPG signal."""
    peak_indices = argrelextrema(ppg_signal, np.greater)[0]
    trough_indices = argrelextrema(ppg_signal, np.less)[0]
    
    all_indices = sorted([(i, 'peak') for i in peak_indices] + [(i, 'trough') for i in trough_indices])
    new_peak_indices, new_trough_indices = [], []
    prev_index, prev_type = all_indices[0]

    for curr_index, curr_type in all_indices[1:]:
        if curr_type != prev_type and abs(curr_index - prev_index) <= peak_trough_distance:
            prev_index, prev_type = curr_index, curr_type
            continue
        else:
            (new_peak_indices if prev_type == 'peak' else new_trough_indices).append(prev_index)
        prev_index, prev_type = curr_index, curr_type

    (new_peak_indices if prev_type == 'peak' else new_trough_indices).append(prev_index)
    frequency = None
    if len(new_peak_indices) >= 4:
        frequency = 1 / np.mean(np.diff(new_peak_indices[:4])) if np.mean(np.diff(new_peak_indices[:4])) != 0 else 0

    return new_peak_indices, new_trough_indices, frequency

def butter_smooth(ppg_signal, order, cutoff_freq):
    """Applies a Butterworth filter to smooth the PPG signal."""
    b, a = butter(order, cutoff_freq, btype='low')
    return filtfilt(b, a, ppg_signal) if len(ppg_signal) > 3 else ppg_signal

def process_chunk_and_floor_signal(chunk, height=1, distance=10):
    """Processes and floors the signal by normalizing segments between detected peaks."""
    peaks = find_time_peaks(chunk, 1, 2, 3)[0]
    floored_signal = np.copy(chunk)
    for i in range(1, len(peaks)):
        floored_signal[peaks[i-1]:peaks[i]] -= np.min(chunk[peaks[i-1]:peaks[i]])
    return floored_signal

class FrameControl:
    """Controls the real-time visualization and processing of PPG signals."""
    def __init__(self, ax1, ax2, ax3, ax4, data_fps, csv_file, refresh_frames, filter_order, filter_cutoff_freq):
        self.ax1, self.ax2, self.ax3, self.ax4 = ax1, ax2, ax3, ax4
        self.data_fps, self.csv_file, self.refresh_frames = data_fps, csv_file, refresh_frames
        self.filter_order, self.filter_cutoff_freq = filter_order, filter_cutoff_freq
        self.df = pd.read_csv(csv_file)
        self.data, self.time = self.df.iloc[:, 1], self.df.iloc[:, 0]
        self.num_points, self.i, self.prev_data = len(self.data), 0, np.array([])
        self.floor_signal = process_chunk_and_floor_signal(self.data)
        self.floor_signal_peaks = find_time_peaks(self.floor_signal, 1, 2, 3)[0]
        self.floor_signal = self.floor_signal[20:]

    def next_frame(self, event=None):
        self.ax1.clear(), self.ax2.clear(), self.ax3.clear(), self.ax4.clear()
        end = min(self.i + self.refresh_frames, self.num_points)
        current_data = butter_smooth(self.data[self.i:end], self.filter_order, self.filter_cutoff_freq)
        current_time = self.time[self.i:end]
        peak_indices, _, hr = find_time_peaks(current_data, 1, 2, 3)
        
        self.ax1.plot(current_time, current_data, color='blue')
        self.ax1.plot(current_time.iloc[peak_indices], current_data[peak_indices], 'rx', label='Peaks')
        self.ax1.set_title(f'Real-Time Data Plot')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Data')
        
        fft_data = np.fft.fft(current_data)
        psd = np.abs(fft_data) ** 2
        freq = np.fft.fftfreq(len(current_data), 1 / self.data_fps)
        valid_indices = np.where((freq > 0.67) & (freq < 10))
        self.ax2.plot(freq[valid_indices], psd[valid_indices], color='green')
        max_psd_index = np.argmax(psd[valid_indices])
        heart_rate = freq[valid_indices][max_psd_index] * 60
        self.ax2.set_title(f'Fourier Transform, HR: {heart_rate:.2f} BPM')
        self.ax2.set_xlabel('Frequency')
        self.ax2.set_ylabel('Power Spectrum')
        
        self.ax3.plot(current_time, self.floor_signal[self.i:end], color='red')
        self.ax3.set_title('Floored Signal')
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Amplitude')
        
        plt.draw()
        self.i += self.refresh_frames

def plot_real_time_data(data_fps, csv_file, refresh_frames, filter_order, filter_cutoff_freq):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    controller = FrameControl(ax1, ax2, ax3, ax4, data_fps, csv_file, refresh_frames, filter_order, filter_cutoff_freq)
    controller.next_frame()
    button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
    button = widgets.Button(button_ax, 'Next Frame')
    button.on_clicked(controller.next_frame)
    plt.show()

plot_real_time_data(34.8, 'FILENAME.csv', 150, 3, 0.7)
