import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import time
import matplotlib.widgets as widgets 
from scipy.signal import argrelextrema
def center_time(time_data):
    time_data = np.array(time_data)  # Convert to numpy array for vectorized operations
    shift_value = time_data[0]
    new_time_data = time_data - shift_value
    return new_time_data

def find_time_peaks(ppg_signal, skip_val, max_distance, peak_trough_distance):
    comparator = np.greater
    peak_indices = argrelextrema(ppg_signal, comparator)

    comparator = np.less
    trough_indices = argrelextrema(ppg_signal, comparator)

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
        average_period = np.mean(np.diff(new_peak_indices[:4]))
        frequency = 1 / average_period if average_period != 0 else 0  # Avoid division by zero
    else:
        print("Not enough peaks for frequency calculation")
        frequency = None

    return new_peak_indices, new_trough_indices, tolerance_peaks, tolerance_troughs, frequency


def butter_smooth(ppg_signal, order, cutoff_freq):
    N = order
    fc = cutoff_freq
    b, a = butter(N, fc, btype='low')

    try:
        filtered_signal_butter = filtfilt(b, a, ppg_signal)
    except ValueError:
        filtered_signal_butter = ppg_signal

    return filtered_signal_butter 

def find_fake_peaks(signal, height, distance):
    peaks = []
    last_peak_index = -1
    for i in range(1, len(signal) - 1):  
        if signal[i] > height and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if last_peak_index == -1 or i - last_peak_index >= distance: 
                peaks.append(i)
                last_peak_index = i

    # Check if the last point is a peak
    if signal[-1] > height and signal[-1] > signal[-2]:
        if len(peaks) == 0 or (len(signal) - 1) - peaks[-1] >= distance:
            peaks.append(len(signal) - 1)

    return peaks

def process_chunk_and_floor_signal(chunk, height=1, distance=10):
    signals = chunk
    signals = signals.to_numpy()
    peaks = find_fake_peaks(signals, height, distance)

    floored_signal = np.copy(signals)
    for i in range(1, len(peaks)):
        segment = signals[peaks[i-1] : peaks[i]]
        floored_signal[peaks[i-1] : peaks[i]] = segment - np.min(segment)

    return floored_signal


class FrameControl:
    def __init__(self, ax1, ax2, ax3, ax4, data_fps, csv_file, refresh_frames, filter_order, filter_cutoff_freq):
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        self.data_fps = data_fps
        self.csv_file = csv_file
        self.refresh_frames = refresh_frames
        self.filter_order = filter_order
        self.filter_cutoff_freq = filter_cutoff_freq
        self.df = pd.read_csv(csv_file)
        self.data = self.df.iloc[:, 1]
        self.time = self.df.iloc[:, 0]  # Time data from the 0th column
        self.num_points = len(self.data)
        self.i = 0
        self.prev_data = np.array([])
        self.floor_signal = process_chunk_and_floor_signal(self.data)  # Floor the entire data 
        self.floor_signal_peaks = find_time_peaks(self.floor_signal, 1, 2, 3) # find peaks of floored signal
        self.floor_signal = self.floor_signal[20:]  # Remove the first 20 frames

    def next_frame(self, event=None):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        end = min(self.i + self.refresh_frames, self.num_points)
        current_data = self.data[self.i:end]
        current_time = self.time[self.i:end]  # Time data for the current chunk

        floored_signal_chunk = self.floor_signal[self.i:end]  # Retrieve the corresponding floored signal chunk

        if len(self.prev_data) == 0:
            cross_corr = 0
        else:
            cross_corr = np.correlate(
                self.prev_data - np.mean(self.prev_data),
                current_data - np.mean(current_data),
                'valid'
            )
            if len(cross_corr) > 0:
                cross_corr = cross_corr[0] / (
                    len(self.prev_data) * np.std(self.prev_data) * np.std(current_data)
                )
            else:
                cross_corr = 0

        current_data = butter_smooth(current_data, self.filter_order, self.filter_cutoff_freq)  
        peak_indices, trough_indices, tol_peaks, tol_troughs, hr  = find_time_peaks(current_data, 1, 2, 3.2) 


        line1, = self.ax1.plot(current_time, current_data, color='blue')  # Use current_time as x-axis 

        current_time_np = current_time.to_numpy() 
        current_time_period = current_time_np[-1] - current_time_np[0] 
   

        


        fft_data = np.fft.fft(current_data)
        psd = np.abs(fft_data) ** 2
        freq = np.fft.fftfreq(len(current_data), 1 / self.data_fps)

        valid_indices = np.where((freq > 0.67) & (freq < 10))
        line2, = self.ax2.plot(freq[valid_indices], psd[valid_indices], color='green')

        max_psd_index = np.argmax(psd[valid_indices])
        heart_rate = freq[valid_indices][max_psd_index] * 60
        self.ax2.plot(freq[valid_indices][max_psd_index], psd[valid_indices][max_psd_index], 'rx')
        self.ax1.set_title(
            # f'Real-Time Data Plot {current_time_period}s' 
            f'Real-Time Data Plot'
        )
        
        line1, = self.ax1.plot(current_time_np[peak_indices], current_data[peak_indices], "x", label='Peaks', color='r') 
        expected_num_peaks = heart_rate * (1/60) * current_time_period 

        self.ax2.set_title(f'Fourier Transform, Heart Rate: {heart_rate:.2f} BPM, Expected Peaks: {round(expected_num_peaks, 1)}')
        self.ax1.set_xlabel('Time')  # Set x-axis label to 'Time'
        self.ax1.set_ylabel('Data')
        self.ax2.set_xlabel('Frequency')
        self.ax2.set_ylabel('Power Spectrum')

        # Gradient bar graph for cross correlation
        bar_value = abs(cross_corr)
        self.ax4.bar(0, bar_value, color='blue')
        self.ax4.set_xlim(-0.5, 0.5)
        self.ax4.set_ylim(0, 1)
        self.ax4.set_xticks([])
        self.ax4.set_yticks([])
        self.ax4.set_title(f'Cross-Correlation: {abs(cross_corr):.2f}')

        # Plot floored signal and average amplitude
        self.ax3.clear()
        self.ax3.plot(current_time, floored_signal_chunk, color='red') 
        peaks_of_floored_graph = find_time_peaks(floored_signal_chunk, 1, 2, 3)
        self.ax3.set_title(f'Floored Signal, Average Amplitude: {np.mean(floored_signal_chunk):.2f}')
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Amplitude')

        self.ax1.relim()
        self.ax1.autoscale_view(True, True, True)
        self.ax2.set_xlim(0.67, 10)
        self.ax2.relim()
        self.ax2.autoscale_view(True, True, True)

        plt.draw()

        self.prev_data = current_data
        self.i += self.refresh_frames

        if self.num_points - self.i <= 150:
            self.button.label.set_text('Exit Simulation')
            self.button.color = 'red'
            self.button.on_clicked(ending_screen)
        elif self.i >= self.num_points:
            plt.close()
            ending_screen()


def plot_real_time_data(data_fps, csv_file, refresh_frames, filter_order, filter_cutoff_freq):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    controller = FrameControl(ax1, ax2, ax3, ax4, data_fps, csv_file, refresh_frames, filter_order, filter_cutoff_freq)
    controller.next_frame()

    button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
    controller.button = widgets.Button(button_ax, 'Next Frame')
    controller.button.on_clicked(controller.next_frame)

    plt.subplots_adjust(hspace=0.4)
    plt.show()

    return controller


def begin_simulation(event):
    welcome_fig.clear()
    plt.close(welcome_fig)
    plot_real_time_data(34.8, 'TEK00005.csv', 150, 3, 0.6)


def welcome_screen():
    global welcome_fig
    welcome_fig, ax = plt.subplots(figsize=(8, 6))
    welcome_fig.canvas.set_window_title('PPG Analysis Simulation')
    ax.text(0.5, 0.5, 'Welcome to the PPG Analysis Simulation', fontsize=18, ha='center', va='center')
    ax.axis('off')
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    button_ax = plt.axes([0.4, 0.1, 0.2, 0.1])
    button = widgets.Button(button_ax, 'Begin')

    button.on_clicked(begin_simulation)

    plt.show()


def ending_screen():
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.set_window_title('PPG Analysis Simulation')
    ax.text(0.5, 0.5, 'Analysis Complete', fontsize=18, ha='center', va='center')
    ax.axis('off')   
    button_ax = plt.axes([0.4, 0.1, 0.2, 0.1])
    button = widgets.Button(button_ax)
    button.on_clicked

    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.show()


welcome_screen()



plot_real_time_data(34.8, 'FILENAME.csv', 150, 3, 0.7)
