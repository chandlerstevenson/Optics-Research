"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script applies multiple digital filters (Butterworth, Chebyshev I, Chebyshev II,
Elliptic, and Bessel) to a signal extracted from a CSV file. The frequency responses
of these filters are plotted for comparison, with labeled passband and stopband regions.
"""

import pandas as pd
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, lfilter, freqz
import matplotlib.pyplot as plt
import numpy as np

# Filter parameters
order = 6
fs = 30.0       # Sample rate in Hz
lowcut = 3.0    # Desired cutoff frequency of the filter in Hz
wc = lowcut     # Cutoff frequency for normalization

def apply_filter(b, a, data):
    """
    Applies a digital filter to the given data.
    
    Parameters:
    b, a (array-like): Filter coefficients.
    data (array-like): Input signal.
    
    Returns:
    array-like: Filtered signal.
    """
    return lfilter(b, a, data)

def plot_filter_response(b, a, fs, wc, title, passband=None, stopband=None, color='blue'):
    """
    Plots the frequency response of a given filter.
    
    Parameters:
    b, a (array-like): Filter coefficients.
    fs (float): Sampling frequency.
    wc (float): Cutoff frequency.
    title (str): Title for the filter plot.
    passband (float, optional): Passband frequency marker.
    stopband (float, optional): Stopband frequency marker.
    color (str, optional): Plot color.
    """
    w, h = freqz(b, a, worN=8000)
    plt.plot((0.5 * fs * w / np.pi) / wc, np.abs(h), color=color, label=title, linewidth=2)
    if passband:
        plt.text(passband / wc, 0.5, 'Passband', fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    if stopband:
        plt.text(stopband / wc, 0.5, 'Stopband', fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Plot configuration
plt.style.use('classic')  # Use classic MATLAB style
plt.figure(figsize=(10,6))
plt.title('Frequency Response of Various Filters')
plt.xlim(0, 2)
plt.xlabel(r'$\omega / \omega_c$', fontsize=18)
plt.ylabel('Gain', fontsize=14)
plt.grid(True)

# Load signal data from CSV file
csv_file = 'gannon_bessel_0_Time.csv'
data = pd.read_csv(csv_file)
signal = data.iloc[:, 1].values.astype(float)  # Extract and convert signal data

# Define filter passband and stopband markers
passband = 1.5
stopband = 4.5

colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # Colors for different filters

# Apply and plot filter responses
b, a = butter(2, lowcut, fs=fs)
plot_filter_response(b, a, fs, wc, "Butterworth II filter", passband, stopband, color=colors[0])
b, a = cheby1(order, 0.5, lowcut, fs=fs)
plot_filter_response(b, a, fs, wc, "Chebyshev I filter", passband, stopband, color=colors[1])
b, a = cheby2(order, 50, lowcut, fs=fs)
plot_filter_response(b, a, fs, wc, "Chebyshev II filter", passband, stopband, color=colors[2])
b, a = ellip(order, 0.5, 50, lowcut, fs=fs)
plot_filter_response(b, a, fs, wc, "Elliptic filter", passband, stopband, color=colors[3])
b, a = bessel(order, lowcut, fs=fs)
plot_filter_response(b, a, fs, wc, "Bessel filter", passband, stopband, color=colors[4])

plt.legend(loc='best')
plt.tight_layout()
plt.show()
