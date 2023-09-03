import pandas as pd
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, lfilter, freqz
import matplotlib.pyplot as plt
import numpy as np

# Filter requirements.
order = 6
fs = 30.0       # sample rate, Hz
lowcut = 3.0    # desired cutoff frequency of the filter, Hz
wc = lowcut      # cutoff frequency for the filter

def apply_filter(b, a, data):
    y = lfilter(b, a, data)
    return y

def plot_filter_response(b, a, fs, wc, title, passband=None, stopband=None, color='blue'):
    w, h = freqz(b, a, worN=8000)
    plt.plot((0.5*fs*w/np.pi)/wc, np.abs(h), color=color, label=title, linewidth=2)  # w/wc on x-axis, in dB
    if passband:
        plt.text(passband/wc, 0.5, 'Passband', fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    if stopband:
        plt.text(stopband/wc, 0.5, 'Stopband', fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Now, generate the plot
plt.style.use('classic')  # Use classic MATLAB style
plt.figure(figsize=(10,6))
plt.title('Frequency Response of Various Filters')
plt.xlim(0, 2)
plt.xlabel(r'$\omega / \omega_c$', fontsize=18)  # use LaTeX syntax for Greek letter omega
plt.ylabel('Gain', fontsize = 14)
plt.grid(True)

# Your .csv file here
csv_file = 'gannon_bessel_0_Time.csv'
data = pd.read_csv(csv_file)

# Extract the signal data from the CSV (assuming it's in the second column)
signal = data.iloc[:, 1].values.astype(float)  # Convert to numerical array    

# Apply and plot filter responses
passband = 1.5  # Set the frequency value here as per your filter response
stopband = 4.5  # Set the frequency value here as per your filter response

colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # MATLAB-like color sequence

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
