import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def awgn(signal, snr):
    # Calculate the signal power
    signal_power = np.mean(signal**2)
    
    # Convert SNR from dB to a linear scale
    snr_linear = 10**(snr / 10)
    
    # Calculate the noise power based on the signal power and SNR
    noise_power = signal_power / snr_linear
    
    # Generate white Gaussian noise with mean 0 and calculated noise power
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    # Add noise to the original signal and return
    noisy_signal = signal + noise
    return noisy_signal, noise

# Generate a test signal
fs = 100000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
signal = np.sin(2 * np.pi * 5 * t)  # A 5 Hz sine wave

# Add noise with a specific SNR
snr = 10  # Signal-to-Noise Ratio in dB
noisy_signal, noise = awgn(signal, snr)

# Plot the histogram of the noise
plt.figure(figsize=(10, 6))
plt.hist(noise, bins=30, density=True, alpha=0.6, color='g', label='Noise Histogram')

# Overlay the normal distribution curve
mu, std = np.mean(noise), np.std(noise)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Fit results: $\mu$ = %.2f, $\sigma$ = %.2f' % (mu, std))

plt.title('Noise Distribution')
plt.xlabel('Noise Value')
plt.ylabel('Density')
plt.legend()
plt.show()
