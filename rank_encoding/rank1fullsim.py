"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script simulates eigenvalue computations of rank-1 signals under varying Signal-to-Noise Ratios (SNR).
It introduces Additive White Gaussian Noise (AWGN) and computes eigenvalues, ranking them based on their
distance from ideal vectors. The script also evaluates Bit Error Rate (BER) and average distance from the
ideal vector, plotting BER and distance versus SNR to analyze the robustness of eigenvalue classification.
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def awgn(signal, snr):
    """Adds Additive White Gaussian Noise (AWGN) to a signal."""
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def compute_eigenvals_rank1(snr_val):
    """
    Computes eigenvalues of a rank-1 signal with added noise at a given SNR.
    """
    E1, E2, Fs, nu = 1, 1, 10000, 10
    t = np.linspace(0, 1 - 1 / Fs, Fs)
    EHa = E1 * np.cos(2 * np.pi * nu * t) + 1
    y, z = awgn(EHa, snr_val), awgn(EHa, snr_val)
    EVa, EVb = np.zeros_like(EHa), np.zeros_like(EHa)
    G = np.array([[np.mean(y * np.conj(y)), np.mean(y * np.conj(EVa)), np.mean(y * np.conj(z)), np.mean(y * np.conj(EVb))],
                  [np.mean(EVa * np.conj(y)), np.mean(EVa * np.conj(EVa)), np.mean(EVa * np.conj(z)), np.mean(EVa * np.conj(EVb))],
                  [np.mean(z * np.conj(y)), np.mean(z * np.conj(EVa)), np.mean(z * np.conj(z)), np.mean(z * np.conj(EVb))],
                  [np.mean(EVb * np.conj(y)), np.mean(EVb * np.conj(EVa)), np.mean(EVb * np.conj(z)), np.mean(EVb * np.conj(EVb))]])
    G /= np.trace(G)
    return eigh(G, eigvals_only=True), 1

def round_and_sort_elements(original_vector):
    """Rounds and sorts eigenvalues in descending order."""
    return np.sort(np.round(original_vector[:4], 2))[::-1]

ideal_vector = np.array([[1, 0, 0, 0], [0.5, 0.5, 0, 0], [1/3, 1/3, 1/3, 0], [0.25, 0.25, 0.25, 0.25]])

def calculate_distance(input_vector, ideal_vector):
    """Calculates Euclidean distance from ideal vectors."""
    return np.array([np.linalg.norm(vec - input_vector) for vec in ideal_vector])

def calculate_rank(distanced_vector):
    """Determines rank based on the closest ideal vector."""
    return np.argmin(distanced_vector) + 1

def calculate_ber(original_rank, actual_rank):
    """Calculates Bit Error Rate (BER) based on rank deviation."""
    ber_map = {0: 0, 1: 1, 2: 2, 3: 2}
    diff = abs(original_rank - actual_rank)
    return ber_map.get(diff, diff)

def num_simulation_runs(run_times, snr_val):
    """Runs simulations and computes average BER and distance metrics."""
    ber_vals, dist_vals = [], []
    for _ in range(run_times):
        try:
            generated_vector, _ = compute_eigenvals_rank1(snr_val)
            rounded_sorted_vector = round_and_sort_elements(generated_vector)
            dist_vector = calculate_distance(rounded_sorted_vector, ideal_vector)
            calced_rank = calculate_rank(dist_vector)
            ber = calculate_ber(1, calced_rank)
            ber_vals.append(ber)
            dist_vals.append(np.min(dist_vector))
        except Exception as e:
            print(f"Error: {e}")
            continue
    return np.mean(ber_vals), np.mean(dist_vals)

def plot_ber_and_dist_vs_snr(snr_values, num_runs):
    """Plots BER and average distance versus SNR."""
    ber_values, dist_values = [], []
    for snr in snr_values:
        avg_ber, avg_dist = num_simulation_runs(num_runs, snr)
        ber_values.append(avg_ber)
        dist_values.append(avg_dist)
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, ber_values, 'b-o', label='BER vs. SNR')
    plt.title('Bit Error Rate vs. Signal-to-Noise Ratio')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER (log scale)')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, dist_values, 'r-o', label='Average Distance vs. SNR')
    plt.title('Average Distance from Ideal Vector vs. Signal-to-Noise Ratio')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Average Distance')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

snr_range = [1e-45, 1, 2, 4, 10, 20, 30, 40]
plot_ber_and_dist_vs_snr(snr_range, 1000)
