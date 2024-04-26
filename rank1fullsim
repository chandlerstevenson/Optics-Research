import numpy as np
from scipy.linalg import eigh

def compute_eigenvals_rank1(snr_val):
    # Constants
    E1 = 1
    E2 = 1
    Fs = 10000
    t = np.linspace(0, 1 - 1 / Fs, Fs)
    nu = 10  # frequency (monochromatic)

    # Signal definitions
    EHa = E1 * np.cos(2 * np.pi * nu * t) + 1  # added one so that minimum is 0 (not necessary)

    EHb = EHa
    EVa = np.zeros_like(EHa)
    EVb = np.zeros_like(EHa)

    # Function to add white Gaussian noise
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
        return noisy_signal

    SnR = snr_val  # Noise in dB
    y = awgn(EHa, SnR)
    z = awgn(EHb, SnR)

    # Recompute G for noisy signals
    G = np.array([[np.mean(y * np.conj(y)), np.mean(y * np.conj(EVa)), np.mean(y * np.conj(z)), np.mean(y * np.conj(EVb))],
                  [np.mean(EVa * np.conj(y)), np.mean(EVa * np.conj(EVa)), np.mean(EVa * np.conj(z)), np.mean(EVa * np.conj(EVb))],
                  [np.mean(z * np.conj(y)), np.mean(z * np.conj(EVa)), np.mean(z * np.conj(z)), np.mean(z * np.conj(EVb))],
                  [np.mean(EVb * np.conj(y)), np.mean(EVb * np.conj(EVa)), np.mean(EVb * np.conj(z)), np.mean(EVb * np.conj(EVb))]])
    G /= np.trace(G)
    eigenvals = eigh(G, eigvals_only=True)

    return (eigenvals, 1)




# Ensure that rank vector is in descending order 
def round_and_sort_elements(original_vector):   
    original_vector = np.array(original_vector) 
    round_1 = np.round(original_vector[0], 2)
    round_2 = np.round(original_vector[1], 2)
    round_3 = np.round(original_vector[2], 2)
    round_4 = np.round(original_vector[3], 2) 
    rounded_vector = np.array([round_1, round_2, round_3, round_4]) 
    sorted_and_rounded_vector = np.sort(rounded_vector)
    return sorted_and_rounded_vector[::-1]


vector, _ = compute_eigenvals_rank1(1) 

print(vector)
# Ideal rank vector 
ideal_vector = np.array([[1, 0, 0, 0], [0.5, 0.5, 0, 0], [1/3, 1/3, 1/3, 0], [0.25, 0.25, 0.25, 0.25]]) 

# # EUCLIDEAN DISTANCE ---------------------------------------------------------------------------
# Calculates distance of  input vector from ideal vector 
def calculate_distance(input_vector, ideal_vector):
    distance_vector = np.array([])
    for vec in ideal_vector:
        if len(vec) != len(input_vector):
            raise ValueError("Vectors must have the same dimensionality")
        distance = np.sqrt(sum((vec[i] - input_vector[i])**2 for i in range(len(input_vector))))
        distance_vector = np.append(distance_vector, distance)
    return distance_vector 


# rank is the minimum distance to an ideal rank vector
def calculate_rank(distanced_vector):  
    rank_set = [1, 2, 3, 4]
    calculated_rank = rank_set[np.argmin(distanced_vector)] 
    return calculated_rank

def calculate_ber(original_rank, actual_rank): 
    num = np.abs(original_rank-actual_rank) 
    ber_map = {0:0, 1:1, 2:2, 3:2}
    num = np.abs(original_rank - actual_rank)
    
    # Apply the mapping to the difference if the difference exists in the mapping
    if num in ber_map:
        ber_value = ber_map[num]
    else:
        ber_value = num  # Or handle it differently if the num is not in the mapping
    
    return ber_value

def num_simulation_runs(run_times, snr_val):
    ber_vals = []
    dist_vals = []
    
    for i in range(run_times):
        try:
            generated_vector, _ = compute_eigenvals_rank1(snr_val)
            rounded_sorted_vector = round_and_sort_elements(generated_vector)
            dist_vector = calculate_distance(rounded_sorted_vector, ideal_vector)
            calced_rank = calculate_rank(dist_vector)
            ber = calculate_ber(1, calced_rank)  # Assuming num_1 is always 1, as seen before
            
            ber_vals.append(ber)
            dist_vals.append(np.min(dist_vector))  # Store the minimum distance from the ideal vectors
        except Exception as e:
            print(f"An error occurred during run {i}: {e}")
            continue

    avg_ber = np.mean(ber_vals) if ber_vals else 0  # Compute average BER
    avg_dist = np.mean(dist_vals) if dist_vals else 0  # Compute average distance
    return avg_ber, avg_dist


import matplotlib.pyplot as plt

def plot_ber_and_dist_vs_snr(snr_values, num_runs):
    ber_values = []
    dist_values = []
    
    for snr in snr_values:
        avg_ber, avg_dist = num_simulation_runs(num_runs, snr)
        ber_values.append(avg_ber)
        dist_values.append(avg_dist)
    
    # Plot BER vs SNR
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, ber_values, 'b-o', label='BER vs. SNR')
    plt.title('Bit Error Rate vs. Signal-to-Noise Ratio')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER (log scale)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

    # Plot Distance vs SNR
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, dist_values, 'r-o', label='Average Distance vs. SNR')
    plt.title('Average Distance from Ideal Vector vs. Signal-to-Noise Ratio')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Average Distance')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


snr_range = [.00000000000000000000000000000000000000000000001, 1, 2, 4, 10, 20, 30, 40]  # Define your range of SNR values
plot_ber_and_dist_vs_snr(snr_range, 1000)  # 100 runs for each SNR value
