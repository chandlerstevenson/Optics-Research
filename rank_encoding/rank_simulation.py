"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script generates 4D vectors based on predefined constraints, ranks them based on distance
from ideal rank vectors using various distance metrics, and visualizes the ranked vectors in
3D plots. The script provides multiple plotting functions to analyze the distribution of vectors
and their rankings, including isolated rank visualizations and exploded views.
"""

#!/usr/bin/env python3.11 

import numpy as np 
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_vector():
    """Generates a 4D vector satisfying predefined constraints and normalizes it to sum to 1."""
    x = random.uniform(0.25, 1)
    y = random.uniform(0, 0.5)
    z = random.uniform(0, 1/3)
    alpha = random.uniform(0, 0.25)
    total = x + y + z + alpha
    x, y, z, alpha = x / total, y / total, z / total, alpha / total
    if not (0.25 <= x <= 1 and 0 <= y <= 0.5 and 0 <= z <= 1/3 and 0 <= alpha <= 0.25):
        return generate_vector()
    return [x, y, z, alpha]

def round_and_sort_elements(original_vector):
    """Rounds and sorts the vector elements in descending order."""
    rounded_vector = np.round(original_vector[:4], 2)
    return np.sort(rounded_vector)[::-1]

# Ideal rank vectors
ideal_vector = np.array([[1, 0, 0, 0], [0.5, 0.5, 0, 0], [1/3, 1/3, 1/3, 0], [0.25, 0.25, 0.25, 0.25]])

def calculate_distance(input_vector, ideal_vector):
    """Calculates Euclidean distance from ideal vectors."""
    return np.array([np.linalg.norm(vec - input_vector) for vec in ideal_vector])

def calculate_rank(distanced_vector):
    """Determines rank based on the closest ideal vector."""
    return np.argmin(distanced_vector) + 1

def num_simulation_runs(run_times):
    """Runs the simulation a specified number of times and returns ranked vectors."""
    plot_vectors = []
    for _ in range(run_times):
        generated_vector = generate_vector()
        rounded_sorted_vector = round_and_sort_elements(generated_vector)
        dist_vector = calculate_distance(rounded_sorted_vector, ideal_vector)
        rank = calculate_rank(dist_vector)
        plot_vectors.append((np.array(rounded_sorted_vector[:3]), rank))
    return plot_vectors

def percent_rank(data):
    """Calculates and prints the percentage of occurrences for each rank."""
    from collections import Counter
    numbers = [num for _, num in data]
    count = Counter(numbers)
    total = sum(count.values())
    for num in range(1, 5):
        percentage = (count[num] / total) * 100 if num in count else 0
        print(f"Percentage of Rank {num}: {percentage:.2f}%")

# Run simulation and plot results
run_sim = num_simulation_runs(100000)
percent_rank(run_sim)
