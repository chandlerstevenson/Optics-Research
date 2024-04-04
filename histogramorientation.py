#!/usr/bin/env python3.11
import glob
import os
import re
import matplotlib.pyplot as plt

def channel_orient_txt(line):
    """Extracts numerical value from a line describing orientation."""
    match = re.search(r'\d+', line)
    if match:
        return float(match.group())
    else:
        raise ValueError("No number found in line.")

def process_all_files(main_directory):
    """Loops through all .txt files in the given directory and processes them."""
    # Create a pattern to match all .txt files in the main directory
    pattern = os.path.join(main_directory, 'ChannelOrientation_*.txt')
    
    # Get a list of all matching files
    files = glob.glob(pattern)
    # Store orientation numbers from all files
    all_orientations = []
    
    for filepath in files:
        # For each file, read the first two lines and extract orientation numbers
        with open(filepath, 'r') as file:
            lines = file.readlines()[:2]
            file_orientations = []
            for line in lines:
                try:
                    num = channel_orient_txt(line.strip())
                    file_orientations.append(num)
                except ValueError as e:
                    print(f"Error processing line in {filepath}: {line.strip()} - {e}")
            all_orientations.append(file_orientations)
    
    return all_orientations

def plot_orientation_histograms(orientations):
    """Plots histograms for the first and second orientation elements from each file."""
    # Extract the first and second elements
    first_elements = [orientation[0] for orientation in orientations if len(orientation) > 0]
    second_elements = [orientation[1] for orientation in orientations if len(orientation) > 1]
    
    # Plot histograms
    plt.figure(figsize=(14, 6))
    
    # Histogram for the first elements
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.hist(first_elements, bins=20, alpha=0.7)
    plt.title('Histogram of Arm a')
    plt.xlabel('Orientation Value')
    plt.ylabel('Count')
    
    # Histogram for the second elements
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.hist(second_elements, bins=20, alpha=0.7)
    plt.title('Histogram of Arm b')
    plt.xlabel('Orientation Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

main_directory = '/Users/chandlerstevenson/Downloads/4-rank-encoding'
orientations = process_all_files(main_directory)
plot_orientation_histograms(orientations)
