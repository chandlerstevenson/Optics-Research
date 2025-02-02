"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script processes text files containing orientation data from a specified directory.
It extracts numerical orientation values, adjusts them using a predefined conversion vector,
and saves the processed data into a CSV file. Additionally, it visualizes the orientation
values using histograms for further analysis.
"""

#!/usr/bin/env python3.11

import glob  # Used for Unix-style pathname pattern expansion
import os  # Module for interacting with the file system
import re  # Module for working with regular expressions
import matplotlib.pyplot as plt  # Import for plotting
import csv  # Module for handling CSV file operations
import numpy as np  # Import for numerical operations

def save_orientations_to_csv(orientations, file_paths, output_file):
    """
    Saves extracted orientation data and file names to a CSV file.
    
    Parameters:
    orientations (list): List of extracted orientation values.
    file_paths (list): Corresponding file paths.
    output_file (str): Path to save the CSV file.
    """
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'arm_a', 'arm_b']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for orientation, file_path in zip(orientations, file_paths):
            arm_a = orientation[0] if len(orientation) > 0 else ''
            arm_b = orientation[1] if len(orientation) > 1 else ''
            writer.writerow({'file_name': os.path.basename(file_path), 'arm_a': arm_a, 'arm_b': arm_b})

def channel_orient_txt(line):
    """
    Extracts numerical value from a text line describing orientation.
    
    Parameters:
    line (str): A line from the orientation file.
    
    Returns:
    float: Extracted numerical value.
    """
    match = re.search(r'\d+', line)
    if match:
        return float(match.group())
    else:
        raise ValueError("No number found in line.")

def process_all_files(main_directory):
    """
    Processes all orientation text files in the specified directory.
    
    Parameters:
    main_directory (str): Path to the directory containing orientation files.
    
    Returns:
    tuple: List of processed orientations and corresponding file paths.
    """
    pattern = os.path.join(main_directory, 'ChannelOrientation_*.txt')
    files = glob.glob(pattern)
    all_orientations = []
    
    for filepath in files:
        with open(filepath, 'r') as file:
            lines = file.readlines()[:2]  # Read the first two lines from each file
            conversion_vector = [97, 74]  # Define the conversion vector for adjustments
            file_orientations = []
            for line in lines:
                try:
                    num = channel_orient_txt(line.strip())
                    file_orientations.append(num)
                except ValueError as e:
                    print(f"Error processing line in {filepath}: {line.strip()} - {e}")
            adjusted_orientations = [np.abs(orientation - conversion) for orientation, conversion in zip(file_orientations, conversion_vector)]
            all_orientations.append(adjusted_orientations)
    
    return all_orientations, files

def plot_orientation_histograms(orientations):
    """
    Generates histograms for extracted orientation values.
    
    Parameters:
    orientations (list): List of extracted orientation values.
    """
    first_elements = [orientation[0] for orientation in orientations if len(orientation) > 0]
    second_elements = [orientation[1] for orientation in orientations if len(orientation) > 1]
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(first_elements, bins=20, alpha=0.7)
    plt.title('Histogram of Arm a')
    plt.xlabel('Orientation Value')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(second_elements, bins=20, alpha=0.7)
    plt.title('Histogram of Arm b')
    plt.xlabel('Orientation Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

# Define main directory containing orientation files
main_directory = '/Users/chandlerstevenson/Downloads/4-rank-encoding'
orientations, file_paths = process_all_files(main_directory)
plot_orientation_histograms(orientations)

# Define output CSV file path
output_file_path = '/Users/chandlerstevenson/Downloads/4-rank-encoding/orientations_output.csv'
save_orientations_to_csv(orientations, file_paths, output_file_path)
