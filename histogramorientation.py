#!/usr/bin/env python3.11
#specifies the script should run with Python 3.11

import glob  # Used for Unix style pathname pattern expansion
import os  
import re  # Module for working with regular expressions
import matplotlib.pyplot as plt  
import csv   
import numpy as np

def save_orientations_to_csv(orientations, file_paths, output_file):
    """Saves the orientations and file names to a CSV file."""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'arm_a', 'arm_b']  # Define CSV column headers
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # Create a writer object that maps dictionaries onto output rows

        writer.writeheader()  # Write the column headers
        for orientation, file_path in zip(orientations, file_paths):
            # Loop through each orientation and corresponding file path
            arm_a = orientation[0] if len(orientation) > 0 else ''  # Extract arm_a orientation if available
            arm_b = orientation[1] if len(orientation) > 1 else ''  # Extract arm_b orientation if available
            # Write a row to the CSV file
            writer.writerow({'file_name': os.path.basename(file_path), 'arm_a': arm_a, 'arm_b': arm_b})

def channel_orient_txt(line):
    """Extracts numerical value from a line describing orientation."""
    match = re.search(r'\d+', line)  # Search for the first occurrence of a numeric value in the line
    if match:
        return float(match.group())  # Return the found numeric value as a float
    else:
        raise ValueError("No number found in line.")  # If no numeric value is found, raise an error

def process_all_files(main_directory):
    """Loops through all .txt files in the given directory and processes them."""
    pattern = os.path.join(main_directory, 'ChannelOrientation_*.txt')  # Define the search pattern
    files = glob.glob(pattern)  # Find all files matching the pattern
    all_orientations = []  # Initialize a list to store all orientation data
    
    for filepath in files:
        with open(filepath, 'r') as file:
            lines = file.readlines()[:2]  # Read the first two lines from each file
            conversion_vector = [97, 74]  # Define the conversion vector for adjustments
            file_orientations = []  # List to hold orientations from the current file
            for line in lines:
                try:
                    num = channel_orient_txt(line.strip())  # Extract the numeric value
                    file_orientations.append(num)  # Append the numeric value to the list
                except ValueError as e:
                    print(f"Error processing line in {filepath}: {line.strip()} - {e}")
            # Adjust orientations by subtracting conversion vector elements and store them
            adjusted_orientations = [np.abs(orientation - conversion) for orientation, conversion in zip(file_orientations, conversion_vector)]
            all_orientations.append(adjusted_orientations)  # Add adjusted orientations to the main list
    
    return all_orientations, files  # Return the list of all orientations and the list of file paths

def plot_orientation_histograms(orientations):
    """Plots histograms for the first and second orientation elements from each file."""
    first_elements = [orientation[0] for orientation in orientations if len(orientation) > 0]  # Extract first elements
    second_elements = [orientation[1] for orientation in orientations if len(orientation) > 1]  # Extract second elements
    
    plt.figure(figsize=(14, 6))  # Set the figure size
    
    # Plot histogram for the first elements
    plt.subplot(1, 2, 1)
    plt.hist(first_elements, bins=20, alpha=0.7)
    plt.title('Histogram of Arm a')
    plt.xlabel('Orientation Value')
    plt.ylabel('Count')
    
    # Plot histogram for the second elements
    plt.subplot(1, 2, 2)
    plt.hist(second_elements, bins=20, alpha=0.7)
    plt.title('Histogram of Arm b')
    plt.xlabel('Orientation Value')
    plt.ylabel('Count')
    
    plt.tight_layout()  # Adjust the layout
    plt.show()  # Display the histograms

# Specify the main directory containing the orientation text files
main_directory = '/Users/chandlerstevenson/Downloads/4-rank-encoding'
# Process all files and get orientations and file paths
orientations, file_paths = process_all_files(main_directory)
# Plot orientation histograms
plot_orientation_histograms(orientations)

# Specify the output file path for the CSV file
output_file_path = '/Users/chandlerstevenson/Downloads/4-rank-encoding/orientations_output.csv'
# Save orientations to CSV
save_orientations_to_csv(orientations, file_paths, output_file_path)
