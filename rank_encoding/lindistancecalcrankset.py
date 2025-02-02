"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script processes eigenvalue data stored in subfolders of a main directory.
It extracts eigenvalues from text files, sorts and rounds them, calculates their distance
from predefined ideal vectors, determines ranks, and writes the results to a CSV file.
"""

import os  # Import for interacting with the operating system
import csv  # Import for CSV file operations
import numpy as np  # Import for numerical operations

def file_to_array_as_numbers(filepath):
    """
    Reads a text file and converts each line into a numerical value (float).
    
    Parameters:
    filepath (str): Path to the text file.
    
    Returns:
    list: A list of numbers extracted from the file.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
        numbers = []
        for line in lines:
            stripped_line = line.strip()
            try:
                num = float(stripped_line)
            except ValueError:
                print(f"Could not convert line to number: {stripped_line}")
                continue
            numbers.append(num)
    return numbers

def subtract_arrays(array1, array2):
    """
    Performs element-wise subtraction of two arrays.
    
    Parameters:
    array1 (list): First array.
    array2 (list): Second array.
    
    Returns:
    list: A new array with element-wise differences.
    """
    return [a - b for a, b in zip(array1, array2)]

def count_positives(array):
    """
    Counts the number of non-negative values in an array.
    
    Parameters:
    array (list): Array of numerical values.
    
    Returns:
    int: The count of non-negative numbers.
    """
    return len([num for num in array if num >= 0])

def round_and_sort_elements(original_vector):   
    """
    Rounds and sorts the eigenvalues in descending order.
    
    Parameters:
    original_vector (list): List of eigenvalues.
    
    Returns:
    numpy.array: Sorted and rounded eigenvalues.
    """
    rounded_vector = np.round(original_vector[:4], 2)  # Round the first four elements
    return np.sort(rounded_vector)[::-1]  # Sort in descending order

def calculate_distance(input_vector, ideal_vector):
    """
    Calculates the distance between the input vector and a set of ideal vectors.
    
    Parameters:
    input_vector (numpy.array): Processed eigenvalues.
    ideal_vector (numpy.array): Set of predefined ideal eigenvalue distributions.
    
    Returns:
    numpy.array: Distance values between input and ideal vectors.
    """
    distance_vector = np.array([sum(np.abs(vec - input_vector)) for vec in ideal_vector])
    return distance_vector

# Define ideal eigenvalue distributions
ideal_vector = np.array([[1, 0, 0, 0], [0.5, 0.5, 0, 0], [1/3, 1/3, 1/3, 0], [0.25, 0.25, 0.25, 0.25]])

def calc_rank(distanced_vector):
    """
    Determines the rank based on the closest ideal vector.
    
    Parameters:
    distanced_vector (numpy.array): Distance values between input and ideal vectors.
    
    Returns:
    int: The assigned rank (1-4).
    """
    return np.argmin(distanced_vector) + 1  # Rank is index of minimum distance +1

def write_to_csv(file_path, data):
    """
    Writes the rank data to a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    data (list): Data to be written.
    """
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Folder Name', 'Rank'])
        writer.writerows(data)

def process_folders(main_directory):
    """
    Processes each subfolder to compute eigenvalue ranks and stores results in a CSV file.
    
    Parameters:
    main_directory (str): Path to the main directory containing subfolders.
    """
    results = []
    for folder_name in os.listdir(main_directory):
        folder_path = os.path.join(main_directory, folder_name)
        if os.path.isdir(folder_path):
            eigenvalues_path = os.path.join(folder_path, 'Eigenvalues.txt')
            if os.path.isfile(eigenvalues_path):
                eigenvalues = file_to_array_as_numbers(eigenvalues_path)
                sorted_eigs = round_and_sort_elements(eigenvalues)
                dist_eigs = calculate_distance(sorted_eigs, ideal_vector)
                rank = calc_rank(dist_eigs)
                results.append([folder_name, rank])
    
    csv_path = os.path.join(main_directory, 'ranks_outputNEW4.csv')
    write_to_csv(csv_path, results)

if __name__ == "__main__":
    main_directory = '/Users/chandlerstevenson/Downloads/4-rank-encoding/Image0/rank-4'  # Modify as needed
    process_folders(main_directory)
