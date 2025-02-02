"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script processes eigenvalue data stored in subfolders of a main directory.
It calculates the rank of each folder based on eigenvalues read from a text file.
The processed data is then written to a CSV file.
"""

import os  # Import for interacting with the operating system
import csv  # Import for CSV file operations

# Global tolerance value used in rank calculations
tolerance_val = 0.2

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

def calc_rank(rank_array):
    """
    Determines the rank based on eigenvalue data.
    
    Parameters:
    rank_array (list): List of eigenvalues.
    
    Returns:
    str: The rank classification.
    """
    tol_array = [tolerance_val] * 4  # Create an array of tolerance values
    rank_tolerance_array = subtract_arrays(rank_array, tol_array)
    rank_count = ['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4']
    rank_num = count_positives(rank_tolerance_array)
    return rank_count[rank_num - 1]

def write_to_csv(file_path, data):
    """
    Writes data to a CSV file.
    
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
                rank = calc_rank(eigenvalues)
                results.append([folder_name, rank])
    csv_path = os.path.join(main_directory, 'ranks_output.csv')
    write_to_csv(csv_path, results)

if __name__ == "__main__":
    main_directory = '/Users/chandlerstevenson/Downloads/4-rank-encoding/Image0/rank-1'  # Modify as needed
    process_folders(main_directory)
