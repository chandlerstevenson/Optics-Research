#!/usr/bin/env python3.11

import os  # Import the os module for interacting with the operating system
import csv  # Import the csv module for CSV file operations

# Define a global tolerance value that will be used in rank calculations
tolerance_val = .2

# Function to read a text file and convert each line into a numerical value (float)
def file_to_array_as_numbers(filepath):
    """Converts lines in a text file to a list of numbers (floats)."""
    with open(filepath, 'r') as file:  # Open the file in read mode
        lines = file.readlines()  # Read all lines from the file
        numbers = []  # Create an empty list to store the converted numbers
        for line in lines:  # Loop over each line in the file
            stripped_line = line.strip()  # Remove any leading/trailing whitespace
            try:
                # Try to convert the line to a float
                num = float(stripped_line)
            except ValueError:
                # If conversion to float fails, print an error message
                print(f"Could not convert line to number: {stripped_line}")
                continue  # Skip to the next line
            numbers.append(num)  # Append the converted number to the list
    return numbers  # Return the list of numbers

# Function to subtract two arrays of numbers element-wise
def subtract_arrays(array1, array2):
    """Subtracts elements of array2 from array1 element-wise and returns a new array."""
    # Use list comprehension to subtract elements pairwise and return a new list
    return [a - b for a, b in zip(array1, array2)]

# Function to count how many numbers in an array are positive
def count_positives(array):
    """Returns the count of positive numbers in an array."""
    # Use list comprehension to filter positive numbers and return their count
    return len([num for num in array if num >= 0])

# Function to calculate rank based on a given array of numbers
def calc_rank(rank_array):
    """Determines rank based on the array of numbers after adjusting with tolerance."""
    tol_array = 4 * [tolerance_val]  # Create an array of tolerance values
    # Subtract tolerance array from the rank array to get the adjusted values
    rank_tolerance_array = subtract_arrays(rank_array, tol_array)
    # Define the possible ranks
    rank_count = ['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4']
    # Count the number of positive values after adjustment
    rank_num = count_positives(rank_tolerance_array)
    # Return the corresponding rank
    return rank_count[rank_num - 1]

# Function to write data to a CSV file
def write_to_csv(file_path, data):
    """Writes the provided data to a CSV file at the specified path."""
    with open(file_path, 'w', newline='') as csvfile:  # Open the CSV file in write mode
        writer = csv.writer(csvfile)  # Create a CSV writer object
        writer.writerow(['Folder Name', 'Rank'])  # Write the header row
        writer.writerows(data)  # Write all the data rows at once

# Function to process each subfolder in the provided main directory
def process_folders(main_directory):
    """Calculates the rank for Eigenvalues in each subfolder and writes the results to a CSV file."""
    results = []  # Create an empty list to store the results
    # Loop over each item in the main directory
    for folder_name in os.listdir(main_directory):
        folder_path = os.path.join(main_directory, folder_name)  # Get the full path of the item
        if os.path.isdir(folder_path):  # Check if the item is a directory
            eigenvalues_path = os.path.join(folder_path, 'Eigenvalues.txt')  # Path to the Eigenvalues.txt file
            if os.path.isfile(eigenvalues_path):  # Check if the Eigenvalues.txt file exists
                # Read the file and convert its contents to numbers
                eigenvalues = file_to_array_as_numbers(eigenvalues_path)
                rank = calc_rank(eigenvalues)  # Calculate the rank based on the eigenvalues
                results.append([folder_name, rank])  # Append the result to the results list

    # Write the results to a CSV file in the main directory
    csv_path = os.path.join(main_directory, 'ranks_output.csv')
    write_to_csv(csv_path, results)

# This condition checks if the script is being run as the main program
if __name__ == "__main__":
    # Define the main directory path
    main_directory = '/Users/chandlerstevenson/Downloads/4-rank-encoding/Image0/rank-1'  # Path to be updated as needed
    process_folders(main_directory)  # Call the process_folders function to start the processing
