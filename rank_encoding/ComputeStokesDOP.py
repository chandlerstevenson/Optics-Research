"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script processes images from subdirectories within a given main directory.
It calculates the brightness of each image relative to a baseline image,
and then computes the Stokes parameters to determine the degree of polarization.
The results are written to a CSV file for further analysis.
"""

import os
import csv
from PIL import Image, ImageStat 
import math 

def list_directory_contents(directory):
    """
    Retrieves a list of subdirectories within the specified directory.
    Returns a list of tuples containing (directory_name, full_directory_path).
    """
    return [(d, os.path.join(directory, d)) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def calculate_brightness(image_path):
    """
    Calculates the brightness of an image by converting it to grayscale
    and computing the mean pixel intensity.
    """
    with Image.open(image_path) as img:
        grayscale_image = img.convert('L')  # Convert image to grayscale
        stat = ImageStat.Stat(grayscale_image)  # Get image statistics
        return stat.mean[0]  # Return mean pixel intensity

def process_images(directory_name, directory_path, baseline_filepath, image_files):
    """
    Processes all images in a given directory by calculating their brightness values.
    Computes the relative brightness by subtracting the baseline brightness value.
    Groups images into two separate rows based on alternating index positions.
    
    Returns:
    row1 - List containing the first set of relative brightness values.
    row2 - List containing the second set of relative brightness values.
    """
    relative_brightness_values = []
    baseline_brightness = calculate_brightness(baseline_filepath)  # Get baseline brightness

    # Compute relative brightness for each image
    for filename in image_files:
        filepath = os.path.join(directory_path, filename)
        pic_brightness = calculate_brightness(filepath)
        relative_brightness = pic_brightness - baseline_brightness
        relative_brightness_values.append(relative_brightness)

    # Distribute brightness values into two separate rows based on order
    row1 = [directory_name] + relative_brightness_values[::2]  # First set of images (a_1, b_1, c_1, d_1)
    row2 = [directory_name] + relative_brightness_values[1::2]  # Second set of images (a_2, b_2, c_2, d_2)
    return row1, row2 

def compute_stokes(intensity_vector):  
    """
    Computes the Stokes parameters (s_0, s_1, s_2, s_3) based on the given intensity vector.
    Calculates the degree of polarization using these parameters.
    
    Inputs:
    intensity_vector - A list of four intensity values [CD, ID, IH, IV]
    
    Returns:
    degree_of_pol - Computed degree of polarization.
    """
    circular = intensity_vector[0]  # Circularly polarized intensity
    diagonal = intensity_vector[1]  # Diagonally polarized intensity
    horizontal = intensity_vector[2]  # Horizontally polarized intensity
    vertical = intensity_vector[3]  # Vertically polarized intensity

    # Compute Stokes parameters
    s_0 = horizontal + vertical  # Total intensity
    s_1 = horizontal - vertical  # Difference between horizontal and vertical intensities
    s_2 = (2 * diagonal) - s_0  # Difference between diagonal and total intensity
    s_3 = s_0 - (2 * circular)  # Difference between total and circular intensities

    # Compute degree of polarization
    degree_of_pol = math.sqrt((s_1 ** 2) + (s_2 ** 2) + (s_3 ** 2)) / s_0 
    return degree_of_pol

def write_to_csv(file_path, data, header=None):
    """
    Writes image brightness data to a CSV file, including the degree of polarization.
    
    Inputs:
    file_path - Path to the CSV file.
    data - List of rows containing image brightness values.
    header - (Optional) Header row to include in the CSV file.
    """
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If the file is empty, write the header first
        if header and os.path.getsize(file_path) == 0:
            writer.writerow(header)
        
        # Compute degree of polarization for each row and append to CSV
        for row in data:
            intensity_vector = row[1:5]  # Extract intensity values
            degree_of_pol = compute_stokes(intensity_vector)  # Compute polarization
            row_with_pol = row + [degree_of_pol]  # Append to row
            writer.writerow(row_with_pol)

# CSV header specifying columns
header = ['Folder', 'CD', 'ID', 'IH', 'IV', 'Degree of Polarization'] 

# Define the main directory containing image folders
main_directory = '/Users/chandlerstevenson/Downloads/HH'  # Change as necessary
subdirectories = list_directory_contents(main_directory)  # Get subdirectories
baseline_filepath = os.path.join(main_directory, 'baseline.bmp')  # Path to baseline image

# Define expected image filenames for processing
image_files = [
    'I_CD_1.bmp', 'I_CD_2.bmp', 'I_D_1.bmp', 'I_D_2.bmp',
    'I_H_1.bmp', 'I_H_2.bmp', 'I_V_1.bmp', 'I_V_2.bmp'
]

# Define output CSV file path
csv_file_path = os.path.join(main_directory, 'stokes_calcs.csv')

# Ensure the CSV file exists and write header if needed
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write header

# Process each subdirectory and compute values
for directory_name, directory_path in subdirectories:
    row1, row2 = process_images(directory_name, directory_path, baseline_filepath, image_files)
    write_to_csv(csv_file_path, [row1, row2], header=header)  # Write results to CSV

print('Stokes + DOP values have been written to', csv_file_path)
