"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script processes image files within subdirectories, identifying and counting pixels
with a specific value (default 255). It records the pixel locations and writes the data
to a CSV file for further analysis.
"""

#!/usr/bin/env python3.11

import os
import csv
from PIL import Image

def find_pixel_locations(image_path, value=255):
    """
    Finds and returns the (x, y) coordinates of pixels in an image with the given value.
    
    Parameters:
    image_path (str): Path to the image file.
    value (int): Pixel intensity value to locate (default is 255 for white pixels).
    
    Returns:
    list: A list of (x, y) coordinates of pixels matching the given value.
    """
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        width, height = img.size
        return [(x, y) for y in range(height) for x in range(width) if img.getpixel((x, y)) == (value, value, value)]

def process_image_files(image_dir, image_files):
    """
    Processes a list of images in a directory, counting the occurrences of a specific pixel value
    and storing their locations.
    
    Parameters:
    image_dir (str): Directory containing image files.
    image_files (list): List of image file names.
    
    Returns:
    list: A list of tuples with pixel count and pixel locations for each image.
    """
    data = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        pixel_locations = find_pixel_locations(image_path)
        locations_str = ';'.join(f'({x},{y})' for x, y in pixel_locations)
        data.append((len(pixel_locations), locations_str))
    return data

def write_to_csv(file_path, data, header=None):
    """
    Writes processed data to a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    data (list): Data to be written.
    header (list, optional): Header row for the CSV file.
    """
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if header:
            writer.writerow(header)
        writer.writerows(data)

if __name__ == "__main__":
    main_directory = '/Users/chandlerstevenson/Downloads/HH'  # Modify as needed
    subdirectories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    image_files = [
        'I_CD_1.bmp', 'I_CD_2.bmp', 'I_D_1.bmp', 'I_D_2.bmp',
        'I_H_1.bmp', 'I_H_2.bmp', 'I_V_1.bmp', 'I_V_2.bmp'
    ]
    
    csv_file_path = os.path.join(main_directory, 'pixel_counts_and_locations.csv')
    header = ['Folder']
    for file in image_files:
        header.extend([f'{file} Pixel 255 Count', f'{file} Pixel 255 Locations'])
    
    if not os.path.exists(csv_file_path):
        write_to_csv(csv_file_path, [], header=header)
    
    for subdir in subdirectories:
        subdir_path = os.path.join(main_directory, subdir)
        image_data = process_image_files(subdir_path, image_files)
        pixel_data = [item for sublist in image_data for item in sublist]
        row = [subdir] + pixel_data
        write_to_csv(csv_file_path, [row], header=None)
    
    print('Pixel count and locations for each image have been written to', csv_file_path)
