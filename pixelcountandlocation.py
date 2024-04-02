#!/usr/bin/env python3.11
import os
import csv
from PIL import Image

# Function to find the (x, y) locations of pixels with a specific value in an image.
def find_pixel_locations(image_path, value=255):
    """Finds and returns the locations of pixels in an image with the given value (default 255)."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        width, height = img.size
        return [(x, y) for y in range(height) for x in range(width) if img.getpixel((x, y)) == (value, value, value)]

# Function to process each image file and collect data.
def process_image_files(image_dir, image_files):
    """Processes a list of image files in a directory, calculates the count of pixels with value 255, 
    and finds their locations, then returns the data as a list."""
    data = []
    for image_file in image_files:
        # Create the full file path for the image
        image_path = os.path.join(image_dir, image_file)
        # Find the locations of pixels with the maximum value (255)
        pixel_locations = find_pixel_locations(image_path)
        # Format the pixel locations as a string "(x1,y1);(x2,y2);..."
        locations_str = ';'.join(f'({x},{y})' for x, y in pixel_locations)
        # Add the count and locations to the data list
        data.append((len(pixel_locations), locations_str))
    return data

# Function to write the data to a CSV file, now with pixel locations.
def write_to_csv(file_path, data, header=None):
    """Writes the provided data to a CSV file."""
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if header:  # Write header if provided
            writer.writerow(header)
        writer.writerows(data)  # Write data rows

# Main script execution.
if __name__ == "__main__":
    main_directory = '/Users/chandlerstevenson/Downloads/HH'  # Change as necessary
    subdirectories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    image_files = [
        'I_CD_1.bmp', 'I_CD_2.bmp', 'I_D_1.bmp', 'I_D_2.bmp',
        'I_H_1.bmp', 'I_H_2.bmp', 'I_V_1.bmp', 'I_V_2.bmp'
    ]
    
    # Define the CSV file path and write the header.
    csv_file_path = os.path.join(main_directory, 'pixel_counts_and_locations.csv')
    header = ['Folder']
    for file in image_files:
        header.extend([f'{file} Pixel 255 Count', f'{file} Pixel 255 Locations'])
    
    if not os.path.exists(csv_file_path):
        write_to_csv(csv_file_path, [], header=header)
    
    # Process each subdirectory and write the results to the CSV.
    for subdir in subdirectories:
        subdir_path = os.path.join(main_directory, subdir)
        image_data = process_image_files(subdir_path, image_files)
        # Flatten the list of tuples to a single list
        pixel_data = [item for sublist in image_data for item in sublist]
        row = [subdir] + pixel_data
        write_to_csv(csv_file_path, [row], header=None)
    
    print('Pixel count and locations for each image have been written to', csv_file_path)
