import os
import csv
from PIL import Image, ImageStat 
import math 

def list_directory_contents(directory):
    """Returns a list of full paths of all subdirectories in the given directory."""
    return [(d, os.path.join(directory, d)) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def calculate_brightness(image_path):
    """Calculates the brightness of an image."""
    with Image.open(image_path) as img:
        grayscale_image = img.convert('L')
        stat = ImageStat.Stat(grayscale_image)
        return stat.mean[0]

def process_images(directory_name, directory_path, baseline_filepath, image_files):
    """Processes all images in a directory and compares brightness to the baseline."""
    relative_brightness_values = []
    baseline_brightness = calculate_brightness(baseline_filepath)

    for filename in image_files:
        filepath = os.path.join(directory_path, filename)
        pic_brightness = calculate_brightness(filepath)
        relative_brightness = pic_brightness - baseline_brightness
        relative_brightness_values.append(relative_brightness)

    # Since the values are in order a_1, a_2, b_1, b_2, c_1, c_2, d_1, d_2
    # We want to take the 1st, 3rd, 5th, etc. values for the first row
    # and the 2nd, 4th, 6th, etc. values for the second row.
    row1 = [directory_name] + relative_brightness_values[::2]  # a_1, b_1, c_1, d_1
    row2 = [directory_name] + relative_brightness_values[1::2]  # a_2, b_2, c_2, d_2
    return row1, row2 

def compute_stokes(intensity_vector):  
    circular = intensity_vector[0]  
    diagonal = intensity_vector[1]
    horizontal = intensity_vector[2]
    vertical = intensity_vector[3]  

    s_0 = horizontal + vertical  
    s_1 = horizontal - vertical  
    s_2 = (2*diagonal) - s_0 
    s_3 = s_0 - (2*circular) 

    degree_of_pol = math.sqrt((s_1**2)+(s_2**2)+(s_3**2))/s_0 
    return degree_of_pol


def write_to_csv(file_path, data, header=None):
    """Writes the brightness values to a CSV file, including the degree of polarization."""
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Only write header if file is empty
        if header and os.path.getsize(file_path) == 0:  
            writer.writerow(header)
        # Compute degree of polarization and append to rows before writing
        for row in data:
            intensity_vector = row[1:5]  # Assuming the order is [CD, ID, IH, IV]
            degree_of_pol = compute_stokes(intensity_vector)
            row_with_pol = row + [degree_of_pol]  # Append degree of polarization to the row
            writer.writerow(row_with_pol)

header = ['Folder', 'CD', 'ID', 'IH', 'IV', 'Degree of Polarization'] 


# Main directory and subdirectories
main_directory = '/Users/chandlerstevenson/Downloads/HH'#change as necessary 
subdirectories = list_directory_contents(main_directory) #necessary for folder type
baseline_filepath = os.path.join(main_directory, 'baseline.bmp')
image_files = [
    'I_CD_1.bmp', 'I_CD_2.bmp', 'I_D_1.bmp', 'I_D_2.bmp',
    'I_H_1.bmp', 'I_H_2.bmp', 'I_V_1.bmp', 'I_V_2.bmp'
]

# Output CSV file
csv_file_path = os.path.join(main_directory, 'stokes_calcs.csv')

# Ensure the header is written if the CSV does not exist
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Use the updated header

# Process each subdirectory
for directory_name, directory_path in subdirectories:
    row1, row2 = process_images(directory_name, directory_path, baseline_filepath, image_files)
    # Calculate degree of polarization and write rows to CSV
    write_to_csv(csv_file_path, [row1, row2], header=header)

print('Stokes + DOP values have been written to', csv_file_path)
