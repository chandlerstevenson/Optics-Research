#!/usr/bin/env python3.11 

import pandas as pd
import matplotlib.pyplot as plt

def fold_nums(num): 
    return ((num + 45) % (90)) - 45

def return_num(num):
    if num in [-45, 45]:
        modified_num = num  
    elif num in [135, 180]:
        modified_num = 180 - num
        if modified_num == 135:  
            modified_num = 45
    else:
        modified_num = fold_nums(num) 
    return modified_num  

def plot_histogram(csv_path):
    """Reads a CSV, applies 'return_num' to the first column, and plots a histogram."""
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Assuming the first column contains the numerical values to be modified
        modified_values = df.iloc[:, 0].apply(return_num)
        
        # Plotting the histogram of the modified values
        plt.figure(figsize=(10, 6))
        plt.hist(modified_values, bins=8, color='blue', edgecolor='black')
        plt.title('Histogram of Modified First Column Values')
        plt.xlabel('Channel Orientations')
        plt.ylabel('Frequency')
        
        # Display the histogram
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")


plot_histogram('/Users/chandlerstevenson/Downloads/orientation_values.csv')

