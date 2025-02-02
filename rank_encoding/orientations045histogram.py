"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script processes numerical values from a CSV file, applies a transformation function,
and plots a histogram of the modified values. It is specifically designed to analyze
channel orientation data and map values within a defined range.
"""

#!/usr/bin/env python3.11 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fold_nums(num): 
    """
    Folds numerical values into a range of -45 to 45 degrees.
    
    Parameters:
    num (float): Input numerical value.
    
    Returns:
    float: Folded numerical value within the range.
    """
    return ((num + 45) % (90)) - 45

def return_num(num):
    """
    Adjusts orientation values according to predefined folding rules.
    
    Parameters:
    num (float): Input numerical value.
    
    Returns:
    float: Absolute modified numerical value.
    """
    if num in [-45, 45]:
        modified_num = num  
    elif num in [135, 180]:
        modified_num = 180 - num
        if modified_num == 135:  
            modified_num = 45
    else:
        modified_num = fold_nums(num) 
    return np.abs(modified_num)  

def plot_histogram(csv_path):
    """
    Reads a CSV file, applies 'return_num' transformation, and plots a histogram.
    
    Parameters:
    csv_path (str): Path to the CSV file.
    """
    try:
        df = pd.read_csv(csv_path)  # Load CSV file
        modified_values = df.iloc[:, 0].apply(return_num)  # Apply transformation to first column
        
        plt.figure(figsize=(10, 6))
        plt.hist(modified_values, bins=8, color='blue', edgecolor='black')
        plt.title('Histogram of Modified First Column Values')
        plt.xlabel('Channel Orientations')
        plt.ylabel('Frequency')
        
        plt.show()  # Display histogram
    except Exception as e:
        print(f"An error occurred: {e}")

# Example execution
plot_histogram('/Users/chandlerstevenson/Downloads/orientation_values.csv')
