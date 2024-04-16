#!/usr/bin/env python3.11 
import numpy as np 
import math 
import matplotlib  
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Criteria [x, y, z, alpha]   
# x >= y >= z >= alpha 
# Limits on x: [0.25, 1]
# Limits on y: [0, .5] 
# Limits on z: [0, 1/3]
# Limits on alpha: [0, .25] 


def generate_vector():
    # Start by generating initial random values within their full allowed ranges
    x = random.uniform(0.25, 1)
    y = random.uniform(0, 0.5)
    z = random.uniform(0, 1/3)
    alpha = random.uniform(0, 0.25)
    
    # Calculate their sum
    total = x + y + z + alpha
    
    # Adjust values to sum to 1
    x, y, z, alpha = x / total, y / total, z / total, alpha / total
    
    # Ensure x >= y >= z >= alpha and all constraints are still met
    if not (0.25 <= x <= 1 and 0 <= y <= 0.5 and 0 <= z <= 1/3 and 0 <= alpha <= 0.25):
        return generate_vector()  # Regenerate if conditions fail
    
    return [x, y, z, alpha]

# Ensure that rank vector is in descending order 
def round_and_sort_elements(original_vector):   
    original_vector = np.array(original_vector) 
    round_1 = np.round(original_vector[0], 2)
    round_2 = np.round(original_vector[1], 2)
    round_3 = np.round(original_vector[2], 2)
    round_4 = np.round(original_vector[3], 2) 
    rounded_vector = np.array([round_1, round_2, round_3, round_4]) 
    sorted_and_rounded_vector = np.sort(rounded_vector)
    return sorted_and_rounded_vector[::-1]

vector = generate_vector() 

# Ideal rank vector 
ideal_vector = np.array([[1, 0, 0, 0], [0.5, 0.5, 0, 0], [1/3, 1/3, 1/3, 0], [0.25, 0.25, 0.25, 0.25]]) 

# Calculates distance of  input vector from ideal vector 
def calculate_distance(input_vector, ideal_vector):
    distance_vector = np.array([])
    for vec in ideal_vector:
        if len(vec) != len(input_vector):
            raise ValueError("Vectors must have the same dimensionality")
        distance = np.sqrt(sum((vec[i] - input_vector[i])**2 for i in range(len(input_vector))))
        distance_vector = np.append(distance_vector, distance)
    return distance_vector

# rank is the minimum distance to an ideal rank vector
def calculate_rank(distanced_vector):  
    rank_set = [1, 2, 3, 4]
    calculated_rank = rank_set[np.argmin(distanced_vector)] 
    return calculated_rank

# runs the simulation run_times times 
def num_simulation_runs(run_times):
    plot_vectors = []
    # Check if ideal_vector is defined
    try:
        ideal_vector 
    except NameError:
        print("Error: 'ideal_vector' is not defined.")
        return []
    for i in range(run_times):
        try:
            generated_vector = generate_vector()
            # print(f"Generated Vector: {generated_vector}")
            rounded_sorted_vector = round_and_sort_elements(generated_vector)
            # print(f"Rounded and Sorted Vector: {rounded_sorted_vector}")
            dist_vector = calculate_distance(rounded_sorted_vector, ideal_vector)
            # print(f"Distance Vector: {dist_vector}")
            rank = calculate_rank(dist_vector)
            # print(f"Rank: {rank}")
            plot_vectors.append((np.array(rounded_sorted_vector[0:3]), rank))
        except Exception as e:
            print(f"An error occurred during run {i}: {e}")
            continue

    return plot_vectors

# Testing Area (as necessary)
# -------------------------------------------------------------------------------------------
# print(vector)                                                                             |
# print(round_and_sort_elements(vector))                                                    |
# print(calculate_distance(round_and_sort_elements(vector), ideal_vector))                  |
# print(calculate_rank(calculate_distance(round_and_sort_elements(vector), ideal_vector)))  |
# print(run_sim)                                                                            |
# -------------------------------------------------------------------------------------------



# plot 3d data
def plot_efficient_3d_data(data, point_size=10):
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}
    
    # Separate data into X, Y, Z and colors
    X = np.array([coords[0] for coords, _ in data])
    Y = np.array([coords[1] for coords, _ in data])
    Z = np.array([coords[2] for coords, _ in data])
    colors = np.array([color_map[marker] for _, marker in data])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c=colors, marker='o', edgecolors='black', linewidths=.3, s=point_size)  


    # Set labels for axes
    ax.set_xlabel('λ1')
    ax.set_ylabel('λ2')
    ax.set_zlabel('λ3')

    # Set the ranges for each axis
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.5])
    ax.set_zlim([0, 0.5])

    # Set the ticks for each axis
    ax.set_xticks(np.linspace(0, 1, num=6))
    ax.set_yticks(np.linspace(0, 0.5, num=6))
    ax.set_zticks(np.linspace(0, 0.5, num=6))

    # Show plot
    plt.show()

# plot 3d data but without graphing background
def plot_blank(data, point_size=10):
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}
    
    # Separate data into X, Y, Z and colors
    X = np.array([coords[0] for coords, _ in data])
    Y = np.array([coords[1] for coords, _ in data])
    Z = np.array([coords[2] for coords, _ in data])
    colors = np.array([color_map[marker] for _, marker in data])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c=colors, marker='o', edgecolors='black', linewidths=.3, s=point_size)  

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Remove grid and axis lines
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Turn off the axes
    ax.set_axis_off()

    # Show plot
    plt.show()

# isolate rank 1
def plot_rank_1(data, point_size=10):
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}

    # Filter the data to include only rank 2 points
    filtered_data = [(coords, marker) for coords, marker in data if marker == 1]
    
    # Separate data into X, Y, Z and colors
    X = np.array([coords[0] for coords, _ in filtered_data])
    Y = np.array([coords[1] for coords, _ in filtered_data])
    Z = np.array([coords[2] for coords, _ in filtered_data])
    colors = np.array([color_map[marker] for _, marker in filtered_data])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for filtered data
    ax.scatter(X, Y, Z, c=colors, marker='o', edgecolors='black', linewidths=0.3, s=point_size)  

    # Set labels for axes
    ax.set_xlabel('λ1')
    ax.set_ylabel('λ2')
    ax.set_zlabel('λ3')

    # Set the ranges for each axis
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.5])
    ax.set_zlim([0, 0.5])

    # Set the ticks for each axis
    ax.set_xticks(np.linspace(0, 1, num=6))
    ax.set_yticks(np.linspace(0, 0.5, num=6))
    ax.set_zticks(np.linspace(0, 0.5, num=6))

    # Show plot
    plt.show()

# isolate rank 2
def plot_rank_2(data, point_size=10):
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}

    # Filter the data to include only rank 2 points
    filtered_data = [(coords, marker) for coords, marker in data if marker == 2]
    
    # Separate data into X, Y, Z and colors
    X = np.array([coords[0] for coords, _ in filtered_data])
    Y = np.array([coords[1] for coords, _ in filtered_data])
    Z = np.array([coords[2] for coords, _ in filtered_data])
    colors = np.array([color_map[marker] for _, marker in filtered_data])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for filtered data
    ax.scatter(X, Y, Z, c=colors, marker='o', edgecolors='black', linewidths=0.3, s=point_size)  

    # Set labels for axes
    ax.set_xlabel('λ1')
    ax.set_ylabel('λ2')
    ax.set_zlabel('λ3')

    # Set the ranges for each axis
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.5])
    ax.set_zlim([0, 0.5])

    # Set the ticks for each axis
    ax.set_xticks(np.linspace(0, 1, num=6))
    ax.set_yticks(np.linspace(0, 0.5, num=6))
    ax.set_zticks(np.linspace(0, 0.5, num=6))

    # Show plot
    plt.show()
# isolate rank 3
def plot_rank_3(data, point_size=10):
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}

    # Filter the data to include only rank 2 points
    filtered_data = [(coords, marker) for coords, marker in data if marker == 3]
    
    # Separate data into X, Y, Z and colors
    X = np.array([coords[0] for coords, _ in filtered_data])
    Y = np.array([coords[1] for coords, _ in filtered_data])
    Z = np.array([coords[2] for coords, _ in filtered_data])
    colors = np.array([color_map[marker] for _, marker in filtered_data])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for filtered data
    ax.scatter(X, Y, Z, c=colors, marker='o', edgecolors='black', linewidths=0.3, s=point_size)  

    # Set labels for axes
    ax.set_xlabel('λ1')
    ax.set_ylabel('λ2')
    ax.set_zlabel('λ3')

    # Set the ranges for each axis
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.5])
    ax.set_zlim([0, 0.5])

    # Set the ticks for each axis
    ax.set_xticks(np.linspace(0, 1, num=6))
    ax.set_yticks(np.linspace(0, 0.5, num=6))
    ax.set_zticks(np.linspace(0, 0.5, num=6))

    # Show plot
    plt.show()

# isolate rank 4
def plot_rank_4(data, point_size=10):
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}

    # Filter the data to include only rank 2 points
    filtered_data = [(coords, marker) for coords, marker in data if marker == 4]
    
    # Separate data into X, Y, Z and colors
    X = np.array([coords[0] for coords, _ in filtered_data])
    Y = np.array([coords[1] for coords, _ in filtered_data])
    Z = np.array([coords[2] for coords, _ in filtered_data])
    colors = np.array([color_map[marker] for _, marker in filtered_data])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for filtered data
    ax.scatter(X, Y, Z, c=colors, marker='o', edgecolors='black', linewidths=0.3, s=point_size)  

    # Set labels for axes
    ax.set_xlabel('λ1')
    ax.set_ylabel('λ2')
    ax.set_zlabel('λ3')

    # Set the ranges for each axis
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.5])
    ax.set_zlim([0, 0.5])

    # Set the ticks for each axis
    ax.set_xticks(np.linspace(0, 1, num=6))
    ax.set_yticks(np.linspace(0, 0.5, num=6))
    ax.set_zticks(np.linspace(0, 0.5, num=6))

    # Show plot
    plt.show()


# plot small dots graph
def plot_theoretical_small(data, point_size=1):
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}
    
    # Separate data into X, Y, Z and colors
    X = np.array([coords[0] for coords, _ in data])
    Y = np.array([coords[1] for coords, _ in data])
    Z = np.array([coords[2] for coords, _ in data])
    colors = np.array([color_map[marker] for _, marker in data])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c=colors, marker='o', edgecolors='none', linewidths=.1, s=point_size)  


    # Set labels for axes
    ax.set_xlabel('λ1')
    ax.set_ylabel('λ2')
    ax.set_zlabel('λ3')

    # Set the ranges for each axis
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.5])
    ax.set_zlim([0, 0.5])

    # Set the ticks for each axis
    ax.set_xticks(np.linspace(0, 1, num=6))
    ax.set_yticks(np.linspace(0, 0.5, num=6))
    ax.set_zticks(np.linspace(0, 0.5, num=6))

    # Show plot
    plt.show()

# modify vectors to make exploded graph
def modify_vectors(data):
    # Define modifications for each value of n
    modifications = {
        1: [0, 0, 0],
        2: [0, 0.1, 0.1],
        3: [0, 0, 0.1],
        4: [-0.1, 0, 0]
    }
    
    # Create an empty list to store the modified vectors
    modified_data = []
    
    # Process each vector and its associated number
    for vector, n in data:
        # Extract coordinates of the vector
        x, y, z = vector
        
        # Retrieve the modification vector based on the value of n
        dx, dy, dz = modifications.get(n, [0, 0, 0])
        
        # Compute the new coordinates by adding the modifications
        new_vector = (x + dx, y + dy, z + dz)
        
        # Append the modified vector and original number to the result list
        modified_data.append((new_vector, n))
    
    return modified_data

# Set number of plot points 
run_sim = num_simulation_runs(100000)
modified_sim = modify_vectors(run_sim) 

def percent_rank(data):
    from collections import Counter
    
    # Extract numbers from the second position in each tuple
    numbers = [num for _, num in data]
    
    # Count occurrences of each number
    count = Counter(numbers)
    
    # Calculate total number of elements
    total = sum(count.values())
    
    # Calculate and print percentage of each number from 1 to 4
    for num in range(1, 5):
        if num in count:
            percentage = (count[num] / total) * 100
        else:
            percentage = 0
        print(f"Percentage of Rank {num}: {percentage:.2f}%")


# Call the function to plot the data  
# print(run_sim)
percent_rank(run_sim)
plot_rank_1(run_sim)
plot_rank_2(run_sim)
plot_rank_3(run_sim)
plot_rank_4(run_sim) 
plot_theoretical_small(run_sim)
plot_blank(modified_sim)
plot_efficient_3d_data(run_sim)
