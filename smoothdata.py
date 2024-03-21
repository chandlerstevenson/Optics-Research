import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

# Read data from a csv
data = pd.read_csv('Rutendo5Bessel.csv')  # replace with your csv file

# Assume that the first column (0 index) is 'x' (time) and the second column (1 index) is 'y' (values)

x_data = data.iloc[:, 0]
y_data = data.iloc[:, 1]

# Set the cut-off frequency of the filter
cut_off_frequency = 0.01  # adjust this value to your needs

# Design the Butterworth filter
N  = 2    # Filter order
Wn = cut_off_frequency  # Cutoff frequency
B, A = butter(N, Wn, output='ba')

# Apply the filter to y_data
y_data_filtered = filtfilt(B, A, y_data)

# Create traces for the original and filtered data
line = go.Scatter(x=x_data, y=y_data, name='Original')
line_filtered = go.Scatter(x=x_data, y=y_data_filtered, name='Filtered')

# Create a layout for the plot with title and axis labels
layout = go.Layout(title='PPG Signal',
                   xaxis_title='Time (seconds)',
                   yaxis_title='Intensity (AU)',
                   xaxis=dict(range=[-6, 4]) # Set range of x-axis
                   )

# Create a figure from the data and layout, and plot the figure
fig = go.Figure(data=line_filtered, layout=layout)
fig.show()
