"""
Author: Chandler W. Stevenson
Affiliation: Brown University, PROBE Lab

Description:
This script generates a confusion matrix visualization using seaborn and matplotlib.
It takes in true positive (TP), false positive (FP), false negative (FN), and true negative (TN) values,
and creates a visually appealing heatmap representation of the confusion matrix.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_confusion_matrix(TP, FP, FN, TN):
    """
    Generates and visualizes a confusion matrix heatmap.
    
    Parameters:
    TP (float): True Positives
    FP (float): False Positives
    FN (float): False Negatives
    TN (float): True Negatives
    """
    # Create a 2x2 confusion matrix as a NumPy array
    cm = np.array([[TP, FP], [FN, TN]])

    # Convert matrix into a pandas DataFrame for seaborn visualization
    df_cm = pd.DataFrame(cm, index=["Actual\nPositive", "Actual\nNegative"],
                         columns=["Predicted\nPositive", "Predicted\nNegative"])

    # Define colors for different categories in the confusion matrix
    tp_color = "#00008b" if TP > 0.5 else "#add8e6"  # Dark blue for strong TP, light blue otherwise
    tn_color = "#00008b" if TN > 0.5 else "#add8e6"  # Dark blue for strong TN, light blue otherwise
    fp_color = "#8b0000" if FP > 0.5 else "#ff7f7f"  # Dark red for strong FP, light red otherwise
    fn_color = "#8b0000" if FN > 0.5 else "#ff7f7f"  # Dark red for strong FN, light red otherwise
    colors = [tp_color, fp_color, fn_color, tn_color]

    # Create plot and set size
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set(font_scale=1.4)  # Adjust font size for readability
    sns.set_style("whitegrid")  # Set background style

    # Generate heatmap with seaborn
    heatmap = sns.heatmap(df_cm, cmap=colors, cbar=False, linewidths=.5, 
                          linecolor='gray', square=True)

    # Add percentage annotations inside each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            heatmap.text(j + 0.5, i + 0.5, '{:.2%}'.format(cm[i, j]),
                         color='black', ha='center', va='center', 
                         size=20, weight='bold',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                   edgecolor='black', linewidth=1))

    # Set cell background colors
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, 
                                            color=colors[i * 2 + j], edgecolor='gray', lw=1))

    # Adjust tick labels
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=15)
    ax.tick_params(axis='x', which='both', pad=10)

    # Set title and subtitle
    plt.title('Peak Detection Confusion Matrix', size=18, pad=30)
    plt.suptitle('Accuracy: 98.9%', fontsize=18, y=0.855, x=0.51)

    plt.tight_layout()
    plt.show()

# Example execution with sample confusion matrix values
create_confusion_matrix(0.0369, 0.0022, 0.0098, 0.9521)
