import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_confusion_matrix(TP, FP, FN, TN):
    # Create a 2x2 confusion matrix
    cm = np.array([[TP, FP], [FN, TN]])

    # Create a pandas dataframe
    df_cm = pd.DataFrame(cm, index = ["Actual\nPositive", "Actual\nNegative"],
                      columns = ["Predicted\nPositive","Predicted\nNegative"])

    # Colors for true and false values
    tp_color = "#00008b" if TP > 0.5 else "#add8e6"
    tn_color = "#00008b" if TN > 0.5 else "#add8e6"
    fp_color = "#8b0000" if FP > 0.5 else "#ff7f7f"
    fn_color = "#8b0000" if FN > 0.5 else "#ff7f7f"
    colors = [tp_color, fp_color, fn_color, tn_color]

    # Plot style and size
    fig, ax = plt.subplots(figsize=(10,7))
    sns.set(font_scale=1.4) # for label size
    sns.set_style("whitegrid") # set style

    # Create a seaborn heatmap without annotations
    heatmap = sns.heatmap(df_cm, cmap=colors, cbar=False, linewidths=.5, linecolor='gray', square=True)

    # Add annotations manually with white box around
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            heatmap.text(j + 0.5, i + 0.5, '{:.2%}'.format(cm[i, j]), 
                         color = 'black', ha = 'center', va = 'center', 
                         size = 20, weight = 'bold',
                         bbox = dict(boxstyle = 'round,pad=0.5', facecolor = 'white', edgecolor = 'black', linewidth = 1))

    # Set cell colors based on predefined color scheme
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors[i*2+j], edgecolor='gray', lw=1))

    # Set labels and title
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=15)
    ax.tick_params(axis='x', which='both', pad=10)

    plt.title('Peak Detection Confusion Matrix', size=18, pad=30)

    # Add a subtitle
    plt.suptitle('Accuracy: 98.9%', fontsize=18, y=0.855,x=.51)

    plt.tight_layout()
    plt.show()

# Replace these numbers with your actual numbers
create_confusion_matrix(0.0369, 0.0022, 0.0098, .9521)
