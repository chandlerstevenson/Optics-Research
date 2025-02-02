# README - Computational Analysis Tools

## Author: Chandler W. Stevenson
### Affiliation: Brown University, PROBE Lab

---

## Overview
This repository contains a collection of Python scripts developed for computational analysis within the PROBE Lab at Brown University. These scripts provide functionalities such as:
- Confusion matrix visualization
- Peak intensity detection in time-series data
- Fourier transform-based partitioning
- Signal processing and smoothing
- Real-time plotting and GUI applications
- Optimal window size detection for time-series analysis
- Rank calculations and vector analysis
- Pixel analysis and histogram plotting

Each script is designed to perform specific tasks essential for analyzing physiological signals, image processing, and time-series data. The following sections describe each script's functionality, dependencies, and usage instructions.

---

## Table of Contents
1. [Confusion Matrix Visualization](#confusion-matrix-visualization)
2. [Peak Intensity and PPG Analysis](#peak-intensity-and-ppg-analysis)
3. [Signal Processing and Data Smoothing](#signal-processing-and-data-smoothing)
4. [Fourier Transform and Partitioning](#fourier-transform-and-partitioning)
5. [Live Plotting and GUI](#live-plotting-and-gui)
6. [Optimal Window Size Calculation](#optimal-window-size-calculation)
7. [Rank and Distance Calculation](#rank-and-distance-calculation)
8. [Pixel Analysis and Histograms](#pixel-analysis-and-histograms)
9. [Installation and Dependencies](#installation-and-dependencies)
10. [Usage](#usage)

---

## Rank and Distance Calculation
**Scripts:**
- `calcrankset.py`
- `lindistancecalcrankset.py`
- `rank_simulation.py`

**Description:**
These scripts compute and simulate rank distributions based on calculated distances from ideal vectors. They apply various distance metrics to classify vectors into ranks and simulate their distributions.

**Usage:**
```sh
python calcrankset.py
python lindistancecalcrankset.py
python rank_simulation.py
```

**Features:**
- Computes rank classifications using distance metrics
- Simulates rank distributions across large datasets
- Supports different ranking methodologies

---

## Pixel Analysis and Histograms
**Scripts:**
- `pixelcountandlocation.py`
- `histogram_HWP_orientation.py`
- `orientations045histogram.py`
- `plot100krandpoints.py`
- `plotnoisefigure.py`
- `prettyshapegraphs.py`

**Description:**
These scripts analyze pixel distributions in images, compute histograms, and visualize various data distributions. `pixelcountandlocation.py` identifies pixel locations and intensities, while the histogram scripts create detailed visualizations.

**Usage:**
```sh
python pixelcountandlocation.py
python histogram_HWP_orientation.py
python orientations045histogram.py
python plot100krandpoints.py
python plotnoisefigure.py
python prettyshapegraphs.py
```

**Features:**
- Identifies and counts pixel intensities in images
- Generates histograms of orientation and noise distributions
- Produces scatter plots for large datasets

---

## Installation and Dependencies
Ensure you have the following Python packages installed:
```sh
pip install numpy scipy matplotlib pandas seaborn pillow
```

For GUI applications, install:
```sh
pip install mpl_toolkits
```

---

## Usage
Each script is standalone and can be executed via the command line:
```sh
python script_name.py
```
Replace `script_name.py` with the desired script (e.g., `FindPeaksPPG.py`).

---

## Contributions
This repository is maintained by Chandler W. Stevenson for research conducted at Brown University’s PROBE Lab. Contributions and modifications should align with the project's goals and be documented accordingly.

---

## License
This code is provided for research purposes within the PROBE Lab and should not be used for commercial applications without permission.

---

