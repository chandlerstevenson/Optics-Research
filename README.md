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

Each script is designed to perform specific tasks essential for analyzing physiological signals, image processing, and time-series data. The following sections describe each script's functionality, dependencies, and usage instructions.

---

## Table of Contents
1. [Confusion Matrix Visualization](#confusion-matrix-visualization)
2. [Peak Intensity and PPG Analysis](#peak-intensity-and-ppg-analysis)
3. [Signal Processing and Data Smoothing](#signal-processing-and-data-smoothing)
4. [Fourier Transform and Partitioning](#fourier-transform-and-partitioning)
5. [Live Plotting and GUI](#live-plotting-and-gui)
6. [Optimal Window Size Calculation](#optimal-window-size-calculation)
7. [Installation and Dependencies](#installation-and-dependencies)
8. [Usage](#usage)

---

## Confusion Matrix Visualization
**Script:** `ConfusionMatrix.py`

**Description:**
This script generates a confusion matrix visualization based on provided classification metrics (True Positives, False Positives, False Negatives, True Negatives). It allows for quick evaluation of model performance using a heatmap.

**Usage:**
```sh
python ConfusionMatrix.py
```

**Features:**
- Generates a 2x2 confusion matrix
- Supports adjustable color schemes for different classification errors
- Annotates values in percentage format

---

## Peak Intensity and PPG Analysis
**Scripts:**
- `FindPeakIntensity.py`
- `FindPeaksPPG.py`

**Description:**
These scripts analyze photoplethysmography (PPG) signal data to detect peaks and compute their intensities. They are essential for analyzing pulsatile physiological signals and estimating heart rate variability.

**Usage:**
```sh
python FindPeakIntensity.py
python FindPeaksPPG.py
```

**Features:**
- Identifies peaks and troughs in PPG data
- Computes intensity variations over time
- Can be used for noise filtering and smoothing

---

## Signal Processing and Data Smoothing
**Scripts:**
- `Floor_PPG_Data.py`
- `FrequencyResponse.py`

**Description:**
These scripts apply filtering techniques to smooth noisy signals and enhance feature detection. `Floor_PPG_Data.py` processes and normalizes time-series data, while `FrequencyResponse.py` visualizes frequency domain characteristics of the processed signals.

**Usage:**
```sh
python Floor_PPG_Data.py
python FrequencyResponse.py
```

**Features:**
- Implements Butterworth and other smoothing filters
- Computes frequency response of time-series data
- Helps in denoising physiological signals

---

## Fourier Transform and Partitioning
**Scripts:**
- `K_Partitions_Fourier.py`
- `K_Partitions_Time.py`

**Description:**
These scripts apply partitioning techniques to time-series data and compute the Fourier Transform of each partition. This allows for time-localized frequency analysis, useful in detecting transient features in signals.

**Usage:**
```sh
python K_Partitions_Fourier.py
python K_Partitions_Time.py
```

**Features:**
- Divides time-series data into multiple segments
- Applies Fourier Transform to each segment
- Provides insights into non-stationary signal characteristics

---

## Live Plotting and GUI
**Script:** `LivePlotGUI.py`

**Description:**
This script provides a real-time plotting interface for analyzing incoming data streams. It is useful for monitoring physiological signals in real-time.

**Usage:**
```sh
python LivePlotGUI.py
```

**Features:**
- Interactive GUI for live plotting
- Adjustable update intervals for different sampling rates
- Supports multiple data visualization modes

---

## Optimal Window Size Calculation
**Script:** `OptimalWindowSize.py`

**Description:**
This script determines the optimal window size for partitioning time-series data based on variance analysis. It is essential for applications requiring adaptive segmentation, such as heart rate analysis and PPG signal processing.

**Usage:**
```sh
python OptimalWindowSize.py
```

**Features:**
- Computes variance across different partition sizes
- Selects optimal window based on statistical stability
- Enhances performance of subsequent signal analysis

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

