# Heart Rate Extraction Using Motion Sensor Data

## Overview

This project implements a Python-based signal processing pipeline that estimates heart rate using motion sensor data (e.g., accelerometer, gyroscope). The objective is to explore noninvasive heart-rate monitoring using inertial signals instead of traditional optical sensors.

The system processes raw time-series data, isolates physiologically relevant frequencies, and estimates heart rate using frequency-domain analysis.


## Approach

The pipeline consists of the following stages:

### 1. Data Preparation

- Removed anomalous segments caused by sensor initialization artifacts

- Identified corrupted datasets via visualization

- Normalized timestamps and structured time-series data

### 2. Signal Processing

- Applied moving average smoothing

- Implemented Butterworth band-pass filters

- Performed multi-axis signal fusion using L2 norm

### 3. Frequency-Domain Analysis

- Converted signals from time domain to frequency domain using FFT

- Detected spectral peaks within physiological heart-rate bands

- Converted frequency peaks to beats per minute (BPM)

### 4. Evaluation

Performance was evaluated across four sensor types using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Standard deviation of error

## Results

- Achieved approximately 85% heart-rate estimation accuracy
- Demonstrated that meaningful physiological signals can be extracted from motion-based sensors using structured analytical methods


## Data Availability

The original dataset used in this project is not included in this repository due to distribution restrictions. The code is structured to operate on time-series motion sensor data in CSV format, and the pipeline can be adapted to similar inertial datasets.
