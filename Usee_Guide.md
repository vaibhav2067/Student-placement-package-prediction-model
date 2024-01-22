# Placement Package Prediction Analysis User Guide

## Overview

The Placement Package Prediction Analysis tool is designed to analyze and compare the performance of different regression models on placement package prediction data. The supported models include Linear Regression, Multiple Linear Regression, and Artificial Neural Network (ANN).

## Getting Started

1. **Load CSV File:**
   - Click the "Load CSV File" button to open a file dialog.
   - Select a CSV file containing the placement package prediction data.
   - The tool will load the data, and the information label will display the successful loading of the CSV file.

2. **Run Analysis:**
   - After loading the CSV file, click the "Run Analysis" button to perform the regression analysis using different models.
   - The tool will display results for two iterations, each with its own set of model parameters.
   - Results include Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared values for each model.

## Results

The analysis results are presented in two iterations, and each iteration includes the following information:

### Iteration 1 Results

- **Linear Regression:**
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R-squared

- **Multiple Linear Regression:**
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R-squared

- **Artificial Neural Network (ANN):**
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R-squared

### Iteration 2 Results

- **Linear Regression:**
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R-squared

- **Multiple Linear Regression:**
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R-squared

- **Artificial Neural Network (ANN):**
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R-squared

## Graphical Representation

The tool provides graphical representation for each metric (MSE, MAE, R-squared) using bar plots. The plots show the performance of each model in both Iteration 1 and Iteration 2, as well as the difference between the two iterations.

## Models

The supported regression models are:

- Linear Regression
- Multiple Linear Regression
- Neural Network

## Important Notes

- Ensure that the CSV file contains the necessary columns for features and the target variable ("placement_package").
- The tool normalizes the data using Min-Max scaling before performing regression analysis.
- Graphical representations include bar plots for MSE, MAE, and R-squared values for each model.

Enjoy using the Placement Package Prediction Analysis tool for insightful regression model comparisons!
