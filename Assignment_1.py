"""
This file analyzes banknote data by creating a 2D scatter plot and a bar chart.
It demonstrates reading and summarizing data from a CSV file, as well as visualizing
data dynamically using NumPy and Matplotlib.

Author: Sahil Hirpara
Student Number: 000889701
"""

import numpy as np
import matplotlib.pyplot as plt

## Read It
# Read the CSV file and split the data

def read_data():
    """Reads the CSV file and splits it into numeric data and labels.

    Returns:
        headers (ndarray): Array of column headers.
        numeric_data (ndarray): Array of numeric data (first four columns).
        labels (ndarray): Array of labels (last column).
    """
    data = np.genfromtxt('banknote_data.csv', delimiter=',', skip_header=1)
    headers = np.genfromtxt('banknote_data.csv', delimiter=',', max_rows=1, dtype=str)
    numeric_data = data[:, :-1].astype(float)
    labels = data[:, -1].astype(int)
    return headers, numeric_data, labels

headers, numeric_data, labels = read_data()

print("Headers:", headers)
print("Numeric Data (First 5 Rows):\n", numeric_data[:5])
print("Labels (First 5):", labels[:5])

## Summarize It
# Summarize the data for the first four columns

def summarize_data(headers, numeric_data):
    """Prints summary statistics (min, max, mean, median) for each numeric column.

    Args:
        headers (ndarray): Array of column headers.
        numeric_data (ndarray): Array of numeric data.
    """
    print("\nSummary:")
    for i, header in enumerate(headers[:-1]):
        column = numeric_data[:, i]
        print(f"{header} - Min: {np.min(column)}, Max: {np.max(column)}, Mean: {np.mean(column):.2f}, Median: {np.median(column):.2f}")

summarize_data(headers, numeric_data)

## Graph It
# Scatter Plot - Columns 0 and 3 with labels

def plot_scatter(headers, numeric_data, labels):
    """Plots a scatter plot for columns 0 and 3, separating data by labels.

    Args:
        headers (ndarray): Array of column headers.
        numeric_data (ndarray): Array of numeric data.
        labels (ndarray): Array of labels.
    """
    legal = numeric_data[labels == 0]
    counterfeit = numeric_data[labels == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(legal[:, 0], legal[:, 3], color='blue', label='Legal')
    plt.scatter(counterfeit[:, 0], counterfeit[:, 3], color='red', label='Counterfeit')
    plt.title('Scatter Plot of Banknote Data')
    plt.xlabel(headers[0])
    plt.ylabel(headers[3])
    plt.legend()
    plt.grid(True)
    plt.show()

plot_scatter(headers, numeric_data, labels)

# Bar Chart - Mean of the first four columns

def plot_bar_chart(headers, numeric_data):
    """Plots a bar chart showing mean values of the first four columns.

    Args:
        headers (ndarray): Array of column headers.
        numeric_data (ndarray): Array of numeric data.
    """
    means = np.mean(numeric_data, axis=0)

    plt.figure(figsize=(8, 6))
    plt.bar(headers[:-1], means, color=['blue', 'orange', 'green', 'red'])
    plt.title('Mean Values of Numeric Columns')
    plt.xlabel('Columns')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.tight_layout()  # Ensure everything fits nicely
    plt.show()

plot_bar_chart(headers, numeric_data)
