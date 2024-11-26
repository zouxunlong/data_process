import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# Function to analyze clusters and transitions
def analyze_clusters(sequence, window_size=10):
    # Convert sequence to a pandas Series for easier manipulation
    sequence_series = pd.Series(sequence)

    # Smooth the sequence using a rolling mean
    smoothed_sequence = sequence_series.rolling(window=window_size, center=True).mean()

    # Fill NaN values that result from rolling
    smoothed_sequence_filled = smoothed_sequence.fillna(method="bfill").fillna(method="ffill")

    # Find local maxima and minima (transition points)
    local_maxima = argrelextrema(smoothed_sequence_filled.values, np.greater)[0]
    local_minima = argrelextrema(smoothed_sequence_filled.values, np.less)[0]

    # Combine both minima and maxima for transition points
    transition_indices = sorted(np.concatenate([local_maxima, local_minima]))
    cluster_centers = smoothed_sequence.iloc[transition_indices].to_numpy()

    # Create a DataFrame with the results
    result_df = pd.DataFrame({
        'Transition Index': transition_indices,
        'Cluster Center': cluster_centers
    })

    return result_df, smoothed_sequence

# Example usage
if __name__ == "__main__":
    # Given sequence of numbers
    sequence = [
        3.32, 5.06, 0.03, 0.13, 0.07, 1.54, -0.38, 6.80, 0.02, -0.01, 4.61, -7.01, -7.79, 0.17, 0.10, 3.39, 3.96, 3.51, -8.14, -14.45, -5.65, -0.10, 6.94, 18.75, 18.87, 18.24, 21.45, 9.61, 7.31, 9.34, 6.06, 13.00, 3.58, -1.94, 0.18, 1.32, 0.29, 0.16, -0.02, 0.11, -0.21, 0.16, 8.23, 0.08, 0.03, 5.17, 0.01, 4.60, 12.99, 0.11, -0.06, 0.07, 0.16, 1.76, 0.05, 0.14, 0.18, 3.11, 3.87, -0.07, 0.21, 3.10, 0.83, 1.85, -0.02, 0.24, -0.01, 0.09, 0.11, 0.01, -0.51, 1.21, 0.14, 1.20, 3.13, 3.49, 3.06, 0.14, 1.85, 7.26, 12.48, 15.30, 0.10, -0.12, 0.11, 0.10, -0.34, 0.05, 0.01, 0.06, 3.42, 4.54, 4.84, 0.11, 0.87, 0.21, -4.69, 0.14, -0.06, 0.09, 0.03, 1.39, 1.38, 0.12, 1.76, 0.10, 1.75, 4.04, 0.08, 2.21, 0.21, 0.06, 0.19, 0.09, 5.88, 3.04, -0.16, 0.14, 1.25, 0.17, 0.02, 0.17, -0.23, 0.73, -0.22, 0.02, 4.80, 2.67, 4.13, -0.08, 0.02, 10.54, 14.24, 12.56, 25.81, 11.74, 0.11, 0.20, 0.13, -2.07, -0.43, 2.38, 4.38, 0.11, 21.75, 23.39, 0.20, 2.08, -0.15, 1.40, 0.22, -0.11, -4.42, -9.81, 0.02, -2.77, 0.10, 0.04, -0.02, 0.11, 0.02, 0.15, 0.20, -1.82, 24.62, 0.25, -0.03, 0.18, 0.14, 0.24, 0.05, 0.19, 0.23, 5.72, 1.78, -0.02, 0.21, 0.15, -0.02, 0.17, 0.16, 0.16, 0.17, 0.41, 0.14, 1.91, -0.12, 0.01, 0.19, -0.16, 0.00, 1.30, 0.23, -14.10, -6.74, 4.70, 0.06, 21.50, 0.15, 12.87, -2.86, 1.24, 0.20, -9.54, -0.18, -2.33, -2.17, 0.01, 12.75, 0.09, -0.02, -5.47, -4.03, -3.78, 0.67, -3.67, -6.66, -6.63, -6.59, -6.49, -6.58, -6.51, -5.00, -6.53, -6.65, -6.51, -6.61, -6.74, -18.02, -21.05, -20.11, -14.71, -6.73, -3.25, -0.14, -6.57, -4.34, -6.57, 11.51, 18.34, 5.78, -6.47, -7.33, -6.60, -6.49, -6.82, -3.31, -6.55, 13.93, 18.61, 20.50, 16.54, 8.26, -6.54, 4.56, 4.69, 0.65, -7.04, -6.57, 39.66, -6.64, 0.53, 13.88, 20.69, 14.15, -3.20, -1.91, -2.26, -6.54, 3.15, 5.54, -6.73, -2.65, -3.29, 2.49, 4.53, 6.51, 3.27, -6.73, -10.18, -16.80, -26.02, -2.42, -6.64, -6.55, -6.59, -6.52, -6.52, -1.00, -6.59, -1.38, -6.40, -6.64, -6.37, -6.56, -6.38, 2.23, 2.54, 17.26, 1.47, -12.57, -6.49, -3.71, 8.37, -6.57, -7.68, -7.22, -8.69, -7.12, -6.41, -5.58, -6.61, -6.46, -5.78, -6.70, -1.18, -3.35, -6.72, -7.03, -11.74, 0.48, -6.55, -6.62, -5.20, -6.75, -6.63, -6.61, -6.48, -6.48, -6.57, -2.22, 7.92, -6.51, -4.86, -2.12, -6.46, -6.59, -7.07, 11.76, -6.44, -3.84, -13.64, -6.80, -5.95, -6.41, -6.50, 4.57, 15.00, 14.64, -28.10, -6.51, -6.34, -6.55, -6.42, -6.64, -6.49, -0.70, -6.39, -4.79, 3.98, -6.56, -6.50, -7.57, -6.57, 17.50, 22.61, 9.65, -3.05, 0.88, -11.19, -6.54, 0.99, -6.53, -6.13, 2.31, -11.74, 1.79, -6.58, 9.63, 15.25, -6.54, -6.47, -6.36, -6.44, -6.40, -6.50, -5.70, 10.99, -0.94, -6.30, -6.45, -6.61, -6.56, -6.53, -1.23, -6.57, -6.73, -6.42, -4.22, -3.21, -2.92, -11.73, -3.92, -6.57, -6.61, -6.41, 10.02, -6.53, -6.50, -6.37, -1.79, 3.39, -15.15, -14.84, -2.28, 27.89, 12.20, 15.41, 7.16, -6.74, -6.52, -6.61, -6.45, -4.24, -6.48, -9.79, -6.37, -6.44, -6.75, -8.38, -5.47, 9.90, -31.43, -7.02, -6.44, 2.67, -6.21, -5.36, -2.42, -1.43, -9.23, -4.68, -6.46, -6.41, -6.58, -6.46, -2.18, -2.18, -10.65, -10.63, -9.32, -8.09, -6.29, -6.54, -6.44, -5.74, -6.44, -3.27, 22.69, 27.31, 21.32, 26.26, -6.42, -6.53, -4.03, -1.27, -6.37, -6.59, -1.9209502320185627
    ]

    # Analyze the sequence for clusters and transition points
    result_df, smoothed_sequence = analyze_clusters(sequence, window_size=10)

    # Display the results
    print(result_df)

    # Plotting the original and smoothed sequence for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(sequence, label='Original Sequence', alpha=0.5)
    plt.plot(smoothed_sequence, label='Smoothed Sequence', color='red')
    plt.scatter(result_df['Transition Index'], result_df['Cluster Center'], color='green', marker='o', label='Cluster Centers')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Cluster Centers and Transitions in Sequence')
    plt.show()
