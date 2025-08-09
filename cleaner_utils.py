# SharkAgeEstimation - Shark age estimation using Fourier analysis
# Copyright (C) 2025 Angus Lewis
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from typing import Optional, Union

import emd 

# Needed to convert between distance and pixels
CONVERT_UM_TO_PIXEL = 5.35026744635231 # um per pixel.

def replace_outliers_with_neighbour_avg(data: np.ndarray) -> np.ndarray:
    """
    Replace outlier values in a 1D array with the average of their immediate neighbors.

    An outlier is any value more than 3 standard deviations from the mean.
    - If the outlier is the first element, replace it with the next element.
    - If the outlier is the last element, replace it with the previous element.
    - Otherwise, replace it with the average of its two adjacent values.

    Parameters
    ----------
    data : array-like
        Input 1D array or sequence of numerical values.

    Returns
    -------
    numpy.ndarray
        Array with outliers replaced by the average of their neighbors.
    """
    data_array: np.ndarray = np.asarray(data)
    mean: float = np.mean(data_array)
    std: float = np.std(data_array)
    cleaned_data: np.ndarray = data_array.copy()

    # Find indices of outliers (>3 std from mean)
    outlier_indices: np.ndarray = np.where(np.abs(data_array - mean) > 3 * std)[0]

    for idx in outlier_indices:
        if idx == 0:
            next_idx = idx + 1
            while next_idx in outlier_indices and next_idx < len(data)-1:
                next_idx += 1
            cleaned_data[idx] = data[next_idx]
        elif idx == len(data) - 1:
            prev_idx = idx - 1
            while prev_idx in outlier_indices and prev_idx >= 0:
                prev_idx -= 1
            cleaned_data[idx] = data[prev_idx]
        else:
            next_idx = idx + 1
            while next_idx in outlier_indices and next_idx < len(data)-1:
                next_idx += 1
            prev_idx = idx - 1
            while prev_idx in outlier_indices and prev_idx >= 0:
                prev_idx -= 1
            cleaned_data[idx] = (data[prev_idx] + data[next_idx]) / 2

    return cleaned_data

def remove_outlier_with_emd(data: Union[np.ndarray, list]) -> Optional[np.ndarray]:
    """
    Removes outliers from a 1D data array using Empirical Mode Decomposition (EMD).

    This function first detrends the input data by extracting the first intrinsic mode function (IMF)
    using EMD, which is approximated by removing the mean of the upper and lower envelopes.
    It then detects and replaces outliers in the detrended data using the
    `replace_outliers_with_neighbour_avg` function, and finally restores the trend.

    Args:
        data (np.ndarray or list): 1D array or list of numerical data to be processed.

    Returns:
        np.ndarray or None: Array with outliers removed and trend restored.
        Returns None if the data has no variation (i.e., upper envelope is None).

    Notes:
        - Requires the `emd` package for envelope computation.
        - Assumes `replace_outliers_with_neighbour_avg` is defined elsewhere.
        - Prints a message and returns None if the input data has no variation.
    """

    # Compute upper and lower envelopes
    upper_env: Optional[np.ndarray] = emd.sift.interp_envelope(data, mode='upper')
    lower_env: Optional[np.ndarray] = emd.sift.interp_envelope(data, mode='lower')

    if upper_env is None:
        print("No variation")
        data_cleaned: Optional[np.ndarray] = None
    else:
        # Compute average envelope
        avg_env = (upper_env + lower_env) / 2

        # Remove first average envelope
        data_cleaned = np.asarray(data) - avg_env

        # Remove outliers with first envelope removed then add it back in
        data_cleaned = replace_outliers_with_neighbour_avg(data_cleaned)
        data_cleaned = data_cleaned + avg_env

    return data_cleaned

def average_around_0_for_log(
    data: np.ndarray, thres: float = 1e-100
) -> np.ndarray:
    """
    Replace near-zero values in a NumPy array with the average of their immediate neighbors.

    Useful for preparing data for logarithmic transformations, where zero or near-zero values can cause issues.
    For each element whose absolute value is less than `thres`, replace it with the average of its two nearest neighbors.
    Edge cases (first and last elements) are handled by copying the adjacent value.

    Parameters:
        data (np.ndarray): Input 1D NumPy array.
        thres (float, optional): Threshold below which values are considered zero (default: 1e-100).

    Returns:
        np.ndarray: Array with near-zero values replaced.
    """
    zero_indices: np.ndarray = np.where(np.abs(data) < thres)[0]

    cleaned_data: np.ndarray = data.copy()

    for idx in zero_indices:
        if idx == 0:
            next_idx = idx + 1
            while next_idx in zero_indices and next_idx < len(data)-1:
                next_idx += 1
            cleaned_data[idx] = data[next_idx]
        elif idx == len(data) - 1:
            prev_idx = idx - 1
            while prev_idx in zero_indices and prev_idx >= 0:
                prev_idx -= 1
            cleaned_data[idx] = data[prev_idx]
        else:
            next_idx = idx + 1
            while next_idx in zero_indices and next_idx < len(data)-1:
                next_idx += 1
            prev_idx = idx - 1
            while prev_idx in zero_indices and prev_idx >= 0:
                prev_idx -= 1
            cleaned_data[idx] = (data[prev_idx] + data[next_idx]) / 2

    return cleaned_data

def down_sampling_arrays(
    data: pd.DataFrame, 
    pixels: np.ndarray, 
    col: str
) -> pd.DataFrame:
    """
    Downsamples a DataFrame by averaging values within specified pixel intervals along a reference column.
    For each value in `pixels`, this function selects rows from `data` where the value in the `col` column
    falls within the interval (p - diff, p], where `diff` is the difference between the first two elements
    of `pixels`. It then computes the mean of these rows and concatenates the results into a new DataFrame.
    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data to be downsampled.
        pixels (array-like): A sequence of reference values (e.g., pixel positions) used to define intervals.
        col (str): The name of the column in `data` to use as the reference for interval selection.
    Returns:
        pd.DataFrame: A DataFrame containing the averaged values for each interval, with rows corresponding
                      to each value in `pixels`. Rows with all zeros are removed.
    """

    diff = pixels[1] - pixels[0]  # Interval size
    df = pd.DataFrame()

    for p in pixels:
        # Select rows in the interval and compute mean
        df1 = pd.DataFrame(data.loc[(data[col] > (p - diff)) & (data[col] <= p), ].mean()).T
        df = pd.concat([df, df1], ignore_index=True)

    return df