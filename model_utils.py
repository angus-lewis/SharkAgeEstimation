# Implement tools to help modelling sharks.
# Copyright (C) 2025  Angus Lewis

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

def truncated_geometric_pdf(p: float, k: int, trunc: int) -> float:
    """Evaluate the truncated geometric pdf with paramter p and trucnation point trunc at k
    
    The support of k is 1,2,3,...,trunc
    """
    assert trunc >= 0, f"expected truncation point, trunc, to be positive, got {trunc}"
    if k <= 0 or k > trunc:
        return 0.0
    norm_const = p * (1 - (1 - p)**trunc)/(1-(1-p))
    return p * (1 - p)**(k-1) / norm_const

def find_peaks(x: np.ndarray) -> list[int]:
    """
    Find the indices of local maxima (peaks) in a 1D array.

    Args:
        x (np.ndarray): Input 1D array in which to find peaks.

    Returns:
        list[int]: List of indices where local maxima occur.

    Notes:
        - A peak is defined as a point that is greater than its immediate neighbors.
        - The first and last elements are not considered as peaks.

    Example:
        >>> find_peaks(np.array([0, 2, 1, 3, 1]))
        [1, 3]
    """
    locations: list[int] = []
    for i in range(1, len(x) - 1):
        # Check if current element is greater than its neighbors
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            locations.append(i)
    return locations

def make_peak_prior(X, low_freq_ix, mortality_rate):
    """Return a geometric prior distribution function for age, coef -> peak_prior(coef).

    Given coefs, the prior distribution constructs a signal using only the coefs where
    low_freq_ix is true, then counts the number of peaks in this signal to determine age.
    """
    def peak_prior(coef):
        low_freq_coef = coef * low_freq_ix[:, np.newaxis]
        smoothed = np.dot(X, low_freq_coef)
        prior_pdf = np.zeros(smoothed.shape[1])
        for model_ix in range(smoothed.shape[1]):
            peaks = find_peaks(smoothed[:,model_ix])
            age = len(peaks)
            prior_pdf[model_ix] = truncated_geometric_pdf(mortality_rate, age)
        return prior_pdf
    return peak_prior