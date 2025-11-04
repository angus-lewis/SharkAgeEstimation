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
import count

def truncated_geometric_pdf(p: float, k: int, trunc: int = None) -> float:
    """Evaluate the truncated geometric pdf with paramter p and trucnation point trunc at k
    
    The support of k is 1,2,3,...,trunc
    """
    if trunc is not None:
        assert trunc >= 0, f"expected truncation point, trunc, to be positive, got {trunc}"
        norm_const = p * (1 - (1 - p)**trunc)/(1-(1-p))
    else:
        norm_const = 1
    if k <= 0 or k > trunc:
        return 0.0
    return p * (1 - p)**(k-1) / norm_const

def make_peak_prior(X, low_freq_ix, mortality_rate, max_age):
    """Return a geometric prior distribution function for age, coef -> peak_prior(coef).

    Given coefs, the prior distribution constructs a signal using only the coefs where
    low_freq_ix is true, then counts the number of peaks in this signal to determine age.
    """
    def peak_prior(coef):
        low_freq_coef = coef * low_freq_ix[:, np.newaxis]
        smoothed = np.dot(X, low_freq_coef)
        prior_pdf = np.zeros(smoothed.shape[1])
        for model_ix in range(smoothed.shape[1]):
            peaks = count.find_peaks(smoothed[:,model_ix])
            age = len(peaks)
            prior_pdf[model_ix] = truncated_geometric_pdf(mortality_rate, age, max_age)
        return prior_pdf
    return peak_prior