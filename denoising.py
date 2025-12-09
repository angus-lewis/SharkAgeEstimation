# Implement basis pursuit denoising with LassoLars.
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
import LassoLarsBIC

def _get_bpdn_eps(signal, dictionary):
    if np.issubdtype(signal.dtype, np.floating):
        signal_eps = np.finfo(signal.dtype).eps
    else: 
        signal_eps = np.finfo(float).eps
    if np.issubdtype(dictionary.dtype, np.floating):
        dictionary_eps = np.finfo(dictionary.dtype).eps
    else: 
        dictionary_eps = np.finfo(float).eps
    eps = max(signal_eps, dictionary_eps)
    return eps

class Dictionary:
    @classmethod
    def make_dictionary_scales(cls, signal_len, scales, shifts):
        """Expand a vector of unique scales by repeating each value shift/signal_len for shift in shifts times

        Args:
            length (int): Length of the signal.
            scales (array-like): Scales to use for the Ricker wavelets.
                Default 2**np.arange(np.log2(max(2, length // 2 + 1)), step=0.25).
            shifts (array-like): Shifts to use for the Ricker wavelets at each scale.
                Default np.ones(length), a shift of 1 pt between each wavelet at all scales.
        """
        if signal_len <= 0 or not isinstance(signal_len, int):
            raise ValueError(
                f"signal_len must be a positive integer, got {signal_len}"
            )
        n_atoms = Dictionary.get_num_atoms(signal_len, scales, shifts)
        
        expanded_scales = np.zeros(n_atoms, dtype=scales.dtype)
        start_ix = 0
        for (i, scale) in enumerate(scales):
            n_atoms_at_scale_i = Dictionary.get_num_atoms_at_scale(signal_len, shifts[i])
            expanded_scales[start_ix:(start_ix+n_atoms_at_scale_i)] = scale
            start_ix += n_atoms_at_scale_i
        
        return expanded_scales

    @classmethod
    def make_dictionary_shifts(cls, signal_len, scales, shifts):
        """Determine the shifts for all functions in a dictionary,

        Args:
            length (int): Length of the signal.
            scales (array-like): Scales to use for the Ricker wavelets.
                Default 2**np.arange(np.log2(max(2, length // 2 + 1)), step=0.25).
            shifts (array-like): Shifts to use for the Ricker wavelets at each scale.
                Default np.ones(length), a shift of 1 pt between each wavelet at all scales.
        """
        tmin, tmax = Dictionary.get_tlims(signal_len)
        expanded_shifts = []
        for (i, scale) in enumerate(scales):
            delta = shifts[i]
            n = Dictionary.get_num_atoms_at_scale(signal_len, delta)
            centers = np.linspace(tmin+delta/2, tmax-delta/2, num=n, dtype=float)
            expanded_shifts.append(centers)
        expanded_shifts = np.concatenate(expanded_shifts)
        return expanded_shifts
    
    @classmethod
    def fractional_dyadic_grid(cls, signal_len, fraction):
        """Create a list of points 2^(x) for x = 0, 2f, 3f, 4f, ..., log2(N)
        where f = fraction and N = signal_len.

        E.g., for f=1 returns a list of dyadic numbers 1,2,4,...,2^(N-1).

        Args:
            signal_len (int): Length of the signal.
            fraction (float): Step size for dyadic grid.

        """
        if fraction <= 0:
            raise ValueError(
                f"fraction must be positive, got {fraction}"
            )
        if signal_len <= 0 or not isinstance(signal_len, int):
            raise ValueError(
                f"signal_len must be a positive integer, got {signal_len}"
            )
        grid = 2**np.arange(np.log2(signal_len), step=fraction)
        return grid
    
    @classmethod
    def get_tlims(cls, length):
        tmin = -(length//2) + (1-length%2)/2
        tmax = length//2 - (1-length%2)/2
        return tmin, tmax
    
    @classmethod
    def get_num_atoms_at_scale(cls, length, shift):
        return int((length-shift)//shift)

    @classmethod
    def get_num_atoms(cls, length, scales, shifts):
        """Compute the total number of basis functions.
        
        Args:
            length (int): Length of the signal.
            scales (array-like): Scales to use for the Ricker wavelets.
                Default 2**np.arange(np.log2(max(2, length // 2 + 1)), step=0.25).
            shifts (array-like): Shifts to use for the Ricker wavelets at each scale.
                Default np.ones(length), a shift of 1 pt between each wavelet at all scales.
        """
        num_atoms = 0
        for i in range(len(scales)):
            num_atoms += Dictionary.get_num_atoms_at_scale(length, shifts[i])
        return num_atoms
    
    @classmethod
    def ricker_wavelet(cls, x, sigma, shift):
        """
        Sample the Ricker (Mexican Hat) wavelet.

        Args:
            x (array-like): Input time points.
            sigma (float): Scale parameter.
            shift (float): Shift parameter.

        Returns:
            ndarray: Ricker wavelet values at input points.
        """
        a = 2 / (np.sqrt(3 * sigma) * (np.pi**0.25))
        b = 1 - ((x - shift) / sigma)**2
        c = np.exp(-((x - shift)**2) / (2 * sigma**2))
        return a * b * c
    
    @classmethod 
    def _validate_ricker_dict_inputs(cls, length, scales, shifts):
        """
        Args:
            length (int): Length of the signal.
            scales (array-like): Scales to use for the Ricker wavelets.
                Default 2**np.arange(np.log2(max(2, length // 2 + 1)), step=0.25).
            shifts (array-like): Shifts to use for the Ricker wavelets at each scale.
                Default np.ones(length), a shift of 1 pt between each wavelet at all scales.
        """
        # Validate length
        if not (isinstance(length, int) and length > 0):
            raise ValueError("length must be a positive integer")

        # Default scales
        if scales is None:
            scales = np.arange(1, max(2, length // 4 + 1))

        # Default shifts
        if shifts is None:
            shifts = np.ones(len(scales))

        # Convert and validate scales
        scales = np.asarray(scales, dtype=float)
        if scales.ndim != 1:
            raise ValueError("scales must be a 1-D array-like")
        if not np.all(np.isfinite(scales)):
            raise ValueError("scales must contain finite values")
        if not np.all(scales > 0):
            raise ValueError("scales must be positive")
        
        # Convert and validate scales
        shifts = np.asarray(shifts, dtype=float)
        if shifts.ndim != 1:
            raise ValueError("shifts must be a 1-D array-like")
        if not np.all(np.isfinite(shifts)):
            raise ValueError("shifts must contain finite values")
        if not np.all(shifts > 0):
            raise ValueError("shifts must be positive")
        
        if len(scales) != len(shifts):
            raise ValueError(f"shifts anc scales must be the same length, got {len(scales)} and {len(shifts)}, respectively")

        # Safety check for memory blow-up
        num_atoms = Dictionary.get_num_atoms(length, scales, shifts)
        if num_atoms > 10_000_000:
            raise MemoryError(
                f"Dictionary would have {num_atoms:,} atoms (length={length}, scales={scales.size}, shifts={shifts.size}). "
                "Reduce length or number of scales and/or shifts."
            )
        return length, scales, shifts
    
    def make_ricker_cwt_dictionary(self):
        """
        Generate a dictionary of Ricker (Mexican Hat) wavelet basis functions.

        Args:
            length (int): Length of the signal.
            scales (array-like): Scales to use for the Ricker wavelets.
                Default 2**np.arange(np.log2(max(2, length // 2 + 1)), step=0.25).
            shifts (array-like): Shifts to use for the Ricker wavelets at each scale.
                Default np.ones(length), a shift of 1 pt between each wavelet at all scales.

        Returns:
            ndarray: Dictionary matrix (atoms as columns) with shape (length, sum(shifts)).
        """
        
        # Preallocate for speed and memory predictability: shape (length, num_atoms)
        self.dictionary = np.empty((self.signal_len, self.n_atoms), dtype=np.float64)

        # time stamps of the data to be processed, symmetric around 0
        t = np.linspace(self.tmin, self.tmax, num=self.signal_len, dtype=float)

        # Fill columns
        col = 0
        for i in range(len(self.dictionary_scales)):
            scale = self.dictionary_scales[i]
            shift = self.dictionary_shifts[i]
            self.dictionary[:, col] = Dictionary.ricker_wavelet(t, scale, shift)
            # no need to normalise as the should all have the same norm and mean 0
            col += 1

        return self.dictionary

    def __init__(self, signal_len, scales, shifts):
        signal_len, scales, shifts = Dictionary._validate_ricker_dict_inputs(signal_len, scales, shifts)

        self.signal_len = signal_len
        self.scales = scales
        self.shifts = shifts
        self.n_atoms = Dictionary.get_num_atoms(self.signal_len, self.scales, self.shifts)

        self.tmin, self.tmax = Dictionary.get_tlims(self.signal_len)

        self.dictionary_scales = Dictionary.make_dictionary_scales(self.signal_len, self.scales, self.shifts)
        self.dictionary_shifts = Dictionary.make_dictionary_shifts(self.signal_len, self.scales, self.shifts)

        assert len(self.dictionary_scales)==self.n_atoms, f"internal logic error, expected len(self.dictionary_scales_)==num_atoms but got {len(self.dictionary_scales_)}, {self.n_atoms}"
        assert len(self.dictionary_shifts)==self.n_atoms, f"internal logic error, expected len(self.dictionary_shifts_)==num_atoms but got {len(self.dictionary_shifts_)}, {self.n_atoms}"

        self.dictionary = self.make_ricker_cwt_dictionary()
        return
    
    def dot(self, coef):
        return np.dot(self.dictionary, coef)
    
def basis_pursuit_denoising(signal, dictionary: Dictionary, prior=LassoLarsBIC._no_penalty, fit_intercept=False, max_iter=500, verbose=0, eps=None, precompute=False):
    """
    Perform Basis Pursuit Denoising using LassoLars regression.

    Args:
        signal (array-like): Noisy input signal.
        dictionary: Dictionary matrix (atoms as columns).
        prior (callable [optional]): a prior distribution on the regression coefficients
        fit_intercept (bool [optional]): see LassoLarsIC in sklearn
        max_iter (int [optional]): see LassoLarsIC in sklearn
        verbose (int [optional]): see LassoLarsIC in sklearn
        eps (float [optional]): see LassoLarsIC in sklearn
        precompute ([optional]): see LassoLarsIC in sklearn
        copy_dictionary ([optional]): see copy_X LassoLarsIC in sklearn
        
    Returns:
        ndarray: Sparse coefficient vector.
        ndarray: Reconstructed signal.
    """
    assert len(signal.shape)==1, f"Expected signal to be a numpy vector with len(signal.shape)==1, got {len(signal.shape)}."
    assert dictionary.dictionary.shape[0]==signal.shape[0], f"Expected dictionary to be a numpy ndarray with dictionary.shape[0]==signal.shape[0], got {dictionary.dictionary.shape[0]} and {signal.shape[0]}, respectively."

    if eps is None:
        eps = _get_bpdn_eps(signal, dictionary.dictionary)
    
    # fit lasso
    lasso = LassoLarsBIC.LassoLarsBIC(fit_intercept=fit_intercept, 
                                      verbose=verbose, 
                                      max_iter=max_iter, 
                                      precompute=precompute, 
                                      eps=eps)
    lasso.fit(dictionary.dictionary, signal, prior)
    coef = lasso.coef_
    
    # reconstruct smoothed signal
    reconstructed = dictionary.dot(coef)

    # estimate of noise around lasso fit
    sigma2 = lasso.variance_estimate
    return coef, reconstructed, sigma2, lasso
