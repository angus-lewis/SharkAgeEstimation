# ===============================================================================
# filter_utils.py
# ===============================================================================
# About
# -----
# This module provides utilities for signal processing and model selection using
# Fourier basis functions. It includes classes and functions for constructing
# Fourier-based linear models, performing stepwise model selection (forward or
# backward) using AIC or BIC criteria, smoothing 1D data series, and detecting
# peaks in signals. The primary use case is for applications such as automated
# age estimation in biological signals (e.g., shark age estimation from banded
# structures), but the tools are general-purpose for any 1D signal analysis.
# Classes
# - FourierModelBuilder: Stepwise model selection using Fourier basis functions.
# - DFTStats: Encapsulates a fitted linear model and computes model criteria.
# - DFTOperator: Projects signals onto selected Fourier basis functions.
# Functions
# ---------
# - find_peaks(x, smooth): Finds local maxima in a 1D array.
# - smooth(series, bandwidth, ...): Smooths a 1D series using stepwise Fourier selection.
# - age_shark(series, max_age, ...): Estimates "age" by counting peaks in a smoothed signal.
# License
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
#
# SharkAgeEstimation - Shark age estimation using Fourier analysis
# Copyright (C) 2025 Angus Lewis

import numpy as np
import pyfftw as fftw
from typing import Optional

import scipy.stats as stats

class FourierModelBuilder:
    """
    FourierModelBuilder performs stepwise model selection (forward or backward)
    using Fourier basis functions and AIC/BIC criteria.

    Parameters
    ----------
    N : int
        Signal length (non-negative).
    max_model_size : int
        Max number of Fourier basis functions (≤ N/2, non-negative).
    mode : str
        "forward" (add terms) or "backward" (remove terms).
    delta_criteria_threshold : float, optional
        Minimum criterion improvement to accept update (≥ 0, default 2).
    aic_or_bic : str, optional
        Model selection criterion: "aic" or "bic" (default "aic").

    Attributes
    ----------
    N : int
    max_model_size : int
    mode : str
    model_size : int or None
    projection_operator : DFTOperator
    dft_stats : DFTStats or None
    aic_or_bic : str
    delta_criteria_threshold : float
    y : np.ndarray or None

    Methods
    -------
    reset()
        Reset internal state.
    initialise(y)
        Set data and fit initial model.
    iteration()
        Perform one step of selection.
    build()
        Run selection to completion and return fitted model.

    Raises
    ------
    ValueError
        For invalid arguments.
    RuntimeError
        If called before initialization.
    """
    def __init__(
        self, 
        N: int, 
        max_model_size: int, 
        mode: str, 
        delta_criteria_threshold: float = 2, 
        aic_or_bic: str = "aic"
    ) -> None:
        if N < 0:
            raise ValueError(f"N must be non-negative, got {N}")
        if max_model_size > N / 2:
            raise ValueError(f"max_model_size must be at most N/2={N/2}, got {max_model_size}")
        if max_model_size < 0:
            raise ValueError(f"max_model_size must be non-negative, got {max_model_size}")
        if aic_or_bic not in ("aic", "bic"):
            raise ValueError(f"aic_or_bic must be either 'aic' or 'bic', got {aic_or_bic}")
        if delta_criteria_threshold < 0:
            raise ValueError(f"delta_criteria_threshold must be non-negative, got {delta_criteria_threshold}")

        self.N: int = N
        self.max_model_size: int = max_model_size
        self.mode: str = mode
        self.model_size: Optional[int] = None
        self.projection_operator: DFTOperator = DFTOperator(self.N, self.max_model_size)
        self.dft_stats: Optional[DFTStats] = None
        self.aic_or_bic: str = aic_or_bic
        self.delta_criteria_threshold: float = delta_criteria_threshold
        self.basis_idx: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def _init_proj(self) -> None:
        """
        Initializes the basis indices and model size for the projection operator according to the current mode.

        In 'forward' mode:
            - Sets the basis index array to all False except the first element, which is set to True.
            - Sets the model size to 0.

        In 'backward' mode:
            - Sets the basis index array to all True.
            - Sets the model size to the maximum allowed model size.

            Modifies self.basis_idx and self.model_size attributes.
        """
        # Get the bandwidth (number of Fourier basis functions)
        bw: int = self.projection_operator.bandwidth
        if self.mode == "forward":
            self.basis_idx: np.ndarray = np.zeros(bw + 1, dtype=bool)
            self.basis_idx[0] = True
            self.model_size = 0
        elif self.mode == "backward":
            self.basis_idx: np.ndarray = np.ones(bw + 1, dtype=bool)
            self.model_size = self.max_model_size
        else:
            raise ValueError("mode can be 'forward' or 'backward' only")
    
    def reset(self) -> None:
        """
        Reset the internal state of the FourierModelBuilder.

        This method reinitializes the projection basis and model size by calling the internal
        `_init_proj()` method, and clears any existing dft_stats or data by setting
        `self.dft_stats`, `self.model_size`, `self.basis_idx`, and `self.y` to None.

        Side Effects:
            - Resets the projection state and model size.
            - Removes any previously fitted or assigned dft_stats and data.

        Example:
            >>> builder.reset()
        """
        self.dft_stats = None
        self.model_size = None
        self.basis_idx = None
        self.y = None
        self._init_proj()

    def initialise(self, y: np.ndarray) -> None:
        """
        Initialise model with data.

        Args:
            y: 1D array of observations.

        Side effects:
            Resets state, sets self.y, fits initial model.
        """
        self.reset()
        self.y = y
        self.projection_operator.set_signal(y)
        self.dft_stats = self.projection_operator.project(self.basis_idx, True)

    def iteration(self):
        """
        Perform a single model update step based on the current mode.

        Raises:
           RuntimeError: If DFT stats are uninitialized.

        Modes:
            - "forward": Iterates if model_size < max_model_size.
            - "backward": Iterates if model_size > 0.

        Assumes:
            Attributes `dft_stats`, `mode`, `model_size`, `max_model_size` exist,
            and a private method `_iteration()` performs the update.
        """
        if self.dft_stats is None:
            raise RuntimeError("Expected DFT stats to be initialised")
        if self.mode == "forward" and self.model_size < self.max_model_size:
            return self._iteration()
        elif self.mode == "backward" and self.model_size > 0:
            return self._iteration()
        return False

    def _iteration(self) -> bool:
        """
        Performs a single step of forward or backward stepwise model selection.

        In forward mode, attempts to add one new Fourier basis function to the model
        that most improves the selection criterion (AIC/BIC). In backward mode,
        attempts to remove one basis function that least degrades the criterion.
        The update is accepted only if the improvement/degradation exceeds the
        delta_criteria_threshold.

        Returns:
            bool: True if the model was updated (i.e., a basis function was added or removed),
                  False otherwise.

        Side Effects:
            - Updates self.dft_stats, self.model_size, and the projection operator's basis.

        Raises:
            None
        """
        new_basis: np.ndarray = np.copy(self.basis_idx)
        best_stats: Optional[DFTStats] = None
        best_criterion: float = np.inf
        best_idx: Optional[int] = None

        # Add/rm basis functions one at a time
        for idx, is_active in enumerate(new_basis):
            # Always include constant/DC (idx=0)
            if (
                idx == 0
                or (self.mode=="forward" and is_active) # cannot add if is_active, skip
                or (self.mode=="backward" and not is_active) # cannot rm if not is_active, skip
            ):
                continue

            # Toggle basis function and fit model
            new_basis[idx] = not new_basis[idx]
            new_dft_stats: DFTStats = self.projection_operator.project(new_basis, True)

            # Evaluate criterion (AIC/BIC)
            crit: float = new_dft_stats.criterion(self.aic_or_bic)
            if crit < best_criterion:
                best_criterion = crit
                best_stats = new_dft_stats
                best_idx = idx

            # Revert toggle for next iteration
            new_basis[idx] = not new_basis[idx]
        
        # Compute criterion difference
        # in fwd mode:
        # - self.dft_stats.criterion(self.aic_or_bic) is simpler -> larger AIC/BIC
        # - best_stats is more complex -> smaller AIC/BIC
        # - delta_criteria = larger - smaller > 0
        # - if delta_criteria > threshold the new model is better, keep going
        # in bkwd mode:
        # - self.dft_stats.criterion(self.aic_or_bic) is more complex -> smaller AIC/BIC
        # - best_stats is simpler -> larger AIC/BIC
        # - delta_criteria = smaller - larger < 0
        # - if delta_criteria > -threshold the new model is still good enough, keep going
        delta_criteria: float = self.dft_stats.criterion(self.aic_or_bic) - best_criterion
        if self.mode == "forward" and (delta_criteria > self.delta_criteria_threshold):
            self.dft_stats = best_stats
            self.model_size += 1
            new_basis[best_idx] = True
            self.basis_idx = new_basis
            return True
        elif self.mode == "backward" and (delta_criteria > -self.delta_criteria_threshold):
            self.dft_stats = best_stats
            self.model_size -= 1
            new_basis[best_idx] = False
            self.basis_idx = new_basis
            return True
        return False

    def build(self):
        """
        Fit the model by iterating until convergence.

        Returns:
            (np.ndarray, DFTStats): Fitted signal and model stats.
        """
        self.initialise(self.y)
        while self.iteration():
            pass
        dft_stats, fitted = self.projection_operator.project(self.basis_idx, False)
        return fitted, dft_stats

class DFTStats:
    """
    DFTStats holds statistics from fitting a linear model, such as variance, parameter count, and data size.
    Provides AIC/BIC model selection criteria.

        Residual variance estimate.
        Number of model coefficients.
        Number of data points.

    Methods
    -------
    count_params()
        Return number of model parameters.
    aic()
        Compute Akaike Information Criterion.
    bic()
        Compute Bayesian Information Criterion.
    criterion(which)
        Return AIC or BIC by name.
    """

    def __init__(
        self,
        s2: float,
        N_coeffs: int,
        N_data: int,
    ) -> None:
        """
        Initialize the DFTStats.

        Args:
            s2 (float): Estimated variance of residuals.
            N_coeffs (int): Number of coefficients in the model.
            N_data (int): Number of data points used in the model.
        """
        self.s2: float = s2
        self.N_coeffs: int = N_coeffs
        self.N_data: int = N_data
        self._aic: Optional[float] = None
        self._bic: Optional[float] = None

    def count_params(self) -> int:
        """
        Count the number of parameters in the model.

        Returns:
            int: Number of parameters.
        """
        return self.N_coeffs

    def aic(self) -> float:
        """
        Compute the Akaike Information Criterion (AIC) for the model.

        Returns:
            float: AIC value.
        """
        if self._aic is None:
            k = self.count_params()
            self._aic = 2 * k + self.N_data * np.log(self.s2)
        return self._aic

    def bic(self) -> float:
        """
        Compute the Bayesian Information Criterion (BIC) for the model.

        Returns:
            float: BIC value.
        """
        if self._bic is None:
            k = self.count_params()
            n = len(self.y)
            self._bic = k * np.log(n) + self.N_data * np.log(self.s2)
        return self._bic

    def criterion(self, which: str) -> float:
        """
        Return the requested model selection criterion.

        Args:
            which (str): Either "aic" or "bic".

        Returns:
            float: The requested criterion value.

        Raises:
            ValueError: If 'which' is not "aic" or "bic".
        """
        if which == "aic":
            return self.aic()
        elif which == "bic":
            return self.bic()
        else:
            raise ValueError(f'Expected which argument to be either "aic" or "bic", got {which}')

class DFTOperator:
    """
    DFTOperator projects signals onto a selected subset of Fourier basis functions using FFT.

    This class manages the efficient computation of the Discrete Fourier Transform (DFT) and
    its inverse for real-valued signals, and allows for projection onto a user-specified set
    of Fourier basis functions (frequencies). It supports setting the basis, projecting the
    signal, and computing model fit statistics.

    Attributes
    ----------
    N : int
        Length of the signal (number of samples).
    bandwidth : int
        Number of Fourier basis functions to use (maximum frequency index).
    x : np.ndarray
        Aligned buffer for the input signal.
    X : np.ndarray
        Buffer for the DFT of the signal.
    X_cached : np.ndarray
        Cached DFT of the original signal.
    ifft : pyfftw.FFTW
        Inverse FFT operator.
    xhat : np.ndarray
        Buffer for the reconstructed (projected) signal.
    projection_idx : np.ndarray[bool]
        Boolean mask indicating which Fourier basis functions are included in the projection.
    residuals_calc_malloc : np.ndarray
        Buffer for intermediate calculations.
    x_sumofsquares : float
        Sum of squares of the original signal (for variance estimation).

    Methods
    -------
    __init__(N: int, bw: int)
        Initialize the operator for signals of length N and bandwidth bw.
    set_basis(basis_idx: np.ndarray[bool]) -> None
        Set which Fourier basis functions to include in the projection.
    set_signal(y: np.ndarray) -> None
        Set the signal to be projected and compute its DFT.
    sum_of_squares(idx: Optional[np.ndarray[bool]] = None) -> float
        Compute the sum of squares of the (optionally masked) Fourier coefficients.
    rss() -> float
        Compute the residual sum of squares for the current projection.
    project(basis_idx: np.ndarray[bool]) -> DFTStats
        Project the signal onto the selected basis and return model fit statistics.

    Notes
    -----
    - Uses pyfftw for fast FFT and inverse FFT operations.
    - The projection is performed by zeroing out unused Fourier coefficients and reconstructing
      the signal via inverse FFT.
    - The sum of squares and variance estimates are computed efficiently using the FFT buffers.
    """
    def __init__(self, N: int, bw: int) -> None:
        """
        Initialize the DFTOperator.

        Sets up FFTW-aligned buffers and FFT/IFFT operators for efficient projection of real-valued
        signals of length N onto a subset of Fourier basis functions up to the specified bandwidth.

        Args:
            N (int): Length of the signal (number of samples).
            bw (int): Bandwidth, i.e., the maximum frequency index (number of Fourier basis functions).

        Attributes set:
            N (int): Signal length.
            bandwidth (int): Number of Fourier basis functions (maximum frequency index).
            x (np.ndarray): FFTW-aligned buffer for the input signal.
            fft (pyfftw.FFTW): Real FFT operator.
            X (np.ndarray): Buffer for the DFT of the signal.
            X_cached (np.ndarray): Cached DFT of the original signal.
            ifft (pyfftw.FFTW): Inverse real FFT operator.
            xhat (np.ndarray): Buffer for the reconstructed (projected) signal.
            residuals_calc_malloc (np.ndarray): Buffer for intermediate calculations.
            x_sumofsquares (float): Sum of squares of the original signal.
            projection_idx (np.ndarray[bool]): Boolean mask for selected Fourier basis functions.
        """

        self.x: np.ndarray = fftw.empty_aligned(N, dtype=np.float64)
        self.fft: fftw.FFTW = fftw.builders.rfft(self.x, N, threads=1)

        self.X: np.ndarray = self.fft()
        self.X_cached: np.ndarray = np.copy(self.X)
        self.ifft: fftw.FFTW = fftw.builders.irfft(self.X, N, threads=1)
        self.xhat: np.ndarray = self.ifft()

        self.residuals_calc_malloc: np.ndarray = fftw.empty_aligned(len(self.X), dtype=np.float64)
        self.x_sumofsquares: float = 0.0

        self.N: int = N
        self.bandwidth: int = bw
        self.projection_idx: np.ndarray = np.zeros(len(self.X), dtype=bool)
        return None
    
    def set_basis(self, basis_idx: Optional[np.ndarray] = None) -> None:
        """
        Set the Fourier basis functions to be used for projection.

        Args:
            basis_idx (np.ndarray[bool]): Boolean array of length (bandwidth + 1) indicating which
            Fourier basis functions to include (True = include). If None, all basis functions up to
            the bandwidth are included.

        Raises:
            RuntimeError: If the length of basis_idx does not match (bandwidth + 1).

        Side Effects:
            Updates the projection_idx attribute, which determines which Fourier coefficients are
            retained during projection.

        Notes:
            - The projection_idx array is used to mask the DFT coefficients for projection.
            - Only the selected basis functions (frequencies) are included in the reconstructed signal.
        """
        if basis_idx is None:
            # By default, include all basis functions up to the bandwidth
            basis_idx = np.ones(self.bandwidth + 1, dtype=bool)
        if len(basis_idx) != self.bandwidth + 1:
            raise RuntimeError(f"length of basis must be {self.bandwidth + 1} got {len(basis_idx)}.")

        # Construct the full projection index for the DFT matrix:
        # - Use selected basis indices (basis_idx)
        # - Pad with zeros for unused frequencies
        # - Mirror the basis indices (excluding DC) for conjugate symmetry
        self.projection_idx[:len(basis_idx)] = basis_idx
        return None

    def sum_of_squares(self, idx: Optional[np.ndarray[bool]] = None) -> float:
        """
        Computes the sum of squares of the Fourier coefficients.

        Args:
            idx (Optional[np.ndarray[bool]]): Optional boolean mask to select which coefficients to include.

        Returns:
            float: The sum of squares of the coefficients.
        """
        if idx is None:
            np.abs(self.X_cached, out=self.residuals_calc_malloc)
            np.pow(self.residuals_calc_malloc, 2, out=self.residuals_calc_malloc)
            s = np.sum(self.residuals_calc_malloc)
        elif len(idx) != len(self.X):
            raise ValueError(f"Expected idx to have length {len(self.X)}, got {len(idx)}")
        else:
            np.abs(self.X_cached, out=self.residuals_calc_malloc, where=idx)
            np.pow(self.residuals_calc_malloc, 2, out=self.residuals_calc_malloc, where=idx)
            s = np.sum(self.residuals_calc_malloc, where=idx)
        
        sos = 2.0*s
        if idx is None or idx[0]:
            sos -= self.residuals_calc_malloc[0]
        if (idx is None or idx[-1]) and self.N%2 == 0:
            sos -= self.residuals_calc_malloc[-1]
        return sos

    def set_signal(self, y: np.ndarray) -> None:
        """
        Sets the signal for which the projection will be computed.

        Args:
            y (np.ndarray): The input signal to be projected (1D array of length N).

        Raises:
            ValueError: If the length of y does not match the expected signal length N.

        Side Effects:
            - Overwrites the internal signal array `self.x` with the values from `y`.
            - Computes and updates the DFT of the signal, storing the result in `self.X` and `self.X_cached`.
            - Updates `self.x_sumofsquares` with the sum of squares of the new signal for variance estimation.
        """
        if len(y) != self.N:
            raise ValueError(f"Expected signal length {self.N}, got {len(y)}")
        self.x[:] = y
        # Compute the DFT of the signal self.x, sets self.X
        self.fft()
        self.X_cached[:] = self.X

        # Compute the sum of squares of the signal for variance estimation
        self.x_sumofsquares = self.sum_of_squares()
        return None
    
    def rss(self) -> float:
        """
        Calculate and return the residual sum of squares (RSS) for the current projection.

        The RSS is computed as the difference between the total sum of squares (`self.x_sumofsquares`)
        and the sum of squares for the projected data (`self.sum_of_squares(self.projection_idx)`),
        normalized by the number of samples (`self.N`).

        Returns:
            float: The residual sum of squares for the current projection.
        """
        # Parsevals theorem
        rss = (self.x_sumofsquares - self.sum_of_squares(self.projection_idx))/self.N
        return rss

    def project(self, basis_idx: np.ndarray[bool], lazy=False) -> DFTStats:
        """
        Project the current signal onto the selected Fourier basis and return model fit statistics.

        Args:
            basis_idx (np.ndarray[bool]): Boolean mask indicating which Fourier basis functions to include.

        Returns:
            DFTStats: Object containing residual variance, number of coefficients, and data size.

        Notes:
            - The projection is performed by zeroing out unused Fourier coefficients and reconstructing
              the signal via inverse FFT.
            - The number of model parameters is twice the number of selected basis functions minus one
              (for real-valued signals).
            - The variance estimate is based on the residual sum of squares after projection.
            - Side effect: This method updates the `xhat` attribute with the reconstructed signal after projection.
        """
        self.set_basis(basis_idx)

        # Get only the coefficients corresponding to the selected basis functions
        self.X[:] = np.float64(0)
        self.X[self.projection_idx] = self.X_cached[self.projection_idx]
        
        s2 = self.rss()/self.N

        if lazy:
            return DFTStats(s2, 2*np.sum(basis_idx)-1, self.N)
        
        # Project the signal onto the selected basis functions (sets self.xhat)
        self.ifft()

        return DFTStats(s2, 2*np.sum(basis_idx)-1, self.N), self.xhat

def find_peaks(x: np.ndarray, smooth: int = 0) -> list[int]:
    """
    Find the indices of local maxima (peaks) in a 1D array.

    Args:
        x (np.ndarray): Input 1D array in which to find peaks.
        smooth (int, optional): Not used (placeholder for future smoothing). Default is 0.

    Returns:
        list[int]: List of indices where local maxima occur.

    Notes:
        - A peak is defined as a point that is greater than its immediate neighbors.
        - The first and last elements are not considered as peaks.
        - The 'smooth' argument is currently unused.

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

def _check_smooth_arguments(
    series: np.ndarray,
    bandwidth: int,
    mode: str,
    threshold: float,
    criterion: str
) -> None:
    """
    Validates the arguments for the smooth function.

    Args:
        series (np.ndarray): Input 1D data array to be smoothed.
        bandwidth (int): The number of Fourier basis functions to use.
        mode (str): Stepwise selection mode, either "forward" or "backward".
        threshold (float): Threshold for model selection criterion improvement.
        criterion (str): Model selection criterion, either "aic" or "bic".

    Raises:
        ValueError: If any argument does not meet the expected requirements.

    Returns:
        None
    """
    if len(series.shape) != 1:
        raise ValueError(f"Expected series to have dimension 1, got {len(series.shape)}")
    if bandwidth < 0 or not isinstance(bandwidth, int):
        raise ValueError(f"Expected bandwidth to be a non-negative integer, got {bandwidth} of type {type(bandwidth)}")
    if not (mode == "forward" or mode == "backward"):
        raise ValueError(f'Expected mode to be "forward" or "backward", got {mode}')
    if threshold <= 0:
        raise ValueError(f"Expected threshold to be positive, got {threshold}")
    if not (criterion == "aic" or criterion == "bic"):
        raise ValueError(f'Expected criterion to be "aic" or "bic", got {criterion}')
    return None

def smooth(
    series: np.ndarray,
    bandwidth: int,
    mode: str = "forward",
    threshold: float = 2.0,
    criterion: str = "aic"
) -> DFTStats:
    """
    Smooth a 1D data series using stepwise selection of Fourier basis functions.

    This function fits a smoothed version of the input series by selecting a subset of Fourier
    basis functions via forward or backward stepwise selection, using AIC or BIC as the model
    selection criterion. The selection process stops when the improvement in the criterion is
    less than the specified threshold.

    Args:
        series (np.ndarray): Input 1D array to be smoothed.
        bandwidth (int): Maximum number of Fourier basis functions to use (not counting DC).
        mode (str, optional): Stepwise selection mode, "forward" (add terms) or "backward" (remove terms). Default is "forward".
        threshold (float, optional): Minimum improvement in model selection criterion to accept an update. Default is 2.0.
        criterion (str, optional): Model selection criterion, "aic" or "bic". Default is "aic".

    Returns:
        tuple[np.ndarray, DFTStats]: The smoothed (fitted) signal and model statistics.

    Raises:
        ValueError: If any argument is invalid.

    Example:
        >>> fitted, stats = smooth(np.array([1, 2, 3, 4]), 2)
        >>> print(fitted)
        [ ... ]
    """
    _check_smooth_arguments(series, bandwidth, mode, threshold, criterion)

    N: int = len(series)
    model_builder: FourierModelBuilder = FourierModelBuilder(
        N, bandwidth, mode, threshold, criterion
    )
    model_builder.initialise(series)
    fitted, dft_stats = model_builder.build()
    return fitted, dft_stats

def age_shark(
    series: np.ndarray,
    max_age: int,
    mode: str = "forward",
    threshold: float = 2.0,
    criterion: str = "aic"
) -> tuple[int, list[int], np.ndarray]:
    """
    Estimate the number of peaks ("age") in a 1D data series after smoothing with stepwise Fourier basis selection.

    Args:
        series (np.ndarray): Input 1D data array to analyze.
        max_age (int): Maximum number of Fourier basis functions (interpreted as max age).
        mode (str, optional): Stepwise selection mode, either "forward" or "backward". Default is "forward".
        threshold (float, optional): Minimum improvement in model selection criterion to accept an update. Default is 2.0.
        criterion (str, optional): Model selection criterion, either "aic" or "bic". Default is "aic".

    Returns:
        tuple[int, list[int], np.ndarray]:
            - Estimated number of peaks (int).
            - List of indices where peaks occur.
            - The smoothed (fitted) signal as a numpy array.

    Example:
        >>> age, peak_indices, fitted = age_shark(series, max_age=10)
        >>> print(age)
        5

    Notes:
        - The input series is smoothed using the `smooth` function with stepwise Fourier basis selection.
        - Peaks are detected in the fitted (smoothed) signal using `find_peaks`.
        - The number of detected peaks is returned as the estimated age.
        - The `max_age` parameter sets the maximum number of Fourier basis functions (not necessarily the maximum number of peaks).
        - The function returns the count of detected peaks, their indices, and the fitted signal.
    """
    # Smooth the input series using stepwise Fourier basis selection
    fitted, dft_stats = smooth(series, max_age, mode, threshold, criterion)
    # Find peaks in the smoothed (fitted) signal
    peaks: list[int] = find_peaks(fitted)
    # Return the number of peaks, their indices, and the fitted signal
    return len(peaks), peaks, fitted
