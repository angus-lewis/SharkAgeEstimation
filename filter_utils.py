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
# - LinearModel: Encapsulates a fitted linear model and computes model criteria.
# - FourierProjectionOperator: Projects signals onto selected Fourier basis functions.
# Functions
# ---------
# - find_peaks(x, smooth): Finds local maxima in a 1D array.
# - smooth(series, bandwidth, ...): Smooths a 1D series using stepwise Fourier selection.
# - age_shark(series, max_age, ...): Estimates "age" by counting peaks in a smoothed signal.
# License
# MIT License
# Copyright (c) 2025 Angus Lewis
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from typing import Optional

import scipy.stats as stats

class FourierModelBuilder:
    """
    FourierModelBuilder constructs a linear model using a subset of Fourier basis functions,
    with model selection performed via forward or backward stepwise selection based on AIC or BIC criteria.
    Parameters
    ----------
    N : int
        The number of data points (length of the signal). Must be non-negative.
    max_model_size : int
        The maximum number of Fourier basis functions (excluding the DC component) to include in the model.
        Must be non-negative and at most N/2.
    mode : str
        The stepwise selection mode, either "forward" (start with minimal model and add terms) or "backward"
        (start with full model and remove terms).
    delta_criteria_threshold : float, optional
        The threshold for the change in the selection criterion (AIC/BIC) to accept a model update.
        Must be non-negative. Default is 2.
    aic_or_bic : str, optional
        The model selection criterion to use, either "aic" or "bic". Default is "aic".
    Attributes
    ----------
    N : int
        Number of data points.
    max_model_size : int
        Maximum number of Fourier basis functions in the model.
    mode : str
        Selection mode ("forward" or "backward").
    model_size : int or None
        Current number of basis functions in the model.
    projection_operator : FourierProjectionOperator
        Operator for projecting data onto the selected Fourier basis.
    linear_model : object or None
        The current linear model fitted to the data.
    aic_or_bic : str
        Model selection criterion in use.
    delta_criteria_threshold : float
        Threshold for accepting model updates.
    y : array-like or None
        The data vector being modeled.
    Methods
    -------
    reset()
        Resets the model builder to its initial state.
    initialise(y)
        Initializes the model builder with data `y` and fits the initial model.
    iteration()
        Performs one step of forward or backward selection, updating the model if the criterion improves sufficiently.
    build()
        Runs the full model selection process and returns the final fitted linear model.
    Raises
    ------
    ValueError
        If input parameters are out of valid range or mode/criterion is invalid.
    RuntimeError
        If iteration is attempted before the model is initialized.
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
        self.projection_operator: FourierProjectionOperator = FourierProjectionOperator(self.N, self.max_model_size)
        self.linear_model: Optional[LinearModel] = None
        self.aic_or_bic: str = aic_or_bic
        self.delta_criteria_threshold: float = delta_criteria_threshold
        self.y: Optional[np.ndarray] = None

    def _init_proj(self) -> None:
        """
        Initializes the projection operator's basis indices and model size based on the current mode.

        In 'forward' mode, initializes the basis index array with only the first element set to True,
        and sets the model size to 0. In 'backward' mode, initializes the basis index array with all
        elements set to True, and sets the model size to the maximum allowed model size.

        Raises:
            ValueError: If the mode is not 'forward' or 'backward'.

        Side Effects:
            Updates the basis indices in the projection operator and sets the model size attribute.
        """
        # Get the bandwidth (number of Fourier basis functions)
        bw: int = self.projection_operator.bandwidth
        if self.mode == "forward":
            basis_idx: np.ndarray = np.zeros(bw + 1, dtype=bool)
            basis_idx[0] = True
            self.model_size = 0
        elif self.mode == "backward":
            basis_idx: np.ndarray = np.ones(bw + 1, dtype=bool)
            self.model_size = self.max_model_size
        else:
            raise ValueError("mode can be 'forward' or 'backward' only")
        self.projection_operator.set_basis(basis_idx)

    def reset(self):
        """
        Resets the internal state of the object.

        This method reinitializes the projection by calling the internal `_init_proj()` method,
        and clears any existing linear model by setting `self.linear_model` to `None`.

        Side Effects:
            - The projection state is reset.
            - Any previously fitted or assigned linear model is removed.

        Usage:
            Call this method to restore the object to its initial state, typically before
            starting a new filtering or modeling operation.
        """
        self._init_proj()
        self.linear_model = None

    def initialise(self, y: np.ndarray) -> None:
        """
        Initialise the filter with the provided observation data.

        This method sets the observation data `y` as an instance variable,
        resets the internal state of the filter, and applies the projection
        operator to the observation data to initialise the linear model.

        Args:
            y (np.ndarray): The observation data to initialise the filter with.

        Returns:
            None

        Side Effects:
            - Sets `self.y` to the provided observation data.
            - Calls `self.reset()` to reset the filter's internal state.
            - Sets `self.linear_model` by applying the projection operator to `y`.

        Example:
            >>> filter_instance.initialise(observation_array)
        """
        self.y = y
        self.reset()
        self.linear_model = self.projection_operator.apply(y)

    def iteration(self):
        """
        Performs a single iteration step for updating the linear model based on the current mode.

        Returns:
            bool: True if an iteration was performed, False otherwise.

        Raises:
            RuntimeError: If the linear model has not been initialized.

        The method checks the current mode of operation:
            - In "forward" mode, it performs an iteration if the model size is less than the maximum allowed.
            - In "backward" mode, it performs an iteration if the model size is greater than zero.
            - If neither condition is met, no iteration is performed and False is returned.

        Note:
            This method assumes that the attributes `linear_model`, `mode`, `model_size`, and `max_model_size`
            are defined on the instance, and that a private method `_iteration()` exists to perform the actual update.
        """
        if self.linear_model is None:
            raise RuntimeError("Expected linear model to be initialised")
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
            - Updates self.linear_model, self.model_size, and the projection operator's basis.

        Raises:
            None
        """
        new_basis: np.ndarray = np.copy(self.projection_operator.basis_idx)
        best_model: Optional[LinearModel] = None
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
            self.projection_operator.set_basis(new_basis)
            new_lm: LinearModel = self.projection_operator.apply(self.y)

            # Evaluate criterion (AIC/BIC)
            crit: float = new_lm.criterion(self.aic_or_bic)
            if crit < best_criterion:
                best_criterion = crit
                best_model = new_lm
                best_idx = idx

            # Revert toggle for next iteration
            new_basis[idx] = not new_basis[idx]
        
        # Compute criterion difference
        # in fwd mode:
        # - self.linear_model.criterion(self.aic_or_bic) is simpler -> larger AIC/BIC
        # - best_model is more complex -> smaller AIC/BIC
        # - delta_criteria = larger - smaller > 0
        # - if delta_criteria > threshold the new model is better, keep going
        # in bkwd mode:
        # - self.linear_model.criterion(self.aic_or_bic) is more complex -> smaller AIC/BIC
        # - best_model is simpler -> larger AIC/BIC
        # - delta_criteria = smaller - larger < 0
        # - if delta_criteria > -threshold the new model is still good enough, keep going
        delta_criteria: float = self.linear_model.criterion(self.aic_or_bic) - best_criterion
        if self.mode == "forward" and (delta_criteria > self.delta_criteria_threshold):
            self.linear_model = best_model
            self.model_size += 1
            new_basis[best_idx] = True
            self.projection_operator.set_basis(new_basis)
            return True
        elif self.mode == "backward" and (delta_criteria > -self.delta_criteria_threshold):
            self.linear_model = best_model
            self.model_size -= 1
            new_basis[best_idx] = False
            self.projection_operator.set_basis(new_basis)
            return True
        return False

    def build(self):
        """
        Builds and fits the linear model using an iterative process.

        This method initializes the model with the provided target variable (`self.y`),
        then repeatedly calls the `iteration()` method until it returns False, indicating
        convergence or completion. After the process, the fitted linear model is returned.

        Returns:
            object: The fitted linear model.

        Notes:
            - The `initialise` method is expected to prepare the model with the target data.
            - The `iteration` method should implement a single step of the fitting process and
              return a boolean indicating whether further iterations are needed.
            - The attribute `self.linear_model` should contain the final fitted model.
        """
        self.initialise(self.y)
        while self.iteration():
            pass
        return self.linear_model
    
def is_complex_eltype(a: np.ndarray) -> bool:
    """
    Checks if the numpy array has a complex floating-point data type.

    Args:
        a (np.ndarray): Input numpy array.

    Returns:
        bool: True if the array's dtype is a subtype of np.complexfloating, False otherwise.

    Example:
        >>> is_complex_eltype(np.array([1+2j, 3+4j]))
        True
        >>> is_complex_eltype(np.array([1.0, 2.0]))
        False
    """
    return np.issubdtype(a.dtype, np.complexfloating)

class LinearModel:
    """
    LinearModel encapsulates the results of fitting a linear model to data,
    including coefficients, fitted values, residuals, and variance estimate.
    Provides methods to compute log-likelihood, AIC, and BIC criteria.

    Attributes
    ----------
    coeffs : np.ndarray
        The estimated coefficients of the model.
    fitted : np.ndarray
        The fitted values (model predictions).
    y : np.ndarray
        The original observed data.
    residuals : np.ndarray
        The residuals (y - fitted).
    s2 : float
        The estimated variance of the residuals.
    _ll : float or None
        Cached log-likelihood value.
    _aic : float or None
        Cached AIC value.
    _bic : float or None
        Cached BIC value.
    """

    def __init__(
        self,
        coeffs: np.ndarray,
        fitted: np.ndarray,
        y: np.ndarray,
        residuals: np.ndarray,
        s2: float
    ) -> None:
        """
        Initialize the LinearModel.

        Args:
            coeffs (np.ndarray): Model coefficients.
            fitted (np.ndarray): Fitted values.
            y (np.ndarray): Observed data.
            residuals (np.ndarray): Residuals (y - fitted).
            s2 (float): Estimated variance of residuals.
        """
        self.coeffs: np.ndarray = coeffs
        self.fitted: np.ndarray = fitted
        self.y: np.ndarray = y
        self.residuals: np.ndarray = residuals
        self.s2: float = s2
        self._ll: Optional[float] = None
        self._aic: Optional[float] = None
        self._bic: Optional[float] = None

    def count_params(self) -> int:
        """
        Count the number of parameters in the model.

        Returns:
            int: Number of parameters (doubled if coefficients are complex).
        """
        if is_complex_eltype(self.coeffs):
            k = 2 * len(self.coeffs)
        else:
            k = len(self.coeffs)
        return k

    def loglikelihood(self) -> float:
        """
        Compute the log-likelihood of the model under a normal error assumption.

        Returns:
            float: Log-likelihood value.
        """
        if self._ll is None:
            # Use the sum of log pdfs for the residuals
            self._ll = np.sum(
                stats.norm.logpdf(self.residuals, scale=np.sqrt(self.s2))
            )
        return self._ll

    def aic(self) -> float:
        """
        Compute the Akaike Information Criterion (AIC) for the model.

        Returns:
            float: AIC value.
        """
        if self._aic is None:
            k = self.count_params()
            self._aic = 2 * k - 2 * self.loglikelihood()
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
            self._bic = k * np.log(n) - 2 * self.loglikelihood()
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

class FourierProjectionOperator:
    """
    Operator for projecting signals onto a subset of Fourier basis functions.
    This class constructs a projection operator using the Discrete Fourier Transform (DFT)
    matrix, allowing for projection of input signals onto a specified set of Fourier basis
    vectors (frequencies). It supports setting the basis, applying the projection, and
    returning a linear model fit.
    Attributes:
        N (int): Length of the signal (number of samples).
        bandwidth (int): Number of Fourier basis functions to use (bandwidth).
        dft_mat (np.ndarray): The DFT matrix of size (N, N).
        design (Optional[np.ndarray]): The design matrix for the selected basis.
        coef_operator (Optional[np.ndarray]): Operator to compute Fourier coefficients.
        basis_idx (Optional[np.ndarray]): Boolean array indicating selected basis indices.
        projection_operator (Optional[np.ndarray]): The projection matrix for the selected basis.
    Methods:
        __init__(self, N: int, bw: int) -> None
            Initializes the FourierProjectionOperator with signal length N and bandwidth bw.
        set_basis(self, basis_idx: Optional[np.ndarray] = None) -> None
            Sets the basis for projection. If basis_idx is None, uses all basis functions up to bandwidth.
            Raises RuntimeError if basis_idx is too long.
        apply(self, y: np.ndarray) -> LinearModel
            Projects the input signal y onto the selected Fourier basis.
            Returns a LinearModel containing coefficients, projection, residuals, and variance estimate.
            Raises RuntimeError if basis is not set.
    Notes:
        - The projection is performed using the DFT matrix and selected basis indices.
        - The variance estimate (s2) is computed differently for complex and real coefficients.
        - Requires external functions/classes: is_complex_eltype, LinearModel.
    """
    def __init__(self, N: int, bw: int) -> None:
        """
        Initialize the FourierProjectionOperator.

        Constructs the Discrete Fourier Transform (DFT) matrix for a signal of length N,
        and sets up attributes for managing the projection onto a subset of Fourier basis functions.

        Args:
            N (int): Length of the signal (number of samples).
            bw (int): Bandwidth, i.e., the maximum number of Fourier basis functions to use.

        Attributes set:
            N (int): Signal length.
            bandwidth (int): Number of Fourier basis functions (bandwidth).
            dft_mat (np.ndarray): DFT matrix of shape (N, N).
            design (Optional[np.ndarray]): Design matrix for the selected basis (initially None).
            coef_operator (Optional[np.ndarray]): Operator to compute Fourier coefficients (initially None).
            basis_idx (Optional[np.ndarray]): Boolean array indicating selected basis indices (initially None).
            projection_operator (Optional[np.ndarray]): Projection matrix for the selected basis (initially None).
        """
        # Compute the DFT matrix for the given signal length
        dft_mat: np.ndarray = np.fft.fft(np.eye(N))
        self.N: int = N
        self.bandwidth: int = bw
        self.dft_mat: np.ndarray = dft_mat

        # Attributes for basis selection and projection (initialized as None)
        self.design: Optional[np.ndarray] = None
        self.coef_operator: Optional[np.ndarray] = None
        self.basis_idx: Optional[np.ndarray] = None
        self.projection_operator: Optional[np.ndarray] = None
        return None
    
    def set_basis(self, basis_idx: Optional[np.ndarray] = None) -> None:
        """
        Sets the Fourier basis functions to be used for projection.

        Args:
            basis_idx (Optional[np.ndarray]): Boolean array of length (bandwidth + 1) indicating which
                Fourier basis functions to include (True = include). If None, all basis functions up to
                the bandwidth are included.

        Raises:
            RuntimeError: If the length of basis_idx exceeds (bandwidth + 1).

        Side Effects:
            Updates the design matrix, coefficient operator, basis index, and projection operator
            attributes of the object.

        Notes:
            - The projection_idx array is constructed by mirroring the selected basis indices to
              ensure conjugate symmetry for real-valued signals.
            - The design matrix is a subset of the DFT matrix rows corresponding to the selected basis.
            - The coefficient operator computes the least-squares Fourier coefficients.
            - The projection operator projects input data onto the selected Fourier basis.
        """
        if basis_idx is None:
            # By default, include all basis functions up to the bandwidth
            basis_idx = np.ones(self.bandwidth + 1, dtype=bool)
        if len(basis_idx) > self.bandwidth + 1:
            raise RuntimeError(f"length of basis must be less than {self.bandwidth + 1}.")

        # Construct the full projection index for the DFT matrix:
        # - Use selected basis indices (basis_idx)
        # - Pad with zeros for unused frequencies
        # - Mirror the basis indices (excluding DC) for conjugate symmetry
        projection_idx = np.concatenate(
            (
                basis_idx,
                np.zeros(self.N - self.bandwidth * 2 - 1, dtype=bool),
                basis_idx[:0:-1]
            )
        )

        # Select the rows of the DFT matrix corresponding to the chosen basis functions
        design: np.ndarray = self.dft_mat[projection_idx, :]

        # Compute the coefficient operator: (X'X)^-1 X'
        # For orthogonal DFT basis, X'X = N*I, so divide by number of samples
        coef_operator: np.ndarray = design / design.shape[1]

        # Compute the projection operator: X (X'X)^-1 X'
        projection_operator: np.ndarray = np.transpose(design.conj()) @ coef_operator

        # Store the computed matrices and basis index
        self.design = design
        self.coef_operator = coef_operator
        self.basis_idx = np.copy(basis_idx)
        self.projection_operator = projection_operator
        return None

    def apply(self, y: np.ndarray) -> LinearModel:
        """
        Projects the input signal y onto the selected Fourier basis and returns a fitted LinearModel.

        Args:
            y (np.ndarray): The input signal to be projected (1D array of length N).

        Returns:
            LinearModel: An object containing the fitted coefficients, projected values,
                         residuals, and variance estimate.

        Raises:
            RuntimeError: If the basis has not been set (coef_operator is None).

        Notes:
            - The coefficients are computed as (X'X)^-1 X' y, where X is the design matrix.
            - The projection is computed as X (X'X)^-1 X' y.
            - The variance estimate s2 is adjusted for the number of parameters and whether
              the coefficients are complex or real.
        """
        if self.coef_operator is None:
            raise RuntimeError("Choose basis first (call set_basis)")
        # Compute Fourier coefficients: (X'X)^-1 X' y
        coeffs: np.ndarray = self.coef_operator @ y
        # Compute projected (fitted) values: X (X'X)^-1 X' y
        proj: np.ndarray = np.real(self.projection_operator @ y)
        # Compute residuals: y - yhat
        residuals: np.ndarray = y - proj
        # Compute sum of squared errors
        sse: float = np.sum(np.power(residuals, 2))
        # Estimate variance, adjusting for number of parameters and complex coefficients
        if is_complex_eltype(coeffs):
            s2: float = sse / (len(y) - 2 * len(coeffs))
        else:
            s2: float = sse / (len(y) - len(coeffs))
        return LinearModel(coeffs, proj, y, residuals, s2)

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
) -> LinearModel:
    """
    Smooths a 1D data series using stepwise selection of Fourier basis functions.

    This function fits a linear model to the input series using a subset of Fourier basis
    functions, selected via forward or backward stepwise selection based on the specified
    model selection criterion (AIC or BIC). The process continues until the improvement in
    the criterion falls below the given threshold.

    Args:
        series (np.ndarray): Input 1D data array to be smoothed.
        bandwidth (int): Maximum number of Fourier basis functions to use (excluding DC).
        mode (str, optional): Stepwise selection mode, either "forward" or "backward". Default is "forward".
        threshold (float, optional): Threshold for improvement in model selection criterion. Default is 2.0.
        criterion (str, optional): Model selection criterion, either "aic" or "bic". Default is "aic".

    Returns:
        LinearModel: The fitted linear model containing coefficients, fitted values, residuals, and variance estimate.

    Raises:
        ValueError: If any argument does not meet the expected requirements.

    Example:
        >>> lm = smooth(np.array([1, 2, 3, 4]), 2)
        >>> lm.fitted
        array([...])
    """
    _check_smooth_arguments(series, bandwidth, mode, threshold, criterion)

    N: int = len(series)
    model_builder: FourierModelBuilder = FourierModelBuilder(
        N, bandwidth, mode, threshold, criterion
    )
    model_builder.initialise(series)
    model_builder.build()
    return model_builder.linear_model

def age_shark(
    series: np.ndarray,
    max_age: int,
    mode: str = "forward",
    threshold: float = 2.0,
    criterion: str = "aic"
) -> tuple[int, list[int], np.ndarray]:
    """
    Estimates the age of a shark from a 1D data series by counting the number of peaks
    in the smoothed signal using stepwise Fourier basis selection.

    Args:
        series (np.ndarray): Input 1D data array representing the signal to analyze.
        max_age (int): Maximum number of Fourier basis functions (interpreted as max age).
        mode (str, optional): Stepwise selection mode, either "forward" or "backward". Default is "forward".
        threshold (float, optional): Threshold for improvement in model selection criterion. Default is 2.0.
        criterion (str, optional): Model selection criterion, either "aic" or "bic". Default is "aic".

    Returns:
        tuple[int, list[int], np.ndarray]:
            - Estimated age (number of peaks found in the smoothed signal).
            - List of indices where peaks occur.
            - The smoothed (fitted) signal as a numpy array.

    Example:
        >>> age, peak_indices, fitted = age_shark(series, max_age=10)
        >>> print(age)
        5

    Notes:
        - The function first smooths the input series using the `smooth` function,
          which applies stepwise Fourier basis selection.
        - Peaks are detected in the fitted (smoothed) signal using `find_peaks`.
        - The number of detected peaks is returned as the estimated age.
    """
    # Smooth the input series using stepwise Fourier basis selection
    lm: LinearModel = smooth(series, max_age, mode, threshold, criterion)
    # Find peaks in the smoothed (fitted) signal
    peaks: list[int] = find_peaks(lm.fitted)
    # Return the number of peaks, their indices, and the fitted signal
    return len(peaks), peaks, lm.fitted
