import pywt
import numpy as np
from sklearn.linear_model import LassoLars, LassoLarsIC, lars_path
import matplotlib.pyplot as plt

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from math import log
from numbers import Real
import numpy as np

from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import validate_data
from sklearn.linear_model._base import LinearRegression, _preprocess_data, _fit_context

class MyLassoLarsIC(LassoLarsIC, LassoLars):
    """This is the same as LassoLarsIC from sklearn, except the AIC/BIC is calculated in 
    a different way.
    """

    _parameter_constraints: dict = {
        **LassoLarsIC._parameter_constraints,
    }

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, penalty, copy_X=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        copy_X : bool, default=None
            If provided, this parameter will override the choice
            of copy_X made at instance creation.
            If ``True``, X will be copied; else, it may be overwritten.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if copy_X is None:
            copy_X = self.copy_X
        X, y = validate_data(self, X, y, force_writeable=True, y_numeric=True)

        X, y, Xmean, ymean, Xstd = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, copy=copy_X
        )

        Gram = self.precompute

        alphas_, _, coef_path_, self.n_iter_ = lars_path(
            X,
            y,
            Gram=Gram,
            copy_X=copy_X,
            copy_Gram=True,
            alpha_min=0.0,
            method="lasso",
            verbose=self.verbose,
            max_iter=self.max_iter,
            eps=self.eps,
            return_n_iter=True,
            positive=self.positive,
        )

        n_samples = X.shape[0]

        if self.criterion == "aic":
            criterion_factor = 2
        elif self.criterion == "bic":
            criterion_factor = log(n_samples)
        else:
            raise ValueError(
                f"criterion should be either bic or aic, got {self.criterion!r}"
            )

        preds_ = np.dot(X, coef_path_)
        residuals = y[:, np.newaxis] - preds_
        residuals_sum_squares = np.sum(residuals**2, axis=0)
        degrees_of_freedom = np.zeros(coef_path_.shape[1], dtype=int)
        for k, coef in enumerate(coef_path_.T):
            mask = np.abs(coef) > np.finfo(coef.dtype).eps
            # get the number of degrees of freedom equal to:
            # Xc = X[:, mask]
            # Trace(Xc * inv(Xc.T, Xc) * Xc.T) ie the number of non-zero coefs
            # plus the variance if it is estimated too
            degrees_of_freedom[k] = np.sum(mask) + (self.noise_variance is None)

        self.alphas_ = alphas_

        if self.noise_variance is None:
            self.noise_variance_ = residuals_sum_squares / n_samples
            self.criterion_ = (
                n_samples*np.log(self.noise_variance_) 
                + criterion_factor*degrees_of_freedom
                - 2*np.log(penalty(degrees_of_freedom))
            )
        else:
            self.noise_variance_ = np.full(self.noise_variance, coef_path_.shape[1])
            self.criterion_ = (
                n_samples * np.log(2 * np.pi * self.noise_variance_)
                + residuals_sum_squares / self.noise_variance_
                + criterion_factor * degrees_of_freedom
                - 2*np.log(penalty(degrees_of_freedom))
            )

        n_best = np.argmin(self.criterion_)

        self.alpha_ = alphas_[n_best]
        self.coef_ = coef_path_[:, n_best]
        self._set_intercept(Xmean, ymean, Xstd)
        return self

def swt_transform(signal, wavelet='db1', level=3):
    """
    Perform Stationary Wavelet Transform (SWT) on a 1D signal.

    Args:
        signal (array-like): Input signal.
        wavelet (str): Wavelet type (default 'db1').
        level (int): Number of decomposition levels.

    Returns:
        list of tuples: Each tuple contains (approximation, detail) coefficients for each level.
    """
    coeffs = pywt.swt(signal, wavelet, level=level)
    return coeffs

def cwt_transform(signal, wavelet='morl', scales=None):
    """
    Perform Continuous Wavelet Transform (CWT) on a 1D signal.

    Args:
        signal (array-like): Input signal.
        wavelet (str): Wavelet type (default 'morl').
        scales (array-like): Scales to use for the CWT.

    Returns:
        ndarray: CWT coefficients (matrix of shape [len(scales), len(signal)]).
    """
    if scales is None:
        scales = np.arange(1, len(signal)//4+1)
        # scales = np.logspace(0, np.log2(len(signal)//4), base=2)
        print(scales[0:4])
        print(scales[-4:])
    coeffs, _ = pywt.cwt(signal, scales, wavelet, method='fft', precision=24)
    return coeffs

def _ricker_wavelet(x, sigma, shift):
    """
    Generate a Ricker (Mexican Hat) wavelet.

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

def ricker_cwt_dictionary(length, scales=None):
    """
    Generate a dictionary of Ricker (Mexican Hat) wavelet basis functions.

    Args:
        length (int): Length of the signal.
        scales (array-like): Scales to use for the Ricker wavelets.

    Returns:
        ndarray: Dictionary matrix (atoms as columns) with shape (length, len(scales)*length).
    """

    # Validate length
    if not (isinstance(length, int) and length > 0):
        raise ValueError("length must be a positive integer")

    # Default scales
    if scales is None:
        scales = np.arange(1, max(2, length // 4 + 1))

    # Convert and validate scales
    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1:
        raise ValueError("scales must be a 1-D array-like")
    if not np.all(np.isfinite(scales)):
        raise ValueError("scales must contain finite values")
    if not np.all(scales > 0):
        raise ValueError("scales must be positive")

    # Ensure _ricker_wavelet exists and is callable
    if "_ricker_wavelet" not in globals() or not callable(globals()["_ricker_wavelet"]):
        raise NameError("_ricker_wavelet is not defined or not callable in this module")

    # Safety check for memory blow-up
    num_atoms = length * scales.size
    if num_atoms > 10_000_000:
        raise MemoryError(
            f"Dictionary would have {num_atoms:,} atoms (length={length}, scales={scales.size}). "
            "Reduce length or number of scales."
        )

    t = np.arange(length, dtype=float)

    # Preallocate for speed and memory predictability: shape (length, num_atoms)
    basis = np.empty((length, num_atoms), dtype=float)

    # Fill columns
    col = 0
    for scale in scales:
        for shift in range(length):
            basis[:, col] = _ricker_wavelet(t, scale, shift + 0.5)
            basis[:,col] -= np.mean(basis[:,col])
            basis[:, col] /= np.linalg.norm(basis[:, col])
            col += 1

    return basis

def _no_prior(x):
    return 1

def basis_pursuit_denoising(signal, dictionary=None, prior=_no_prior, alpha=0.0):
    """
    Perform Basis Pursuit Denoising using LassoLars regression.

    Args:
        signal (array-like): Noisy input signal.
        dictionary (ndarray): Dictionary matrix (atoms as columns).
        alpha (float): Regularization parameter.

    Returns:
        ndarray: Sparse coefficient vector.
        ndarray: Reconstructed signal.
    """
    if dictionary is None:
        scales = 2**np.arange(np.log2(len(signal)), step=0.25)
        print(len(scales))
        dictionary = ricker_cwt_dictionary(len(signal), scales)
    lasso = MyLassoLarsIC(criterion='bic', fit_intercept=False)
    lasso.fit(dictionary, signal, prior)
    coef = lasso.coef_
    reconstructed = np.dot(dictionary, coef)
    return coef, reconstructed

# Example usage:
if __name__ == "__main__":
    t = np.linspace(0, 1, 256)
    signal = np.sin(2 * np.pi * 7 * t) + np.random.normal(0, 0.2, t.shape)
    
    # Denoising example
    # Create a dictionary of mexican hat wavelets at different scales and shifts
    n = signal.size

    coef, reconstructed = basis_pursuit_denoising(signal, None, alpha=0.005)
    filtered_coef = coef#np.where(np.concatenate([np.zeros(len(coef)-len(coef)//8), np.ones(len(coef)//8)]), coef, 0)
    filtered_reconstructed = np.dot(cwt_transform(np.eye(n), 'mexh').reshape(-1, n).T, filtered_coef)

    print(f"Sparse coefficients shape: {coef.shape}")
    print(f"Sparse coefficients active: {np.count_nonzero(coef)}")
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(signal, label='Noisy Signal')
    plt.plot(reconstructed, label='Denoised Signal', linewidth=2)
    plt.plot(filtered_reconstructed, label='Filtered Denoised Signal', linewidth=2)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(coef, label='Sparse Coefficients')
    plt.plot(filtered_coef, label='Filtered Coefficients')
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt

N = 16

for j in range(N):
    plt.figure()
    t = np.zeros(N)
    t[j] = 1
    c = cwt_transform(t, 'mexh')
    print(c)

    for i in range(c.shape[0]):
        if j%2==0:
            plt.plot(c[i], label=f'Scale {i+1}, Shift {j}')
        else:
            pass
        
# plt.legend(loc='upper right')
plt.show()

ts = cwt_transform(np.eye(16), 'mexh')
plt.figure()
for i in range(ts[0].shape[0]):
    plt.plot(ts[0][i], label=f'Scale {i+1}')
plt.legend(loc='upper right')
plt.show()
# [scales][signal_shifts][time]

X = ts.reshape(-1, ts.shape[2]).T  # Reshape to (time, scales*shifts)

plt.figure()
plt.matshow(X, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Dictionary Matrix')
plt.xlabel('Time')
plt.ylabel('Dictionary Atoms')
plt.show()

plt.figure()
plt.plot(X[:, -(N//2)], label='Last basis Function')
plt.show()


