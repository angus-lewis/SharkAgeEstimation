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
import lasso_lars_bic as lasso


class WaveletFamily:
    mother_effective_support = (-np.inf, np.inf)
    min_energy_fraction = 0.5
    period = None

    @classmethod
    def quadrature_weights(cls, t):
        """
        Compute second-order accurate quadrature weights for non-uniform sampling.
        t: array of sample locations (monotonic increasing)
        """
        t = np.asarray(t)
        N = len(t)
        w = np.zeros_like(t)

        # Endpoint weights
        w[0] = (t[1] - t[0]) / 1
        w[-1] = (t[-1] - t[-2]) / 1

        # Interior weights
        w[1:-1] = (t[2:] - t[:-2]) / 2

        return np.ones(N) #w

    @classmethod
    def standardize_wavelet(cls, t, psi, valid_idx=None):
        """
        Normalize a wavelet atom for non-uniform sampling and boundaries.
        
        Parameters
        ----------
        t : array_like
            Sample locations (non-uniform grid)
        psi : array_like
            Wavelet evaluated at t
        valid_idx : array_like, optional
            Boolean or index array indicating the portion of psi inside the data.
            If None, all indices are considered valid.
            
        Returns
        -------
        atom : np.ndarray or None
            Normalized atom (length len(t)), or None if atom discarded.
        """
        t = np.asarray(t)
        psi = np.asarray(psi.copy())
        N = len(t)

        if valid_idx is None:
            valid_idx = np.ones(N, dtype=bool)
        valid_idx = np.asarray(valid_idx)

        # Compute quadrature weights
        w = cls.quadrature_weights(t)

        # Weighted zero-mean correction
        weighted_mean = np.sum(psi * w) / np.sum(w[valid_idx])
        psi[valid_idx] -= weighted_mean

        # Weighted L2 normalization
        norm = np.sqrt(np.sum((psi[valid_idx] ** 2) * w[valid_idx]))
        if norm > 0:
            psi[valid_idx] /= norm
        else:
            return None  # discard atom if too small

        return psi

    def __init__(self):
        return 
    
    def mother(self, *args, **kwargs):
        raise NotImplementedError()
    
    def wavelet(self, x, scale, shift, standardize=True):
        x = np.asarray(x)
        t = (x-shift)/scale
        idx = (self.mother_effective_support[0] <= t) * (t <= self.mother_effective_support[1])

        psi = np.zeros_like(t, dtype=float)
        psi[idx] = self.mother(t[idx])

        if standardize:
            psi = self.standardize_wavelet(x, psi, idx)

        return psi
    
    def __call__(self, x, scale, shift):
        return self.wavelet(x, scale, shift)
    
    def default_scales(self, length):
        return np.arange(1, length//4+1)

    
class RickerWaveletFamily(WaveletFamily):
    mother_effective_support = (-4, 4)
    period = 2*np.sqrt(3)

    def mother(self, t):
        """
        Evaluate the Ricker (Mexican Hat) wavelet.

        Args:
            x (array-like): Input time points.

        Returns:
            ndarray: Ricker wavelet values at input points.
        """
        return (1-t**2)*np.exp(-t**2/2)

ricker = ricker_wavelet = RickerWaveletFamily()


class MorletWaveletFamily(WaveletFamily):
    mother_effective_support = (-4, 4)
    period = None

    def __init__(self, n_cycles):
        super().__init__()
        self.n = n_cycles
        self.period = 2*np.pi/self.n
        return
    
    def mother(self, t):
        w = np.exp(-0.5*t**2)
        s = np.cos(self.n*t) - np.exp(-self.n**2/2)
        return w * s
    
    def default_scales(self, length):
        return np.arange(self.n, length//4 + 1)

morlet1 = MorletWaveletFamily(1)
morlet2 = MorletWaveletFamily(2)
morlet3 = MorletWaveletFamily(3)
morlet4 = MorletWaveletFamily(4)
morlet5 = MorletWaveletFamily(5)
morlet6 = MorletWaveletFamily(6)
morlet7 = MorletWaveletFamily(7)


class Dictionary:
    @classmethod
    def _get_bpdn_eps(cls, signal, dictionary):
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
    
    @classmethod
    def make_shifts(cls, signal_len, shift, n_shifts):
        """Determine the shifts for all functions in a dictionary,

        Args:
            length (int): Length of the signal.
            shift (array-like): Shift to use for the Ricker wavelets at each scale.
                Default np.ones(length), a shift of 1 pt between each wavelet at all scales.
        """
        tmin, tmax = Dictionary.get_tlims(signal_len)
        delta = shift
        shifts = np.linspace(tmin+delta/2, tmax-delta/2, num=n_shifts, dtype=float)
        return shifts
    
    @classmethod
    def get_tlims(cls, length):
        tmin = -(length//2) + (1-length%2)/2
        tmax = length//2 - (1-length%2)/2
        return tmin, tmax
    
    @classmethod 
    def _validate_dict_inputs(cls, length, wavelets, scales, shift):
        """
        Args:
            length (int): Length of the signal.
            scales (array-like): Scales to use for the wavelets.
                Default np.arange(1, max(2, length // 4 + 1)).
            shift (real): Shift to use for the wavelets at each scale.
                Default 1, a shift of 1 pt between each wavelet at all scales.
        """
        # Validate length
        if not (isinstance(length, int) and length > 0):
            raise ValueError("length must be a positive integer")
        
        if wavelets is None:
            wavelets = [ricker]
        
        for w in wavelets:
            assert isinstance(w, WaveletFamily), f"wavelets must be a tuple of instances of WaveletFamily."

        # Default scales
        if scales is None:
            scales = [w.default_scales(length) for w in wavelets]

        assert len(wavelets)==len(scales), f"wavelets and scales objects must be the same length"

        # Default shifts
        if shift is None:
            shift = 1

        # Convert and validate scales
        for (i,scale) in enumerate(scales):
            scale = np.asarray(scale, dtype=float)
            if scale.ndim != 1:
                raise ValueError("scales must be a 1-D array-like")
            if not np.all(np.isfinite(scale)):
                raise ValueError("scales must contain finite values")
            if not np.all(scale > 0):
                raise ValueError("scales must be positive")
            scales[i] = scale
        
        # Validate scales
        if not np.isfinite(shift):
            raise ValueError("shift must be finite")
        if not (shift > 0):
            raise ValueError("shifts must be positive")
        
        n_shifts = int((length-shift)//shift)

        return length, wavelets, scales, shift, n_shifts

    def __init__(self, signal_len, *, scales=None, shift=None, wavelets=None, max_corr=None):
        self.signal_len, self.wavelets, self.scales, self.shift, self.n_shifts = (
            Dictionary._validate_dict_inputs(signal_len, wavelets, scales, shift)
        )

        self.n_shifts = int((self.signal_len-self.shift)//self.shift)
        self.shifts = self.make_shifts(self.signal_len, self.shift, self.n_shifts)

        self.tmin, self.tmax = Dictionary.get_tlims(self.signal_len)

        self.make_wavelet_dictionary(max_corr)
        return
    
    def make_wavelet_dictionary(self, max_corr=None):
        """
        Generate a dictionary of wavelet basis functions.

        Args:
            max_corr (float): Number between 0 and 1 which specifies the maximum correlation 
                between vectors in the dictionary. If max_corr=1 (or None) all vectors will 
                remain in the dictionary. If max_corr<1 then vectors with corr(v_i, v_j)>max_corr
                will not be in the dictionary.

        Returns:
            ndarray: Dictionary matrix (atoms as columns) with shape (length, sum(shifts)).
        """
        # add vectors to dict in row-wise order for speed
        max_atoms = sum([len(scales) for scales in self.scales]) * self.n_shifts
        # Safety check for memory blow-up
        if max_atoms > 10_000_000:
            raise MemoryError(
                f"Dictionary would have {max_atoms} atoms (length={self.signal_len},"
                "num wavelets={len(self.wavelets)}, scales={self.scales.size}, shift={self.shift}). "
                "Reduce length or number of scales and/or increase shift."
            )
        X = np.empty((max_atoms, self.signal_len), dtype=np.float64)
        wavelet_idx = np.empty((max_atoms,), dtype=int)
        dict_scales = np.empty((max_atoms,), dtype=np.float64)
        dict_shifts = np.empty((max_atoms,), dtype=np.float64)

        # time stamps of the data to be processed, symmetric around 0
        t = np.linspace(self.tmin, self.tmax, num=self.signal_len, dtype=float)
        # Fill columns
        row = 0
        for (w_idx, wavelet) in enumerate(self.wavelets):
            for scale in self.scales[w_idx]:
                for shift_ix in range(self.n_shifts):
                    shift = self.shifts[shift_ix]
                    v = wavelet(t, scale, shift)
                    
                    X[row] = v

                    wavelet_idx[row] = w_idx
                    dict_scales[row] = scale
                    dict_shifts[row] = shift
                    
                    row += 1
        
        # throw out vectors which are too correlated
        if max_corr is not None:
            corrs = np.dot(X, X.T)
            np.fill_diagonal(corrs, 0)
            max_corrs = np.max(np.abs(corrs), axis=1)
            keep_idx = max_corrs <= max_corr 
        else:
            keep_idx = np.ones((max_atoms,), dtype=bool)

        # transpose so that columns are X vectors, as is standard for regression
        self.X = np.transpose(X[keep_idx])
        self.n_atoms = np.sum(keep_idx)
        self.wavelet_idx = wavelet_idx[keep_idx]
        self.dict_scales = dict_scales[keep_idx]
        self.dict_shifts = dict_shifts[keep_idx]

        return self.X
    
    def dot(self, coef):
        return np.dot(self.X, coef)


class Denoiser(lasso.LassoLarsBIC):
    def __init__(self, signal_len, dictionary=None, prior=None, **lasso_kwargs):
        if dictionary is None:
            dictionary = Dictionary(signal_len)
        elif isinstance(dictionary, Dictionary):
            assert dictionary.X.shape[0]==signal_len, f"Expected dictionary to be a numpy ndarray with X.shape[0]==signal_len, got {dictionary.X.shape[0]} and {signal_len}, respectively."
        else:
            assert dictionary.shape[0]==signal_len, f"Expected dictionary to be a numpy ndarray with X.shape[0]==signal_len, got {dictionary.shape[0]} and {signal_len}, respectively."
        if prior is None: 
            prior = lasso._no_penalty

        self.lasso_kw_args = lasso_kwargs

        super().__init__(**lasso_kwargs)
        self.signal_len = signal_len
        self.dictionary = dictionary
        self.prior = prior
        return
    
    def fit(self, signal, eps=None, copy_X=None):
        assert len(signal.shape)==1, f"Expected signal to be a vector (1-dimensional array), got len(signal.shape)={len(signal.shape)}"
        assert signal.shape[0]==self.signal_len, f"Expected signal to be a length self.signal_len, got signal.shape[0]={signal.shape} but self.signal_len={self.signal_len}"
        
        X = self.get_X()

        if eps is None: 
            self.eps = Dictionary._get_bpdn_eps(signal, X)
        
        super().fit(X, signal, self.prior, copy_X)

        # reconstruct smoothed signal
        self.reconstructed = np.dot(X, self.coef_)
        return self
    
    def dot(self, coef):
        X = self.get_X()
        return np.dot(X, coef)

    def get_X(self):
        if isinstance(self.dictionary, Dictionary):
            X = self.dictionary.X
        else:
            X = self.dictionary
        return X