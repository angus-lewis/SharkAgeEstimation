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

import os
import numpy as np
import lasso_lars_bic as lasso
import pyfftw as fftw

def fftw_threads():
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count()

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS"):
        if var in os.environ:
            n = min(n, int(os.environ[var]))

    return max(1, n)

class WaveletFamily:
    mother_effective_support = (-np.inf, np.inf)
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
    
    def __call__(self, x, scale, shift, standardize=True):
        return self.wavelet(x, scale, shift, standardize=standardize)
    
    def default_scales(self, length):
        return np.arange(1, length//4+1)

    
class RickerWaveletFamily(WaveletFamily):
    mother_effective_support = (-64, 64)
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
    mother_effective_support = (-64, 64)
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
    _fft_batch_size: int = 64
    _fft_num_threads: int = fftw_threads()

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
            wavelets = [ricker, morlet5]
        
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

    def __init__(self, signal_len, *, scales=None, shift=None, wavelets=None, max_corr=0.975):
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
        max_atoms = sum([len(scales) for scales in self.scales]) * self.n_shifts + 1
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
        X[row] = t/np.linalg.norm(t)
        wavelet_idx[row] = 0 # arbitrary
        dict_scales[row] = np.inf
        dict_shifts[row] = 0
        row += 1

        if max_corr is not None:
            # This section of code determines the correlations (excluding edge effects)
            # between vectors which we are considering adding to the dictionary.
            # Correlations can be computed via convolutions, which are implemented via ffts.

            # amount of padding needed to compute convolution of wavelets with fft
            pad_size = (len(t)+1)//2
            padded_len = 2*pad_size + len(t)
            # number of wavelet functions, without considering shifts plus the linear function
            n_wavelets_scales = 1 + sum([len(scales) for scales in self.scales])
            # round up for fft computations
            n_wavelets_scales_batch_up = ((n_wavelets_scales-1)//self._fft_batch_size + 1)*self._fft_batch_size

            # mask tells us which vectors at each scale and shift that will be kept
            X_mask = np.zeros((n_wavelets_scales_batch_up, padded_len), dtype=np.bool_)
            X_mask[:,(-self.n_shifts+1):] = 1
            X_mask[:,0] = 1
            # always keep the linear part
            X_mask[0][:] = 0
            X_mask[0][0] = 1

            # allocations to compute fft's
            fft_batch_real = fftw.empty_aligned((self._fft_batch_size, padded_len), dtype=np.float64)
            fft_batch_real[:] = 0
            n_fft_coefs = padded_len//2 + 1
            fft_batch_cplx = fftw.empty_aligned((self._fft_batch_size, n_fft_coefs), dtype=np.complex128)
            fft_batch_cplx[:] = 0
            fft_fwd_engine = fftw.FFTW(fft_batch_real, fft_batch_cplx, axes=(1,), direction='FFTW_FORWARD', threads=self._fft_num_threads)
            
            # store the ffts of each wavelet prototype (each scale but no shift)
            X_fft = np.zeros((n_wavelets_scales_batch_up, n_fft_coefs), dtype=np.complex128)
            # compute ffts of prototypes 
            row = 0
            prev_row = 0
            fft_batch_real[row] = 0
            fft_batch_real[row][(pad_size-1):(pad_size-1 + len(t))] = X[0]
            row += 1
            centred_shift = self.shifts[self.n_shifts//2]
            for (w_idx, wavelet) in enumerate(self.wavelets):
                for (_, scale) in enumerate(self.scales[w_idx]):
                    if (row % self._fft_batch_size) == 0:
                        # process fft batch
                        fft_fwd_engine()
                        X_fft[prev_row:row] = fft_batch_cplx
                        prev_row = row
                    fft_batch_real[row % self._fft_batch_size][(pad_size-1):(pad_size-1 + len(t))] = wavelet(t, scale, centred_shift)
                    row += 1
            # process any remaining ffts
            fft_fwd_engine()
            X_fft[prev_row:] = fft_batch_cplx

            # Store the ffts of the mask at each scale here
            X_mask_fft = np.zeros((n_wavelets_scales_batch_up, n_fft_coefs), dtype=np.complex128)

            # allocations for upcoming calculations
            is_high_corr = np.zeros_like(fft_batch_real, dtype=np.bool_)
            batch_mask = np.zeros_like(fft_batch_real, shape=(fft_batch_real.shape[1],))
            fft_batch_real[:] = 0
            fft_batch_cplx[:] = 0
            fft_bkwd_engine = fftw.FFTW(fft_batch_cplx, fft_batch_real, axes=(1,), direction='FFTW_BACKWARD', threads=self._fft_num_threads)
            fft_single_real = fft_batch_real[0]
            fft_single_cplx = fft_batch_cplx[0]
            fft_single_bkwd_engine = fftw.FFTW(fft_single_cplx, fft_single_real, direction='FFTW_BACKWARD')
            fft_single_fwd_engine = fftw.FFTW(fft_single_real, fft_single_cplx, direction='FFTW_FORWARD')

            for row_mask in range(1,n_wavelets_scales):
                prev_row = 0
                fft_single_real[:] = X_mask[row_mask-1]
                fft_single_fwd_engine()
                X_mask_fft[row_mask-1] = fft_single_cplx
                for row in range(row_mask): # only need to process vectors up to the current
                    if (row > 0) and (row % self._fft_batch_size) == 0: # process fft batch
                        # convolve all previous vectors with current vector in fft space to get correlations
                        # at each shift
                        np.multiply(fft_batch_cplx,  X_fft[row_mask], out=fft_batch_cplx)
                        fft_bkwd_engine()
                        # determine which correlations are too large
                        np.abs(fft_batch_real, out=fft_batch_real)
                        np.less(max_corr, fft_batch_real, out=is_high_corr)
                        # keep only the correlations which are with vectors which are already 
                        # going to be added to the dictionary - convolve with mask then any 
                        # convolutions which are positive are too highly correlated.
                        fft_batch_real[:] = is_high_corr
                        fft_fwd_engine()
                        np.multiply(fft_batch_cplx, X_mask_fft[prev_row:row,:].conj(), out=fft_batch_cplx)
                        fft_bkwd_engine()
                        # Columns correspond to shifts, so add down the columns to determin
                        # if correlations are too high at each shift
                        np.sum(fft_batch_real, axis=(0,), out=fft_batch_real[0])
                        # Any elements that are 1 or greater are too highly correlated at that shift
                        np.less(fft_batch_real[0], 0.5, out=batch_mask)
                        np.logical_and(X_mask[row_mask], batch_mask, out=X_mask[row_mask])
                        prev_row = row
                    # reversal in time is equivalent to reversal in freq which is 
                    # equivalent to conjugation for real signals
                    fft_batch_cplx[row % self._fft_batch_size] = X_fft[row].conj()
                # process any remaining rows
                # set any remaining rows to 0
                fft_batch_cplx[((row+1) % self._fft_batch_size):] = 0.0
                # logic here is the same as in the loop, see comments above
                np.multiply(fft_batch_cplx,  X_fft[row_mask], out=fft_batch_cplx)
                fft_bkwd_engine()
                np.abs(fft_batch_real, out=fft_batch_real)
                np.less(max_corr, fft_batch_real, out=is_high_corr)
                fft_batch_real[:] = is_high_corr
                fft_fwd_engine()
                np.multiply(fft_batch_cplx, X_mask_fft[prev_row:(prev_row+self._fft_batch_size),:].conj(), out=fft_batch_cplx)
                fft_bkwd_engine()
                np.sum(fft_batch_real, axis=(0,), out=fft_batch_real[0])
                np.less(fft_batch_real[0], 0.5, out=batch_mask)
                np.logical_and(X_mask[row_mask], batch_mask, out=X_mask[row_mask])

                if not X_mask[row_mask].any():
                    # no vectors to keep at this scale
                    continue

                # determine autocorrelations
                np.multiply(X_fft[row_mask].conj(), X_fft[row_mask], out=fft_single_cplx)
                fft_single_bkwd_engine()
                np.abs(fft_single_real, out=fft_single_real)
                self_corr_idx = 0
                fft_single_real[self_corr_idx] = 0 # zero out correlation with self

                # determine mask for autocorrelation
                max_idx = np.argmax(fft_single_real)
                if fft_single_real[max_idx] <= max_corr:
                    # no additional masking at this scale
                    continue
                min_idx = np.argmin(fft_single_real)
                if fft_single_real[min_idx] > max_corr:
                    # all masked except one
                    idx = np.argmax(X_mask[row_mask]) # finds first True idx (one must exist as its checked above)
                    X_mask[row_mask][:] = 0
                    X_mask[row_mask][idx] = 1
                    continue
                # mask some shifts at this scale due to autocorrelation
                self_mask = fft_single_real <= max_corr
                for i in range(X_mask.shape[1]):
                    if X_mask[row_mask][-i]:
                        np.logical_and(X_mask[row_mask], self_mask, out=X_mask[row_mask])
                    self_mask = np.roll(self_mask, -1)

        row = 1
        row_mask = 0
        keep_idx = [0]
        for (w_idx, wavelet) in enumerate(self.wavelets):
            for (scale_idx, scale) in enumerate(self.scales[w_idx]):
                # add new vectors to dictionary at each shift
                row_mask += 1
                for shift_ix in range(self.n_shifts):
                    shift = self.shifts[shift_ix]
                    v = wavelet(t, scale, shift)

                    # determine if this vector is too correlated with previously added ones
                    if max_corr is None or X_mask[row_mask][-shift_ix]:
                        X[row] = v

                        wavelet_idx[row] = w_idx
                        dict_scales[row] = scale
                        dict_shifts[row] = shift
                        keep_idx.append(row)
                        
                        row += 1
        
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
    def __init__(self, signal_len, dictionary=None, prior=None, max_iter=None, criterion='bic', **lasso_kwargs):
        if max_iter is None:
            max_iter = signal_len
        self.max_iter = max_iter
        if dictionary is None:
            dictionary = Dictionary(signal_len)
        elif isinstance(dictionary, Dictionary):
            assert dictionary.X.shape[0]==signal_len, f"Expected dictionary to be a numpy ndarray with X.shape[0]==signal_len, got {dictionary.X.shape[0]} and {signal_len}, respectively."
        else:
            assert dictionary.shape[0]==signal_len, f"Expected dictionary to be a numpy ndarray with X.shape[0]==signal_len, got {dictionary.shape[0]} and {signal_len}, respectively."
        if prior is None: 
            prior = lasso._no_penalty

        self.lasso_kw_args = lasso_kwargs

        super().__init__(max_iter=max_iter, criterion=criterion, **lasso_kwargs)
        self.signal_len = signal_len
        self.dictionary = dictionary
        self.prior = prior
        return
    
    def fit(self, signal, X=None, eps=None, copy_X=None):
        assert len(signal.shape)==1, f"Expected signal to be a vector (1-dimensional array), got len(signal.shape)={len(signal.shape)}"
        assert signal.shape[0]==self.signal_len, f"Expected signal to be a length self.signal_len, got signal.shape[0]={signal.shape} but self.signal_len={self.signal_len}"
        
        if X is None: 
            X = self.get_X()

        assert X.shape[0]==self.signal_len, f"Expected X to have self.signal_len rows, got X.shape[0]={X.shape[0]} but self.signal_len={self.signal_len}"

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