# Interface for basic band counting.
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import shared_memory, Pool

import denoising
import model_utils

class SmoothedSignal:
    def __init__(self, signal, coef, smoothed, noise_variance):
        self.signal = signal.copy()
        self.coef = coef.copy()
        self.smoothed = smoothed.copy()
        self.noise_variance = noise_variance
        return

class BandCounter:
    def __init__(self, signal, *, max_bands=None, mortality_rate=None, scales=None, max_corr=0.8, wavelets=None):
        assert len(signal.shape)==1, f"Expected signal to be 1-d array, for shape {signal.shape}."

        # de-mean signal
        self.signal = np.asarray(signal.copy(), dtype=np.float64)
        self.signal -= np.mean(self.signal)

        dictionary = denoising.Dictionary(len(self.signal),
                                          wavelets=wavelets,
                                          scales=scales,
                                          max_corr=max_corr)

        # dictionary terms with scale=n can represent bands with period approx period*n points.
        # Bands which are too short can be filtered out to help smooth the signal.
        # Determine which frequencies we want to keep.
        if max_bands is not None:
            assert max_bands>1, f"expected max_bands to be greater than 1, got {max_bands}"
            min_pts_per_year = len(self.signal)/max_bands
            wavelet_periods = np.asarray([w.period for w in dictionary.wavelets])
            pts_per_mound = np.asarray(wavelet_periods[dictionary.wavelet_idx]) * dictionary.dict_scales
            # Keep these frequencies (low freq only)
            self.is_low_freq_scales = pts_per_mound > (min_pts_per_year)
        else:
            self.is_low_freq_scales = np.full(dictionary.n_atoms, True, dtype=bool)

        if mortality_rate is not None:
            self.prior = model_utils.make_peak_prior(dictionary, self.is_low_freq_scales, mortality_rate, max_bands)
        else:
            self.prior = None
        
        self.denoiser = denoising.Denoiser(len(self.signal), dictionary=dictionary, prior=self.prior, criterion='bic')

        self.smoothed = None
        self.low_freq_smoothed = None
        self.denoiser_info = None
        return
    
    def set_signal(self, signal):
        assert len(signal)==len(self.signal), "New signal must be the same length as existing signal"
        self.signal = np.asarray(signal.copy(), dtype=np.float64)
        self.signal -= np.mean(self.signal)
        self.smoothed = None
        self.low_freq_smoothed = None
        self.denoiser_info = None
        return 
    
    def get_denoiser_info(self):
        if self.denoiser_info is None:
            self.denoiser_info = self.denoiser.fit(self.signal)
        return self.denoiser_info
    
    def get_smoothed(self, filter=False):
        if self.smoothed is None:
            # basis pursuit smoothing
            print("computing smooth...", end="", flush=True)
            self.denoiser_info = self.denoiser.fit(self.signal)
            self.denoiser_n_best_iter_ = self.denoiser.n_best
            self.smoothed = SmoothedSignal(self.signal, 
                                           self.denoiser_info.coef_, 
                                           self.denoiser_info.reconstructed, 
                                           self.denoiser_info.variance_estimate)
            print("done.", flush=True)
        if filter:
            # further band limited smoothing 
            if self.low_freq_smoothed is None:
                coef = self.smoothed.coef * self.is_low_freq_scales
                smoothed = self.denoiser.dot(coef)
                sigma2 = np.mean((smoothed-self.signal)**2)
                self.low_freq_smoothed = SmoothedSignal(self.signal, coef, smoothed, sigma2)
            smoothed_signal = self.low_freq_smoothed
        else:
            smoothed_signal = self.smoothed
        return smoothed_signal        

    def get_count_estimate(self, filter=True):
        smoothed_signal = self.get_smoothed(filter)

        locations = model_utils.count.find_peaks(smoothed_signal.smoothed)
        band_count_est = len(locations)

        return locations, band_count_est
    
    def plot(self, filter=True):
        # Run this to compute BPDN smoothing if it has not yet been done
        smoothed = self.get_smoothed(filter)
        locations, band_count = self.get_count_estimate(filter)
        
        x1 = range(len(smoothed.signal))
        x2 = range(len(smoothed.signal))
        x3 = locations
        
        y1 = smoothed.signal
        y2 = smoothed.smoothed
        y3 = smoothed.smoothed[locations]

        plt.figure()
        plt.plot(x1, y1, label="Signal", color="grey")
        plt.plot(x2, y2, label="Smoothed", color="black")
        plt.title(f"Band Count: {band_count}")
        plt.xlabel("Sample index")
        p = plt.scatter(x3, y3, label=f"Peaks: {band_count}", marker='o', s=50, color='black', zorder=5)
        return p

    def get_count_distribution(self, nboot, filter=True, seed=None, boot_method=None, boot_max_iter=None, boot_min_alpha=None, n_workers=1):
        if seed is None:
            seed = np.random.randint(1,2**21)
        rng = np.random.default_rng(seed)
        
        # get unfiltered smooth
        smoothed = self.get_smoothed(False)
        # to speed up the bootstrap, use 2denoiser_n_best_iter_ from initial run as a reasonable guess at 
        # an upper limit on number of lars iters needed for the bootstrap and use alpha/4 as a guess at
        # min alpha
        max_iter = self.denoiser.max_iter
        if boot_max_iter is None:
            self.denoiser.max_iter = min(max(2*self.denoiser_n_best_iter_, 50), max_iter)
        else:
            self.denoiser.max_iter = boot_max_iter
        if boot_min_alpha is None:
            # TODO: a justification for this choice
            boot_min_alpha = self.denoiser.alpha_/4
        
        active_set = denoising.lasso.get_active_set(smoothed.coef)
        n_active = np.sum(active_set)
        if n_active >= self.denoiser.dictionary.X.shape[0]:
            raise ValueError("Estimated smoothed model has too many non-zero coefficients to construct the posterior")
        elif n_active == 0:
            return [None]*nboot, np.zeros(nboot, dtype=int), np.zeros((len(smoothed.smoothed), nboot), dtype=float)
        
        match boot_method:
            case None | 'ols':
                # ols fit
                X = self.denoiser.get_X()[:,active_set]
                XTX = X.T @ X
                hat_matrix = X @ np.linalg.solve(XTX, X.T)
                leverage = np.diag(hat_matrix)
                ols_pred = hat_matrix @ self.signal
                
                # less-biased estimate of mean function of data (still conditions on selection event)
                mean_fn = ols_pred
                # use residuals from the ols fit, as they are consistent
                raw_resids = self.signal - mean_fn
                modified_resids = raw_resids/np.sqrt(1 - leverage)
                # recenter in case adjusting for leverage changes the mean of the distribution of residuals
                resids = modified_resids - np.mean(modified_resids)
            case 'lasso':
                mean_fn = smoothed.smoothed
                resids = self.signal - mean_fn
            case 'pairs':
                pass
                

        print("bootstrapping...", flush=True)
        if n_workers==1:
            smoothed_boot = np.zeros((nboot, len(self.signal)), dtype=float)
            locations_boot = []
            band_count_boot = np.zeros(nboot, dtype=int)
            sd = np.sqrt(np.var(resids)/len(resids))
            for i in tqdm(range(nboot)):
                # parametric bootstrap sample of data 
                X = None
                match boot_method:
                    case None | 'ols' | 'lasso':
                        sim = mean_fn + rng.choice(resids, size=len(resids), replace=True)
                    case 'ols+smooth' | 'lasso+smooth':
                        # add small amount of noise to residuals to reduce effects of discreteness
                        sim = mean_fn + rng.choice(resids, size=len(resids), replace=True) + rng.normal(0, sd, len(resids))
                    case 'pairs':
                        n = len(self.signal)
                        idx = rng.choice(range(n), size=n, replace=True)
                        sim = self.signal[idx]
                        X = self.denoiser.get_X()[idx,:]
                    case _:
                        raise ValueError("Unknown method parameter")
                # refit smoother
                denoiser_info = self.denoiser.fit(sim, X, min_alpha=boot_min_alpha)
                if filter:
                    coef = denoiser_info.coef_ * self.is_low_freq_scales
                else:
                    coef = denoiser_info.coef_
                smoothed_b = self.denoiser.dot(coef)
                # get statistics
                locations = model_utils.count.find_peaks(smoothed_b)
                band_count = len(locations)

                smoothed_boot[i] = smoothed_b
                locations_boot.append(locations)
                band_count_boot[i] = band_count
        elif n_workers==int(n_workers) and n_workers > 1:
            if self.prior is not None:
                raise(ValueError("bootstrap with n_workers>1 not implemented with a prior: expected prior=None"))
            match boot_method:
                case None | 'ols' | 'lasso':
                    pass
                case 'pairs':
                    raise ValueError("boot method 'pairs' not implemented with n_worker>1")
                case _:
                    raise ValueError("Unknown method parameter")
            
            X = self.denoiser.get_X()
            shared_mem = shared_memory.SharedMemory(create=True,
                                                    size=X.nbytes)
            shared_X = np.ndarray(X.shape, dtype=X.dtype, buffer=shared_mem.buf)
            shared_X[:] = X[:]

            sims = np.asarray([mean_fn + rng.choice(resids, size=len(resids), replace=True) for i in range(nboot)])
            shared_sims_mem = shared_memory.SharedMemory(create=True,
                                                         size=sims.nbytes)
            shared_sims = np.ndarray(sims.shape, dtype=sims.dtype, buffer=shared_sims_mem.buf)
            shared_sims[:] = sims[:]

            shared_fil_mem = shared_memory.SharedMemory(create=True,
                                                         size=self.is_low_freq_scales.nbytes)
            shared_fil = np.ndarray(self.is_low_freq_scales.shape, dtype=self.is_low_freq_scales.dtype, buffer=shared_fil_mem.buf)
            shared_fil[:] = self.is_low_freq_scales[:]

            coef_out_mem = shared_memory.SharedMemory(create=True,
                                                         size=self.denoiser.coef_.nbytes*nboot)
            coef_out = np.ndarray((nboot, len(self.denoiser.coef_)), dtype=self.denoiser.coef_.dtype, buffer=coef_out_mem.buf)
            coef_out[:] = 0

            smoothed_boot = np.zeros((nboot, len(self.signal)), dtype=float)
            locations_boot = [None] * nboot
            band_count_boot = np.zeros(nboot, dtype=int)
            # prepare args for each iteration
            args_list = [(i, boot_min_alpha, filter)
                         for i in range(nboot)]
            with Pool(
                processes=n_workers,
                initializer=init_worker,
                initargs=(shared_mem.name, X.shape, X.dtype, 
                          shared_sims_mem.name, sims.shape, sims.dtype, 
                          shared_fil_mem.name, self.is_low_freq_scales.shape, self.is_low_freq_scales.dtype,
                          coef_out_mem.name, coef_out.shape, coef_out.dtype, len(self.signal), self.denoiser.max_iter),
            ) as pool:
                for i in tqdm(
                    pool.imap_unordered(bootstrap_worker_unpack, args_list),
                    total=nboot
                ):
                    pass
            
            for i in range(nboot):
                smoothed_boot[i] = self.denoiser.dot(coef_out[i])
                locations_boot[i] = model_utils.count.find_peaks(smoothed_boot[i])
                band_count_boot[i] = len(locations_boot[i])

            shared_mem.close()
            shared_mem.unlink()
            shared_sims_mem.close()
            shared_sims_mem.unlink()
            shared_fil_mem.close()
            shared_fil_mem.unlink()
            coef_out_mem.close()
            coef_out_mem.unlink()

        else:
            raise(ValueError("n_worker must be a positive integer"))
        
        self.denoiser.max_iter = max_iter
        
        return locations_boot, band_count_boot, smoothed_boot

def init_worker(shared_X_name, shared_X_shape, shared_X_dtype, shared_sims_name, shared_sims_shape, shared_sims_dtype,
                shared_fil_name, fil_shape, fil_dtype, coef_out_name, coef_out_shape, coef_out_dtype, len_sim, max_iters):
    global _shared_X, _shared_X_mem, _shared_sims, _shared_sims_mem, _denoiser, _shared_fil, _shared_fil_mem, _coef_out, _coef_out_mem

    from multiprocessing import shared_memory
    import numpy as np
    import denoising

    _shared_X_mem = shared_memory.SharedMemory(name=shared_X_name)
    _shared_X = np.ndarray(shared_X_shape, dtype=shared_X_dtype, buffer=_shared_X_mem.buf)

    _shared_sims_mem = shared_memory.SharedMemory(name=shared_sims_name)
    _shared_sims = np.ndarray(shared_sims_shape, dtype=shared_sims_dtype, buffer=_shared_sims_mem.buf)

    _shared_fil_mem = shared_memory.SharedMemory(name=shared_fil_name)
    _shared_fil = np.ndarray(fil_shape, dtype=fil_dtype, buffer=_shared_fil_mem.buf)

    _coef_out_mem = shared_memory.SharedMemory(name=coef_out_name)
    _coef_out = np.ndarray(coef_out_shape, dtype=coef_out_dtype, buffer=_coef_out_mem.buf)

    _denoiser = denoising.Denoiser(len_sim, dictionary=_shared_X, prior=None, criterion='bic')

def bootstrap_worker(i, boot_min_alpha, is_filter):
    # attach to shared memory
    global _shared_sims, _shared_fil, _coef_out, _denoiser

    sim = _shared_sims[i]
    # fit smoother
    denoiser_info = _denoiser.fit(sim, min_alpha=boot_min_alpha)

    if is_filter:
        coef = denoiser_info.coef_ * _shared_fil
    else:
        coef = denoiser_info.coef_
    
    _coef_out[i] = coef

    return i

def bootstrap_worker_unpack(args):
    return bootstrap_worker(*args)