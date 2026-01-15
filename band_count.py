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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
    def __init__(self, signal, *, max_bands=None, mortality_rate=None, scales=None, shift=None, max_corr=0.975, wavelets=None, **denoiser_lasso_kwargs):
        assert len(signal.shape)==1, f"Expected signal to be 1-d array, for shape {signal.shape}."

        # de-mean signal
        self.signal = np.asarray(signal.copy(), dtype=np.float64)
        self.signal -= np.mean(self.signal)

        print("building dictionary...")
        dictionary = denoising.Dictionary(len(self.signal),
                                          wavelets=wavelets,
                                          scales=scales,
                                          shift=shift,
                                          max_corr=max_corr)
        print("...done.")

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
            prior = model_utils.make_peak_prior(dictionary, self.is_low_freq_scales, mortality_rate, max_bands)
        else:
            prior = None
        
        self.denoiser = denoising.Denoiser(len(self.signal), dictionary=dictionary, prior=prior, criterion='bic', **denoiser_lasso_kwargs)

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
            print("computing smooth...")
            self.denoiser_info = self.denoiser.fit(self.signal)
            self.smoothed = SmoothedSignal(self.signal, 
                                           self.denoiser_info.coef_, 
                                           self.denoiser_info.reconstructed, 
                                           self.denoiser_info.variance_estimate)
            print("...done")
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
        # Run this to compute BPDN smoothing and or CWT if it has not yet been done
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

    def get_count_distribution(self, nboot, filter=True, seed=None, boot_method=None):
        if seed is None:
            seed = np.random.randint(1,2**21)
        
        # get unfiltered smooth
        smoothed = self.get_smoothed(False)
        
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
                
        smoothed_boot = np.zeros((nboot, len(self.signal)), dtype=float)
        locations_boot = []
        band_count_boot = np.zeros(nboot, dtype=int)

        for i in range(nboot):
            # parametric bootstrap sample of data 
            X = None
            match boot_method:
                case None | 'ols' | 'lasso':
                    sim = mean_fn + np.random.choice(resids, size=len(resids), replace=True)
                case 'pairs':
                    n = len(self.signal)
                    idx = np.random.choice(range(n), size=n, replace=True)
                    sim = self.signal[idx]
                    X = self.denoiser.get_X()[idx,:]
                case _:
                    raise ValueError("Unknown method parameter")
            # refit smoother
            denoiser_info = self.denoiser.fit(sim, X)
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
        
        return locations_boot, band_count_boot, smoothed_boot
