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
    # When sampling from the posterior, we use SMC.
    # This parameters specifies the number of steps to use when jittering SMC samples.
    # Make this larger if there is evidence of degeneracy in the PF.
    _n_pf_mcmc_steps = 256

    def __init__(self, signal, *, max_bands=None, mortality_rate=None, scales=None, shift=None, max_corr=None, wavelets=None, **denoiser_lasso_kwargs):
        assert len(signal.shape)==1, f"Expected signal to be 1-d array, for shape {signal.shape}."

        # de-mean signal
        self.signal = np.asarray(signal.copy(), dtype=np.float64)
        self.signal -= np.mean(self.signal)

        dictionary = denoising.Dictionary(len(self.signal),
                                          wavelets=wavelets,
                                          scales=scales,
                                          shift=shift,
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
            prior = model_utils.make_peak_prior(dictionary, self.is_low_freq_scales, mortality_rate, max_bands)
        else:
            prior = None
        
        self.denoiser = denoising.Denoiser(len(self.signal), dictionary=dictionary, prior=prior, **denoiser_lasso_kwargs)

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
            self.denoiser_info = self.denoiser.fit(self.signal)
            self.smoothed = SmoothedSignal(self.signal, 
                                           self.denoiser_info.coef_, 
                                           self.denoiser_info.reconstructed, 
                                           self.denoiser_info.variance_estimate)
            smoothed_signal = self.smoothed

        if filter:
            # further band limited smoothing 
            if self.low_freq_smoothed is None:
                coef = self.smoothed.coef * self.is_low_freq_scales
                smoothed = self.denoiser.dot(coef)
                sigma2 = np.mean((smoothed-self.signal)**2)
                self.low_freq_smoothed = SmoothedSignal(self.signal, coef, smoothed, sigma2)
            smoothed_signal = self.low_freq_smoothed
        
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
    
    def _lasso_select_ols_fit(self, X, y, filter=True):
        denoiser = denoising.Denoiser(len(y), X, prior=self.denoiser.prior, **self.denoiser.lasso_kw_args)
        
        # lasso select
        lasso_fit = denoiser.fit(y)
        active_set = denoising.lasso.get_active_set(lasso_fit.coef_)
        if filter:
            active_set = active_set * self.is_low_freq_scales
        X_ols = X[:,active_set]
        if X_ols.shape[0]<=X_ols.shape[1]+1:
            # ols not useful -- residual variance cannot be estimated
            return None

        # ols fit
        lm_fit = LinearRegression(fit_intercept=False).fit(X_ols, y)
        preds = lm_fit.predict(X_ols)
        resid_var = np.sum(((preds - y)**2)/(len(y) - X_ols.shape[1] - 1))
        return lm_fit, active_set, resid_var
    
    def build_data_model(self, seed, fitler):
        # split data into two
        # estimate lasso on one half and keep the selected model
        # refit selected model (unregularised) on the remainder of the data
        # to get an unbiased (conditional on selection) estimate of the true model
        
        # split data
        X = self.denoiser.get_X()
        X1, X2, y1, y2 = train_test_split(
            X, 
            self.signal,
            test_size=0.5,
            random_state=seed,
            shuffle=True)
        
        fit1 = self._lasso_select_ols_fit(X1, y1, filter)
        fit2 = self._lasso_select_ols_fit(X2, y2, filter)
        if fit1 is None or fit2 is None: 
            return None
        model1, active_set1, resid_var1 = fit1
        model2, active_set2, resid_var2 = fit2
        preds1 = model1.predict(X[:,active_set1])
        preds2 = model2.predict(X[:,active_set2])

        return DataModel((preds1, resid_var1), (preds2, resid_var2))

    def get_count_distribution(self, nboot, filter=True, seed=None):
        if seed is None:
            seed = np.random.randint(1,2**21)
        
        smoothed = self.get_smoothed(filter)
        
        active_set = denoising.lasso.get_active_set(smoothed.coef)
        n_active = np.sum(active_set)
        if n_active >= self.denoiser.dictionary.X.shape[0]:
            raise ValueError("Estimated smoothed model has too many non-zero coefficients to construct the posterior")
        elif n_active == 0:
            return [None]*nboot, np.zeros(nboot, dtype=int), np.zeros((len(smoothed.smoothed), nboot), dtype=float)

        smoothed_boot = np.zeros((nboot, len(self.signal)), dtype=float)
        locations_boot = []
        band_count_boot = np.zeros(nboot, dtype=int)

        for i in range(nboot):
            data_model = self.build_data_model(seed, filter)
            if data_model is None:
                return None
            # parametric bootstrap sample of data 
            sim = data_model.draw_sample()
            # refit smoother
            denoiser_info = self.denoiser.fit(sim)
            if filter:
                coef = denoiser_info.coef_ * self.is_low_freq_scales
            else:
                coef = denoiser_info.coef_
            smoothed = self.denoiser.dot(coef)
            # get statistics
            locations = model_utils.count.find_peaks(smoothed)
            band_count = len(locations)

            smoothed_boot[i] = smoothed
            locations_boot.append(locations)
            band_count_boot[i] = band_count
        
        return locations_boot, band_count_boot, smoothed_boot

        
class DataModel:
    def __init__(self, model1, model2):
        self.preds1, self.resid_var1 = model1
        self.preds2, self.resid_var2 = model2
        return 
    
    def draw_sample(self):
        if np.random.random() <= 0.5:
            return self.preds1 + np.random.normal(scale=np.sqrt(self.resid_var1), size=len(self.preds1))
        return self.preds2 + np.random.normal(scale=np.sqrt(self.resid_var2), size=len(self.preds2))
