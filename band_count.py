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
import pywt
import matplotlib.pyplot as plt

import denoising
import count
import model_utils

class SmoothedSignal:
    def __init__(self, signal, coef, smoothed, noise_variance):
        self.signal = signal
        self.coef = coef
        self.smoothed = smoothed
        self.noise_variance = noise_variance
        return

class BandCounter:
    # the distance between the minima of the Ricker wavelet is 2*sqrt(3)
    _pts_per_mound_multiplier = 2*np.sqrt(3)

    @classmethod
    def _build_scales_and_shifts(cls, signal_len, scale_switch, scales, shifts):
        # if not explictly specified, need to determine the scales and shifts at each scale that we want to use
        # by default the scales 1,2,..., len(ts)//4 are used but for long ts (e.g. len(ts)>1500) this can be slow
        # if scale_switch is specified, then the dictionary becomes sparser for scales>scale_switch; every
        # second scales is used and every second shift is used.
        switch_idx = max(2, min(signal_len // 4 + 1, scale_switch + 1))
        last_idx = max(2, signal_len // 4 + 1)

        if scales is None: 
            scales1 = np.arange(1, switch_idx)
            scales2 = np.arange(switch_idx, last_idx, step=2)
            scales = np.concatenate((scales1, scales2))

        if shifts is None: 
            shifts = np.concatenate((np.ones(len(scales1)),2*np.ones(len(scales2))))
        
        return scales, shifts

    def __init__(self, signal, max_bands=None, mortality_rate=None, scale_switch=np.inf, scales=None, shifts=None):
        assert len(signal.shape)==1, f"Expected signal to be 1-d array, for shape {signal.shape}."

        # de-mean signal
        self.signal = np.asarray(signal.copy(), dtype=np.float64)
        self.signal -= np.mean(self.signal)

        # construct function dictionary of ricker wavelets
        scales, shifts = BandCounter._build_scales_and_shifts(len(self.signal), scale_switch, scales, shifts)
        
        self.dictionary = denoising.Dictionary(len(self.signal), scales, shifts)

        # dictionary terms with scale=n are 'mounds' with approx 3.6n points.
        # Mounds which are too short can be filtered out to help smooth the signal.
        # Determine which frequencies we want to keep.
        if max_bands is not None:
            assert max_bands>1, f"expected max_bands to be greater than 1, got {max_bands}"
            self.min_pts_per_year = len(self.signal)/max_bands
            pts_per_mound = self._pts_per_mound_multiplier*self.dictionary.dictionary_scales
            # Keep these frequencies (low freq only)
            self.low_freq_scales_ix = pts_per_mound > (self.min_pts_per_year)
        else:
            self.low_freq_scales_ix = np.full(self.dictionary.n_atoms, True, dtype=np.bool)

        if mortality_rate is not None:
            self.prior = model_utils.make_peak_prior(self.dictionary, self.low_freq_scales_ix, mortality_rate, max_bands)
        else:
            self.prior = denoising.LassoLarsBIC._no_penalty
        
        self.smoothed = None
        self.low_freq_smoothed = None
        return

    def get_count_estimate(self, filter=True):
        if self.smoothed is None:
            # basis pursuit smoothing
            coef, smoothed, sigma2 = denoising.basis_pursuit_denoising(self.signal, self.dictionary, self.prior)
            self.smoothed = SmoothedSignal(self.signal, coef, smoothed, sigma2)
        
        if filter:
            # further band limited smoothing 
            if self.low_freq_smoothed is None:
                coef = self.smoothed.coef * self.low_freq_scales_ix
                smoothed = self.dictionary.dot(coef)
                sigma2 = np.mean((smoothed-self.signal)**2)
                self.low_freq_smoothed = SmoothedSignal(self.signal, coef, smoothed, sigma2)
            smoothed_signal = self.low_freq_smoothed
        else:
            smoothed_signal = self.smoothed
        
        locations = count.find_peaks(smoothed_signal.smoothed)
        band_count_est = len(locations)

        return locations, band_count_est
    
    def plot(self, filter=True):
        # Run this to compute BPDN smoothing and or CWT if it has not yet been done
        locations, band_count = self.get_count_estimate(filter)

        if filter:
            smoothed = self.low_freq_smoothed
        else:
            smoothed = self.smoothed
        
        x1 = range(len(smoothed.signal))
        x2 = range(len(smoothed.signal))
        x3 = locations
        
        y1 = smoothed.signal
        y2 = smoothed.smoothed
        y3 = smoothed.smoothed[locations]

        plt.figure()
        plt.plot(x1, y1, label="Signal", color="grey")
        plt.plot(x2, y2, label="Smoothed", color="blue")
        plt.title(f"Band Count: {band_count}")
        plt.xlabel("Sample index")
        p = plt.scatter(x3, y3, label=f"Peaks: count={band_count}", marker='o', s=80, color='red')
        return p
    
    def get_count_distribution(self, nboot, filter=True):
        if filter:
            smoothed = self.low_freq_smoothed
        else:
            smoothed = self.smoothed
        
        # Construct Laplace approximation to the posterior of active coefs, conditional on non-active coefs = 0
        active_set = denoising.LassoLarsBIC.get_active_set(smoothed.coef)
        if np.sum(active_set)>=self.dictionary.dictionary.shape[0]:
            raise ValueError("Estimated smoothed model has too many non-zero coefficients to construct the Laplace approximation to the posterior")
        X = self.dictionary.dictionary[:, active_set]
        sigma2 = smoothed.noise_variance
        S = sigma2*np.linalg.inv(np.dot(X.T, X))
        mu = smoothed.coef[active_set]

        # Simulate from posterior approximation
        sim_coefs = np.random.multivariate_normal(mu, S, nboot).T

        # Construct signal from simulted coefs
        sim_smoothed = np.dot(X, sim_coefs).T

        # Count peaks for each sim
        band_counts = np.zeros(nboot, dtype=int)
        sim_locations = []
        for i in range(nboot):
            smoothed_signal = sim_smoothed[i]
            locations = count.find_peaks(smoothed_signal)
            sim_locations.append(locations)
            band_counts[i] = len(locations)
        
        return sim_locations, band_counts, sim_smoothed.T


def cwt_count(cwt, signal_len, coef, dict_shifts, dict_scales, tmin):    
    shifts_idx = dict_shifts-tmin
    scales_idx = dict_scales-1

    x, y = count.cwt_path(coef, signal_len, scales_idx, shifts_idx)
    locations, band_count_est = count.count_path(x, y, cwt)
    
    return locations, band_count_est, x, y

def cwt(signal, max_scale):
    cwt, _ = pywt.cwt(signal, np.arange(1, max_scale+1), 'mexh')
    return cwt

def plot_scalogram(cwt, path_x, path_y, band_count):
    plt.figure()
    wlb=np.quantile(cwt,0.1)/10
    wub=np.quantile(cwt,0.9)/10
    plt.imshow(cwt, cmap='PRGn', aspect='auto', vmax=-wlb, vmin=wlb)
    plt.scatter(path_x, path_y, marker='+', color='red')
    plt.title(f"Count: {band_count}")
    plt.xlabel("Sample index")
    plt.ylabel("Scale")
    p = plt.plot(path_x, path_y, color='red', linestyle='dashed')
    return p

def cwt_band_analysis(signal, dictionary):
    max_scale = np.max(dictionary.dictionary_scales)
    cwtmatr = cwt(signal, max_scale)
    locations, band_count, x, y = cwt_count(cwtmatr, 
                                            len(signal), 
                                            dictionary.dictionary_shifts, 
                                            dictionary.dictionary_scales, 
                                            dictionary.tmin)
    return cwtmatr, locations, band_count, x, y 