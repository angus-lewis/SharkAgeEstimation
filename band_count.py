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

class BandCounter:
    def __init__(self, signal, max_age=None, mortality_rate=None, scale_switch=np.inf, scales=None, shifts=None):
        assert len(signal.shape)==1, f"Expected signal to be 1-d array, for shape {signal.shape}."
        self.signal = np.asarray(signal.copy(), dtype=np.float64)
        self.signal -= np.mean(self.signal)

        # construct function dictionary of ricker wavelets
        # if not explictly specified, need to determine the scales and shifts at each scale that we want to use
        # by default the scales 1,2,..., len(ts)//4 are used but for long ts (e.g. len(ts)>1500) this can be slow
        # if scale_switch is specified, then the dictionary becomes sparser for scales>scale_switch; every
        # second scales is used and every second shift is used.
        switch_idx = max(2, min(len(self.signal) // 4 + 1, scale_switch + 1))
        last_idx = max(2, len(self.signal) // 4 + 1)

        if scales is None: 
            scales1 = np.arange(1, switch_idx)
            scales2 = np.arange(switch_idx, last_idx, step=2)
            scales = np.concatenate((scales1, scales2))

        if shifts is None: 
            shifts = np.concatenate((np.ones(len(scales1)),2*np.ones(len(scales2))))
        
        self.dictionary = denoising.ricker_cwt_dictionary(len(self.signal), scales, shifts, dtype=np.float64)
        self.dictionary_scales = denoising.dictionary_scales(len(self.signal), scales, shifts)
        self.dictionary_shifts = denoising.dictionary_shifts(len(self.signal), scales, shifts)

        # dictionary terms with scale=n are 'mounds' with approx 3.6n points.
        # Mounds which are too short can be filtered out to help smooth the signal.
        # Determine which frequencies we want to keep.
        if max_age is not None:
            assert max_age>1, f"expected max_age to be greater than 1, got {max_age}"
            self.min_pts_per_year = len(self.signal)/max_age
            pts_per_mound = 3.6*self.dictionary_scales
            # Keep these frequencies (low freq only)
            self.keep_freq_scales_ix = pts_per_mound > (self.min_pts_per_year)
        else:
            num_atoms = denoising.get_num_atoms(len(self.signal), scales, shifts)
            self.keep_freq_scales_ix = np.full(num_atoms, True, dtype=np.bool)

        if mortality_rate is not None:
            self.prior = model_utils.make_peak_prior(self.dictionary, self.keep_freq_scales_ix, mortality_rate, max_age)
        else:
            self.prior = denoising.LassoLarsBIC._no_penalty
        
        self.coef = None
        self.smoothed = None
        self.low_freq_coef = None
        self.low_freq_smoothed = None
        self.cwt = None
        return

    def estimate_count(self, method = "peaks", filter=True):
        assert isinstance(method, str) and (method in ("peaks", "scalogram")), f"expected method to be one of \'peaks\' or \'scalogram\', got {method}."

        if self.coef is None or self.smoothed is None:
            # basis pursuit smoothing
            self.coef, self.smoothed = denoising.basis_pursuit_denoising(self.signal, self.dictionary, self.prior)
        
        if filter:
            # further band limited smoothing 
            if self.low_freq_coef is None:
                self.low_freq_coef = self.coef * self.keep_freq_scales_ix
                self.low_freq_smoothed = np.dot(self.dictionary, self.low_freq_coef)
            coef = self.low_freq_coef
            signal = self.low_freq_smoothed
        else:
            coef = self.coef
            signal = self.smoothed
        
        if isinstance(method, str) and (method == "peaks"):
            locations = count.find_peaks(signal)
            age_est = len(locations)
        elif isinstance(method, str) and (method == "scalogram"):
            if self.cwt is None:
                max_scale = np.max(self.dictionary_scales)
                self.cwt, _ = pywt.cwt(self.signal, np.arange(1,max_scale+1), 'mexh')
            
            tmin, tmax = denoising.get_tlims(len(self.signal))
            shifts_idx = self.dictionary_shifts-tmin
            scales_idx = self.dictionary_scales-1

            self.cwt_x, self.cwt_y = count.cwt_path(coef, len(self.signal), scales_idx, shifts_idx)
            locations, age_est = count.count_path(self.cwt_x, self.cwt_y, self.cwt)
        else:
            raise ValueError(f"Unknown method")

        return locations, age_est
    
    def plot(self, method = "peaks", filter=True):
        assert isinstance(method, str) and (method in ("peaks", "scalogram")), f"expected method to be one of \'peaks\' or \'scalogram\', got {method}."
        
        locations, age = self.estimate_count(method, filter)
        if filter:
            smoothed = self.low_freq_smoothed
        else:
            smoothed = self.smoothed
        
        if method == "peaks":    
            x1 = range(len(self.signal))
            x2 = range(len(self.signal))
            x3 = locations
            
            y1 = self.signal
            y2 = smoothed
            y3 = smoothed[locations]

            plt.figure()
            plt.plot(x1, y1, label="Signal", color="grey")
            plt.plot(x2, y2, label="Smoothed", color="blue")
            plt.title(f"Age: {age}")
            plt.xlabel("Sample index")
            p = plt.scatter(x3, y3, label=f"Peaks: age={age}", marker='o', s=80, color='red')
            return p
        
        plt.figure()
        wlb=np.quantile(self.cwt,0.1)/10
        wub=np.quantile(self.cwt,0.9)/10
        plt.imshow(self.cwt, cmap='PRGn', aspect='auto', vmax=-wlb, vmin=wlb)
        plt.scatter(self.cwt_x, self.cwt_y, marker='+', color='red')
        plt.title(f"Age: {age}")
        plt.xlabel("Sample index")
        plt.ylabel("Scale")
        p = plt.plot(self.cwt_x, self.cwt_y, color='red', linestyle='dashed')
        return p
            
        



