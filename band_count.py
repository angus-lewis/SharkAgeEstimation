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
import particles
from particles import smc_samplers as ssp
from particles import distributions as dists
from scipy import stats

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

    def get_count_estimate(self, filter=True):
        if self.smoothed is None:
            # basis pursuit smoothing
            self.denoiser_info = self.denoiser.fit(self.signal)
            coef, smoothed, sigma2 = self.denoiser_info.coef_, self.denoiser_info.reconstructed, self.denoiser_info.variance_estimate
            self.smoothed = SmoothedSignal(self.signal, 
                                           self.denoiser_info.coef_, 
                                           self.denoiser_info.reconstructed, 
                                           self.denoiser_info.variance_estimate)
        
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
        
        locations = model_utils.count.find_peaks(smoothed_signal.smoothed)
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
        plt.plot(x2, y2, label="Smoothed", color="black")
        plt.title(f"Band Count: {band_count}")
        plt.xlabel("Sample index")
        p = plt.scatter(x3, y3, label=f"Peaks: {band_count}", marker='o', s=50, color='black', zorder=5)
        return p

    def get_count_distribution(self, nboot, filter=True, seed=None):
        # Sample the posterior of active coefs, conditional on non-active coefs = 0
        # then construct signals for these coeffs and count their peaks

        if seed is None:
            seed = np.random.randint(1,2**21)
        
        # get smoothed signal
        if filter:
            smoothed = self.low_freq_smoothed
        else:
            smoothed = self.smoothed
        
        active_set = denoising.lasso.get_active_set(smoothed.coef)
        n_active = np.sum(active_set)
        if n_active >= self.denoiser.dictionary.X.shape[0]:
            raise ValueError("Estimated smoothed model has too many non-zero coefficients to construct the posterior")
        elif n_active == 0:
            return [None]*nboot, np.zeros(nboot, dtype=int), np.zeros((len(smoothed.smoothed), nboot), dtype=float)

        # Get design matrix of active a set
        X = self.denoiser.dictionary.X[:, active_set]
        sigma2 = smoothed.noise_variance

        # Generate samples from posterior
        beta = self.simulate_posterior(nboot, n_active, X, sigma2)

        # Construct signal from simulated coefs
        sim_smoothed = (X @ beta).T

        # Count peaks for each sim
        band_counts = np.zeros(nboot, dtype=int)
        sim_locations = []
        for i in range(nboot):
            smoothed_signal = sim_smoothed[i]
            locations = model_utils.count.find_peaks(smoothed_signal)
            sim_locations.append(locations)
            band_counts[i] = len(locations)
        
        return sim_locations, band_counts, sim_smoothed
    
    def simulate_posterior(self, nboot, n_active, X, sigma2):        
        # Simulate from posterior approximation which is proportional to a Gaussian multiplied by Laplacian prior
        # The rate parameter of lars_path is scaled so undo scaling here
        alpha = self.denoiser_info.alpha_ * len(self.signal) * 2 / (2 * sigma2)

        mu = np.linalg.solve(np.dot(X.T, X), X.T) @ self.signal

        # preds = X @ mu
        # sigma2 = np.sum((self.signal - preds)**2)/len(self.signal)
        S = sigma2 * np.linalg.inv(np.dot(X.T, X))

        # Model to simulate from
        class SMCBridge(ssp.TemperingBridge):
            # mallocs for calculations
            abs_malloc = np.zeros((nboot, n_active), dtype=float)
            sum_malloc = np.zeros(nboot, dtype=float)
            exp_dist = stats.expon(scale=1/alpha)
            norm_dist = stats.multivariate_normal(mean=mu, cov=S)

            # model-specific implementation of log-density to sample from
            def logtarget(self, theta):
                np.abs(theta, out=self.abs_malloc)
                np.sum(self.abs_malloc, axis=1, out=self.sum_malloc)
                return 0.5*self.exp_dist.logpdf(self.sum_malloc) + self.norm_dist.logpdf(theta)
            
            # particles package doesn't normally get you to redefine this,
            # but here we can avoid the expensive calculation; 
            # the default implementaion is loglik(theta) = logtarget(theta) - prior.logpdf(theta)
            # and logtarget = logpdf-normal + logpdf-exponential, so loglik = logpdf-exponential
            def loglik(self, theta):
                np.abs(theta, out=self.abs_malloc)
                np.sum(self.abs_malloc, axis=1, out=self.sum_malloc)
                return 0.5*self.exp_dist.logpdf(self.sum_malloc)

        base_dist = dists.MvNormal(loc=mu, cov=S)
        smc_bridge = SMCBridge(base_dist=base_dist)
        tempering_model = ssp.AdaptiveTempering(model=smc_bridge, len_chain=self._n_pf_mcmc_steps, wastefree=False)
        alg = particles.SMC(fk=tempering_model, N=nboot, verbose=False)
        alg.run()
        beta = alg.X.theta
        return beta.T
