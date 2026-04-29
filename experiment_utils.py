import matplotlib.pyplot as plt
import numpy as np
import yaml
import json
import scipy.stats as stats
import os
import rpy2.robjects
from rpy2.robjects import pandas2ri
import pandas as pd
import time
from scipy.stats import norm

import band_count

def parse_seeds_input(config_path, yaml_field):
    if isinstance(yaml_field, int):
        return [yaml_field]
    elif isinstance(yaml_field, list):
        return yaml_field
    elif isinstance(yaml_field, str):
        seed_file_name = os.path.join(os.path.dirname(config_path), yaml_field)
        with open(seed_file_name, "r") as io:
            line = io.readline()
            seeds_str = line.strip().split(',')
            return [int(seed) for seed in seeds_str]
    else:
        raise ValueError(f"Expected yaml field to be in list or str, got {type(yaml_field)}")

class Summary:
    lower_quantiles = [i for i in np.arange(0,0.5,0.025)]
    upper_quantiles = [1-i for i in np.arange(0,0.5,0.025)]

    def __init__(self):
        self.n_correct = 0
        self.n_in_posterior_range = 0
        self.n_in_posterior_95pct_range = 0
        self.n_in_posterior_90pct_range = 0
        self.n_in_posterior_85pct_range = 0
        self.n_in_posterior_80pct_range = 0
        self.n_in_posterior_75pct_range = 0
        self.n_in_posterior_70pct_range = 0
        self.n_in_posterior_65pct_range = 0
        self.n_in_posterior_60pct_range = 0
        self.n_in_posterior_55pct_range = 0
        self.n_in_posterior_50pct_range = 0
        self.n_in_posterior_45pct_range = 0
        self.n_in_posterior_40pct_range = 0
        self.n_in_posterior_35pct_range = 0
        self.n_in_posterior_30pct_range = 0
        self.n_in_posterior_25pct_range = 0
        self.n_in_posterior_20pct_range = 0
        self.n_in_posterior_15pct_range = 0
        self.n_in_posterior_10pct_range = 0
        self.n_in_posterior_5pct_range = 0
        self.n_in_bc_posterior_range = 0
        self.n_in_bc_posterior_95pct_range = 0
        self.n_in_bc_posterior_90pct_range = 0
        self.n_in_bc_posterior_85pct_range = 0
        self.n_in_bc_posterior_80pct_range = 0
        self.n_in_bc_posterior_75pct_range = 0
        self.n_in_bc_posterior_70pct_range = 0
        self.n_in_bc_posterior_65pct_range = 0
        self.n_in_bc_posterior_60pct_range = 0
        self.n_in_bc_posterior_55pct_range = 0
        self.n_in_bc_posterior_50pct_range = 0
        self.n_in_bc_posterior_45pct_range = 0
        self.n_in_bc_posterior_40pct_range = 0
        self.n_in_bc_posterior_35pct_range = 0
        self.n_in_bc_posterior_30pct_range = 0
        self.n_in_bc_posterior_25pct_range = 0
        self.n_in_bc_posterior_20pct_range = 0
        self.n_in_bc_posterior_15pct_range = 0
        self.n_in_bc_posterior_10pct_range = 0
        self.n_in_bc_posterior_5pct_range = 0
        self.average_distance_to_point_estimate = 0.0
        self.average_distance_squared_to_point_estimate = 0.0
        self.average_distance_abs_to_point_estimate = 0.0
        self.average_distance_to_posterior_mean = 0.0
        self.average_distance_squared_to_posterior_mean = 0.0
        self.average_distance_abs_to_posterior_mean = 0.0
        self.average_distance_to_posterior_mode = 0.0
        self.average_distance_squared_to_posterior_mode = 0.0
        self.average_distance_abs_to_posterior_mode = 0.0
        self.posterior_prob_of_true_value = []
        self.distance_to_point_estimate = []
        self.distance_to_posterior_mode = []
        self.distance_to_posterior_mean = []
        self.n = 0
        return 

    def bca_ci(self, theta_hat, boot_stats, alpha):
        """
        Approximate BCa bootstrap CI using bootstrap-based acceleration.

        Parameters
        ----------
        theta_hat : float
            Observed statistic
        boot_stats : array-like
            Bootstrap statistics
        alpha : float
            Significance level (e.g., 0.05 for 95% CI)

        Returns
        -------
        (lower, upper)
            BCa confidence interval
        """
        boot_stats = np.asarray(boot_stats)
        B = len(boot_stats)

        # -------------------------
        # 1. Bias correction (z0)
        # -------------------------
        less = np.sum(boot_stats < theta_hat)
        equal = np.sum(boot_stats == theta_hat)

        prop_less = (less + 0.5 * equal) / B

        eps = 1e-10
        prop_less = np.clip(prop_less, eps, 1 - eps)

        z0 = norm.ppf(prop_less)

        # -------------------------
        # 2. Acceleration (bootstrap-based)
        # -------------------------
        centered = boot_stats - np.mean(boot_stats)
        num = np.mean(centered**3)
        den = 6 * (np.mean(centered**2) ** 1.5)

        a = num / den if den != 0 else 0.0

        # -------------------------
        # 3. Adjusted alpha levels (FULL BCa)
        # -------------------------
        z_low = norm.ppf(alpha / 2)
        z_high = norm.ppf(1 - alpha / 2)

        def adj_alpha(z):
            num = z0 + z
            den = 1 - a * (z0 + z)
            return norm.cdf(z0 + num / den)

        alpha1 = adj_alpha(z_low)
        alpha2 = adj_alpha(z_high)

        # Clamp to valid range (important for extreme cases)
        alpha1 = np.clip(alpha1, 0, 1)
        alpha2 = np.clip(alpha2, 0, 1)

        # -------------------------
        # 4. Final CI
        # -------------------------
        lower = np.quantile(boot_stats, alpha1)
        upper = np.quantile(boot_stats, alpha2)

        return lower, upper

    def update(self, true_peak_count, estimate, counts_dist):
        
        estimate_locations, estimate_count = estimate
        self.n_correct += estimate_count == true_peak_count

        self.distance_to_point_estimate.append(estimate_count - true_peak_count)
        self.average_distance_to_point_estimate = (
            self.average_distance_to_point_estimate * (self.n/(self.n+1))
            + (estimate_count - true_peak_count)/(self.n+1)
        )
        self.average_distance_squared_to_point_estimate = (
            self.average_distance_squared_to_point_estimate * (self.n/(self.n+1))
            + (estimate_count - true_peak_count)**2/(self.n+1)
        )
        self.average_distance_abs_to_point_estimate = (
            self.average_distance_abs_to_posterior_mean * (self.n/(self.n+1))
            + np.abs(estimate_count - true_peak_count)/(self.n+1)
        )

        if counts_dist is not None and len(counts_dist)>0:
            lower_quantiles = np.quantile(counts_dist, self.lower_quantiles)
            upper_quantiles = np.quantile(counts_dist, self.upper_quantiles)

            is_between_quantiles = (true_peak_count >= lower_quantiles) & (true_peak_count <= upper_quantiles)

            self.n_in_posterior_range += is_between_quantiles[0]
            self.n_in_posterior_95pct_range += is_between_quantiles[1]
            self.n_in_posterior_90pct_range += is_between_quantiles[2]
            self.n_in_posterior_85pct_range += is_between_quantiles[3]
            self.n_in_posterior_80pct_range += is_between_quantiles[4]
            self.n_in_posterior_75pct_range += is_between_quantiles[5]
            self.n_in_posterior_70pct_range += is_between_quantiles[6]
            self.n_in_posterior_65pct_range += is_between_quantiles[7]
            self.n_in_posterior_60pct_range += is_between_quantiles[8]
            self.n_in_posterior_55pct_range += is_between_quantiles[9]
            self.n_in_posterior_50pct_range += is_between_quantiles[10]
            self.n_in_posterior_45pct_range += is_between_quantiles[11]
            self.n_in_posterior_40pct_range += is_between_quantiles[12]
            self.n_in_posterior_35pct_range += is_between_quantiles[13]
            self.n_in_posterior_30pct_range += is_between_quantiles[14]
            self.n_in_posterior_25pct_range += is_between_quantiles[15]
            self.n_in_posterior_20pct_range += is_between_quantiles[16]
            self.n_in_posterior_15pct_range += is_between_quantiles[17]
            self.n_in_posterior_10pct_range += is_between_quantiles[18]
            self.n_in_posterior_5pct_range += is_between_quantiles[19]

            assert np.all(is_between_quantiles[:-1] >= is_between_quantiles[1:])

            bc_lower_quantiles = [counts_dist.min()]
            bc_upper_quantiles = [counts_dist.max()]

            for i,_ in enumerate(self.lower_quantiles):
                if i==0:
                    continue
                alpha = 1 - (self.upper_quantiles[i]-self.lower_quantiles[i])
                bc_lower_quantile, bc_upper_quantile = self.bca_ci(estimate_count, counts_dist, alpha)
                bc_lower_quantiles.append(bc_lower_quantile)
                bc_upper_quantiles.append(bc_upper_quantile)
            
            bc_lower_quantiles = np.asarray(bc_lower_quantiles)
            bc_upper_quantiles = np.asarray(bc_upper_quantiles)

            is_between_bc_quantiles = (true_peak_count >= bc_lower_quantiles) & (true_peak_count <= bc_upper_quantiles)
            
            self.n_in_bc_posterior_range += is_between_bc_quantiles[0]
            self.n_in_bc_posterior_95pct_range += is_between_bc_quantiles[1]
            self.n_in_bc_posterior_90pct_range += is_between_bc_quantiles[2]
            self.n_in_bc_posterior_85pct_range += is_between_bc_quantiles[3]
            self.n_in_bc_posterior_80pct_range += is_between_bc_quantiles[4]
            self.n_in_bc_posterior_75pct_range += is_between_bc_quantiles[5]
            self.n_in_bc_posterior_70pct_range += is_between_bc_quantiles[6]
            self.n_in_bc_posterior_65pct_range += is_between_bc_quantiles[7]
            self.n_in_bc_posterior_60pct_range += is_between_bc_quantiles[8]
            self.n_in_bc_posterior_55pct_range += is_between_bc_quantiles[9]
            self.n_in_bc_posterior_50pct_range += is_between_bc_quantiles[10]
            self.n_in_bc_posterior_45pct_range += is_between_bc_quantiles[11]
            self.n_in_bc_posterior_40pct_range += is_between_bc_quantiles[12]
            self.n_in_bc_posterior_35pct_range += is_between_bc_quantiles[13]
            self.n_in_bc_posterior_30pct_range += is_between_bc_quantiles[14]
            self.n_in_bc_posterior_25pct_range += is_between_bc_quantiles[15]
            self.n_in_bc_posterior_20pct_range += is_between_bc_quantiles[16]
            self.n_in_bc_posterior_15pct_range += is_between_bc_quantiles[17]
            self.n_in_bc_posterior_10pct_range += is_between_bc_quantiles[18]
            self.n_in_bc_posterior_5pct_range += is_between_bc_quantiles[19]

            self.distance_to_posterior_mean.append(np.mean(counts_dist) - true_peak_count)
            self.distance_to_posterior_mode.append(stats.mode(counts_dist).mode - true_peak_count)
            
            self.average_distance_to_posterior_mean = (
                self.average_distance_to_posterior_mean * (self.n/(self.n+1))
                + (np.mean(counts_dist) - true_peak_count)/(self.n+1)
            )
            self.average_distance_squared_to_posterior_mean = (
                self.average_distance_squared_to_posterior_mean * (self.n/(self.n+1))
                + (np.mean(counts_dist) - true_peak_count)**2/(self.n+1)
            )
            self.average_distance_abs_to_posterior_mean = (
                self.average_distance_abs_to_posterior_mean * (self.n/(self.n+1))
                + np.abs(np.mean(counts_dist) - true_peak_count)/(self.n+1)
            )
            self.average_distance_to_posterior_mode = (
                self.average_distance_to_posterior_mode * (self.n/(self.n+1))
                + (stats.mode(counts_dist).mode - true_peak_count)/(self.n+1)
            )
            self.average_distance_squared_to_posterior_mode = (
                self.average_distance_squared_to_posterior_mode * (self.n/(self.n+1))
                + (stats.mode(counts_dist).mode - true_peak_count)**2/(self.n+1)
            )
            self.average_distance_abs_to_posterior_mode = (
                self.average_distance_abs_to_posterior_mode * (self.n/(self.n+1))
                + np.abs(stats.mode(counts_dist).mode - true_peak_count)/(self.n+1)
            )
            self.posterior_prob_of_true_value.append(
                np.sum(counts_dist == true_peak_count)/len(counts_dist)
            )

        self.n +=1 
        return

class GAMDenoiserInfo:
    coef_ = []
    variance_estimate = None
    
    def __init__(self, fitted):
        self.reconstructed = fitted
        return 
    
class GAMDenoiser:
    _exists = False

    _init_script = """
        # Load mgcv
        library(mgcv)    

        simulate_gam <- NULL
        parametric_bootstrap <- NULL
        residual_bootstrap <- NULL
    """

    _reset_script = """
        df <- NULL
        k <- NULL
        gam_model <- NULL
        smoothed <- NULL
        y_hat_sims <- NULL   
        seed <- NULL
        gc()     
    """

    _fit_script = """
        # Fit GAM in R: 1D thin plate spline smoother
        gam_model <- gam(y ~ s(x, bs="tp", k=k), data=df)
        smoothed <- predict(gam_model)
        # returns
        smoothed
    """

    _posterior_simulation_fn_script = """
        simulate_gam <- function(gam_obj, seed, nsim = 1) {
            set.seed(seed)
            # draw multivariate normal beta:
            Rbeta <- rmvn(nsim,
                            coef(gam_obj),
                            vcov(gam_obj, unconditional = TRUE))
            Xp <- predict(gam_obj, type = "lpmatrix")
            sims <- Xp %*% t(Rbeta)
            sims
        }
    """
    
    _smoother_dist_posterior_method_script = """
        y_hat_sims <- simulate_gam(gam_model, seed=seed, nsim=n_sim)
        # returns 
        y_hat_sims
    """

    _smoother_dist_parametric_boot_fn_script = """
        parametric_bootstrap <- function(gam_object, df, seed, n_sim = 1) {
            set.seed(seed)
            y_sims <- simulate(gam_object, nsim=n_sim, unconditional=FALSE)
            y_hat_sims <- matrix(nrow = nrow(df), ncol = n_sim)
            for( i in 1:n_sim ){
                y <- y_sims[,i]
                gam_fit <- gam(y ~ s(df$x, bs="tp", k=k))
                y_hat_sims[,i] <- predict(gam_fit)
            }
            y_hat_sims <- y_hat_sims
            # returns 
            y_hat_sims
        }
    """

    _smoother_dist_parametric_boot_script = """
        y_hat_sims <- parametric_bootstrap(gam_model, df=df, seed=seed, nsim=n_sim)
        # returns 
        y_hat_sims
    """

    _smoother_dist_redidual_boot_fn_script = """
        residual_bootstrap <- function(gam_object, df, seed, n_sim=1) {
            set.seed(seed)
            fitted <- predict(gam_object)
            k <- gam_object$smooth$bs.dim[1]
            h <- influence(gam_object)
            # modified residuals
            resids <- (df$y - fitted)/sqrt(1-h)
            resids <- resids - mean(resids)
            y_hat_sims <- matrix(nrow = nrow(df), ncol = n_sim)
            for( i in 1:n_sim ){
                y <- fitted + sample(resids, size=nrow(df), replace=TRUE)
                gam_fit <- gam(y ~ s(df$x, bs="tp", k=k))
                y_hat_sims[,i] <- predict(gam_fit)
            }
            y_hat_sims <- y_hat_sims
            # returns 
            y_hat_sims
        }
    """

    _smoother_dist_residual_boot_script = """
        y_hat_sims <- residual_bootstrap(gam_model, df=df, seed=seed, n_sim=n_sim)
        # returns 
        y_hat_sims
    """

    def __init__(self, k=None):
        if self._exists:
            raise RuntimeError("Only one instance of GAMDenoiser can be instantiated")
        
        # insantiate r functions and variables
        rpy2.robjects.r(self._init_script)
        rpy2.robjects.r(self._reset_script)
        rpy2.robjects.r(self._posterior_simulation_fn_script)
        rpy2.robjects.r(self._smoother_dist_parametric_boot_fn_script)
        rpy2.robjects.r(self._smoother_dist_redidual_boot_fn_script)
        if k is None:
            k = np.inf
        self.k = k
        self._exists = True
        return
    
    def fit(self, signal):
        # clean up objects
        rpy2.robjects.r(self._reset_script)

        t = np.arange(0, len(signal))
        df = pd.DataFrame({"x": t, "y": signal})
        with rpy2.robjects.conversion.localconverter(rpy2.robjects.default_converter 
                                                     + pandas2ri.converter):
            rpy2.robjects.globalenv['df'] = rpy2.robjects.conversion.py2rpy(df)
            rpy2.robjects.globalenv['k'] = min(self.k, len(signal)//2)
            smoothed = np.array(rpy2.robjects.r(self._fit_script))

        return GAMDenoiserInfo(smoothed)
    
    def simulate_smooths(self, signal, seed, n_sim, method):
        # clean up objects
        rpy2.robjects.r(self._reset_script)

        self.fit(signal)
        
        with rpy2.robjects.conversion.localconverter(rpy2.robjects.default_converter 
                                                     + pandas2ri.converter):
            rpy2.robjects.globalenv['seed'] = seed
            rpy2.robjects.globalenv['n_sim'] = n_sim

            match method:
                case 'posterior sim':
                    smoothed_boot = rpy2.robjects.r(self._smoother_dist_posterior_method_script)
                case 'parametric boot':
                    smoothed_boot = rpy2.robjects.r(self._smoother_dist_parametric_boot_script)
                case 'residual boot':
                    smoothed_boot = rpy2.robjects.r(self._smoother_dist_residual_boot_script)
                case _:
                    raise ValueError('Unknown simulate method')
        return smoothed_boot.T
    
class GAMBandCounter(band_count.BandCounter):
    def __init__(self, signal, max_age):
        assert len(signal.shape)==1, f"Expected signal to be 1-d array, for shape {signal.shape}."

        # de-mean signal
        self.signal = np.asarray(signal.copy(), dtype=np.float64)
        self.signal -= np.mean(self.signal)

        self.max_age = min(max_age, len(signal)//2)
        self.denoiser = GAMDenoiser(k=max_age)

        self.smoothed = None
        self.low_freq_smoothed = None
        self.denoiser_info = None
        return
    
    def get_smoothed(self, filter=False):
        # filter arg ignored for GAM method
        filter = False
        return super().get_smoothed(filter=filter)
    
    def get_count_distribution(self, nboot, filter=False, seed=None, boot_method=None):
        filter = False
        if seed is None:
            seed = np.random.randint(1,2**21)
        smoothed_boot = self.denoiser.simulate_smooths(self.signal, seed, nboot, boot_method)

        locations_boot = []
        band_count_boot = np.zeros(nboot, dtype=int)

        for i in range(nboot):
            smooth = smoothed_boot[i]

            locations = band_count.model_utils.count.find_peaks(smooth)
            count = len(locations)

            locations_boot.append(locations)
            band_count_boot[i] = count
        
        return locations_boot, band_count_boot, smoothed_boot

def check_experiment_parameters(file_path, cfg):
    path_elts = file_path.split("/")
    file_name = path_elts[-1]
    
    path_parameters = {}
    path_parameters["max_corr"] = None
    for elt in path_elts[:-1]:
        if elt=="sin":
            path_parameters["signal type"] = "sin"
        elif elt=="lin" or (len(elt)>len("lin_peaksize") and elt[:len("lin_peaksize")]=="lin_peaksize") or elt=="lin1" or elt=="lin2" or elt=="lin3":
            path_parameters["signal type"] = "lin"
            if len(elt)>len("lin_peaksize"):
                path_parameters["min_peak_size"] = float( elt[len("lin_peaksize"):])
        elif elt=="gp" or (len(elt)>len("gp_peaksize") and elt[:len("gp_peaksize")]=="gp_peaksize") or elt=="gp1" or elt=="gp2" or elt=="gp3" or elt=="gp4":
            path_parameters["signal type"] = "gp"
            if len(elt)>len("gp_peaksize"):
                path_parameters["min_peak_size"] = float( elt[len("gp_peaksize"):])
        elif elt=="var0.16" or elt=="var0.64" or elt=="var0.36":
            path_parameters["var"] = float(elt[3:])
        elif elt=="peaks4":
            path_parameters["freq"] = 4
            path_parameters["chirp"] = 0
        elif elt=="peaks8":
            path_parameters["freq"] = 8
            path_parameters["chirp"] = 0
        elif elt=="peaks16":
            path_parameters["freq"] = 16
            path_parameters["chirp"] = 0
        elif elt=="freq8":
            path_parameters["freq"] = 8
            path_parameters["chirp"] = 0
        elif elt=="freq16":
            path_parameters["freq"] = 16
            path_parameters["chirp"] = 0
        elif elt=="freq32":
            path_parameters["freq"] = 32
            path_parameters["chirp"] = 0
        elif elt=="multiple" or elt=="multiple2":
            path_parameters["freq"] = "multiple"
            path_parameters["chirp"] = 0
        elif elt=="chirp":
            path_parameters["chirp"] = 6
            path_parameters["freq"] = 4
        elif len(elt)>4 and elt[:4]=="corr":
            if cfg["signal_generator"]["method"]["name"]=="sinusoids":
                if len(cfg["signal_generator"]["method"]["frequencies"]) == 1:
                    path_parameters["freq"] = cfg["signal_generator"]["method"]["frequencies"][0]
                else: 
                    path_parameters["freq"] = "multiple"
                path_parameters["chirp"] = cfg["signal_generator"]["method"]["chirp_rate"]
            elif cfg["signal_generator"]["method"]["name"]=="piecewise_linear":
                path_parameters["freq"] = cfg["signal_generator"]["method"]["n_peaks"]
                path_parameters["chirp"] = 0
            elif cfg["signal_generator"]["method"]["name"]=="gaussian_process":
                path_parameters["freq"] = "multiple"
                path_parameters["chirp"] = 0

            path_parameters["max_corr"] = float(elt[4:])
            path_parameters["min_peak_size"] = 0.6
            path_parameters["var"] = 0.36
            path_parameters["length"] = cfg["signal_generator"]["length"]

    
    filename_parameters = {}
    filename_elts = file_name.split(".")
    # assert filename_elts[0]=="config"
    assert filename_elts[-1]=="yaml"
    filename_elts = ".".join(filename_elts[1:-1])
    filename_elts = filename_elts.split("_")
    filename_parameters["max_corr"] = None
    for elt in filename_elts:
        if elt=="sin":
            filename_parameters["signal type"] = "sin"
        elif elt=="lin":
            filename_parameters["signal type"] = "lin"
        elif elt=="gp":
            filename_parameters["signal type"] = "gp"
        elif elt=="var0.16" or elt=="var0.64" or elt=="var0.36":
            filename_parameters["var"] = float(elt[3:])
        elif elt=="peaks4":
            filename_parameters["freq"] = 4
            filename_parameters["chirp"] = 0
        elif elt=="peaks8":
            filename_parameters["freq"] = 8
            filename_parameters["chirp"] = 0
        elif elt=="peaks16":
            filename_parameters["freq"] = 16
            filename_parameters["chirp"] = 0
        elif elt=="freqs8":
            filename_parameters["freq"] = 8
            filename_parameters["chirp"] = 0
        elif elt=="freqs16":
            filename_parameters["freq"] = 16
            filename_parameters["chirp"] = 0
        elif elt=="freqs32":
            filename_parameters["freq"] = 32
            filename_parameters["chirp"] = 0
        elif elt=="freqsmany":
            filename_parameters["freq"] = "multiple"
            filename_parameters["chirp"] = 0
        elif elt=="chirp":
            filename_parameters["freq"] = 4
            filename_parameters["chirp"] = 6
        elif elt=="len128" or elt=="length128":
            filename_parameters["length"] = 128
        elif elt=="len256" or elt=="length256":
            filename_parameters["length"] = 256
        elif elt=="len512" or elt=="length512":
            filename_parameters["length"] = 512
        elif elt=="len1024" or elt=="length1024":
            filename_parameters["length"] = 1024
        elif elt=="len2048" or elt=="length2048":
            filename_parameters["length"] = 2048
        elif elt=="BPDN":
            filename_parameters["method"] = "BPDN"
        elif elt=="TP":
            filename_parameters["method"] = "TP"
        elif len(elt)>4 and elt[:4]=="corr":
            filename_parameters["max_corr"] = float(elt[4:])
            filename_parameters["min_peak_size"] = 0.6
            filename_parameters["length"] = cfg["signal_generator"]["length"]
    
    cfg_parameters = {}
    if cfg["signal_generator"]["method"]["name"]=="sinusoids":
        cfg_parameters["signal type"] = "sin"
        if len(cfg["signal_generator"]["method"]["frequencies"]) == 1:
            cfg_parameters["freq"] = cfg["signal_generator"]["method"]["frequencies"][0]
        else: 
            cfg_parameters["freq"] = "multiple"
        cfg_parameters["chirp"] = cfg["signal_generator"]["method"]["chirp_rate"]
    elif cfg["signal_generator"]["method"]["name"]=="piecewise_linear":
        cfg_parameters["signal type"] = "lin"
        cfg_parameters["freq"] = cfg["signal_generator"]["method"]["n_peaks"]
        cfg_parameters["chirp"] = 0
        assert cfg["signal_generator"]["method"]["peak_size_shape"] == 1
        assert cfg["signal_generator"]["method"]["peak_size_rate"] == 1
    elif cfg["signal_generator"]["method"]["name"]=="gaussian_process":
        cfg_parameters["signal type"] = "gp"
        cfg_parameters["freq"] = "multiple"
        cfg_parameters["chirp"] = 0
    elif cfg["signal_generator"]["method"]["name"]=="read_from_file":
        cfg_parameters["signal type"] = filename_parameters["signal type"]
        cfg_parameters["freq"] = filename_parameters["freq"]
        cfg_parameters["chirp"] = filename_parameters["chirp"]
        cfg_parameters["length"] = filename_parameters["length"]
    cfg_parameters["length"] = cfg["signal_generator"]["length"]
    cfg_parameters["var"] = cfg["signal_generator"]["noise"]["variance"]
    cfg_parameters["method"] = cfg["inference"]["smooth_method"]
    if cfg_parameters["method"]=="BPDN":
        assert cfg["inference"]["sim_method"] == "ols"
        assert cfg["inference"]["max_bands"] == 20
    elif cfg_parameters["method"]=="TP":
        assert cfg["inference"]["sim_method"] == "residual boot"
        assert cfg["inference"]["max_bands"] == 32
    # assert (cfg["inference"]["n_sims"] == 400 or cfg["inference"]["n_sims"] == 0)
    cfg_parameters["max_corr"] = None
    if "max_corr" in cfg["inference"]:
        cfg_parameters["max_corr"] = cfg["inference"]["max_corr"]
    # assert all fields match
    assert filename_parameters["signal type"] == cfg_parameters["signal type"] == path_parameters["signal type"], f"file path: {file_path}\nfilename_parameters[\"signal type\"] == cfg_parameters[\"signal type\"] == path_parameters[\"signal type\"]: {filename_parameters["signal type"]} == {cfg_parameters["signal type"]} == {path_parameters["signal type"]}"
    assert filename_parameters["var"]         == cfg_parameters["var"],                                           f"file path: {file_path}\nfilename_parameters[\"var\"]         == cfg_parameters[\"var\"]         == path_parameters[\"var\"] {filename_parameters["var"]}         == {cfg_parameters["var"]}         == {path_parameters["var"]}"
    assert filename_parameters["freq"]        == cfg_parameters["freq"],                                          f"file path: {file_path}\nfilename_parameters[\"freq\"]        == cfg_parameters[\"freq\"]        == path_parameters[\"freq\"]: {filename_parameters["freq"]}        == {cfg_parameters["freq"]}        == {path_parameters["freq"]}"
    assert filename_parameters["chirp"]       == cfg_parameters["chirp"],                                         f"file path: {file_path}\nfilename_parameters[\"chirp\"]       == cfg_parameters[\"chirp\"]       == path_parameters[\"chirp\"]: {filename_parameters["chirp"]}       == {cfg_parameters["chirp"]}       == {path_parameters["chirp"]}"
    assert filename_parameters["length"]      == cfg_parameters["length"],                                        f"file path: {file_path}\nfilename_parameters[\"length\"]      == cfg_parameters[\"length\"]: {filename_parameters["length"]}      == {cfg_parameters["length"]}"
    assert filename_parameters["method"]      == cfg_parameters["method"],                                        f"file path: {file_path}\nfilename_parameters[\"method\"]      == cfg_parameters[\"method\"]: {filename_parameters["method"]}      == {cfg_parameters["method"]}"
    # assert filename_parameters["max_corr"]    == cfg_parameters["max_corr"],    f"file path: {file_path}\nfilename_parameters[\"max_corr\"]    == cfg_parameters[\"max_corr\"]: {filename_parameters["max_corr"]}      == {cfg_parameters["max_corr"]}   == {path_parameters["max_corr"]}" 
    

def run_experiment(config_path, outpath):
    experiment_start_time = time.time()
    smooth_time = 0
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    check_experiment_parameters(config_path, cfg)

    sig_gen_cfg = cfg["signal_generator"]

    if "rng_seed" in sig_gen_cfg["method"].keys():
        sig_gen_seeds = parse_seeds_input(config_path, sig_gen_cfg["method"]["rng_seed"])
    else:
        sig_gen_seeds = None
    noise_gen_seeds = parse_seeds_input(config_path, sig_gen_cfg["noise"]["rng_seed"])
    posterior_inference_seeds = parse_seeds_input(config_path, cfg["inference"]["rng_seed"])

    assert sig_gen_seeds is None or (len(noise_gen_seeds)==len(posterior_inference_seeds)==len(sig_gen_seeds)), "Expected the same number of seeds for each rng"
    assert len(noise_gen_seeds)==len(posterior_inference_seeds), "Expected the same number of seeds for each rng"
    
    match cfg["inference"]["smooth_method"]:
        case 'BPDN':
            if "wavelet" in cfg["inference"].keys():
                match cfg["inference"]["wavelet"]:
                    case "ricker":
                        wavelet = (band_count.denoising.ricker,)
                    case "morlet1":
                        wavelet = (band_count.denoising.morlet1,)
                    case "morlet2":
                        wavelet = (band_count.denoising.morlet2,)
                    case "morlet3":
                        wavelet = (band_count.denoising.morlet3,)
                    case "morlet4":
                        wavelet = (band_count.denoising.morlet4,)
                    case "morlet5":
                        wavelet = (band_count.denoising.morlet5,)
                    case "morlet6":
                        wavelet = (band_count.denoising.morlet6,)
                    case "morlet7":
                        wavelet = (band_count.denoising.morlet7,)
                    case "morlet8":
                        wavelet = (band_count.denoising.morlet8,)
                    case "morlet9":
                        wavelet = (band_count.denoising.morlet9,)
                    case "morlet10":
                        wavelet = (band_count.denoising.morlet10,)
                    case "morlet11":
                        wavelet = (band_count.denoising.morlet11,)
                    case "morlet12":
                        wavelet = (band_count.denoising.morlet12,)
                    case "morlet13":
                        wavelet = (band_count.denoising.morlet13,)
                    case "morlet14":
                        wavelet = (band_count.denoising.morlet14,)
                    case "morlet15":
                        wavelet = (band_count.denoising.morlet15,)
                    case "morlet16":
                        wavelet = (band_count.denoising.morlet16,)
                    case "morlet17":
                        wavelet = (band_count.denoising.morlet17,)
                    case "morlet18":
                        wavelet = (band_count.denoising.morlet18,)
                    case "morlet19":
                        wavelet = (band_count.denoising.morlet19,)
                    case "morlet20":
                        wavelet = (band_count.denoising.morlet20,)
                    case "morlet21":
                        wavelet = (band_count.denoising.morlet21,)
                    case "morlet22":
                        wavelet = (band_count.denoising.morlet22,)
                    case "morlet23":
                        wavelet = (band_count.denoising.morlet23,)
                    case "morlet24":
                        wavelet = (band_count.denoising.morlet24,)
                    case "morlet48":
                        wavelet = (band_count.denoising.morlet48,)
            else:
                wavelet = None

            if "max_corr" in cfg["inference"]:
                max_corr = max_corr=cfg["inference"]["max_corr"]
                if max_corr >= 1:
                    max_corr = None
            else:
                max_corr = 0.8
            fft_start_time = time.time()
            counter = band_count.BandCounter(np.zeros(sig_gen_cfg["length"], dtype=float), 
                                             max_bands=cfg["inference"]["max_bands"], wavelets=wavelet,
                                             max_corr=max_corr)
            fft_end_time = time.time()
            fft_elapsed = fft_end_time-fft_start_time
            dict_mem = counter.denoiser.get_X().nbytes
        case 'TP':
            counter = GAMBandCounter(np.zeros(sig_gen_cfg["length"], dtype=float),
                                     max_age=cfg["inference"]["max_bands"])
            fft_elapsed = 0
            dict_mem = 0
        case _:
            raise ValueError("Error in config. Unexpected smooth_method entry.")

    summary = Summary()
    
    for seed_idx in range(len(noise_gen_seeds)):
        match sig_gen_cfg["method"]["name"]:
            case "sinusoids":
                signal = generate_chirp(sig_gen_cfg["length"], 
                                        sig_gen_cfg["method"]["frequencies"], 
                                        sig_gen_cfg["method"]["amplitudes"],
                                        sig_gen_cfg["method"]["phases"],
                                        sig_gen_cfg["method"]["chirp_rate"])
            case "piecewise_linear":
                if "min_peak_size" in sig_gen_cfg["method"].keys():
                    min_peak_size = sig_gen_cfg["method"]["min_peak_size"]
                else:
                    min_peak_size = -1
                signal = generate_piecewise_linear_signal(sig_gen_cfg["length"], 
                                                          sig_gen_seeds[seed_idx],
                                                          sig_gen_cfg["method"]["n_peaks"],
                                                          sig_gen_cfg["method"]["peak_size_shape"],
                                                          sig_gen_cfg["method"]["peak_size_rate"],
                                                          sig_gen_cfg["method"]["min_distance_between_peaks"],
                                                          min_peak_size=min_peak_size)
            case "gaussian_process":
                if "min_peak_size" in sig_gen_cfg["method"].keys():
                    min_peak_size = sig_gen_cfg["method"]["min_peak_size"]
                else:
                    min_peak_size = -1
                signal = generate_gaussian_process_signal(sig_gen_cfg["length"], 
                                                          sig_gen_seeds[seed_idx],
                                                          sig_gen_cfg["method"]["exp2_kernel"]["sigma2"],
                                                          sig_gen_cfg["method"]["exp2_kernel"]["length_scale"],
                                                          sig_gen_cfg["method"]["periodic_kernel"]["sigma2"],
                                                          sig_gen_cfg["method"]["periodic_kernel"]["length_scale"],
                                                          sig_gen_cfg["method"]["periodic_kernel"]["period"],
                                                          min_peak_size)
            case "read_from_file":
                signal = read_signal_from_file(sig_gen_cfg["method"]["file_elts"], seed_idx)
        
        match sig_gen_cfg["noise"]["type"]:
            case "gaussian":
                noise = generate_correlated_gaussian_noise(sig_gen_cfg["length"], 
                                                           noise_gen_seeds[seed_idx], 
                                                           sig_gen_cfg["noise"]["variance"],
                                                           sig_gen_cfg["noise"]["correlation"])
            case "signal_dependent_gaussian":
                noise = generate_signal_dependent_noise(sig_gen_cfg["length"],
                                                        noise_gen_seeds[seed_idx],
                                                        sig_gen_cfg["noise"]["variance"],
                                                        signal)

        noisy_signal = signal + noise
        
        true_peak_locations = band_count.model_utils.count.find_peaks(signal)
        true_peak_count = len(true_peak_locations)
        smooth_start_time = time.time()
        counter.set_signal(noisy_signal)
        estimate = counter.get_count_estimate(True)
        smooth_end_time = time.time()
        smooth_time += smooth_end_time - smooth_start_time
        # smth = counter.get_smoothed(True)
        # plt.plot(smth.smoothed)
        # plt.show()
        
        if cfg["inference"]["n_sims"]>0:
            locations_dist, counts_dist, smoothed_dist = counter.get_count_distribution(
                cfg["inference"]["n_sims"], True, posterior_inference_seeds[seed_idx], cfg["inference"]["sim_method"]
            )
        else:
            locations_dist, counts_dist, smoothed_dist = None, None, None

        summary.update(true_peak_count, estimate, counts_dist)

        write_experiement(outpath, seed_idx, signal, noisy_signal, true_peak_locations, true_peak_count, estimate, locations_dist, counts_dist, smoothed_dist)
        make_plots(outpath, cfg, counter, seed_idx, signal, noisy_signal, true_peak_locations, true_peak_count, counts_dist, smoothed_dist)
    
    experiment_end_time = time.time()
    elapsed =  experiment_end_time - experiment_start_time
    write_summary(outpath, summary, smooth_time, fft_elapsed, elapsed, dict_mem, posterior_inference_seeds[seed_idx])

    plt.close('all')
    return

def bootstrap_mean_ci(data, n_bootstrap=5000, ci=90, random_state=None):
    rng = np.random.default_rng(random_state)
    data = np.array(data)
    
    # Generate bootstrap samples and compute means
    means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        means.append(sample.mean())
    
    means = np.array(means)
    
    # Compute confidence interval
    alpha = 100 - ci
    lower = np.percentile(means, alpha / 2)
    upper = np.percentile(means, 100 - alpha / 2)
    
    return lower, upper

def write_summary(outpath, summary: Summary, smooth_elapsed, fft_elapsed, total_elapsed, dict_mem, seed):
    with open(os.path.join(outpath, f"summary.txt"), "w") as f:
        f.write(f"correct: {summary.n_correct/summary.n} ({summary.n_correct}/{summary.n}).\n")
        f.write(f"in posterior range: {summary.n_in_posterior_range/summary.n} ({summary.n_in_posterior_range}/{summary.n}).\n")
        f.write(f"in posterior 95% range: {summary.n_in_posterior_95pct_range/summary.n} ({summary.n_in_posterior_95pct_range}/{summary.n}).\n")
        f.write(f"in posterior 90% range: {summary.n_in_posterior_90pct_range/summary.n} ({summary.n_in_posterior_90pct_range}/{summary.n}).\n")
        f.write(f"in posterior 85% range: {summary.n_in_posterior_85pct_range/summary.n} ({summary.n_in_posterior_85pct_range}/{summary.n}).\n")
        f.write(f"in posterior 80% range: {summary.n_in_posterior_80pct_range/summary.n} ({summary.n_in_posterior_80pct_range}/{summary.n}).\n")
        f.write(f"in posterior 75% range: {summary.n_in_posterior_75pct_range/summary.n} ({summary.n_in_posterior_75pct_range}/{summary.n}).\n")
        f.write(f"in posterior 70% range: {summary.n_in_posterior_70pct_range/summary.n} ({summary.n_in_posterior_70pct_range}/{summary.n}).\n")
        f.write(f"in posterior 65% range: {summary.n_in_posterior_65pct_range/summary.n} ({summary.n_in_posterior_65pct_range}/{summary.n}).\n")
        f.write(f"in posterior 60% range: {summary.n_in_posterior_60pct_range/summary.n} ({summary.n_in_posterior_60pct_range}/{summary.n}).\n")
        f.write(f"in posterior 55% range: {summary.n_in_posterior_55pct_range/summary.n} ({summary.n_in_posterior_55pct_range}/{summary.n}).\n")
        f.write(f"in posterior 50% range: {summary.n_in_posterior_50pct_range/summary.n} ({summary.n_in_posterior_50pct_range}/{summary.n}).\n")
        f.write(f"in posterior 45% range: {summary.n_in_posterior_45pct_range/summary.n} ({summary.n_in_posterior_45pct_range}/{summary.n}).\n")
        f.write(f"in posterior 40% range: {summary.n_in_posterior_40pct_range/summary.n} ({summary.n_in_posterior_40pct_range}/{summary.n}).\n")
        f.write(f"in posterior 35% range: {summary.n_in_posterior_35pct_range/summary.n} ({summary.n_in_posterior_35pct_range}/{summary.n}).\n")
        f.write(f"in posterior 30% range: {summary.n_in_posterior_30pct_range/summary.n} ({summary.n_in_posterior_30pct_range}/{summary.n}).\n")
        f.write(f"in posterior 25% range: {summary.n_in_posterior_25pct_range/summary.n} ({summary.n_in_posterior_25pct_range}/{summary.n}).\n")
        f.write(f"in posterior 20% range: {summary.n_in_posterior_20pct_range/summary.n} ({summary.n_in_posterior_20pct_range}/{summary.n}).\n")
        f.write(f"in posterior 15% range: {summary.n_in_posterior_15pct_range/summary.n} ({summary.n_in_posterior_15pct_range}/{summary.n}).\n")
        f.write(f"in posterior 10% range: {summary.n_in_posterior_10pct_range/summary.n} ({summary.n_in_posterior_10pct_range}/{summary.n}).\n")
        f.write(f"in posterior 5% range: {summary.n_in_posterior_5pct_range/summary.n} ({summary.n_in_posterior_5pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior range: {summary.n_in_bc_posterior_range/summary.n} ({summary.n_in_bc_posterior_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 95% range: {summary.n_in_bc_posterior_95pct_range/summary.n} ({summary.n_in_bc_posterior_95pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 90% range: {summary.n_in_bc_posterior_90pct_range/summary.n} ({summary.n_in_bc_posterior_90pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 85% range: {summary.n_in_bc_posterior_85pct_range/summary.n} ({summary.n_in_bc_posterior_85pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 80% range: {summary.n_in_bc_posterior_80pct_range/summary.n} ({summary.n_in_bc_posterior_80pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 75% range: {summary.n_in_bc_posterior_75pct_range/summary.n} ({summary.n_in_bc_posterior_75pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 70% range: {summary.n_in_bc_posterior_70pct_range/summary.n} ({summary.n_in_bc_posterior_70pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 65% range: {summary.n_in_bc_posterior_65pct_range/summary.n} ({summary.n_in_bc_posterior_65pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 60% range: {summary.n_in_bc_posterior_60pct_range/summary.n} ({summary.n_in_bc_posterior_60pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 55% range: {summary.n_in_bc_posterior_55pct_range/summary.n} ({summary.n_in_bc_posterior_55pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 50% range: {summary.n_in_bc_posterior_50pct_range/summary.n} ({summary.n_in_bc_posterior_50pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 45% range: {summary.n_in_bc_posterior_45pct_range/summary.n} ({summary.n_in_bc_posterior_45pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 40% range: {summary.n_in_bc_posterior_40pct_range/summary.n} ({summary.n_in_bc_posterior_40pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 35% range: {summary.n_in_bc_posterior_35pct_range/summary.n} ({summary.n_in_bc_posterior_35pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 30% range: {summary.n_in_bc_posterior_30pct_range/summary.n} ({summary.n_in_bc_posterior_30pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 25% range: {summary.n_in_bc_posterior_25pct_range/summary.n} ({summary.n_in_bc_posterior_25pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 20% range: {summary.n_in_bc_posterior_20pct_range/summary.n} ({summary.n_in_bc_posterior_20pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 15% range: {summary.n_in_bc_posterior_15pct_range/summary.n} ({summary.n_in_bc_posterior_15pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 10% range: {summary.n_in_bc_posterior_10pct_range/summary.n} ({summary.n_in_bc_posterior_10pct_range}/{summary.n}).\n")
        f.write(f"in bias corrected posterior 5% range: {summary.n_in_bc_posterior_5pct_range/summary.n} ({summary.n_in_bc_posterior_5pct_range}/{summary.n}).\n")
        f.write(f"average distance to point estimate: {summary.average_distance_to_point_estimate} ({summary.average_distance_to_point_estimate}).\n")
        f.write(f"average distance squared to point estimate: {summary.average_distance_squared_to_point_estimate} ({summary.average_distance_squared_to_point_estimate}).\n")
        f.write(f"average distance abs to point estimate: {summary.average_distance_abs_to_point_estimate} ({summary.average_distance_abs_to_point_estimate}).\n")
        f.write(f"smoothing run time: {smooth_elapsed} seconds.\n")
        f.write(f"dict build run time: {fft_elapsed} seconds.\n")
        f.write(f"total run time: {total_elapsed} seconds.\n")
        f.write(f"run time: {total_elapsed} seconds.\n") # for legacy reasons
        f.write(f"Dict memory: {dict_mem} bytes.\n")
        
        bias_lwr_ci, bias_upr_ci = bootstrap_mean_ci(summary.distance_to_point_estimate, random_state=seed)
        f.write(f"bias 0.05 quantile: {bias_lwr_ci}.\n")
        f.write(f"bias 0.95 quantile: {bias_upr_ci}.\n")

        se_samples = np.array(summary.distance_to_point_estimate)**2
        mse_lwr_ci, mse_upr_ci = bootstrap_mean_ci(se_samples, random_state=seed)
        f.write(f"mse 0.05 quantile: {mse_lwr_ci}.\n")
        f.write(f"mse 0.95 quantile: {mse_upr_ci}.\n")

        accuracy_samples = np.concatenate((np.ones(summary.n_correct), np.zeros(summary.n-summary.n_correct)))
        acc_lwr_ci, acc_upr_ci = bootstrap_mean_ci(accuracy_samples, random_state=seed)
        f.write(f"accuracy 0.05 quantile: {acc_lwr_ci}.\n")
        f.write(f"accuracy 0.95 quantile: {acc_upr_ci}.\n")

    # if len(summary.posterior_prob_of_true_value)>0:
    #     plt.figure()
    #     plt.hist(summary.posterior_prob_of_true_value, weights=[1/len(summary.posterior_prob_of_true_value)]*len(summary.posterior_prob_of_true_value), bins=20)
    #     plt.xlabel("Proportion of Bootstrap samples at true value")
    #     plt.ylabel("Proportion")
    #     plt.savefig(os.path.join(outpath, f"proportion_of_bootstrap_samples_at_true_value.pdf"))

    # plt.figure()
    # plt.hist(summary.distance_to_point_estimate, weights=[1/len(summary.distance_to_point_estimate)]*len(summary.distance_to_point_estimate), 
    #          bins=np.arange(min(summary.distance_to_point_estimate)-0.5, max(summary.distance_to_point_estimate)+2.5))
    # plt.xlabel("Distance from point estimate to true value")
    # plt.ylabel("Proportion")
    # plt.savefig(os.path.join(outpath, f"distance_from_point_estimate_to_true_value.pdf"))

    if len(summary.distance_to_posterior_mean)>0:
        plt.figure()
        plt.hist(summary.distance_to_posterior_mean, weights=[1/len(summary.distance_to_posterior_mean)]*len(summary.distance_to_posterior_mean),
                bins=np.arange(min(summary.distance_to_posterior_mean)-0.5, max(summary.distance_to_posterior_mean)+2.5))
        plt.xlabel("Distance from Bootstrap mean to true value")
        plt.ylabel("Proportion")
        plt.savefig(os.path.join(outpath, f"distance_from_boot_mean_to_true_value.pdf"))

    if len(summary.distance_to_posterior_mode)>0:
        plt.figure()
        plt.hist(summary.distance_to_posterior_mode, weights=[1/len(summary.distance_to_posterior_mode)]*len(summary.distance_to_posterior_mode),
                bins=np.arange(min(summary.distance_to_posterior_mode)-0.5, max(summary.distance_to_posterior_mode)+2.5))
        plt.xlabel("Distance from Bootstrap mode to true value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(outpath, f"distance_from_boot_mode_to_true_value.pdf"))
    return 

def make_plots(outpath, cfg, counter, seed_idx, signal, noisy_signal, true_peak_locations, true_peak_count, counts_dist, smoothed_dist):
    # p1 = counter.plot()
    # p1 = plt.plot(signal, label="True signal", color="red")
    # p1 = plt.scatter(true_peak_locations, signal[true_peak_locations], 
    #                     label=f"True peaks: {true_peak_count}", marker='v', s=50, color='red', zorder=5)
    # p1 = plt.legend()
    # plt.savefig(os.path.join(outpath, f"peak_estimate_seed_{seed_idx}.pdf"))
    # plt.close()

    # p1b = counter.plot(filter=False)
    # p1b = plt.plot(signal, label="True signal", color="red")
    # p1b = plt.scatter(true_peak_locations, signal[true_peak_locations], 
    #                     label=f"True peaks: {true_peak_count}", marker='v', s=50, color='red', zorder=5)
    # p1b = plt.legend()
    # plt.savefig(os.path.join(outpath, f"peak_estimate_unfiltered_seed_{seed_idx}.pdf"))
    # plt.close()

    if counts_dist is not None:
        p2 = plt.figure()
        min_count = min(counts_dist)
        max_count = max(counts_dist)
        bins = np.linspace(min_count-0.5, max_count+0.5, num=max_count-min_count+2)
        p2 = plt.hist(counts_dist, bins=bins)
        p2 = plt.xlabel("Num peaks")
        p2 = plt.ylabel("Frequency")
        p2 = plt.title("Posterior distribution of number of peaks")
        plt.savefig(os.path.join(outpath, f"peak_distribution_seed_{seed_idx}.pdf"))
        plt.close()

    # if smoothed_dist is not None:
    #     p3 = plt.figure()
    #     # each row of smoothed_dist is a sample of the smoothed signal from the posterior
    #     p3 = plt.plot(smoothed_dist.T, color="grey", alpha=10/(100*np.log10(cfg["inference"]["n_sims"])))
    #     p3 = plt.plot(counter.get_smoothed(True).smoothed, color="black")
    #     p3 = plt.plot(signal, color="red")
    #     p3 = plt.plot(range(len(noisy_signal)), noisy_signal, color="green", alpha=1, linestyle=":")
    #     p3 = plt.xlabel("Sample index")
    #     p3 = plt.title("Posterior simulations of smoothed regression line")
    #     plt.savefig(os.path.join(outpath, f"smoothed_signal_posterior_seed_{seed_idx}.pdf"))
    #     plt.close()

def write_experiement(outpath, 
                      seed_idx, 
                      signal, 
                      noisy_signal, 
                      true_peak_locations, 
                      true_peak_count, 
                      estimate, 
                      locations_dist, 
                      counts_dist, 
                      smoothed_dist):
    os.makedirs(outpath, exist_ok=True)

    with open(os.path.join(outpath, f"generated_signal_seed_{seed_idx}.json"), "w") as f:
        json.dump(signal.tolist(), f, indent=2)
        
    with open(os.path.join(outpath, f"generated_noisy_signal_seed_{seed_idx}.json"), "w") as f:
        json.dump(noisy_signal.tolist(), f, indent=2)
        
    # with open(os.path.join(outpath, f"peak_info_seed_{seed_idx}.json"), "w") as f:
    #     json.dump([true_peak_locations, true_peak_count], f, indent=2)
        
    # with open(os.path.join(outpath, f"peak_estimate_seed_{seed_idx}.json"), "w") as f:
    #     estimate_data = [elt for elt in estimate]
    #     json.dump(estimate_data, f, indent=2)
    
    # if locations_dist is not None:
    #     with open(os.path.join(outpath, f"peak_locations_posterior_dist_seed_{seed_idx}.json"), "w") as f:
    #         json.dump(locations_dist, f, indent=2)

    # if counts_dist is not None:        
    #     with open(os.path.join(outpath, f"peak_counts_posterior_dist_seed_{seed_idx}.json"), "w") as f:
    #         json.dump(counts_dist.tolist(), f, indent=2)

    # if smoothed_dist is not None:        
    #     with open(os.path.join(outpath, f"smoothed_signal_posterior_dist_seed_{seed_idx}.json"), "w") as f:
    #         json.dump(smoothed_dist.tolist(), f, indent=2)

def generate_chirp(length, freqs, amplitudes, phases, chirp_rate):
    assert len(freqs)==len(amplitudes)==len(phases), "Expected freqs, amplitudes and phases to have the same length"

    t = np.linspace(0, 1, num=length, endpoint=False)
    
    s = np.zeros(length)
    for i in range(len(freqs)):
        s += amplitudes[i] * np.sin(2*np.pi*(freqs[i] + chirp_rate*t)*t + phases[i])
    
    return s

def generate_piecewise_linear_signal(length, seed, npeaks, peak_size_shape, peak_size_rate, min_distance_between_peaks=None, trough_dist_p=0.5, min_peak_size=-1):
    """
        peak_size_shape: shape parameter of gamma distribution of peak sizes (if integer, its the number of phases in an Erlang distribution)
        peak_size_rate: rate parameter of gamma disttribution or peak sizes (same as the rate parameter of an Erlang distribution)
    """
    if min_distance_between_peaks is None:
        min_distance_between_peaks = 1
    elif min_distance_between_peaks < 1:
        raise ValueError(f"min_distance_between_peaks must be at least 1, got {min_distance_between_peaks}")
    assert 0<trough_dist_p<1, f"trough_dist_p must be a be between 0 and 1, got {trough_dist_p}"
    assert npeaks < (length-1)//2
    assert peak_size_shape > 0, f"peak_size_shape must be non-negative, got {peak_size_shape}"
    assert length > 2, f"length must be greater than 2, got {length}"

    t = np.arange(length, dtype=int)
    
    # pick locations of peaks
    np.random.seed(seed)
    peak_locations = np.zeros(length, dtype=int)
    while np.any(np.diff(peak_locations) <= min_distance_between_peaks):
        peak_locations = np.sort(np.random.choice(t[1:-1], npeaks, replace=False))
    # troughs are in between peaks (rounded down)
    trough_locations = np.zeros(npeaks+1, dtype=int)
    for i in range(1, npeaks):
        # range of locations for preak
        ix = range(peak_locations[i-1]+1, peak_locations[i])
        # binomial location of trough
        j = np.random.binomial(len(ix)-1, trough_dist_p)
        trough_locations[i] = ix[j]
        # uniform random location
        # trough_locations[i] = np.random.choice(ix, 1, replace=False)
    trough_locations[-1] = length-1

    m = -2
    while m < min_peak_size:
        peak_size = np.random.gamma(peak_size_shape, 1/peak_size_rate, npeaks)
        trough_size = -np.random.gamma(peak_size_shape, 1/peak_size_rate, npeaks+1)

        extrema = np.zeros(2*npeaks+1)
        extrema[0::2] = trough_size
        extrema[1::2] = peak_size
        distance = np.abs(np.diff(extrema))
        m = np.min(distance)

    s = np.zeros(length)
    peak_idx = 0
    trough_idx = 0
    is_increasing = True
    prev_extrema = trough_size[0]
    next_extrema = peak_size[0]

    for i in range(length):
        # once past the peak, move to the next trough
        if (peak_idx + (not is_increasing) < len(peak_locations) 
            and t[i] > peak_locations[peak_idx + (not is_increasing)]):

            trough_idx += 1
            prev_extrema = next_extrema
            next_extrema = trough_size[trough_idx]
            is_increasing = False
        # once past the trough move to the next peak
        if t[i] > trough_locations[trough_idx + is_increasing]:
            peak_idx += 1
            prev_extrema = next_extrema
            next_extrema = peak_size[peak_idx]
            is_increasing = True

        peak_location = peak_locations[peak_idx]
        trough_location = trough_locations[trough_idx]
        a = ( 
            (t[i]-min(trough_location, peak_location))
             / abs(trough_location - peak_location)
        )
        s[i] = prev_extrema*(1-a) + next_extrema*a
    
    return s

def d2(x,y): 
    return (x-y)**2

def generate_gaussian_process_signal(length, seed, var_sq_exp, ell_sq_exp, var_periodic, ell_periodic, period, min_peak_size=-1):
    d2_mat = np.zeros((length, length))
    for x in range(length):
        for y in range(length):
            d2_mat[x,y] = d2(x,y)
    cov_sq_exp = var_sq_exp * np.exp(- d2_mat/(2*ell_sq_exp**2))

    d1_mat = np.sqrt(d2_mat)
    cov_periodic = var_periodic * np.exp(-2 * (np.sin(d1_mat*np.pi/period)**2)/(ell_periodic**2))

    cov = cov_sq_exp + cov_periodic
    np.random.seed(seed)
    m = -2
    while m < min_peak_size:
        s = np.random.multivariate_normal(np.zeros(length), cov)
        peak_locs = band_count.model_utils.count.find_peaks(s)
        trough_locs = band_count.model_utils.count.find_peaks(-s)
        extrema_locs = peak_locs + trough_locs
        extrema_locs.sort()
        extrema = s[extrema_locs]
        diffs = np.abs(np.diff(np.asfarray(extrema)))
        m = diffs.min()
    return s

def generate_correlated_gaussian_noise(length, seed, var, corr):
    noise = np.zeros(length)
    np.random.seed(seed)
    stationary_var = np.sqrt(var/(1-corr**2))
    noise[0] = np.random.normal(0.0, stationary_var)
    for i in range(1,length):
        noise[i] = corr*noise[i-1] + np.random.normal(0.0, np.sqrt(var))
    return noise

def generate_signal_dependent_noise(length, seed, base_var, signal):
    # shift the signal to have minimum 1
    shift = np.min(signal) - 1.0
    shifted_signal = signal - shift

    # noise is proportional to shifted signal
    sd = np.sqrt(base_var) * shifted_signal

    np.random.seed(seed)
    noise = np.random.normal(0.0, sd, length)
    return noise

def read_signal_from_file(file_elts,seed_idx):
    filename = str(seed_idx).join(file_elts)
    with open(filename, "r") as f:
        data = json.load(f)
    signal = np.array(data)
    return signal