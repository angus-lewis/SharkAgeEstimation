import matplotlib.pyplot as plt
import numpy as np
import yaml
import json
import os
import rpy2.robjects
from rpy2.robjects import pandas2ri
import pandas as pd

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
    lower_quantiles = [0.0, 0.025, 0.05, 0.1, 0.25]
    upper_quantiles = [1.0, 0.975, 0.85, 0.9, 0.75]

    def __init__(self):
        self.n_correct = 0
        self.n_in_posterior_range = 0
        self.n_in_posterior_95pct_range = 0
        self.n_in_posterior_90pct_range = 0
        self.n_in_posterior_80pct_range = 0
        self.n_in_posterior_50pct_range = 0
        self.average_distance_to_point_estimate = 0.0
        self.average_distance_squared_to_point_estimate = 0.0
        self.average_distance_to_posterior_mean = 0.0
        self.average_distance_squared_to_posterior_mean = 0.0
        self.n = 0
        return 
    
    def update(self, true_peak_count, estimate, counts_dist):
        
        estimate_locations, estimate_count = estimate
        self.n_correct += estimate_count == true_peak_count

        lower_quantiles = np.quantile(counts_dist, self.lower_quantiles)
        upper_quantiles = np.quantile(counts_dist, self.upper_quantiles)

        is_between_quantiles = (estimate_count >= lower_quantiles) * (estimate_count <= upper_quantiles)

        self.n_in_posterior_range += is_between_quantiles[0]
        self.n_in_posterior_95pct_range += is_between_quantiles[1]
        self.n_in_posterior_90pct_range += is_between_quantiles[2]
        self.n_in_posterior_80pct_range += is_between_quantiles[3]
        self.n_in_posterior_50pct_range += is_between_quantiles[4]

        self.average_distance_to_point_estimate = (
            self.average_distance_to_point_estimate * (self.n/(self.n+1))
            + (estimate_count - true_peak_count)/(self.n+1)
        )
        self.average_distance_squared_to_point_estimate = (
            self.average_distance_squared_to_point_estimate * (self.n/(self.n+1))
            + (estimate_count - true_peak_count)**2/(self.n+1)
        )
        self.average_distance_to_posterior_mean = (
            self.average_distance_to_posterior_mean * (self.n/(self.n+1))
            + (estimate_count - np.mean(counts_dist))/(self.n+1)
        )
        self.average_distance_squared_to_posterior_mean = (
            self.average_distance_squared_to_posterior_mean * (self.n/(self.n+1))
            + (estimate_count - np.mean(counts_dist))**2/(self.n+1)
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
        
def run_experiment(config_path, outpath):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

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
            counter = band_count.BandCounter(np.zeros(sig_gen_cfg["length"], dtype=float), 
                                             max_bands=cfg["inference"]["max_bands"])
        case 'TP':
            counter = GAMBandCounter(np.zeros(sig_gen_cfg["length"], dtype=float),
                                     max_age=cfg["inference"]["max_bands"])

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
                signal = generate_piecewise_linear_signal(sig_gen_cfg["length"], 
                                                          sig_gen_seeds[seed_idx],
                                                          sig_gen_cfg["method"]["n_peaks"],
                                                          sig_gen_cfg["method"]["peak_size_shape"],
                                                          sig_gen_cfg["method"]["peak_size_rate"],
                                                          sig_gen_cfg["method"]["min_distance_between_peaks"])
        
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

        counter.set_signal(noisy_signal)
        estimate = counter.get_count_estimate(True)
        # smth = counter.get_smoothed(True)
        # plt.plot(smth.smoothed)
        # plt.show()
        
        locations_dist, counts_dist, smoothed_dist = counter.get_count_distribution(
            cfg["inference"]["n_sims"], True, posterior_inference_seeds[seed_idx], cfg["inference"]["sim_method"]
        )

        summary.update(true_peak_count, estimate, counts_dist)

        write_experiement(outpath, seed_idx, signal, noisy_signal, true_peak_locations, true_peak_count, estimate, locations_dist, counts_dist, smoothed_dist)
        make_plots(outpath, cfg, counter, seed_idx, signal, noisy_signal, true_peak_locations, true_peak_count, counts_dist, smoothed_dist)
    
    write_summary(outpath, summary)
    return

def write_summary(outpath, summary: Summary):
     with open(os.path.join(outpath, f"summary.txt"), "w") as f:
         f.write(f"correct: {summary.n_correct/summary.n} ({summary.n_correct}/{summary.n}).\n")
         f.write(f"in posterior range: {summary.n_in_posterior_range/summary.n} ({summary.n_in_posterior_range}/{summary.n}).\n")
         f.write(f"in posterior 95% range: {summary.n_in_posterior_95pct_range/summary.n} ({summary.n_in_posterior_95pct_range}/{summary.n}).\n")
         f.write(f"in posterior 90% range: {summary.n_in_posterior_90pct_range/summary.n} ({summary.n_in_posterior_90pct_range}/{summary.n}).\n")
         f.write(f"in posterior 80% range: {summary.n_in_posterior_80pct_range/summary.n} ({summary.n_in_posterior_80pct_range}/{summary.n}).\n")
         f.write(f"in posterior 50% range: {summary.n_in_posterior_50pct_range/summary.n} ({summary.n_in_posterior_50pct_range}/{summary.n}).\n")
         f.write(f"average distance to point estimate: {summary.average_distance_to_point_estimate/summary.n} ({summary.average_distance_to_point_estimate}/{summary.n}).\n")
         f.write(f"average distance squared to point estimate: {summary.average_distance_squared_to_point_estimate/summary.n} ({summary.average_distance_squared_to_point_estimate}/{summary.n}).\n")
         f.write(f"average distance to posterior mean: {summary.average_distance_to_posterior_mean/summary.n} ({summary.average_distance_to_posterior_mean}/{summary.n}).\n")
         f.write(f"average distance squared to posterior mean: {summary.average_distance_squared_to_posterior_mean/summary.n} ({summary.average_distance_squared_to_posterior_mean}/{summary.n}).\n")
         return 

def make_plots(outpath, cfg, counter, seed_idx, signal, noisy_signal, true_peak_locations, true_peak_count, counts_dist, smoothed_dist):
    p1 = counter.plot()
    p1 = plt.plot(signal, label="True signal", color="red")
    p1 = plt.scatter(true_peak_locations, signal[true_peak_locations], 
                        label=f"True peaks: {true_peak_count}", marker='v', s=50, color='red', zorder=5)
    p1 = plt.legend()
    plt.savefig(os.path.join(outpath, f"peak_estimate_seed_{seed_idx}.pdf"))
    plt.close()

    p1b = counter.plot(filter=False)
    p1b = plt.plot(signal, label="True signal", color="red")
    p1b = plt.scatter(true_peak_locations, signal[true_peak_locations], 
                        label=f"True peaks: {true_peak_count}", marker='v', s=50, color='red', zorder=5)
    p1b = plt.legend()
    plt.savefig(os.path.join(outpath, f"peak_estimate_unfiltered_seed_{seed_idx}.pdf"))
    plt.close()

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

    p3 = plt.figure()
    # each row of smoothed_dist is a sample of the smoothed signal from the posterior
    p3 = plt.plot(smoothed_dist.T, color="grey", alpha=10/(100*np.log10(cfg["inference"]["n_sims"])))
    p3 = plt.plot(counter.get_smoothed(True).smoothed, color="black")
    p3 = plt.plot(signal, color="red")
    p3 = plt.plot(noisy_signal, color="seagreen", alpha=0.2)
    p3 = plt.xlabel("Sample index")
    p3 = plt.title("Posterior simulations of smoothed regression line")
    plt.savefig(os.path.join(outpath, f"smoothed_signal_posterior_seed_{seed_idx}.pdf"))
    plt.close()

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
        
    with open(os.path.join(outpath, f"peak_info_seed_{seed_idx}.json"), "w") as f:
        json.dump([true_peak_locations, true_peak_count], f, indent=2)
        
    with open(os.path.join(outpath, f"peak_estimate_seed_{seed_idx}.json"), "w") as f:
        estimate_data = [elt for elt in estimate]
        json.dump(estimate_data, f, indent=2)
        
    with open(os.path.join(outpath, f"peak_locations_posterior_dist_seed_{seed_idx}.json"), "w") as f:
        json.dump(locations_dist, f, indent=2)
        
    with open(os.path.join(outpath, f"peak_counts_posterior_dist_seed_{seed_idx}.json"), "w") as f:
        json.dump(counts_dist.tolist(), f, indent=2)
        
    with open(os.path.join(outpath, f"smoothed_signal_posterior_dist_seed_{seed_idx}.json"), "w") as f:
        json.dump(smoothed_dist.tolist(), f, indent=2)

def generate_chirp(length, freqs, amplitudes, phases, chirp_rate):
    assert len(freqs)==len(amplitudes)==len(phases), "Expected freqs, amplitudes and phases to have the same length"

    t = np.linspace(0, 1, num=length, endpoint=False)
    
    s = np.zeros(length)
    for i in range(len(freqs)):
        s += amplitudes[i] * np.sin(2*(np.pi*freqs[i] + chirp_rate*t)*t + phases[i])
    
    return s

def generate_piecewise_linear_signal(length, seed, npeaks, peak_size_shape, peak_size_rate, min_distance_between_peaks=None, trough_dist_p=0.5):
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

    peak_size = np.random.gamma(peak_size_shape, 1/peak_size_rate, npeaks)
    trough_size = -np.random.gamma(peak_size_shape, 1/peak_size_rate, npeaks+1)

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
