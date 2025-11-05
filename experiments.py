import matplotlib.pyplot as plt
import numpy as np
import yaml
import json
import os

import band_count

def run_experiment(config, outpath):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    sig_gen_cfg = cfg["signal_generator"]
    match sig_gen_cfg["method"]["name"]:
        case "tone":
            signal = generate_tone(sig_gen_cfg["length"], 
                                   sig_gen_cfg["method"]["frequencies"], 
                                   sig_gen_cfg["method"]["amplitudes"],
                                   sig_gen_cfg["method"]["phases"],)
    
    match sig_gen_cfg["noise"]["type"]:
        case "gaussian":
            noise = generate_gaussian_noise(sig_gen_cfg["length"], 
                                            sig_gen_cfg["noise"]["rng_seed"], 
                                            sig_gen_cfg["noise"]["variance"])
            noisy_signal = signal + noise
    
    true_peak_locations = band_count.count.find_peaks(signal)
    true_peak_count = len(true_peak_locations)

    counter = band_count.BandCounter(noisy_signal, cfg["inference"]["max_bands"])
    estimate = counter.get_count_estimate(True)
    
    locations_dist, counts_dist, smoothed_dist = counter.get_count_distribution(
        cfg["inference"]["n_sims"], True
    )

    os.makedirs(outpath, exist_ok=True)

    with open(os.path.join(outpath, "generated_signal.json"), "w") as f:
        json.dump(signal.tolist(), f, indent=2)
    
    with open(os.path.join(outpath, "generated_noisy_signal.json"), "w") as f:
        json.dump(noisy_signal.tolist(), f, indent=2)
    
    with open(os.path.join(outpath, "peak_info.json"), "w") as f:
        json.dump([true_peak_locations, true_peak_count], f, indent=2)
    
    with open(os.path.join(outpath, "peak_estimate.json"), "w") as f:
        estimate_data = [elt for elt in estimate]
        json.dump(estimate_data, f, indent=2)
    
    with open(os.path.join(outpath, "peak_locations_posterior_dist.json"), "w") as f:
        json.dump(locations_dist, f, indent=2)
    
    with open(os.path.join(outpath, "peak_counts_posterior_dist.json"), "w") as f:
        json.dump(counts_dist.tolist(), f, indent=2)
    
    with open(os.path.join(outpath, "smoothed_signal_posterior_dist.json"), "w") as f:
        json.dump(smoothed_dist.tolist(), f, indent=2)
    
    p1 = counter.plot()
    p1 = plt.plot(signal, label="Truth", color="red")
    p1 = plt.legend()
    plt.savefig(os.path.join(outpath, "peak_estimate.pdf"))
    plt.close()

    p2 = plt.figure()
    min_count = min(counts_dist)
    max_count = max(counts_dist)
    bins = np.linspace(min_count-0.5, max_count+0.5, num=max_count-min_count+2)
    p2 = plt.hist(counts_dist, bins=bins)
    p2 = plt.xlabel("Num peaks")
    p2 = plt.ylabel("Frequency")
    p2 = plt.title("Posterior distribution of number of peaks")
    plt.savefig(os.path.join(outpath, "peak_distribution.pdf"))
    plt.close()

    p3 = plt.figure()
    p3 = plt.plot(smoothed_dist, color="grey", alpha=10/(100*np.log10(cfg["inference"]["n_sims"])))
    p3 = plt.plot(counter.low_freq_smoothed.smoothed, color="black")
    p3 = plt.plot(signal, color="red")
    p3 = plt.xlabel("Sample index")
    p3 = plt.title("Posterior simulations of smoothed regression line")
    plt.savefig(os.path.join(outpath, "smoothed_signal_posterior.pdf"))
    plt.close()
    
    return

def generate_tone(length, freqs, amplitudes, phases):
    assert len(freqs)==len(amplitudes)==len(phases), "Expected freqs, amplitudes and phases to have the same length"

    t = np.linspace(0, 1, num=length, endpoint=False)
    
    s = np.zeros(length)
    for i in range(len(freqs)):
        s += amplitudes[i] * np.sin(2*np.pi*freqs[i]*t + phases[i])
    
    return s

def generate_gaussian_noise(length, seed, var):
    np.random.seed(seed)
    noise = np.random.normal(0.0, np.sqrt(var), length)
    return noise

run_experiment("experiments/config1.yaml", "experiments/dm")