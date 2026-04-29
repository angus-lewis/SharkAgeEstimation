import json
import os
import matplotlib.pyplot as plt
import band_count
import numpy as np
from scipy import stats 

def read_signals(dir, idxs=range(128)):
    signals = []
    for i in idxs:
        with open(os.path.join(dir,f"generated_signal_seed_{i}.json")) as f: 
            signals.append(json.load(f))
    return signals

def dir_factory(length, varvalue, experiment="perf"):
    if experiment=="perf":
        varname = "peaksize"
        fn = f"./experiments/{experiment}/gp/config.gp_var0.36_freqsmany_length{length}_BPDN_{varname}{varvalue}.yaml.out/"
    else:
        varname = "corr"
        fn = f"./experiments/{experiment}/gp/{varname}{varvalue}/config.gp_var0.36_freqsmany_length{length}_BPDN_{varname}{varvalue}.yaml.out/"
    return fn

def find_peaks(signals):
    npeaks = []
    for signal in signals:
        npeaks.append(
            len(band_count.model_utils.count.find_peaks(signal))
        )
    return npeaks

def integer_bins(x):
    return np.arange(min(x)-0.5, max(x)+1.5)

def get_peak_dist(length, varvalue, experiment="perf", idxs=range(128)):
    dir = dir_factory(length, varvalue, experiment)
    signals = read_signals(dir, idxs)
    peaks = find_peaks(signals)
    return peaks

def get_peak_dist_mode(length, varvalue, experiment="perf", idxs=range(128)):
    """
    returns mode object with fields .mode, .count
    """
    peaks = get_peak_dist(length, varvalue, experiment, idxs)
    mode = stats.mode(peaks)
    return {"mode": mode.mode, "p": mode.count/len(peaks), "count": mode.count}

if __name__ == "__main__":
    dir_128_06 = dir_factory(128, "0.6")
    dir_256_06 = dir_factory(256, "0.6")
    dir_512_06 = dir_factory(512, "0.6")
    signals_128_max_corr06 = read_signals(dir_128_06)
    signals_256_max_corr06 = read_signals(dir_256_06)
    signals_512_max_corr06 = read_signals(dir_512_06)
    peaks_128_06 = find_peaks(signals_128_max_corr06)
    peaks_256_06 = find_peaks(signals_256_max_corr06)
    peaks_512_06 = find_peaks(signals_512_max_corr06)

    plt.hist(peaks_128_06, bins=integer_bins(peaks_128_06), alpha=0.6)
    plt.hist(peaks_256_06, bins=integer_bins(peaks_256_06), alpha=0.6)
    plt.hist(peaks_512_06, bins=integer_bins(peaks_512_06), alpha=0.6)
    plt.show()

    mode = get_peak_dist_mode(128,"0.6")