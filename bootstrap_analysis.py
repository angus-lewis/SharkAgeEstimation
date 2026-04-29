import json 
import matplotlib.pyplot as plt
import numpy as np 

with open("experiments/perf/lin/config.lin_var0.36_peaks8_length128_BPDN_peaksize0.0.yaml.out/peak_counts_posterior_dist_seed_199.json", "r") as f:
    peaks = json.load(f)

plt.hist(peaks, bins=np.arange(min(peaks)-0.5, max(peaks)+0.5, 1))
plt.show()

