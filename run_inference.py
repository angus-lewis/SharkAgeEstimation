import experiment_utils as eu
import sys
import pandas as pd
import numpy as np
import json
import time 
from datetime import datetime
from pathlib import Path

file_path = sys.argv[1] # "data/GreyValues_43.csv"
outpath = sys.argv[2] # "experiments/whitetips/greyvalues43/"
birthmark_idx = int(sys.argv[3]) # 432
downsample_factor = int(sys.argv[4]) # 32
max_bands = float(sys.argv[5]) # 40
max_corr = float(sys.argv[6]) # 0.1
n_boot = int(sys.argv[7]) # 10
seed = int(sys.argv[8]) # 16860

specimen = pd.read_csv(file_path)[birthmark_idx::1].reset_index()
group = np.arange(len(specimen)) // downsample_factor
specimen = specimen.groupby(group).mean().reset_index(drop=True)
signal = np.asfarray(specimen["Gray_Value"])

signal = np.asfarray(specimen["Gray_Value"])
bpdn_counter = eu.band_count.BandCounter(signal, max_bands=max_bands, max_corr=max_corr)
tp_counter = eu.GAMBandCounter(signal, max_age=max_bands*2)

start_time = time.time()

bpdn_estimate = bpdn_counter.get_count_estimate()
bpdn_smooth = bpdn_counter.get_smoothed(True).smoothed
tp_estimate = tp_counter.get_count_estimate()
tp_smooth = tp_counter.get_smoothed().smoothed

# End timer
end_time = time.time()

# Compute elapsed time in seconds
elapsed_time = end_time - start_time
# Get current time
now = datetime.now()
print("Current time:", now.strftime("%Y-%m-%d %H:%M:%S"))
print(f"Estimated time remaining: {n_boot*elapsed_time:.3f} seconds")

bpdn_boot = bpdn_counter.get_count_distribution(n_boot, seed=seed)
tp_boot = tp_counter.get_count_distribution(n_boot, seed=seed, boot_method='residual boot')

path = Path(outpath)
path.mkdir(parents=True, exist_ok=True)

with open(path.joinpath("bpdn_estimates.json"), "w") as f:
    json.dump(bpdn_estimate, f, indent=2)
with open(path.joinpath("tp_estimates.json"), "w") as f:
    json.dump(tp_estimate, f, indent=2)
with open(path.joinpath("bpdn_smooths.json"), "w") as f:
    json.dump(bpdn_smooth.tolist(), f, indent=2)
with open(path.joinpath("tp_smooths.json"), "w") as f:
    json.dump(tp_smooth.tolist(), f, indent=2)
with open(path.joinpath("bpdn_boots_locs.json"), "w") as f:
    json.dump(bpdn_boot[0], f, indent=2)
with open(path.joinpath("bpdn_boots_counts.json"), "w") as f:
    json.dump(bpdn_boot[1].tolist(), f, indent=2)
bpdn_boot_smooths = [bpdn_boot[2][j].tolist() for j in range(n_boot)]
with open(path.joinpath("bpdn_boots_smooths.json"), "w") as f:
    json.dump(bpdn_boot_smooths, f, indent=2)
with open(path.joinpath("tp_boots_locs.json"), "w") as f:
    json.dump(tp_boot[0], f, indent=2)
with open(path.joinpath("tp_boots_counts.json"), "w") as f:
    json.dump(tp_boot[1].tolist(), f, indent=2)
tp_boot_smooths = [tp_boot[2][j].tolist() for j in range(n_boot)]
with open(path.joinpath("tp_boots_smooths.json"), "w") as f:
    json.dump(tp_boot_smooths, f, indent=2)
