# Experiment settings
# periodic signals, sinusoidal chirps, random linear signals
# low, med, high noise, relative to amplitude of peaks
# gaussian noise, noise which increases with the signal

import matplotlib
# Ensure no GUI is used
matplotlib.use("Agg")

import experiment_utils
import os
import sys
import yaml

dir_name = sys.argv[1]
filename = sys.argv[2]


def run_experiment(dir, filename):
    item_path = os.path.join(dir, filename)
    print(f"Checking config {item_path}.\n")
    with open(item_path, "r") as f:
        cfg = yaml.safe_load(f)
    experiment_utils.check_experiment_parameters(item_path, cfg)
    out_path = os.path.join(os.path.dirname(item_path),
                            os.path.basename(item_path) + ".out")
    print(f"Running experiment with config {item_path}.\n")
    experiment_utils.run_experiment(item_path, out_path)
    return

run_experiment(dir_name, filename)
