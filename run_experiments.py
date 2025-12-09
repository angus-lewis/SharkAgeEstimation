# Experiment settings
# periodic signals, sinusoidal chirps, random linear signals
# low, med, high noise, relative to amplitude of peaks
# gaussian noise, noise which increases with the signal

import experiment_utils
import os
import sys

dir_name = sys.argv[1]

experiment_cfg_names = os.listdir(dir_name)

for cfg_name in experiment_cfg_names:
    cfg_path = os.path.join(dir_name, cfg_name)
    if not os.path.basename(cfg_path).startswith("config."):
        continue
    elif os.path.isdir(cfg_path):
        continue
    out_path = os.path.join(os.path.dirname(cfg_path),
                            os.path.basename(cfg_path) + ".out")
    print(f"Running experiment with config {cfg_path}.\n")
    experiment_utils.run_experiment(cfg_path, out_path)