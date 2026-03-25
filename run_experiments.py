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

" Recursively check all files in dir which start with 'config.' and end with '.yaml' for the correct config structure"
def check_all_configs(dir):
    dir_items = os.listdir(dir)
    for item in dir_items:
        item_path = os.path.join(dir, item)
        if os.path.isdir(item_path):
            check_all_configs(item_path)
            continue
        file_elts = item.split(".")
        if file_elts[0]=="config" and file_elts[-1]=="yaml":
            print(f"Checking config {item_path}.\n")
            with open(item_path, "r") as f:
                cfg = yaml.safe_load(f)
            experiment_utils.check_experiment_parameters(item_path, cfg)
    return

" Recursively run all files in dir which start with 'config.' and end with '.yaml'"
def run_experiments_recursive(dir):
    dir_items = os.listdir(dir)
    for item in dir_items:
        item_path = os.path.join(dir, item)
        if os.path.isdir(item_path):
            run_experiments_recursive(item_path)
            continue
        file_elts = item.split(".")
        if file_elts[0]=="config" and file_elts[-1]=="yaml":
            out_path = os.path.join(os.path.dirname(item_path),
                                    os.path.basename(item_path) + ".out")
            print(f"Running experiment with config {item_path}.\n")
            experiment_utils.run_experiment(item_path, out_path)
    return

check_all_configs(dir_name)
run_experiments_recursive(dir_name)