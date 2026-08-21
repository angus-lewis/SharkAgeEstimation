import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os 
from plotnine import *
import numpy as np 

import band_count

DATA_PATH = os.path.join(".", "data", "paper", "csv")

FILE_PREFIX = "GG"

GG_ELEMENTAL_FILE_SUFFIX = "_Glyphisglyphis_ElementalDataBMTrimmed.csv"
GG_GRAY_VALUES_FILE_SUFFIX = "_Values_BMTrim.csv"
GG_SR_FILE_SUFFIX = ".csv"

GG_ELEMENTAL_IDS = ["03", "05", "06", "07", "08", "09", "10"]
GG_GRAY_VALUES_IDS = ["01", "03", "06", "07", "08", "09", "10"]
GG_SR_IDS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

ELEMENTAL_TIME_COLNAME = "Elapsed Time"
ELEMENTAL_SERIES_COLNAME = "Sr88_ppm"

GRAY_VALUES_TIME_COLNAME = "Distance_(pixels)"
GRAY_VALUES_SERIES_COLNAME = "Gray_Value"

SR_TIME_COLNAME = None
SR_SERIES_1_COLNAME = "Sr87_Y"
SR_SERIES_2_COLNAME = "Sr87_86_X"

N_BOOT = 1000
BOOT_SEED = 16860

def read_files_single(ids, suffix, skiprows=0):
    dfs = []
    for id in ids:
        df_ = pd.read_csv(os.path.join(DATA_PATH, f"{FILE_PREFIX}{id}{suffix}"), skiprows=skiprows)
        df_["ID"] = id
        df_["Sample Index"] = range(len(df_))
        dfs.append(df_)
    df = pd.concat(dfs, ignore_index=True)
    return df

def read_files_all(elt_ids=GG_ELEMENTAL_IDS, gray_ids=GG_GRAY_VALUES_IDS, sr_ids=GG_SR_IDS):
    elt_df = read_files_single(elt_ids, GG_ELEMENTAL_FILE_SUFFIX, skiprows=1)
    gray_df = read_files_single(gray_ids, GG_GRAY_VALUES_FILE_SUFFIX)
    sr_df = read_files_single(sr_ids, GG_SR_FILE_SUFFIX)
    return {"elements": elt_df, "gray values": gray_df, "sr": sr_df}

def get_time_series_single(df, seriescolname, tcolname=None):
    if tcolname is not None:
        idx = df[tcolname].values.copy()
    else:
        idx = range(len(df))
    series = pd.DataFrame({
        "t": idx, 
        "Sample Index": df["Sample Index"].values.copy(),
        "Value": df[seriescolname].values.copy(), 
        "Variable": seriescolname,
        "ID": df["ID"].values.copy(),
    })
    return series

def missing_series(variable, id):
    return pd.DataFrame({
        "t": [0], 
        "Sample Index": [0],
        "Value": [np.nan], 
        "Variable": [variable],
        "ID": [id],
    })

def get_time_series_all(dfs):
    elt_series = get_time_series_single(dfs["elements"], ELEMENTAL_SERIES_COLNAME, ELEMENTAL_TIME_COLNAME)
    elt_missing_series = pd.concat([missing_series(ELEMENTAL_SERIES_COLNAME, id) for id in GG_SR_IDS if id not in GG_ELEMENTAL_IDS])

    gray_series = get_time_series_single(dfs["gray values"], GRAY_VALUES_SERIES_COLNAME, GRAY_VALUES_TIME_COLNAME)
    gray_missing_series = pd.concat([missing_series(GRAY_VALUES_SERIES_COLNAME, id) for id in GG_SR_IDS if id not in GG_GRAY_VALUES_IDS])

    sr_series = get_time_series_single(dfs["sr"], SR_SERIES_1_COLNAME, SR_TIME_COLNAME)
    sr_ratio_series = get_time_series_single(dfs["sr"], SR_SERIES_2_COLNAME, SR_TIME_COLNAME)

    series = pd.concat((elt_series, elt_missing_series, gray_series, gray_missing_series, sr_series, sr_ratio_series), ignore_index=True)
    return series

def demean_series(series):
    means = []
    out = []

    for (ID, Variable), subdf in series.groupby(["ID", "Variable"]):
        subdf = subdf.copy()

        if all(pd.isna(subdf["Value"].values)):
            out.append(subdf)
            continue

        mean = subdf["Value"].mean()
        subdf["Value"] = subdf["Value"] - mean

        out.append(subdf)

        means.append(pd.DataFrame({
            "ID": [ID],
            "Variable": [Variable],
            "mean": [mean]
        }))

    means = pd.concat(means, ignore_index=True)
    out = pd.concat(out, ignore_index=True)

    return out, means

def impute_nan(series):
    out = []
    total_nans = 0
    for (ID, Variable), subdf in series.groupby(["ID", "Variable"]):
        subdf = subdf.copy()

        if all(pd.isna(subdf["Value"].values)):
            out.append(subdf)
            continue
        elif not any(pd.isna(subdf["Value"].values)):
            out.append(subdf)
            continue

        mean = subdf["Value"].mean()
        nan_idx = pd.isna(subdf["Value"])
        total_nans += sum(nan_idx)
        subdf.loc[nan_idx,"Value"] = mean

        out.append(subdf)
    
    out = pd.concat(out, ignore_index=True)

    return out, total_nans

def _run_counter(counter, variable, ID):
    locs_estimate, count_estimate = counter.get_count_estimate()
    out = pd.DataFrame({
        "ID": [ID], 
        "Variable": [variable], 
        "Estimate": [count_estimate], 
        "Type": "estimate", 
        "Boot Index": "estimate"
    })
    return out

def _run_boot(counter, variable, ID, subdf):
    locs_boot, counts_boot, smooths_boot = counter.get_count_distribution(N_BOOT, seed=BOOT_SEED)
    estimates_out = pd.DataFrame({
        "ID": ID,
        "Variable": variable,
        "Estimate": counts_boot,
        "Type": "bootstrap sample",
        "Boot Index": [str(i) for i in range(N_BOOT)]
    })
    smooths_out = [
        pd.DataFrame({
            "ID": ID,
            "Variable": variable,
            "Value": smooth,
            "t": subdf["t"].values.copy(),
            "Sample Index": subdf["Sample Index"].values.copy(),
            "Boot Index": str(i),
        })
        for (i,smooth) in enumerate(smooths_boot)
    ]
    return estimates_out, smooths_out
    
def run_counter(series, max_ages):
    count_estimates = []
    smooths_boots = []
    for (ID, Variable), subdf in series.groupby(["ID", "Variable"]):
        if all(pd.isna(subdf["Value"].values)):
            continue
        counter = band_count.BandCounter(subdf["Value"].values, max_bands=max_ages[ID])
        count_estimates.append(_run_counter(counter, Variable, ID))
        
        boot_estimates, boot_smooths = _run_boot(counter, Variable, ID, subdf)
        count_estimates.append(boot_estimates)
        smooths_boots.extend(boot_smooths)
    
    count_estimates = pd.concat(count_estimates, ignore_index=True)
    smooths_boots = pd.concat(smooths_boots, ignore_index=True)

    return count_estimates, smooths_boots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run band counting with a max age parameter.")
    parser.add_argument(
        "--max_age",
        type=int,
        required=True,
        help="Maximum age (max bands) to use for all IDs"
    )
    args = parser.parse_args()

    max_age = args.max_age
    series = get_time_series_all(read_files_all())
    estimates = pd.read_csv(os.path.join(DATA_PATH, "estimates.csv"), dtype={"ID": str})
    # max_ages = {ID: max(5, 4*estimates[estimates["ID"]==ID]["Estimate"].max()) for ID in GG_SR_IDS}
    max_ages = {ID: max_age for ID in GG_SR_IDS}

    series, means = demean_series(series)
    series, n_nan = impute_nan(series)

    count_estimates_filename = os.path.join(".","out",f"count_estimates_max_age_{max_age}.csv")
    smooth_boots_filename = os.path.join(".","out",f"smooth_boots_max_age_{max_age}.csv")

    if os.path.isfile(count_estimates_filename) and os.path.isfile(smooth_boots_filename):
        count_estimates = pd.read_csv(count_estimates_filename, index_col=0, dtype={"ID": str, "Boot Index": str})
        smooth_boots = pd.read_csv(smooth_boots_filename, index_col=0, dtype={"ID": str, "Boot Index": str})
    else:
        count_estimates, smooth_boots = run_counter(series, max_ages)
        count_estimates.to_csv(count_estimates_filename)
        smooth_boots.to_csv(smooth_boots_filename)

    smooth_boots["Group"] = smooth_boots["ID"]+smooth_boots["Boot Index"]

    estimates = pd.read_csv(os.path.join(DATA_PATH, "estimates.csv"), dtype={"ID": str})
    estimates.loc[estimates["Method"]=="TLOM","Estimate"] = estimates[estimates["Method"]=="TLOM"]["Estimate"]+0.15
    estimates.loc[estimates["Method"]=="microXRF","Estimate"] = estimates[estimates["Method"]=="microXRF"]["Estimate"]-0.15

    series["Type"] = "Data"
    smooth_boots["Type"] = "Bootstrap"

