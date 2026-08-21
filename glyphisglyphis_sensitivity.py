import glyphisglyphis_inference as ggi

import pandas as pd
import matplotlib.pyplot as plt
import os 
from plotnine import *
import numpy as np


my_theme = (
    theme_bw()
    + theme(
        figure_size=(5, 6),
        legend_position="bottom",
        axis_title=element_text(size=12),
        axis_text=element_text(size=10),
        strip_text=element_text(size=11)
    )
)

def plot_data_and_smooths(series, smooth_boots, ID):
    data = series[series["ID"] == ID].dropna().copy()

    # Compute per-Variable mean and std
    stats = (
        data.groupby("Variable")["Value"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "y_mean", "std": "y_std"})
        .reset_index()
    )

    # Merge stats back
    data = data.merge(stats, on="Variable", how="left")

    # Clip values to ±5 std
    data["Value_clipped"] = data["Value"].clip(
        lower=data["y_mean"] - 4 * data["y_std"],
        upper=data["y_mean"] + 4 * data["y_std"]
    )

    p1 = (
        ggplot(data, aes(x="Sample Index", y="Value_clipped", color=""))
        + geom_point(size=1.5, alpha=0.8)
        + facet_wrap("Variable", ncol=1, scales="free")
        + my_theme
        + labs(x="Sample Index", y="")
    )

    p1 = (
        p1
        + geom_line(smooth_boots[smooth_boots["ID"]==ID], 
                    aes(x="Sample Index", y="Value", group="Group", color=""), 
                    alpha=0.2)
        + scale_color_manual(values=["#000000", "#D55E00", "#0072B2"])
    )
    return p1

def step2breaks(x):
    return np.arange(
        np.floor(x[0]),
        np.ceil(x[1]) + 1,
        2
    )

def plot_hists(count_estimates, estimates, ID):
    id_idx = count_estimates["ID"]==ID
    boot_idx = count_estimates["Type"]=="bootstrap sample"
    boot_estimates = count_estimates[boot_idx & id_idx].copy()
    p2 = (ggplot(boot_estimates, aes(x="Peaks/Bands Estimate", fill=""))
        + geom_histogram(binwidth=1, center=0, alpha=0.5, position="identity")
        + facet_wrap("~Variable", ncol=1, scales='free_x')
        + my_theme
        + scale_fill_manual(values=["#D55E00", "#0072B2"]))
    p2 = (p2 
        + geom_vline(
            data=count_estimates[(~boot_idx) & id_idx],
            mapping=aes(xintercept="Peaks/Bands Estimate", color=""),
            linetype="dashed", size=1.5)
        + scale_color_manual(values=[ "#D55E00", "#0072B2", "#CC79A7", "#009E73"]))
    p2 = (p2 
        + geom_vline(
            data=estimates[estimates["ID"]==ID],
            mapping=aes(xintercept="Peaks/Bands Estimate", color=""),
            linetype="dashed", size=1.5))
    x_min = count_estimates[id_idx]["Peaks/Bands Estimate"].min()
    x_max = count_estimates[id_idx]["Peaks/Bands Estimate"].max()
    vline_min = estimates[estimates["ID"]==ID]["Peaks/Bands Estimate"].min()
    vline_max = estimates[estimates["ID"]==ID]["Peaks/Bands Estimate"].max()
    xmin = round(min(x_min, vline_min))
    xmax = round(max(x_max, vline_max))
    p2 = p2 + scale_x_continuous(breaks=step2breaks, limits=(xmin - 1, xmax + 1))
    p2 = p2 + labs(y="")
    return p2

def add_plot_to_ax(p, ax):
    p_fig = p.draw()
    p_fig.canvas.draw() 
    ax.imshow(p_fig.canvas.buffer_rgba())
    ax.axis("off")

def clean_labels(df):
    df = df.rename(columns={
        "Estimate": "Peaks/Bands Estimate",
    })
    if "Variable" in df.columns:
        df["Variable"] = df["Variable"].map({
            "Gray_Value": "Gray Value",
            "Sr87_86_X": "Sr isotopic ratio",
            "Sr87_Y": "Sr87 (Volts)",
            "Sr88_ppm": "Sr88 (ppm)"
        })
    df["ID"] = df["ID"].map({
        id: f"Specimen {id}" for id in df["ID"].unique()
    })
    return df

def get_point_boot_estimates(df):
    point_estimates = df[df["Type"]!="bootstrap sample"].copy()
    boot_estimates = df[df["Type"]=="bootstrap sample"].copy()
    return point_estimates, boot_estimates

def get_clean_estimates(fn):
    df = pd.read_csv(fn, index_col=0, dtype={"ID": str, "Boot Index": str})
    df = clean_labels(df)
    point_estimates, boot_estimates = get_point_boot_estimates(df)
    return point_estimates, boot_estimates

if __name__ == "__main__":
    count_estimates_filename = os.path.join(".","out","count_estimates_max_bands_3x_age_estimates.csv")
    count_estimates_40_filename = os.path.join(".","out","count_estimates_max_age_40.csv")
    smooth_boots_filename = os.path.join(".","out","smooth_boots_max_bands_3x_age_estimates.csv")
    smooth_boots_40_filename = os.path.join(".","out","smooth_boots_max_age_40.csv")

    series = ggi.get_time_series_all(ggi.read_files_all())
    series = clean_labels(series)

    series, means = ggi.demean_series(series)
    series, n_nan = ggi.impute_nan(series)

    ce = pd.concat(get_clean_estimates(count_estimates_filename), ignore_index=True)
    ce["Boot Index"] = ce["Boot Index"] + " (m=3x)"
    ce[""] = "Boot\n(m=3x)"
    ce.loc[ce["Type"]=="estimate",""] = "Count\n(m=3x)"
    ce40 = pd.concat(get_clean_estimates(count_estimates_40_filename), ignore_index=True)
    ce40["Boot Index"] = ce40["Boot Index"] + " (m=40)"
    ce40[""] = "Boot\n(m=40)"
    ce40.loc[ce40["Type"]=="estimate",""] = "Count\n(m=40)"

    count_estimates = pd.concat((ce, ce40), ignore_index=True)

    smooth_boots_thin_factor = 20
    sb = pd.read_csv(smooth_boots_filename, index_col=0, dtype={"ID": str, "Boot Index": str})
    sb = sb[(np.asarray(sb["Boot Index"], dtype=int) % smooth_boots_thin_factor) == 0]
    sb["Boot Index"] = sb["Boot Index"] + " (m=3x)"
    sb[""] = "Boot\n(m=3x)"
    sb40 = pd.read_csv(smooth_boots_40_filename, index_col=0, dtype={"ID": str, "Boot Index": str})
    sb40 = sb40[(np.asarray(sb40["Boot Index"], dtype=int) % smooth_boots_thin_factor) == 0]
    sb40["Boot Index"] = sb40["Boot Index"] + " (m=40)"
    sb40[""] = "Boot\n(m=40)"

    smooth_boots = pd.concat((sb, sb40), ignore_index=True)
    smooth_boots = clean_labels(smooth_boots)

    smooth_boots["Group"] = smooth_boots["ID"]+smooth_boots["Boot Index"]

    estimates = pd.read_csv(os.path.join(ggi.DATA_PATH, "estimates.csv"), dtype={"ID": str})
    estimates.loc[estimates["Method"]=="TLOM","Estimate"] = estimates[estimates["Method"]=="TLOM"]["Estimate"]+0.15
    estimates.loc[estimates["Method"]=="microXRF","Estimate"] = estimates[estimates["Method"]=="microXRF"]["Estimate"]-0.15
    estimates[""] = estimates["Method"]
    estimates = clean_labels(estimates)
    estimates[""] = estimates[""].replace({"microXRF": "uXRF     "})

    series[""] = "Data"
    series["Group"] = "data group"

    for ID in ggi.GG_SR_IDS:
        id = f"Specimen {ID}"
        p1 = plot_data_and_smooths(series, smooth_boots, id)
        p2 = plot_hists(count_estimates, estimates, id)

        fig = plt.figure(figsize=(10, 9))
        ax1 = fig.add_subplot(1, 2, 1)
        add_plot_to_ax(p1, ax1)
        ax2 = fig.add_subplot(1, 2, 2)
        add_plot_to_ax(p2, ax2)

        plt.tight_layout()
        plt.savefig(f"out/GG{ID}_estimates_sensitivity{count_estimates_filename.replace(".", "").replace("/", "_")}_5_6.pdf")
