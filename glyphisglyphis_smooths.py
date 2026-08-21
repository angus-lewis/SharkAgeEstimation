import glyphisglyphis_inference as ggi

import pandas as pd
import matplotlib.pyplot as plt
import os 
from plotnine import *
import numpy as np
import band_count


my_theme = (
    theme_bw()
    + theme(
        figure_size=(5, 9),
        legend_position="bottom",
        axis_title=element_text(size=12),
        axis_text=element_text(size=10),
        strip_text=element_text(size=11)
    )
)

def plot_smooths(p1, smooth_boots, ID):
    p1 = (
        p1
        + geom_line(smooth_boots[smooth_boots["ID"]==ID], 
                    aes(x="Sample Index", y="Value", group="Group", color=""), 
                    alpha=1, size=1.5)
    )
    return p1

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
        + geom_point(size=1.5, alpha=1)
        + facet_wrap("Variable", ncol=1, scales="free")
        + my_theme
        + labs(x="Sample Index", y="")
    )

    p1 = plot_smooths(p1, smooth_boots, ID) + scale_fill_discrete(guide=guide_legend(nrow=2))
    return p1

def breaks2(x):
    return np.arange(np.floor(x[0]), np.ceil(x[1]) + 1, 2)

def _get_xmin_xmax_breaks(count_estimates, estimates, ID, id_idx):
    x_min = count_estimates[id_idx]["Peaks/Bands Estimate"].min()
    x_max = count_estimates[id_idx]["Peaks/Bands Estimate"].max()
    vline_min = estimates[estimates["ID"]==ID]["Peaks/Bands Estimate"].min()
    vline_max = estimates[estimates["ID"]==ID]["Peaks/Bands Estimate"].max()
    xmin, xmax = round(min(x_min, vline_min)), round(max(x_max, vline_max))
    return scale_x_continuous(breaks=breaks2, limits=(xmin - 1, xmax + 1))

def plot_hists(count_estimates, estimates, ID):
    id_idx = (count_estimates["ID"]==ID)
    boot_idx = (count_estimates["Method"]=="bootstrap sample")
    boot_estimates = count_estimates[boot_idx & id_idx].copy()
    p2 = ggplot(boot_estimates, aes(x="Peaks/Bands Estimate"))
    p2 = p2 + geom_histogram(aes(color="", fill=""), binwidth=1, center=0, alpha=0.6, position="identity")
    p2 = p2 + facet_wrap("~Variable", ncol=1, scales='free_x')
    p2 = p2 + my_theme
    p2 = p2 + geom_vline(data=count_estimates[(~boot_idx) & id_idx], mapping=aes(xintercept="Peaks/Bands Estimate", color="Method"), fill=None, linetype="dashed", size=1.5, inherit_aes=False)
    p2 = p2 + geom_vline(data=estimates[estimates["ID"]==ID], mapping=aes(xintercept="Peaks/Bands Estimate", color="Method"), fill=None, linetype="dashed", size=1.5, inherit_aes=False)
    p2 = p2 + scale_fill_discrete(guide=guide_legend(nrow=2))
    p2 = p2 + labs(y="") + _get_xmin_xmax_breaks(count_estimates, estimates, ID, id_idx)
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

VAR_SUBSET = ["Gray Value", "Sr isotopic ratio"]

if __name__ == "__main__":
    filename = "out/series_and_smooths.csv"
    if os.path.exists(filename):
        series = pd.read_csv(filename)
        series[""] = series['Unnamed: 5']
    else:
        series = ggi.get_time_series_all(ggi.read_files_all())
        series = clean_labels(series)
        series = series[series["Variable"].isin(VAR_SUBSET)]
        estimates = pd.read_csv(os.path.join(ggi.DATA_PATH, "estimates.csv"), dtype={"ID": str})

        series, means = ggi.demean_series(series)
        series, n_nan = ggi.impute_nan(series)

        series[""] = "Data"

        max_ages = {f"Specimen {ID}": max(5, 3*estimates[estimates["ID"]==ID]["Estimate"].max()) for ID in estimates["ID"].unique()}
    
        for ID in series["ID"].unique():
            for var in series["Variable"].unique():
                y = series[(series["Variable"]==var) & (series["ID"]==ID)]["Value"].values
                series.loc[(series["Variable"]==var) & (series["ID"]==ID), "Sample Index"] = np.linspace(0, 1, num=len(y), endpoint=True)
                if np.all(np.isnan(y)):
                    continue

                counter = band_count.BandCounter(y, max_bands=max_ages[ID])
                s = counter.get_smoothed(True).smoothed
                s_df = pd.DataFrame({
                    "Variable": var,
                    "ID": ID,
                    "Value": s,
                    "Sample Index": np.linspace(0, 1, num=len(y), endpoint=True),
                    "": "Smooth (m=3x)",
                })
                series = pd.concat(
                    (series, s_df),
                    ignore_index=True
                )

                counter = band_count.BandCounter(y, max_bands=40)
                s40 = counter.get_smoothed(True).smoothed
                s40_df = pd.DataFrame({
                    "Variable": var,
                    "ID": ID,
                    "Value": s40,
                    "Sample Index": np.linspace(0, 1, num=len(y), endpoint=True),
                    "": "Smooth (m=40)",
                })
                series = pd.concat(
                    (series, s40_df),
                    ignore_index=True
                )
                series.to_csv(filename, index=False)
    
    IDs = ggi.GG_SR_IDS
    ids_sets = [[f"Specimen {ID}" for ID in IDs[:5]], [f"Specimen {ID}" for ID in IDs[5:]]]
    for i, ids in enumerate(ids_sets):
        subset = series[series["ID"].isin(ids)].copy()
        subset["Facet"] = subset["ID"] + " - " + subset["Variable"]
        data = subset[subset[""]=="Data"]
        smooths = subset[subset[""]!="Data"]

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
        data["Value"] = data["Value"].clip(
            lower=data["y_mean"] - 4 * data["y_std"],
            upper=data["y_mean"] + 4 * data["y_std"]
        )

        smooths = smooths.merge(stats, on="Variable", how="left")
        smooths["Value"] = smooths["Value"].clip(
            lower=smooths["y_mean"] - 4 * smooths["y_std"],
            upper=smooths["y_mean"] + 4 * smooths["y_std"]
        )
        smooths_3x = smooths[smooths[""]=="Smooth (m=3x)"]
        smooths_40 = smooths[smooths[""]=="Smooth (m=40)"]

        p1 = (
            ggplot(smooths[smooths["Variable"]=="Gray Value"], aes(x="Sample Index", y="Value", color=""))
            + geom_point(data[data["Variable"]=="Gray Value"], aes(x="Sample Index", y="Value"), color="grey", alpha=0.5)
            + geom_line(smooths_3x[smooths_3x["Variable"]=="Gray Value"], aes(x="Sample Index", y="Value"), color="red", size=2.5, alpha=1)
            + geom_line(smooths_40[smooths_40["Variable"]=="Gray Value"], aes(x="Sample Index", y="Value"), color="navy", size=1.25, alpha=1)
            + facet_grid("ID~Variable", scales='free_y')
            + my_theme + theme(figure_size=(4,8), panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
                                   axis_ticks_major=element_blank(), axis_ticks_minor=element_blank(), axis_ticks=element_blank(), axis_text=element_blank())
            + labs(y="")
        )

        p2 = (
            ggplot(smooths[smooths["Variable"]=="Sr isotopic ratio"], aes(x="Sample Index", y="Value", color=""))
            + geom_point(data[data["Variable"]=="Sr isotopic ratio"], aes(x="Sample Index", y="Value"), color="grey", alpha=0.5)
            + geom_line(smooths_3x[smooths_3x["Variable"]=="Sr isotopic ratio"], aes(x="Sample Index", y="Value"), color="red", size=2.5, alpha=1)
            + geom_line(smooths_40[smooths_40["Variable"]=="Sr isotopic ratio"], aes(x="Sample Index", y="Value"), color="navy", size=1.25, alpha=1)
            + facet_grid("ID~Variable", scales='free_y')
            + my_theme + theme(figure_size=(4,8), panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
                                   axis_ticks_major=element_blank(), axis_ticks_minor=element_blank(), axis_ticks=element_blank(), axis_text=element_blank())
            + labs(y="")
        )

        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        add_plot_to_ax(p1, ax1)
        ax2 = fig.add_subplot(1, 2, 2)
        add_plot_to_ax(p2, ax2)

        plt.tight_layout()
        plt.savefig(f"out/GG_smooths_IDSet_{i}.pdf")
