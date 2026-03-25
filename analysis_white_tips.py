import pandas as pd
import numpy as np
import json
import subprocess
from pathlib import Path
from plotnine import * 
import sys 

MAX_BANDS = 40
SEED = 16860
NBOOT = 400
DOWNSAMPLE_FACTOR = int(sys.argv[1])
MAX_BANDS = 40
MAX_CORR = 0.4

birthmark_idx = [432, 549, 298, 266]
specimen_ids = ["43", "44", "55", "56"]
files = [f"data/GreyValues_{i}.csv" for i in specimen_ids]
outpath_stem = f"experiments/whitetips_downsample_factor_{DOWNSAMPLE_FACTOR}_moving_average_ssd_variance/"
outpaths = [f"{outpath_stem}greyvalues{i}/" for i in specimen_ids]

subprocesses = []
for i in range(len(files)):
    p = subprocess.Popen([
        "python3",
        "run_inference.py", 
        files[i],
        outpaths[i],
        str(birthmark_idx[i]),
        str(DOWNSAMPLE_FACTOR),
        str(MAX_BANDS),
        str(MAX_CORR),
        str(NBOOT),
        str(SEED),
    ])
    subprocesses.append(p)

for p in subprocesses:
    p.wait()

bpdn_estimates = []
tp_estimates = []
bpdn_smooths = []
tp_smooths = []
bpdn_boots = []
tp_boots = []

for i, specimen_id in enumerate(specimen_ids):
    path = Path(outpaths[i])

    with open(path.joinpath("bpdn_estimates.json"), "r") as f:
        bpdn_estimates.append(json.load(f))
    with open(path.joinpath("tp_estimates.json"), "r") as f:
        tp_estimates.append(json.load(f))
    
    with open(path.joinpath("bpdn_smooths.json"), "r") as f:
        bpdn_smooths.append(json.load(f))
    with open(path.joinpath("tp_smooths.json"), "r") as f:
        tp_smooths.append(json.load(f))

for i, specimen_id in enumerate(specimen_ids):
    path = Path(outpaths[i])
    with open(path.joinpath("bpdn_boots_locs.json"), "r") as f:
        bpdn_boots_locs = json.load(f)
    with open(path.joinpath("bpdn_boots_counts.json"), "r") as f:
        bpdn_boots_counts = json.load(f)
    with open(path.joinpath("bpdn_boots_smooths.json"), "r") as f:
        bpdn_boot_smooths = json.load(f)
    bpdn_boot = [bpdn_boots_locs, bpdn_boots_counts, bpdn_boot_smooths]
    bpdn_boots.append(bpdn_boot)

for i, specimen_id in enumerate(specimen_ids):
    path = Path(outpaths[i])
    with open(path.joinpath("tp_boots_locs.json"), "r") as f:
        tp_boots_locs = json.load(f)
    with open(path.joinpath("tp_boots_counts.json"), "r") as f:
        tp_boots_counts = json.load(f)
    with open(path.joinpath("tp_boots_smooths.json"), "r") as f:
        tp_boot_smooths = json.load(f)
    tp_boot = [tp_boots_locs, tp_boots_counts, tp_boot_smooths]
    tp_boots.append(tp_boot)

print(len(bpdn_estimates))
print(len(tp_estimates))
print(len(bpdn_smooths))
print(len(tp_smooths))
print(len(bpdn_boots))
print(len(tp_boots))

def demean(v):
    return v - np.mean(v)

specimen_names = [f"Specimen {i}" for i in specimen_ids]
specimens = [pd.read_csv(file)[birthmark_idx[i]::DOWNSAMPLE_FACTOR].reset_index() for (i,file) in enumerate(files)]


# ----------------------------
# Build tidy dataframe
# ----------------------------
rows = []
peak_rows = []

for i in range(4):

    specimen = specimen_names[i]
    raw = demean(specimens[i]["Gray_Value"].values)
    bpdn = bpdn_smooths[i]
    tp = tp_smooths[i]

    n = len(raw)

    for x in range(n):

        rows.append({
            "Pixel": float(x),
            "Gray Value": raw[x],
            "Specimen": specimen,
            "Type": "Sample"
        })

        rows.append({
            "Pixel": float(x),
            "Gray Value": bpdn[x],
            "Specimen": specimen,
            "Type": "BPDN"
        })

        rows.append({
            "Pixel": float(x),
            "Gray Value": tp[x],
            "Specimen": specimen,
            "Type": "Thin-plate"
        })

    # Add peak markers
    bpdn_idx = bpdn_estimates[i][0]
    for ix in bpdn_idx:
        peak_rows.append({
            "Pixel": float(ix),
            "Gray Value": bpdn[ix],
            "Specimen": specimen,
            "Type": "BPDN"
        })

    tp_idx = tp_estimates[i][0]
    for ix in tp_idx:
        peak_rows.append({
            "Pixel": float(ix),
            "Gray Value": tp[ix],
            "Specimen": specimen,
            "Type": "Thin-plate"
        })

df = pd.DataFrame(rows)
df_peaks = pd.DataFrame(peak_rows)

p = (
    ggplot(df, aes(x="Pixel", y="Gray Value", color="Type", linetype="Type"))

    + geom_line(size=1)

    + geom_point(
        data=df_peaks,
        size=2.5
    )

    + facet_wrap("~Specimen", nrow=4, ncol=1, scales="free")

    + scale_color_manual(values={
        "Sample": "gray",
        "BPDN": "#1f77b4",
        "Thin-plate": "#ff7f0e"
    })

    + scale_linetype_manual(values={
        "Sample": "solid",
        "BPDN": "dashed",
        "Thin-plate": "dashed"
    })

    + scale_shape_manual(values={
        "Sample": None,        # none
        "BPDN": "^",         # triangle
        "Thin-plate": "s"    # square
    })

    + theme_bw()

    + theme(
        legend_position="bottom",
        legend_title=element_blank(),
        figure_size=(8, 12),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13)
    )

    + labs(
        x="Pixel",
        y="Gray Value"
    )
)

p.save(f"{outpath_stem}/white_tips_estimates.png")

boot_rows = []
true_rows = []

for i in range(4):

    specimen = specimen_names[i]

    # True signal
    true_signal = demean(specimens[i]["Gray_Value"].values)
    n = len(true_signal)

    for x in range(n):
        true_rows.append({
            "Pixel": x,
            "Gray Value": true_signal[x],
            "Specimen": specimen,
            "Type": "True"
        })

    # Bootstrap smooths
    # bpdn_boots[i][2] and tp_boots[i][2] are lists of smoothed signals
    bpdn_boot_smooths = bpdn_boots[i][2]
    tp_boot_smooths = tp_boots[i][2]

    # BPDN bootstraps
    for b, smooth in enumerate(bpdn_boot_smooths):
        for x in range(len(smooth)):
            boot_rows.append({
                "Pixel": x,
                "Gray Value": smooth[x],
                "Specimen": specimen,
                "Method": "BPDN",
                "Boot_ID": f"BPDN_{b}"
            })

    # Thin-plate bootstraps
    for b, smooth in enumerate(tp_boot_smooths):
        for x in range(len(smooth)):
            boot_rows.append({
                "Pixel": x,
                "Gray Value": smooth[x],
                "Specimen": specimen,
                "Method": "Thin-plate",
                "Boot_ID": f"TP_{b}"
            })

df_boot = pd.DataFrame(boot_rows)
df_true = pd.DataFrame(true_rows)

df_bpdn = df_boot[df_boot["Method"] == "BPDN"].copy()
df_tp   = df_boot[df_boot["Method"] == "Thin-plate"].copy()
p_bpdn = (
    ggplot()

    # Bootstrap smooths
    + geom_line(
        df_bpdn,
        aes(
            x="Pixel",
            y="Gray Value",
            group="Boot_ID"
        ),
        color="#1f77b4",
        alpha=0.15,
        size=0.5
    )

    # True signal
    + geom_line(
        df_true,
        aes(x="Pixel", y="Gray Value"),
        color="black",
        size=1.2
    )

    + facet_wrap("~Specimen", nrow=4, ncol=1, scales="free")

    + theme_bw()

    + theme(
        figure_size=(8, 12),
        legend_position="none",
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13)
    )

    + labs(
        title="BPDN Bootstrap Smooths",
        x="Pixel",
        y="Gray Value"
    )
)

p_bpdn.save(f"{outpath_stem}//white_tips_boot_smooth_bpdn")

p_tp = (
    ggplot()

    # Bootstrap smooths
    + geom_line(
        df_tp,
        aes(
            x="Pixel",
            y="Gray Value",
            group="Boot_ID"
        ),
        color="#ff7f0e",
        alpha=0.15,
        size=0.5
    )

    # True signal
    + geom_line(
        df_true,
        aes(x="Pixel", y="Gray Value"),
        color="black",
        size=1.2
    )

    + facet_wrap("~Specimen", nrow=4, ncol=1, scales="free")

    + theme_bw()

    + theme(
        figure_size=(8, 12),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13),
        legend_position="none"
    )

    + labs(
        title="Thin-Plate Bootstrap Smooths",
        x="Pixel",
        y="Gray Value"
    )
)

p_tp.save(f"{outpath_stem}//white_tips_boot_smooth_tp")


# Build long dataframe for both methods
rows = []

for i, specimen in enumerate(specimen_names):
    # BPDN bootstraps
    for value in bpdn_boots[i][1]:
        rows.append({
            "Specimen": specimen,
            "Peak Count": value,
            "Method": "BPDN",
        })
    # Thin-plate bootstraps
    for value in tp_boots[i][1]:
        rows.append({
            "Specimen": specimen,
            "Peak Count": value,
            "Method": "Thin-plate",
        })

df_boot = pd.DataFrame(rows)

# Build point estimates dataframe
df_est = pd.DataFrame([
    {"Specimen": specimen_names[i], "Estimate": bpdn_estimates[i][1], "Method": "BPDN"}
    for i in range(4)
] + [
    {"Specimen": specimen_names[i], "Estimate": tp_estimates[i][1], "Method": "Thin-plate"}
    for i in range(4)
])

# Plot
p_combined = (
    ggplot(df_boot, aes(x="Peak Count", fill="Method"))
    
    # Semi-transparent histograms
    + geom_histogram(
        alpha=0.4,
        position="identity",
        binwidth=1,
        boundary=-0.5
    )

    # Point estimates as vertical lines
    + geom_vline(
        df_est,
        aes(xintercept="Estimate", color="Method"),
        size=1.2
    )

    + facet_wrap("~Specimen", nrow=4, ncol=1, scales="free_y")

    + scale_fill_manual(values={"BPDN": "#1f77b4", "Thin-plate": "#ff7f0e"})
    + scale_color_manual(values={"BPDN": "#1f77b4", "Thin-plate": "#ff7f0e"})

    + theme_bw()
    + theme(
        figure_size=(8, 12),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13),
        legend_position="bottom",
    )

    + labs(
        title="Bootstrap Distribution of Peak Count",
        x="Number of Peaks",
        y="Frequency"
    )

    # Zoom x-axis from 5 to 25
    + coord_cartesian(xlim=(5, 25))
)

p_combined.save(f"{outpath_stem}//white_tips_boot_hist")
