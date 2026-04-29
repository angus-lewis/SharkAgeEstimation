import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *
from plotnine.scales import (
    scale_color_manual,
    scale_linetype_manual,
    scale_shape_manual
)
from mizani.formatters import number_format
from mizani.breaks import log_breaks
import pandas as pd

import summary_utils as utl
import band_count

signal = "gp"
lengths = [128, 256, 512, 1024]
if signal=="gp":
    corrs = ["0.3", "0.4", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "0.99"]#, "0.999"]
else:
    corrs = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "0.99"]

def files_list(lengths, corrs, model):
    freqs = "freqsmany" if model=="gp" else "peaks8"
    files = [f"experiments/max_corr/timing/{model}/corr{corr}/config.{model}_var0.36_{freqs}_length{N}_BPDN_corr{corr}.yaml.out/summary.txt" for corr in corrs for N in lengths]
    return files

find_fields = [
    "correct",
    "average distance to point estimate",
    "average distance squared to point estimate",
    "run time",
    "Dict memory",
    "smoothing run time",
    "dict build run time",
]

names = [
    " acc ", 
    " bias ",
    " mse ",
    " run time ",
    " memory ",
    " smoothing time ",
    " dict build time ",
]

sep = ' & '
newline = '\\\\\n'
inc_name = True
inc_scenario = False
wrap_every = 7
keep_brackets = False  
est_var = False


summary_data, data = utl.get_summary(files_list(lengths, corrs, signal), sep, newline, inc_name, inc_scenario, wrap_every, keep_brackets, est_var, find_fields, names)
summary_data_1, data_1 = utl.get_summary(files_list(lengths[:-1], ["1.0"], signal), sep, newline, inc_name, inc_scenario, wrap_every, keep_brackets, est_var, find_fields, names)

print(summary_data)
data[' run time '] = np.asarray(data[' run time '], dtype=float)

data_1[' run time '] = np.asarray(data_1[' run time '], dtype=float)

x = [float(corr) for corr in corrs]

metrics = [' acc ', ' bias ', ' mse ', ' run time ', 
           " memory ", " smoothing time ", " dict build time "]
metric_names = ['Accuracy', 'Bias', 'MSE', 'Run Time',
                'Memory (Bytes)', 'Smooth Time (sec)', 'Dict Pruning Time (sec)']

rows = []
n = len(lengths)
for metric, metric_label in zip(metrics, metric_names):
    y = np.asarray(data[metric], dtype=float).reshape((-1, n))

    for i in range(n):

        if i == 0:
            method, resolution = "BPDN", " 128"
        elif i == 1:
            method, resolution = "BPDN", " 256"
        elif i == 2:
            method, resolution = "BPDN", " 512"
        elif i == 3:
            method, resolution = "BPDN", "1024"
        elif i == 4:
            method, resolution = "BPDN", "2048"

        model = f"{resolution}"
        for xi, yi in zip(x, y[:, i]):
            rows.append({
                'Maximum Correlation': xi,
                'Value': yi,
                'Metric': metric_label,
                'Model': model
            })

df = pd.DataFrame(rows)


rows1 = []
n = len(lengths)-1
for metric, metric_label in zip(metrics, metric_names):
    y = np.asarray(data_1[metric], dtype=float).reshape((-1, n))

    for i in range(n):

        if i == 0:
            method, resolution = "BPDN", " 128"
        elif i == 1:
            method, resolution = "BPDN", " 256"
        elif i == 2:
            method, resolution = "BPDN", " 512"
        elif i == 3:
            method, resolution = "BPDN", "1024"
        elif i == 4:
            method, resolution = "BPDN", "2048"

        model = f"{resolution}"
        for xi, yi in zip([1.0], y[:, i]):
            rows1.append({
                'Maximum Correlation': xi,
                'Value': yi,
                'Metric': metric_label,
                'Model': model
            })

df1 = pd.DataFrame(rows1)
df1 = pd.concat((
    df1, 
    pd.DataFrame([{"Maximum Correlation": 1.0, "Value": 2145394688, "Metric": 'Memory (Bytes)', "Model": "1024"}])
), ignore_index=True)

df = pd.concat((df,df1), ignore_index=True)

print(df.head())
df["Model"] = df["Model"].astype("category")

def y_ticks_labels(x):
    ls = [f"{xi}×" for xi in x]
    return ls

# Filter only the "Relative Run Time" rows
mask = (df['Metric'] == 'Smooth Time (sec)')
df.loc[mask, "Value"] = df.loc[mask, "Value"]/128
# Group by Model and normalize
new_rows = df.loc[mask & (df["Maximum Correlation"]!=1.0)].copy()
new_rows["Value"] = (
    new_rows.groupby("Model")["Value"]
            .transform(lambda x: x.max() / x)
)
new_rows["Metric"] = 'Smooth Speed-up vs Max Corr 0.99'
df = pd.concat([df, new_rows], ignore_index=True)

# Filter only the "Relative Run Time" rows
mask = df['Metric'] == 'Memory (Bytes)'
# Group by Model and normalize
new_rows = df.loc[mask].copy()
new_rows["Value"] = new_rows["Value"]/(1024**3) # convert to GiB
new_rows["Metric"] = "Memory (GiB)"
df = pd.concat([df, new_rows], ignore_index=True)
new_rows["Value"] = (
    new_rows.groupby("Model")["Value"]
            .transform(lambda x: x.max() / x)
)
new_rows["Metric"] = "Memory Efficiency (Compression)"
df = pd.concat([df, new_rows], ignore_index=True)

mask1 = df['Metric'] == 'Smooth Time (sec)'
mask2 = df['Metric'] == 'Dict Pruning Time (sec)'
new_rows = df.loc[mask1].copy()
# Group by Model and normalize
new_rows["Value"] = (
    np.asarray(df.loc[mask1, "Value"]) + np.asarray(df.loc[mask2, "Value"])
)
new_rows["Metric"] = "Total Time (sec)"
df = pd.concat([df, new_rows], ignore_index=True)

mask1 = df['Metric'] == 'Smooth Time (sec)'
new_rows = df.loc[mask1].copy()
new_rows["Value"] = (
    new_rows.groupby("Model")["Value"]
            .transform(lambda x: x / x.min())
)
new_rows["Metric"] = "Total Time Speed-up"
df = pd.concat([df, new_rows], ignore_index=True)

time_metrics = df[df["Metric"].isin([
    'Smooth Speed-up vs Max Corr 0.99', 'Total Time Speed-up'
])]
p = (
    ggplot(time_metrics, aes(
        x='Maximum Correlation',
        y='Value',
        color='Model',
        shape='Model'
    ))
    + geom_point(size=2.5, alpha=0.85)
    + geom_line()
    + facet_wrap('~Metric', nrow=1, ncol=2, scales='free_y')
    + theme_bw()
    + theme(
        figure_size=(8, 4),
        legend_position='bottom',
        legend_title=element_blank(),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13)
    )
    + labs(
        x='Maximum Correlation',
        y=''
    )
)

p.show()
p.save(f"experiments/max_corr_rel_time_{signal}.png", dpi=300)

time_metrics = df[df["Metric"].isin([
    'Smooth Speed-up vs Max Corr 0.99', 'Smooth Time (sec)', 'Total Time (sec)'
])]
p = (
    ggplot(time_metrics, aes(
        x='Maximum Correlation',
        y='Value',
        color='Model',
        shape='Model'
    ))
    + geom_point(size=2.5, alpha=0.85)
    + geom_line()
    + facet_wrap('~Metric', nrow=1, scales='free_y')
    + scale_y_continuous(trans='log2', labels=number_format(), breaks=log_breaks(n=8))
    + theme_bw()
    + theme(
        figure_size=(12, 4),
        legend_position='bottom',
        legend_title=element_blank(),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13)
    )
    + labs(
        x='Maximum Correlation',
        y=''
    )
)

p.show()
p.save(f"experiments/max_corr_times_{signal}.png", dpi=300)

time_metrics = df[df["Metric"].isin([
    'Smooth Time (sec)', 'Dict Pruning Time (sec)', 'Total Time (sec)'
])].copy()
# time_metrics = time_metrics[time_metrics["Model"]==512].copy()
p = (
    ggplot(time_metrics, aes(
        x='Maximum Correlation',
        y='Value',
        color='Metric',
        shape='Metric'
    ))
    + geom_point(size=2.5, alpha=0.85)
    + geom_line()
    + facet_wrap('~Model', nrow=1, scales='free_y')
    + scale_y_continuous(trans='log2', labels=number_format(), breaks=log_breaks(n=8))
    + theme_bw()
    + theme(
        figure_size=(12, 3),
        legend_position='bottom',
        legend_title=element_blank(),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13)
    )
    + labs(
        x='Maximum Correlation',
        y='Time (sec)'
    )
)

p.show()
p.save(f"experiments/max_corr_time_{signal}.png", dpi=300)

time_metrics = df[df["Metric"].isin([
    "Memory Efficiency (Compression)", "Memory (GiB)"
])]
p = (
    ggplot(time_metrics, aes(
        x='Maximum Correlation',
        y='Value',
        color='Model',
        shape='Model'
    ))
    + geom_point(size=2.5, alpha=0.85)
    + geom_line()
    + facet_wrap('~Metric', nrow=1, ncol=2, scales='free_y')
    + scale_y_continuous(trans='log2', labels=number_format(), breaks=log_breaks(n=8))
    + theme_bw()
    + theme(
        figure_size=(8, 4),
        legend_position='bottom',
        legend_title=element_blank(),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13)
    )
    + labs(
        x='Maximum Correlation',
        y=''
    )
)

p.show()
p.save(f"experiments/max_corr_mem_{signal}.png", dpi=300)
