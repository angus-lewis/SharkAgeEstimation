import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *
from plotnine.scales import (
    scale_color_manual,
    scale_linetype_manual,
    scale_shape_manual
)
import pandas as pd
from scipy import stats

import summary_utils as utl
import prior_peak_dist as ppd

signal = "gp"
lengths = [128, 256, 512, 1024]
if signal=="gp":
    corrs = ["0.3", "0.4", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "0.99"]
else:
    corrs = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "0.99"]

def files_list(lengths, corrs, model):
    freqs = "freqsmany" if model=="gp" else "peaks8"
    files = [f"experiments/max_corr/{model}/corr{corr}/config.{model}_var0.36_{freqs}_length{N}_BPDN_corr{corr}.yaml.out/summary.txt" for corr in corrs for N in lengths]
    files = files
    return files


find_fields = [
    "correct",
    "average distance to point estimate",
    "average distance squared to point estimate",
    "bias 0.05 quantile",
    "bias 0.95 quantile",
    "mse 0.05 quantile",
    "mse 0.95 quantile",
    "accuracy 0.05 quantile",
    "accuracy 0.95 quantile",
]

names = [
    " acc ", 
    " bias ",
    " mse ",
    " bias 0.05 quantile ",
    " bias 0.95 quantile ",
    " mse 0.05 quantile ",
    " mse 0.95 quantile ",
    " accuracy 0.05 quantile ",
    " accuracy 0.95 quantile ",
]

sep = ' & '
newline = '\\\\\n'
inc_name = True
inc_scenario = False
wrap_every = 9
keep_brackets = False  
est_var = False

summary_data, data = utl.get_summary(files_list(lengths, corrs, signal), sep, newline, inc_name, inc_scenario, wrap_every, keep_brackets, est_var, find_fields, names)
print(summary_data)

summary_data_1, data_1 = utl.get_summary(files_list(lengths[:-1], ["1.0"], signal), sep, newline, inc_name, inc_scenario, wrap_every, keep_brackets, est_var, find_fields, names)

x = [float(corr) for corr in corrs]

metrics = [' acc ', ' bias ', ' mse ', 
            " bias 0.05 quantile ",
            " bias 0.95 quantile ",
            " mse 0.05 quantile ",
            " mse 0.95 quantile ",
            " accuracy 0.05 quantile ",
            " accuracy 0.95 quantile "
]
metric_names = ['Accuracy', 'Bias', 'MSE',
    "bias 0.05 quantile",
    "bias 0.95 quantile",
    "mse 0.05 quantile",
    "mse 0.95 quantile",
    "accuracy 0.05 quantile",
    "accuracy 0.95 quantile",
]

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

stat_metrics = df[df["Metric"].isin(['Accuracy', 'Bias', 'MSE'])].copy()
stat_metrics["ymin"] = 0.
stat_metrics["ymax"] = 0.
stat_metrics.loc[stat_metrics["Metric"]=="Accuracy","ymin"] = df.loc[df["Metric"]=="accuracy 0.05 quantile", "Value"].values
stat_metrics.loc[stat_metrics["Metric"]=="Accuracy","ymax"] = df.loc[df["Metric"]=="accuracy 0.95 quantile", "Value"].values
stat_metrics.loc[stat_metrics["Metric"]=="Bias","ymin"] = df.loc[df["Metric"]=="bias 0.05 quantile", "Value"].values
stat_metrics.loc[stat_metrics["Metric"]=="Bias","ymax"] = df.loc[df["Metric"]=="bias 0.95 quantile", "Value"].values
stat_metrics.loc[stat_metrics["Metric"]=="MSE","ymin"] = df.loc[df["Metric"]=="mse 0.05 quantile", "Value"].values
stat_metrics.loc[stat_metrics["Metric"]=="MSE","ymax"] = df.loc[df["Metric"]=="mse 0.95 quantile", "Value"].values

p = (
    ggplot(stat_metrics, aes(
        x='Maximum Correlation',
        y='Value',
        color='Model',
        shape='Model'
    ))
    + geom_ribbon(
        aes(ymin='ymin', ymax='ymax', fill='Model'),
        alpha=0.2,
        color=None
    )
    + geom_point(size=2.5, alpha=0.85)
    + geom_line()
    + facet_wrap('~Metric', nrow=1, ncol=3, scales='free_y')
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

peaks = []
# for corr in corrs:
for length in lengths:
    peaks.extend(ppd.get_peak_dist(length, 0.8, "max_corr", range(256)))

mode = stats.mode(peaks, keepdims=True)

acc = mode.count[0] / len(peaks)
bias = np.mean(mode.mode[0] - peaks)
mse = np.mean((peaks - mode.mode[0])**2)

p = (
    p
    + geom_hline(
        data=pd.DataFrame({'Metric': ['Accuracy'], 'yintercept': [acc]}),
        mapping=aes(yintercept='yintercept'),
        linetype='dotted',
        color='black',
        inherit_aes=False
    )
)

p = (
    p
    + geom_hline(
        data=pd.DataFrame({'Metric': ['Bias'], 'yintercept': [bias]}),
        mapping=aes(yintercept='yintercept'),
        linetype='dotted',
        color='black',
        inherit_aes=False
    )
)

p = (
    p
    + geom_hline(
        data=pd.DataFrame({'Metric': ['MSE'], 'yintercept': [mse]}),
        mapping=aes(yintercept='yintercept'),
        linetype='dotted',
        color='black',
        inherit_aes=False
    )
)

p.show()
p.save(f"experiments/max_corr_perf_{signal}.png", dpi=300)
