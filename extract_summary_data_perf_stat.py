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
import summary_utils as utl
from scipy import stats
import prior_peak_dist as ppd


lengths = ["128", "256", "512", "1024"]
mdl = "gp"
peakszs = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"] if mdl=="lin" else ["0.0", "0.2", "0.4", "0.6"]
def files_list(lengths, peakszs, model):
    freqs = "freqsmany" if model=="gp" else "peaks8"
    files = [f"experiments/perf/{model}/config.{model}_var0.36_{freqs}_length{N}_BPDN_peaksize{peaksz}.yaml.out/summary.txt" for peaksz in peakszs for N in lengths]
    return files

sig_levels =  [str(i) for i in np.arange(50, 100, 5, dtype=int)] + ["99.75"]
find_fields = (
    [
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
    + [f"in posterior {a}% range" for a in sig_levels[:-1]] + ["in posterior range"]
)

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
] + [" "+a+"% " for a in sig_levels]

sep = ' & '
newline = '\\\\\n'
inc_name = False
inc_scenario = False
wrap_every = len(find_fields)
keep_brackets = False
est_var = False

fl = files_list(lengths, peakszs, mdl)
summary_data, data = utl.get_summary(fl, sep, newline, inc_name, inc_scenario, wrap_every, keep_brackets, est_var, find_fields, names)
print(summary_data)

# -----------------------
# Build dataframe
# -----------------------

x = [float(ps) for ps in peakszs]

metrics = names
metric_names = ['Accuracy', 'Bias', 'MSE', 
    "bias 0.05 quantile",
    "bias 0.95 quantile",
    "mse 0.05 quantile",
    "mse 0.95 quantile",
    "accuracy 0.05 quantile",
    "accuracy 0.95 quantile"
]
ci_names = [f'{a}% CI Coverage' for a in sig_levels]
all_names = metric_names+ci_names

rows = []
n = len(lengths)
for metric, metric_label in zip(metrics, all_names):
    y = np.asarray(data[metric], dtype=float).reshape((-1, n))

    for i in range(n):

        if i == 0:
            method = " 128"
        elif i == 1:
            method = " 256"
        elif i == 2:
            method = " 512"
        elif i == 3:
            method = "1024"
        else:
            raise ValueError()

        model = method

        for xi, yi in zip(x, y[:, i]):
            rows.append({
                'Minimum peak size': xi,
                'Value': yi,
                'Metric': metric_label,
                'Model': model
            })

df = pd.DataFrame(rows)
df["Model"] = df["Model"].astype("category")

# -----------------------
# Plot
# -----------------------

df['Minimum peak size'] = df['Minimum peak size'].astype(float)

stat_metrics = df[df["Metric"].isin(metric_names[:3])].copy()
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
        x='Minimum peak size',
        y='Value',
        color='Model',
        # linetype='Model',
        shape='Model'
    ))

    + geom_point(size=2.5, alpha=0.85) 
    + geom_line()
    + geom_ribbon(
        aes(ymin='ymin', ymax='ymax', fill='Model'),
        alpha=0.2,
        color=None
    )
    + facet_wrap('~Metric', nrow=1, scales='free_y')

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
        x='Minimum peak size',
        y=''
    )
)

peaks = []
for mps in peakszs:
    for length in lengths:
        peaks.extend(ppd.get_peak_dist(length, mps))
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
p.save(f"experiments/perf_stats_by_peaksize_{mdl}.png", dpi=300)

stat_metrics['Length'] = stat_metrics['Model'].astype(float)
stat_metrics['Minimum peak size'] = stat_metrics['Minimum peak size'].astype("category")

p = (
    ggplot(stat_metrics, aes(
        x='Length',
        y='Value',
        color='Minimum peak size',
        # linetype='Minimum peak size',
        shape='Minimum peak size'
    ))
    + geom_ribbon(
        aes(ymin='ymin', ymax='ymax', fill='Minimum peak size'),
        alpha=0.2,
        color=None
    )
    + geom_point(size=2.5, alpha=0.85) 
    + geom_line()

    + facet_wrap('~Metric', nrow=1, scales='free_y')
    
    + scale_x_continuous(breaks=[2**i for i in range(0, 15)])
    
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
        x='Length',
        y=''
    )
)

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
p.save(f"experiments/perf_stat_by_length_{mdl}.png", dpi=300)


df["Nominal Coverage"] = np.nan
for a in sig_levels:
    df.loc[df["Metric"]==f"{a}% CI Coverage", "Nominal Coverage"] = float(a)/100

stat_metrics = df[df["Metric"].isin(ci_names)].copy()

p = (
    ggplot(stat_metrics, aes(
        x='Minimum peak size',
        y='Value',
        color='Model',
        # linetype='Model',
        shape='Model'
    ))

    + geom_point(size=2.5, alpha=0.85) 
    + geom_line()

    + facet_wrap('~Metric', nrow=2, scales='free_y')

    + theme_bw()

    + theme(
        figure_size=(12, 6),
        legend_position='bottom',
        legend_title=element_blank(),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13)
    )

    + labs(
        x='Minimum peak size',
        y=''
    )
)

# -----------------------
# Reference lines
# -----------------------

for a in sig_levels:
    p = (
        p
        + geom_hline(
            data=pd.DataFrame({'Metric': [f'{a}% CI Coverage'], 'yintercept': [float(a)/100]}),
            mapping=aes(yintercept='yintercept'),
            linetype='dotted',
            color='black',
            inherit_aes=False
        )
    )

p.show()
p.save(f"experiments/perf_ci_by_peaksize_{mdl}.png", dpi=300)

stat_metrics['Length'] = stat_metrics['Model'].astype(float)
stat_metrics['Minimum peak size'] = stat_metrics['Minimum peak size'].astype("category")

p = (
    ggplot(stat_metrics, aes(
        x='Length',
        y='Value',
        color='Minimum peak size',
        # linetype='Minimum peak size',
        shape='Minimum peak size'
    ))

    + geom_point(size=2.5, alpha=0.85) 
    + geom_line()

    + facet_wrap('~Metric', nrow=2, scales='free_y')
    
    + scale_x_continuous(breaks=[2**i for i in range(0, 15)])
    
    + theme_bw()

    + theme(
        figure_size=(12, 6),
        legend_position='bottom',
        legend_title=element_blank(),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13)
    )

    + labs(
        x='Length',
        y=''
    )
)
# -----------------------
# Reference lines
# -----------------------

for a in sig_levels:
    p = (
        p
        + geom_hline(
            data=pd.DataFrame({'Metric': [f'{a}% CI Coverage'], 'yintercept': [float(a)/100]}),
            mapping=aes(yintercept='yintercept'),
            linetype='dotted',
            color='black',
            inherit_aes=False
        )
    )

p.show()

p.save(f"experiments/perf_ci_by_length_{mdl}.png", dpi=300)

summary = (
    stat_metrics
    .groupby(["Minimum peak size", "Nominal Coverage"], as_index=False)
    ["Value"]
    .mean()
    .assign(Model="Average", Length="Average", N=128*4)
)
stat_metrics["N"] = 128

stat_metrics = pd.concat((stat_metrics, summary), ignore_index=True)

# Clean columns
stat_metrics["Length"] = stat_metrics["Length"].astype("category")
stat_metrics["MPS"] = stat_metrics["Minimum peak size"].astype("category")


prior_coverage_rows = []
for cov in sig_levels:
    alpha = 1-float(cov)/100
    print(alpha)
    lwradj = np.quantile(peaks, alpha/2, method='lower')
    upradj = np.quantile(peaks, 1-alpha/2, method='higher')
    lwr = np.quantile(peaks, alpha/2)
    upr = np.quantile(peaks, 1-alpha/2)
    in_ci = (lwr <= peaks) & (peaks <= upr)
    prior_coverage_rows.append({
        'Value': np.mean(in_ci),
        'Nominal Coverage': float(cov)/100,
        'lwr': lwr,
        'upr': upr,
        'lwradj': lwradj,
        'upradj': upradj
    })

prior_coverage_df = pd.DataFrame(prior_coverage_rows)

stat_metrics["ymin"] = np.nan
stat_metrics["ymax"] = np.nan

for i,row in stat_metrics.iterrows():
    N = row["N"]
    coverage_data = np.zeros(N, dtype=int)
    coverage_data[:round(row["Value"]*N)] = 1
    lwr, upr, _ = utl.bootstrap_ci(coverage_data, 5000, np.mean, 0.1, 16860)
    stat_metrics.loc[i,"ymin"] = lwr
    stat_metrics.loc[i,"ymax"] = upr

from mizani.palettes import hue_pal

levels = list(stat_metrics["Length"].cat.categories)

# build default palette
palette = hue_pal()(len(levels))

color_map = dict(zip(levels, palette))
color_map["Average"] = "black"   # override

p = (
    ggplot(stat_metrics[stat_metrics["Length"]!="Average"], aes(
        x='Nominal Coverage',
        y='Value',
        color='Length',
        group='Length'
    ))
    + geom_point(size=2.5, alpha=0.85)
    + geom_ribbon(
        aes(ymin='ymin', ymax='ymax', fill='Length'),
        alpha=0.2,
        color=None
    )
    + geom_line() 

    # reference diagonal
    + geom_abline(intercept=0, slope=1, color='black', linetype='dashed')

    + facet_wrap("MPS", nrow=1, scales='free_y')

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
    + labs(y='True Coverage')
    + scale_color_manual(values=color_map)
    + scale_fill_manual(values=color_map)
)

p = (
    p
    + geom_line(
        stat_metrics[stat_metrics["Length"]=="Average"],
        aes(x='Nominal Coverage', y='Value', group=1),
        color='black',
        size=1.2,
        inherit_aes = False
    )
    + geom_point(
        stat_metrics[stat_metrics["Length"]=="Average"],
        aes(x='Nominal Coverage', y='Value'),
        color='black',
        size=2.5,
        inherit_aes = False
    )
    + geom_ribbon(
        stat_metrics[stat_metrics["Length"]=="Average"],
        aes(x='Nominal Coverage', ymin='ymin', ymax='ymax'),# fill='Length'),#, color='black'),
        alpha=0.2,
        fill='black',
        inherit_aes = False
    )
)
p = (
    p 
    + geom_line(
        data=prior_coverage_df,
        mapping=aes(x='Nominal Coverage', y='Value'),
        linetype='dotted',
        color='black',
        inherit_aes=False
    )
)

p.show()
p.save(f"experiments/perf_ci_quantiles_{mdl}.png", dpi=300)

plt.hist(peaks, bins=ppd.integer_bins(peaks))
plt.show()
