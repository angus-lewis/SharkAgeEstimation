import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
from plotnine import *
from plotnine.scales import (
    scale_color_manual,
    scale_linetype_manual,
    scale_shape_manual
)
import pandas as pd


def get_summary(files, sep, newline, inc_name, inc_scenario, wrap_every, keep_brackets, est_var):
    data = {}
    summary_data = ""
    lineno = 0
    file_ix = 0
    for file in files:
        scenario = file.split("/")[-2]
        scenario = scenario.split(".")
        scenario = "".join(scenario[1:-2])
        with open(file, "r") as f:
            lines = [""]*len(find_fields)
            for line in f:
                fields = line.split(":")
                name = fields[0]
                value = fields[1].strip()
                value = value[:-1]
                if name in find_fields:
                    name_ix = find_fields.index(name)
                    lines[name_ix] = line
            for line in lines:
                fields = line.split(":")
                name = fields[0]
                value = fields[1].strip()
                value = value[:-1]
                if name in find_fields:
                    name_ix = find_fields.index(name)
                    lineno += 1
                    name = names[name_ix]
                    if inc_scenario:
                        scenario = scenario + sep
                    else:
                        scenario = ""
                    if lineno%wrap_every==0:
                        nl = newline
                    else:
                        nl = sep
                    if keep_brackets:
                        value = value.split(" ")
                        v = float(value[0])
                        if name == " bias " or name == " mse ":
                            v = v*100
                        value[0] = str(round(v, ndigits=3))
                        value = "".join(value)
                    else:
                        v = float(value.split(" ")[0])
                        if name == " bias " or name == " mse ":
                            v = v*100
                        value = str(round(v, ndigits=3))
                    if est_var:
                        v = float(value.split(" ")[0])
                        var = round(pow(v*(1-v)/64, 0.5),3)
                        value = value + " (" + str(var) + ")"
                    if not name in data.keys():
                        data[name] = [0]*len(files)
                    data[name][file_ix] = value
                    if inc_name:
                        name = name + sep
                    else:
                        name = ""
                    summary_data = summary_data + scenario + name + value + nl
        file_ix += 1
    return summary_data, data

files = [
    # "experiments/sin/var0.36/freq8/config.sin_freqs8_len128_var0.36_BPDN.yaml.out/summary.txt",
    # "experiments/sin/var0.36/freq8/config.sin_freqs8_len128_var0.36_TP.yaml.out/summary.txt",
    # "experiments/sin/var0.36/freq8/config.sin_freqs8_len256_var0.36_BPDN.yaml.out/summary.txt",
    # "experiments/sin/var0.36/freq8/config.sin_freqs8_len256_var0.36_TP.yaml.out/summary.txt",
    # "experiments/sin/var0.36/chirp/config.sin_chirp_len128_var0.36_BPDN.yaml.out/summary.txt",
    # "experiments/sin/var0.36/chirp/config.sin_chirp_len128_var0.36_TP.yaml.out/summary.txt",
    # "experiments/sin/var0.36/chirp/config.sin_chirp_len256_var0.36_BPDN.yaml.out/summary.txt",
    # "experiments/sin/var0.36/chirp/config.sin_chirp_len256_var0.36_TP.yaml.out/summary.txt",
    # "experiments/sin/var0.36/multiple/config.sin_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    # "experiments/sin/var0.36/multiple/config.sin_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    # "experiments/sin/var0.36/multiple/config.sin_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    # "experiments/sin/var0.36/multiple/config.sin_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    # "experiments/sin/var0.36/multiple2/config.sin_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    # "experiments/sin/var0.36/multiple2/config.sin_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    # "experiments/sin/var0.36/multiple2/config.sin_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    # "experiments/sin/var0.36/multiple2/config.sin_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/config.lin_var0.36_peaks4_length128_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/config.lin_var0.36_peaks4_length128_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/config.lin_var0.36_peaks4_length256_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/config.lin_var0.36_peaks4_length256_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/_config.lin_var0.36_peaks4_length512_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/_config.lin_var0.36_peaks4_length512_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/_config.lin_var0.36_peaks8_length512_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/_config.lin_var0.36_peaks8_length512_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.03/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.03/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.03/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.03/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.06/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.06/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.06/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.06/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.09/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.09/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.09/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.09/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.12/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.12/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.12/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.12/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.15/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.15/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.15/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.15/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.18/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.18/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.18/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.18/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.21/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.21/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.21/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.21/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.30/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.30/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.30/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.30/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.40/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.40/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.40/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.40/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.50/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.50/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.50/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.50/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.60/var0.36/multiple/config.gp_freqsmany_len128_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.60/var0.36/multiple/config.gp_freqsmany_len128_var0.36_TP.yaml.out/summary.txt",
    "experiments/gp_peaksize0.60/var0.36/multiple/config.gp_freqsmany_len256_var0.36_BPDN.yaml.out/summary.txt",
    "experiments/gp_peaksize0.60/var0.36/multiple/config.gp_freqsmany_len256_var0.36_TP.yaml.out/summary.txt"
]

find_fields = [
    "correct",
    "average distance to point estimate",
    "average distance squared to point estimate",
    "in posterior 50% range",
    "in posterior 80% range",
    "in posterior 90% range"
    # "average posterior prob of true value"
    
]

names = [
    " acc ", 
    " bias ",
    " mse ",
    " 50% ", 
    " 80% ",
    " 90% ", 
    # " 95% ",
    # " confidence ",
]

sep = ' & '
newline = '\\\\\n'
inc_name = False
inc_scenario = False
wrap_every = 6
keep_brackets = False
est_var = False

# def var(value):
#     v = value.split(" ")
#     p = float(v[0])
#     var = pow(p*(1-p)/200, 0.5)
#     value = value + " (" + str(var) + ")"
#     return value


summary_data, data = get_summary(files, sep, newline, inc_name, inc_scenario, wrap_every, keep_brackets, est_var)
print(summary_data)


# -----------------------
# Build dataframe
# -----------------------

x = [
    0.00, 0.03, 0.06, 0.09, 0.12, 0.15,
    0.18, 0.21, 0.30, 0.40, 0.50, 0.60
]

metrics = [' acc ', ' bias ', ' mse ', ' 50% ', ' 80% ', ' 90% ']
metric_names = ['Accuracy', 'Bias', 'MSE',
                '50% CI Coverage', '80% CI Coverage', '90% CI Coverage']

rows = []

for metric, metric_label in zip(metrics, metric_names):
    y = np.asarray(data[metric], dtype=float).reshape((-1, 4))

    for i in range(4):

        if i == 0:
            method, resolution = "BPDN", "128"
        elif i == 1:
            method, resolution = "Thin-plate", "128"
        elif i == 2:
            method, resolution = "BPDN", "256"
        elif i == 3:
            method, resolution = "Thin-plate", "256"

        model = f"{method}-{resolution}"

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

p = (
    ggplot(df, aes(
        x='Minimum peak size',
        y='Value',
        color='Model',
        linetype='Model',
        shape='Model'
    ))

    + geom_point(size=2.5, alpha=0.85)
    + geom_smooth(method='loess', span=0.8, se=False, size=1.2)
    # + geom_smooth(method='lm', formula='y ~ x + I(x**2)', se=False, size=1.2)

    + facet_wrap('~Metric', nrow=2, ncol=3, scales='free_y')

    # Manual styling per model
    + scale_color_manual(values={
        'BPDN-128': '#1f77b4',
        'BPDN-256': '#1f77b4',
        'Thin-plate-128': '#ff7f0e',
        'Thin-plate-256': '#ff7f0e'
    })

    + scale_linetype_manual(values={
        'BPDN-128': 'dashed',
        'Thin-plate-128': 'dashed',
        'BPDN-256': 'solid',
        'Thin-plate-256': 'solid'
    })

    + scale_shape_manual(values={
        'BPDN-128': 'o',
        'Thin-plate-128': 'o',
        'BPDN-256': '^',
        'Thin-plate-256': '^'
    })

    + theme_bw()

    + theme(
        figure_size=(10, 6),
        legend_position='bottom',
        legend_title=element_blank(),
        subplots_adjust={'wspace': 0.25},
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

p = (
    p
    + geom_hline(
        data=pd.DataFrame({'Metric': ['50% CI Coverage'], 'yintercept': [0.5]}),
        mapping=aes(yintercept='yintercept'),
        linetype='dotted',
        color='black',
        inherit_aes=False
    )
    + geom_hline(
        data=pd.DataFrame({'Metric': ['80% CI Coverage'], 'yintercept': [0.8]}),
        mapping=aes(yintercept='yintercept'),
        linetype='dotted',
        color='black',
        inherit_aes=False
    )
    + geom_hline(
        data=pd.DataFrame({'Metric': ['90% CI Coverage'], 'yintercept': [0.9]}),
        mapping=aes(yintercept='yintercept'),
        linetype='dotted',
        color='black',
        inherit_aes=False
    )
)

p.save("experiments/gp_perfomance.png", dpi=300)





files = [
    # "experiments/lin/var0.36/peaks4/config.lin_var0.36_peaks4_length128_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/config.lin_var0.36_peaks4_length128_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/config.lin_var0.36_peaks4_length256_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/config.lin_var0.36_peaks4_length256_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/_config.lin_var0.36_peaks4_length512_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks4/_config.lin_var0.36_peaks4_length512_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/_config.lin_var0.36_peaks8_length512_BPDN.yaml.out/summary.txt",
    # "experiments/lin/var0.36/peaks8/_config.lin_var0.36_peaks8_length512_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.05/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.05/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.05/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.05/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.10/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.10/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.10/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.10/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.15/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.15/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.15/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.15/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.20/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.20/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.20/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.20/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.25/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.25/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.25/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.25/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.30/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.30/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.30/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.30/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.40/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.40/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.40/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.40/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.50/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.50/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.50/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.50/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.60/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.60/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.60/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.60/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.80/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.80/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize0.80/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize0.80/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize1.00/var0.36/peaks8/config.lin_var0.36_peaks8_length128_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize1.00/var0.36/peaks8/config.lin_var0.36_peaks8_length128_TP.yaml.out/summary.txt",
    "experiments/lin_peaksize1.00/var0.36/peaks8/config.lin_var0.36_peaks8_length256_BPDN.yaml.out/summary.txt",
    "experiments/lin_peaksize1.00/var0.36/peaks8/config.lin_var0.36_peaks8_length256_TP.yaml.out/summary.txt",
]

find_fields = [
    "correct",
    "average distance to point estimate",
    "average distance squared to point estimate",
    "in posterior 50% range",
    "in posterior 80% range",
    "in posterior 90% range"
    # "average posterior prob of true value"
    
]

names = [
    " acc ", 
    " bias ",
    " mse ",
    " 50% ", 
    " 80% ",
    " 90% ", 
    # " 95% ",
    # " confidence ",
]

sep = ' & '
newline = '\\\\\n'
inc_name = False
inc_scenario = False
wrap_every = 6
keep_brackets = False
est_var = False

summary_data, data = get_summary(files, sep, newline, inc_name, inc_scenario, wrap_every, keep_brackets, est_var)
print(summary_data)


# -----------------------
# Build dataframe
# -----------------------

x = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.60, 0.80, 1.00
]

metrics = [' acc ', ' bias ', ' mse ', ' 50% ', ' 80% ', ' 90% ']
metric_names = ['Accuracy', 'Bias', 'MSE',
                '50% CI Coverage', '80% CI Coverage', '90% CI Coverage']

rows = []

for metric, metric_label in zip(metrics, metric_names):
    y = np.asarray(data[metric], dtype=float).reshape((-1, 4))

    for i in range(4):

        if i == 0:
            method, resolution = "BPDN", "128"
        elif i == 1:
            method, resolution = "Thin-plate", "128"
        elif i == 2:
            method, resolution = "BPDN", "256"
        elif i == 3:
            method, resolution = "Thin-plate", "256"

        model = f"{method}-{resolution}"

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
# Create a small DataFrame for the extra point
bias = df['Value'][df["Metric"]=='Bias']
extra_point = pd.DataFrame({
    'Minimum peak size': [0, 0],  # x-coordinate
    'Value': [-max(abs(bias)), max(abs(bias))],              # y-coordinate
    'Metric': ['Bias','Bias'],           # facet it belongs to
    'Model': ['','']            # model/label
})

p = (
    ggplot(df, aes(
        x='Minimum peak size',
        y='Value',
        color='Model',
        linetype='Model',
        shape='Model'
    ))

    + geom_point(size=2.5, alpha=0.85)
    + geom_point(data=extra_point, mapping=aes(x='Minimum peak size', y='Value', color='Model', shape='Model'), size=0, alpha=0, show_legend=False)
    + geom_smooth(method='loess', span=0.8, se=False, size=1.2)
    # + geom_smooth(method='lm', formula='y ~ x + I(x**2)', se=False, size=1.2)

    + facet_wrap('~Metric', nrow=2, ncol=3, scales='free_y')

    # Manual styling per model
    + scale_color_manual(values={
        'BPDN-128': '#1f77b4',
        'BPDN-256': '#1f77b4',
        'Thin-plate-128': '#ff7f0e',
        'Thin-plate-256': '#ff7f0e',
        '': '#ffffff'
    })

    + scale_linetype_manual(values={
        'BPDN-128': 'dashed',
        'Thin-plate-128': 'dashed',
        'BPDN-256': 'solid',
        'Thin-plate-256': 'solid',
        '': 'solid'
    })

    + scale_shape_manual(values={
        'BPDN-128': 'o',
        'Thin-plate-128': 'o',
        'BPDN-256': '^',
        'Thin-plate-256': '^',
        '': 'o'
    })

    + theme_bw()

    + theme(
        figure_size=(10, 6),
        legend_position='bottom',
        legend_title=element_blank(),
        subplots_adjust={'wspace': 0.25},
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

p = (
    p
    + geom_hline(
        data=pd.DataFrame({'Metric': ['50% CI Coverage'], 'yintercept': [0.5]}),
        mapping=aes(yintercept='yintercept'),
        linetype='dotted',
        color='black',
        inherit_aes=False
    )
    + geom_hline(
        data=pd.DataFrame({'Metric': ['80% CI Coverage'], 'yintercept': [0.8]}),
        mapping=aes(yintercept='yintercept'),
        linetype='dotted',
        color='black',
        inherit_aes=False
    )
    + geom_hline(
        data=pd.DataFrame({'Metric': ['90% CI Coverage'], 'yintercept': [0.9]}),
        mapping=aes(yintercept='yintercept'),
        linetype='dotted',
        color='black',
        inherit_aes=False
    )
)

p.save("experiments/lin_perfomance.png", dpi=300)
