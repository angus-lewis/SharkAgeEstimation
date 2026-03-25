import matplotlib.pyplot as plt
import json
import pandas as pd
import band_count
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import pacf, acf


MAX_BANDS = 40
MAX_CORR = 0.6
DOWNSAMPLE_FACTOR = 4

specimen = 44
brithmark_idx = 549 # 432 # 298 # 549 # 432 # 266

grayvalues = pd.read_csv(f"data/GreyValues_{specimen}.csv")[brithmark_idx::1].reset_index()["Gray_Value"]

group = np.arange(len(grayvalues)) // DOWNSAMPLE_FACTOR
grayvalues_subsampled = grayvalues.groupby(group).mean().reset_index(drop=True)
# grayvalues = grayvalues_subsampled

bpdn_counter = band_count.BandCounter(np.asfarray(grayvalues), max_bands=MAX_BANDS, max_corr=MAX_CORR)
bpdn_boots = bpdn_counter.get_count_distribution(100, True, 16860)
smooth = bpdn_counter.get_smoothed(True).smoothed

with open(f"experiments/whitetips_downsample_factor_1/greyvalues{specimen}/bpdn_boots_smooths.json", "r") as f:
    boots = json.load(f)
with open(f"experiments/whitetips_downsample_factor_1/greyvalues{specimen}/bpdn_smooths.json", "r") as f:
    smooths = json.load(f)

with open(f"experiments/whitetips_downsample_factor_1/greyvalues{specimen}/tp_boots_smooths.json", "r") as f:
    tp_boots = json.load(f)
with open(f"experiments/whitetips_downsample_factor_1/greyvalues{specimen}/tp_smooths.json", "r") as f:
    tp_smooths = json.load(f)

plt.figure()
plt.subplot(3,1,1)
plt.plot(np.arange(len(grayvalues)*DOWNSAMPLE_FACTOR, step=DOWNSAMPLE_FACTOR), grayvalues.tolist()-grayvalues.mean(), label="Data")
# plt.plot(smooths, label="BPDN unfiltered")
# plt.plot(np.arange(len(grayvalues)*DOWNSAMPLE_FACTOR, step=DOWNSAMPLE_FACTOR), smooth, label="BPDN filtered dwn")
# plt.plot(np.arange(len(grayvalues)*DOWNSAMPLE_FACTOR, step=DOWNSAMPLE_FACTOR), bpdn_counter.get_smoothed(False).smoothed, label="BPDN unf (dwn)")
# plt.plot(tp_smooths, label="TP")
plt.legend()
plt.subplot(3,1,2)
for i in range(len(boots)):
    plt.plot(boots[i], color="gray", alpha=0.2)
plt.subplot(3,1,3)
for i in range(len(boots)):
    plt.plot(tp_boots[i], color="gray", alpha=0.2)
plt.show()

plt.hist(bpdn_boots[1])
plt.show()


plot_acf(grayvalues[::1], lags=20)
plot_pacf(grayvalues[::1], lags=20)
plt.show()

plot_acf(grayvalues.diff()[1:], lags=20)
plot_pacf(grayvalues.diff()[1:], lags=20)
plt.show()

plot_acf(grayvalues_subsampled, lags=20)
plot_pacf(grayvalues_subsampled, lags=20)
plt.show()

plot_acf(grayvalues_subsampled.diff()[1:], lags=20)
plot_pacf(grayvalues_subsampled.diff()[1:], lags=20)
plt.show()

plt.plot(grayvalues_subsampled)
plt.show()