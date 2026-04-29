import matplotlib.pyplot as plt
import numpy as np
from plotnine import *
from plotnine.scales import (
    scale_color_manual,
    scale_linetype_manual,
    scale_shape_manual
)
import band_count
import pandas as pd 

x = np.arange(-4,4,0.01)
y1 = band_count.denoising.ricker(x, 1, 0)
y2 = band_count.denoising.morlet5(x, 1, 0)
dfw1 = pd.DataFrame({
    "t": x,
    "Value": y1,
    "Wavelet": np.full(x.shape, "Ricker")
})
dfw2 = pd.DataFrame({
    "t": x,
    "Value": y2,
    "Wavelet": np.full(x.shape, "Morlet 5")
})
dfw = pd.concat((dfw1, dfw2))

# Ricker points
r_p = band_count.denoising.ricker.period / 2
r_y = band_count.denoising.ricker([r_p, -r_p], 1, 0, standardize=False)/np.sqrt(np.sum(band_count.denoising.ricker(x, 1, 0, standardize=False)**2))

df_points_ricker = pd.DataFrame({
    "t": [r_p, -r_p],
    "Value": r_y,
    "Wavelet": "Ricker"
})

# Morlet points
m_p = band_count.denoising.morlet5.period / 2
m_y = band_count.denoising.morlet5([m_p, -m_p], 1, 0, standardize=False)/np.sqrt(np.sum(band_count.denoising.morlet5(x, 1, 0, standardize=False)**2))

df_points_morlet = pd.DataFrame({
    "t": [m_p, -m_p],
    "Value": m_y,
    "Wavelet": "Morlet 5"
})

df_points = pd.concat([df_points_ricker, df_points_morlet])

plot = (
    ggplot(dfw, aes(x='t', y='Value', color='Wavelet')) 
    + geom_line()
    + theme_bw()
    + theme(
        figure_size=(5, 4),
        legend_position='bottom',
        legend_title=element_blank(),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13),
        panel_grid_major_x=element_blank(),  # remove vertical grid
        panel_grid_minor_x=element_blank(),   # remove vertical minor grid
        panel_grid_major_y=element_line(),    # keep horizontal grid
        panel_grid_minor_y=element_blank(),   # optional: remove minor y grid
    )
    + geom_point(
        aes(x='t', y='Value', color='Wavelet'),
        data=df_points,
        size=3,
        shape='x'
    )
)
plot.show()
plot.save(f"experiments/morelet_ricker.png", dpi=300)


df = pd.DataFrame()
for rho in [0.85, 0.9, 0.95, 0.99]:
    D = band_count.denoising.Dictionary(1024, max_corr=rho)
    df_ = pd.DataFrame({
        "Scale": D.dict_scales,
        "Wavelet": D.wavelet_idx,
        "MaxCorr": np.full(D.dict_scales.shape, rho)
    })
    df = pd.concat((df, df_))

df["MaxCorr"] = df["MaxCorr"].astype("category")
df["Wavelet"] = df["Wavelet"].map(lambda x: "Ricker" if x==0 else "Morlet 5").astype("category")

plot = (
    ggplot(df, aes(x='Scale', fill='Wavelet')) 
    + geom_histogram(binwidth=1, alpha=0.3, boundary=0.5, position='identity')
    + facet_wrap('~MaxCorr', nrow=1)
    + theme_bw()
    + theme(
        figure_size=(12, 4),
        legend_position='bottom',
        legend_title=element_blank(),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_text=element_text(size=12),
        strip_text=element_text(size=13),
        panel_grid_major_x=element_blank(),  # remove vertical grid
        panel_grid_minor_x=element_blank(),   # remove vertical minor grid
        panel_grid_major_y=element_line(),    # keep horizontal grid
        panel_grid_minor_y=element_blank(),   # optional: remove minor y grid
    )
)
plot.show()

N = 256

# band_count.denoising.Dictionary._fft_batch_size = 3

D = band_count.denoising.Dictionary(N, max_corr=0.85)
tmin, tmax = D.get_tlims(N)
x = np.arange(tmin, tmax+1)
z1 = np.zeros((N//4+1, D.n_shifts))
for i,scale in enumerate(D.dict_scales[1:]):
    shift = int(D.dict_shifts[i+1] - np.min(D.dict_shifts))
    z1[int(scale),shift] = 1

fig, ax = plt.subplots(1, 2, figsize=(10,3))
plt.subplot(1,2,1)
plt.imshow(z1, cmap='gray_r')
plt.ylabel("Scale", fontsize=14)
plt.title("MaxCorr 0.85", fontsize=14)
plt.xlabel("Shift", fontsize=14)

D = band_count.denoising.Dictionary(N, max_corr=0.99)
z1 = np.zeros((N//4+1, D.n_shifts))
for i,scale in enumerate(D.dict_scales[1:]):
    shift = int(D.dict_shifts[i] - np.min(D.dict_shifts))
    z1[int(scale),shift] = 1

plt.subplot(1,2,2)
plt.imshow(z1, cmap='gray_r')
plt.ylabel("Scale", fontsize=14)
ax[1].set_yticks([])
ax[1].set_ylabel("")
plt.title("MaxCorr 0.99", fontsize=14)
plt.xlabel("Shift", fontsize=14)
plt.tight_layout()
plt.savefig("experiments/pruned_dictionary_vis.png")
plt.show()

# DD = band_count.denoising.Dictionary(1024, max_corr=None)
