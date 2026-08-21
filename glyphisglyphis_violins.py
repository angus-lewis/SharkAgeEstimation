import glyphisglyphis_inference as ggi
import glyphisglyphis_analysis as gga

import pandas as pd
import matplotlib.pyplot as plt
import os 
from plotnine import *
import numpy as np

VAR_SUBSET = ["Gray Value", "Sr isotopic ratio"]

if __name__ == "__main__":
    count_estimates_filename = os.path.join(".","out","count_estimates_max_bands_3x_age_estimates.csv")
    point_estimates, boot_estimates = gga.get_clean_estimates(count_estimates_filename)
    point_estimates = point_estimates[point_estimates["Variable"].isin(VAR_SUBSET)]
    boot_estimates = boot_estimates[boot_estimates["Variable"].isin(VAR_SUBSET)]
    
    estimates = pd.read_csv(os.path.join(ggi.DATA_PATH, "estimates.csv"), dtype={"ID": str})
    estimates["Variable"] = "Sr isotopic ratio" # "Sr88_ppm"
    estimates = gga.clean_labels(estimates)

    offset = 0.15
    estimates.loc[estimates["Method"]=="TLOM","Peaks/Bands Estimate"] = estimates[estimates["Method"]=="TLOM"]["Peaks/Bands Estimate"]+offset
    estimates.loc[estimates["Method"]=="microXRF","Peaks/Bands Estimate"] = estimates[estimates["Method"]=="microXRF"]["Peaks/Bands Estimate"]-offset

    labels = estimates[["Readability", "Method", "ID", "Peaks/Bands Estimate"]].copy()
    labels = labels.fillna(method='ffill')
    labels["Peaks/Bands Estimate"] = labels["Peaks/Bands Estimate"] - 8*offset + 14*offset*np.asarray(estimates["Method"]=="TLOM")
    labels["Variable"] = "Sr isotopic ratio" # "Sr88 (ppm)"
    labels["Readability"] = labels["Readability"].astype(str)

    p = ggplot(boot_estimates, aes(y="Peaks/Bands Estimate", x="Variable", fill="Variable")) + geom_violin(bw=0.5) + facet_wrap("ID", nrow=2)
    p = p + geom_point(point_estimates, aes(y="Peaks/Bands Estimate", x="Variable"), size=1, alpha=0.7)
    p = p + geom_hline(estimates, aes(yintercept="Peaks/Bands Estimate", color="Method"), size=1, alpha=0.7)
    p = p + gga.my_theme + theme(figure_size=(12,7), axis_text_x=element_text(angle=45, hjust=1))
    p = p + scale_y_continuous(breaks=range(0, 22, 2))
    p = p + geom_text(labels, aes(x="Variable", y="Peaks/Bands Estimate", label="Readability", color="Method"), nudge_x = 0.5, size=12, show_legend=False)
    p = p + labs(x="")
    # p.show()
    p.save(os.path.join(".", "out", f"GG_subset{count_estimates_filename.replace(".", "").replace("/", "_")}.pdf"))

    point_estimates["Max Age"] = "3x"
    boot_estimates["Max Age"] = "3x"

    count_estimates_40_filename = os.path.join(".","out","count_estimates_max_age_40.csv")
    point_estimates_40, boot_estimates_40 = gga.get_clean_estimates(count_estimates_40_filename)
    point_estimates_40 = point_estimates_40[point_estimates_40["Variable"].isin(VAR_SUBSET)]
    boot_estimates_40 = boot_estimates_40[boot_estimates_40["Variable"].isin(VAR_SUBSET)]
    point_estimates_40["Max Age"] = "40"
    boot_estimates_40["Max Age"] = "40"

    count_estimates_20_filename = os.path.join(".","out","count_estimates_max_age_20.csv")
    point_estimates_20, boot_estimates_20 = gga.get_clean_estimates(count_estimates_20_filename)
    point_estimates_20 = point_estimates_20[point_estimates_20["Variable"].isin(VAR_SUBSET)]
    boot_estimates_20 = boot_estimates_20[boot_estimates_20["Variable"].isin(VAR_SUBSET)]
    point_estimates_20["Max Age"] = "20"
    boot_estimates_20["Max Age"] = "20"
    
    point_estimates_combined = pd.concat((point_estimates, point_estimates_40), ignore_index=True)
    boot_estimates_combined = pd.concat((boot_estimates, boot_estimates_40), ignore_index=True)

    p = ggplot(boot_estimates_combined, aes(y="Peaks/Bands Estimate", x="Variable", fill="Max Age")) 
    p = p + geom_violin(bw=0.5) + facet_wrap("ID", nrow=2)
    p = p + geom_point(point_estimates_combined, aes(y="Peaks/Bands Estimate", x="Variable", fill="Max Age"), size=1, alpha=0.7, position=position_dodge(width=0.9))
    p = p + geom_hline(estimates, aes(yintercept="Peaks/Bands Estimate", color="Method"), size=1, alpha=0.7)
    p = p + gga.my_theme + theme(figure_size=(12,7), axis_text_x=element_text(angle=45, hjust=1))
    p = p + geom_text(labels, aes(x="Variable", y="Peaks/Bands Estimate", label="Readability", color="Method"), nudge_x = 0.5, size=12, show_legend=False, inherit_aes=False)
    p = p + scale_y_continuous(breaks=range(0, 28, 2))
    p = p + labs(x="")
    p.show()
    p.save(os.path.join(".", "out", f"GG_subset_sensitivity{count_estimates_filename.replace(".", "").replace("/", "_")}.pdf"))
