import numpy as np 
import pywt

import denoising
import count
import model_utils

def estimate_count(ts, max_age=-1, mortality_rate=None, scale_switch=np.inf, scales=None, shifts=None, verbose=False):
    # center the data
    ts = np.asarray(ts, dtype=np.float64)
    ts -= np.mean(ts)

    # construct function dictionary of ricker wavelets
    # if not explictly specified, need to determine the scales and shifts at each scale that we want to use
    # by default the scales 1,2,..., len(ts)//4 are used but for long ts (e.g. len(ts)>1500) this can be slow
    # if scale_switch is specified, then the dictionary becomes sparser for scales>scale_switch; every
    # second scales is used and every second shift is used.
    switch_idx = max(2, min(len(ts) // 4 + 1, scale_switch + 1))
    last_idx = max(2, len(ts) // 4 + 1)

    if scales is None: 
        scales1 = np.arange(1, switch_idx)
        scales2 = np.arange(switch_idx, last_idx, step=2)
        scales = np.concatenate((scales1, scales2))

    if shifts is None: 
        shifts = np.concatenate((np.ones(len(scales1)),2*np.ones(len(scales2))))
    
    dictionary = denoising.ricker_cwt_dictionary(len(ts), scales, shifts, dtype=np.float64)
    expanded_scales = denoising.dictionary_scales(len(ts), scales, shifts)
    expanded_shifts = denoising.dictionary_shifts(len(ts), scales, shifts)

    # dictionary terms with scale=n are 'mounds' with approx 3.6n points.
    # Mounds which are too short can be filtered out to help smooth the signal.
    # Determine which frequencies we want to keep.
    if max_age > 0:
        min_pts_per_year = len(ts)/max_age
        if verbose:
            print(f"Min pts per year: {min_pts_per_year}.")
        pts_per_mound = 3.6*expanded_scales
        # Keep these frequencies (low freq only)
        keep_freq_scales_ix = pts_per_mound > (min_pts_per_year)
    else:
        num_atoms = denoising.get_num_atoms(len(ts), scales, shifts)
        keep_freq_scales_ix = np.full(num_atoms, True, dtype=np.bool)

    if mortality_rate is None:
        prior = model_utils.make_peak_prior(dictionary, keep_freq_scales_ix, mortality_rate)
    else:
        prior = denoising.LassoLarsBIC._no_penalty
    
    # basis pursuit smoothing
    coef, smoothed = denoising.basis_pursuit_denoising(ts, dictionary, prior, verbose=verbose)

    # determine the number of peaks (i.e., age)
    age_smoothed = count.find_peaks(smoothed)
    
    # further band limited smoothing 
    low_freq_coef = coef * keep_freq_scales_ix
    if max_age > 0:
        low_freq_smoothed = np.dot(dictionary, low_freq_coef)
        # determine the number of peaks in smoothed signal
        age_filtered = count.find_peaks(low_freq_smoothed)
    else:
        low_freq_smoothed = None

    cwt, _ = pywt.cwt(ts, np.arange(1,np.maximum(scales)+1), 'mexh')
    cwt_count = count.detect_path()



