# Implement laminae detection based on WLCount methodlogy (see [1])
# Copyright (C) 2025  Angus Lewis

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# [1] Oriani F., Treble P. C., Baker A., Mariethoz G., WlCount: Geological 
# lamination detection and counting using an image analysis approach, 
# Computers & Geoscience, https://doi.org/10.1016/j.cageo.2022.105037

import numpy as np

def lin_interp(a,b,t):
    """Linearly interpolate between a and b.
    
    Args:
        a (real): start point
        b (real): end point
        t (array-like): points to evaluate the interpolation function.
    """
    return a + (b-a)*t

def get_seg(x1,y1,x2,y2):
    """Determine integer indices of line segment between (x1,y1) and (x2,y2)
    
    Interpolates (by repeating values) the shorter of the x and y segments to be 
    the same length as the longer segment.

    Args:
        x1 (real): time stamp index of first point (if non-integer then it will be rounded to nearest int).
        y1 (real): scale index of the first point (if non-integer then it will be rounded to nearest int).
        x2 (real): time stamp index of second point (if non-integer then it will be rounded to nearest int).
        y2 (real): scale index of the second point (if non-integer then it will be rounded to nearest int).
    """

    # if y1>y2 then need to step backwards through y's
    ystep = 1 - 2*(y1>y2) # +1 if y1 <= y2 and -1 if y1>y2

    # construct line segment from (x1,y1) to (x2,y2)
    # segment x coordinate
    segx = np.arange(x1, x2) 
    # segment y coordinate
    segy = np.arange(y1, y2, step=ystep)

    # need x and y to be the same length, interpolate the shorter one
    if len(segx)>len(segy):
        # interpolate y to the same length as x
        t = np.linspace(0, 1, len(segx))
        segy = lin_interp(y1, y2, t)
        segy = np.array(segy).round().astype('int')
    elif len(segx)<len(segy):
        # interpolate x to the same length as y
        t = np.linspace(0, 1, len(segy))
        segx = lin_interp(x1, x2, t)
        segx = np.array(segx).round().astype('int')
    
    return segx, segy

def _check_detect_seg_args(x1,y1,x2,y2,cwtmatr):
    assert len(cwtmatr.shape)==2, f"expected cwtmatr to be a matrix, but got shape {cwtmatr.shape}"
    assert x1 >=0, f"x1 must be non-negative, got {x1}."
    assert y1 >=0, f"y1 must be non-negative, got {y1}."
    assert x2 <=cwtmatr.shape[1]-1, f"x2 must be less than or equal to cwtmatr.shape[1]-1, got x2={x2} and cwtmatr.shape[1]={cwtmatr.shape[1]}."
    assert y2 <=cwtmatr.shape[0]-1, f"y2 must be less than or equal to cwtmatr.shape[0]-1, got y2={y2} and cwtmatr.shape[0]={cwtmatr.shape[0]}."
    assert x1<=x2, f"expected x1<x2, got x1={x1} and x2={x2}, respectively."
    return 

def _count_seg(xs,ys,cwtmatr):
    """For cwtmatr, count the number of upcrossings of its mean along the path defined by xs, ys
    """
    # obtain time-series from wavelet over the line segment from (x1,y1) to (x2,y2)
    cwt_coef=cwtmatr[xs, ys] 
    
    # average of the time-series for threshold
    seg_mean=np.nanmean(cwt_coef)        
    # band switch idx (either 0 or 1)
    sw=0 
    # band counter
    nb=0 

    #counting the number of times cwt_coef up-crosses its mean
    for j in range(len(cwt_coef)):
        if cwt_coef[j]>seg_mean and sw==0:
            sw=1
            nb = nb+1
        elif cwt_coef[j]<seg_mean and sw==1:
            sw=0
    return nb

def detect_seg(x1,y1,x2,y2,cwtmatr):
    """Detect and count bands structures from a continuous wavelet transform between
    the point (x1,y1) and (x2,y2)

    Uses the methodology of WLCount in [1]. This code has been adapted from theirs.

    [1] Oriani F., Treble P. C., Baker A., Mariethoz G., WlCount: Geological 
    lamination detection and counting using an image analysis approach, 
    Computers & Geoscience, https://doi.org/10.1016/j.cageo.2022.105037

    Args:
        x1 (real): time stamp index of first point (if non-integer then it will be rounded to nearest int).
        y1 (real): scale index of the first point (if non-integer then it will be rounded to nearest int).
        x2 (real): time stamp index of second point (if non-integer then it will be rounded to nearest int).
        y2 (real): scale index of the second point (if non-integer then it will be rounded to nearest int).
        cwtmatr: matrix containing cwt coefficients (as output by pywt.cwt).
            The scale of the transform varies with the rows and 
            the shift of the transform varies with the columns.
            E.g., if the list of scales s is used to transfrom the signal x
            then the (i,j) entry of cwtmatr will be the coefficient of the 
            scale s[i] wavelet shifted by time j.
    """
    _check_detect_seg_args(x1,y1,x2,y2,cwtmatr)
    
    segx, segy = get_seg(x1,y1,x2,y2)
    
    return _count_seg(segx, segy, cwtmatr)

def _check_detect_path_args(x,y,cwtmatr):
    assert len(x.shape)==1, f"expected x to be a vector, but got shape {x.shape}"
    assert len(y.shape)==1, f"expected y to be a vector, but got shape {y.shape}"
    assert len(cwtmatr.shape)==2, f"expected cwtmatr to be a matrix, but got shape {cwtmatr.shape}"
    assert len(x)==len(y), f"expected x and y to be the same length, got {len(x)} and {len(y)}"
    for (i,x1) in enumerate(x):
        assert x1 >=0, f"x[i] must be non-negative for all i, got {x1} at index {i}."
        assert x1 <=cwtmatr.shape[1]-1, f"x[i] must be less than or equal to cwtmatr.shape[1]-1, got x[{i}]={x1} and cwtmatr.shape[1]={cwtmatr.shape[1]}."
    for (i,y1) in enumerate(y):
        assert y1 >=0, f"y[i] must be non-negative, got {y1} at index {i}."
        assert y1 <=cwtmatr.shape[0]-1, f"y[i] must be less than or equal to cwtmatr.shape[0]-1, got y[{i}]={y1} and cwtmatr.shape[0]={cwtmatr.shape[0]}."
    assert np.all(np.diff(x)>=0), f"expected x to be sorted."
    return 

def detect_path(xs,ys,cwtmatr):
    """Detect and count bands structures from a continuous wavelet transform between
    the point along the piecewise linear path defined by xs, ys

    Uses the methodology of WLCount in [1]. This code has been adapted from theirs.

    [1] Oriani F., Treble P. C., Baker A., Mariethoz G., WlCount: Geological 
    lamination detection and counting using an image analysis approach, 
    Computers & Geoscience, https://doi.org/10.1016/j.cageo.2022.105037

    Args:
        x1 (real): time stamp index of first point (if non-integer then it will be rounded to nearest int).
        y1 (real): scale index of the first point (if non-integer then it will be rounded to nearest int).
        x2 (real): time stamp index of second point (if non-integer then it will be rounded to nearest int).
        y2 (real): scale index of the second point (if non-integer then it will be rounded to nearest int).
        cwtmatr: matrix containing cwt coefficients (as output by pywt.cwt).
            The scale of the transform varies with the rows and 
            the shift of the transform varies with the columns.
            E.g., if the list of scales s is used to transfrom the signal x
            then the (i,j) entry of cwtmatr will be the coefficient of the 
            scale s[i] wavelet shifted by time j.
    """
    _check_detect_path_args(xs,ys,cwtmatr)

    # determine the idices of the line segments between (x[i],y[i]) and (x[i+1], y[i+1])
    linex = []
    liney = []
    for i in range(len(xs)-2): # exclude the last segment
        x1, x2 = xs[i:i+1]
        y1, y2 = ys[i:i+1]
        # exclude end point as it is included in next segment
        segx, segy = get_seg(x1,y1,x2,y2)
        linex.append(segx)
        liney.append(segy)
    # do the last segment
    i += 1
    segx, segy = get_seg(x1,y1,x2+1,y2+1) # include the last point
    linex.append(segx)
    liney.append(segy)
    
    linex = np.concatenate(linex)
    liney = np.concatenate(liney)

    return _count_seg(linex, liney, cwtmatr)