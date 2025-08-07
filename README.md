# SharkAgeEstimation
This module provides utilities for signal processing and model selection using
Fourier basis functions. It includes functions and classes for constructing
Fourier-based linear models, performing stepwise model selection (forward or
backward) using AIC or BIC criteria, smoothing 1D data series, and detecting
peaks in signals. The primary use case is for applications such as automated
age estimation in biological signals (e.g., shark age estimation from banded
structures), but the tools are general-purpose for any 1D signal analysis.

# Basic usage

Use the function ```age_shark``` to estimate the age of a shark from a 1-D signal
by counting the number of peaks in the smoothed signal.

In python 

```py
series = np.asarray([
    1, 3, 7, 6, 2, 5, 8, 7, 3, 1, 3, 7, 6, 2, 5, 8, 7, 3,
    1, 3, 7, 6, 2, 5, 8, 7, 3, 1, 3, 7, 6, 2, 5, 8, 7, 3
])
age, peak_indices, fitted = filter_utils.age_shark(series, max_age=10)
```

In R

```r
series <- c(1, 3, 7, 6, 2, 5, 8, 7, 3, 1, 3, 7, 6, 2, 5, 8, 7, 3,
    1, 3, 7, 6, 2, 5, 8, 7, 3, 1, 3, 7, 6, 2, 5, 8, 7, 3)
result <- age_shark(series, max_age = 10)
print(result$age)
print(result$peak_indices)
plot(series, type = "l")
lines(result$fitted, col = "blue")
```

See also filter_demo.py and filter_demo.R

# Python dependencies

To generate dependencies 

```zsh
pip freeze > requirements.txt
```

To install packages from requirements 

```zsh
pip install -r ./requirements.txt
```

You might want to make a new venv first - cd to the location of this README.md and run

```zsh
python3 -m venv ./venv
```

To activate the venv use 

```zsh
source ./venv/bin/activate
```