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
# Source the filter utilities directly from GitHub
source("https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/filter_utils.R")

series <- c(1, 3, 7, 6, 2, 5, 8, 7, 3, 1, 3, 7, 6, 2, 5, 8, 7, 3,
    1, 3, 7, 6, 2, 5, 8, 7, 3, 1, 3, 7, 6, 2, 5, 8, 7, 3)
result <- age_shark(series, max_age = 10)
print(result$age)
print(result$peak_indices)
plot(series, type = "l")
lines(result$fitted, col = "blue")
```

See also filter_demo.py and filter_demo.R

# Installation and Usage

## Local Installation

Clone the repository and source the files locally:

```bash
git clone https://github.com/angus-lewis/SharkAgeEstimation.git
cd SharkAgeEstimation
```

## Remote Usage (R)

Source the R utilities directly from GitHub without cloning:

```r
# Method 1: Direct sourcing (simplest)
source("https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/filter_utils.R")

# Method 2: Using devtools (recommended for reliability)
if (!require(devtools)) install.packages("devtools")
devtools::source_url("https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/filter_utils.R")
```

## Remote Usage (Python)

Import Python modules directly from GitHub:

```python
import requests
import tempfile
import importlib.util

# Download and import filter_utils from GitHub
url = "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/filter_utils.py"
response = requests.get(url)
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(response.text)
    temp_path = f.name

spec = importlib.util.spec_from_file_location("filter_utils", temp_path)
filter_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_utils)
```

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