# SharkAgeEstimation
This module provides functions and classes to estimate trends/smooth 
time series data using Basis Pursuit Denoising (a.k.a., Lasso).
The intended use case if for identifying peaks/banded structures in
time series, such as those that arise in ageing sharks via their
vertebrae (sclerochronology).


# Installation and Usage
Built under:
    Python 3.14.0

## Local Installation

Clone the repository and source the files locally:

```bash
git clone https://github.com/angus-lewis/SharkAgeEstimation.git
cd SharkAgeEstimation
```

## Remote Usage (Python)

Import Python modules directly from GitHub:

```python
import requests
import tempfile
import importlib.util

# Download and import files from GitHub
urls = {
    "LassoLarsBIC" : "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/LassoLarsBIC.py",
    "denoising" : "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/denoising.py",
    "count" : "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/count.py",
    "model_utils" : "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/model_utils.py",
    "band_count" : "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/band_count.py"
}
for name, url in urls.items():
    print(f"Loading module {name} at {url}.")
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(response.text)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("module", temp_path)
    globals()[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(globals()[name])

# Now the modules should be available in this script
# Example usage:
import numpy as np
x = [
    1,2,3,4,5,4,3,2,1,1,2,3,4,5,5,5,4,3,2,1,1,2,3,4,5,4,3,2,1,1,2,3,4,5,5,5,4,3,2,1
]
series = np.asarray(x, dtype=float)
max_bands = 40
counter = band_count.BandCounter(series, max_bands)
peak_locations, estimated_count = counter.get_count_estimate()
print(f"Peak locations: {peak_locations}")
print(f"Peak count: {estimated_count}")
```

# Python dependencies

To generate dependencies 

```zsh
pip freeze --exclude-editable > requirements.txt
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

# License
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

See the [LICENSE](./LICENSE) file for details.

This project uses third-party libraries under various permissive licenses 
(MIT, BSD, Apache 2.0, ISC). See [THIRD_PARTY_LICENSES](./THIRD_PARTY_LICENSES) 
for more information.

## TODO
+ check that active set is not empty
+ check estimate of var is not zero