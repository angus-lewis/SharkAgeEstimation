# SharkAgeEstimation
This module provides functions and classes to estimate trends/smooth 
time series data using Basis Pursuit Denoising (a.k.a., Lasso).
The intended use case if for identifying peaks/banded structures in
time series, such as those that arise in ageing sharks via their
vertebrae (sclerochronology).


# Installation and Usage
Built under:
    Python 3.13.0

## Local Installation
Download from github manually 

OR

Clone the repository and source the files locally:

```bash
git clone https://github.com/angus-lewis/SharkAgeEstimation.git
cd SharkAgeEstimation
```

## Python and dependencies (Windows)
Download Python 3.13 installer: https://www.python.org/downloads/

Run installer:
    + “Add Python to PATH”
    + “Install pip”

Verify and install requirements:
```
# In powershell, cd into SharkAgeEstimation directory
python -3.13 -m pip install -r requirements.txt
```

## Python and dependencies (Mac)
```
# In terminal
# Python 3.13
brew install python@3.13

# ensure pip
python3.13 -m ensurepip --upgrade

# install requirements (assuming you cd into SharkAgeEstimation directory)
python3.13 -m pip install -r requirements.txt
```

## VSCode and notebooks
Install python first.

Go to: https://code.visualstudio.com. Download and install (accept defaults)

Open VS Code. Go to Extensions (Ctrl+Shift+X). Install:
    Python (by Microsoft),
    Jupyter (by Microsoft),
    
These enable .ipynb support.

# Python dependencies
To generate dependencies (this is for dev's only, you probably don't need to do this)

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
