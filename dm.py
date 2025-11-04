import requests
import tempfile
import importlib.util

# Download and import files from GitHub
def remote_import(url):
    print(f"Loading module at ${url}.")
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(response.text)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return 

urls = [
    "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/LassoLarsBIC.py",
    "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/denoising.py",
    "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/count.py",
    "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/model_utils.py",
    "https://raw.githubusercontent.com/angus-lewis/SharkAgeEstimation/main/band_count.py"
]
for url in urls:
    remote_import(url)