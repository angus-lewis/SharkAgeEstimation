import numpy as np

def get_summary(files, sep, newline, inc_name, inc_scenario, wrap_every, keep_brackets, est_var, find_fields, names):
    data = {}
    summary_data = ""
    lineno = 0
    file_ix = 0
    for file in files:
        print(file)
        scenario = file.split("/")[-2]
        scenario = scenario.split(".")
        scenario = "".join(scenario[1:-2])
        with open(file, "r") as f:
            lines = [""]*len(find_fields)
            for line in f:
                print(line)
                fields = line.split(":")
                name = fields[0]
                value = fields[1].strip()
                value = value[:-1]
                if name in find_fields:
                    name_ix = find_fields.index(name)
                    lines[name_ix] = line
            for line in lines:
                print(line)
                fields = line.split(":")
                name = fields[0]
                value = fields[1].strip()
                value = value[:-1]
                if name in find_fields:
                    name_ix = find_fields.index(name)
                    lineno += 1
                    name = names[name_ix]
                    if inc_scenario:
                        scenario = scenario + sep
                    else:
                        scenario = ""
                    if lineno%wrap_every==0:
                        nl = newline
                    else:
                        nl = sep
                    if keep_brackets:
                        value = value.split(" ")
                        v = float(value[0])
                        if name == " bias " or name == " mse ":
                            v = v
                        value[0] = str(round(v, ndigits=3))
                        value = "".join(value)
                    else:
                        v = float(value.split(" ")[0])
                        if name == " bias " or name == " mse ":
                            v = v
                        value = str(round(v, ndigits=3))
                    if est_var:
                        v = float(value.split(" ")[0])
                        var = round(pow(v*(1-v)/64, 0.5),3)
                        value = value + " (" + str(var) + ")"
                    if not name in data.keys():
                        data[name] = [0]*len(files)
                    data[name][file_ix] = value
                    if inc_name:
                        name = name + sep
                    else:
                        name = ""
                    summary_data = summary_data + scenario + name + value + nl
        file_ix += 1
    return summary_data, data


def bootstrap_ci(samples, n_resamples, stat_func, sig, random_state):
    """
    Perform bootstrap resampling.

    Parameters:
        samples (array-like): Original data sample
        n_resamples (int): Number of bootstrap resamples
        stat_func (callable): Statistic function to apply (e.g., np.mean, np.median)
        random_state (int or None): Seed for reproducibility

    Returns:
        np.ndarray: Bootstrap distribution of the statistic
    """
    rng = np.random.default_rng(random_state)
    samples = np.asarray(samples)
    n = len(samples)

    boot_stats = np.empty(n_resamples)

    for i in range(n_resamples):
        resample = rng.choice(samples, size=n, replace=True)
        boot_stats[i] = stat_func(resample)

    lower = np.quantile(boot_stats, sig/2)
    upper = np.quantile(boot_stats, 1-sig/2)
    return lower, upper, boot_stats