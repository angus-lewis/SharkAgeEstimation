import subprocess

p1 = subprocess.Popen(["python", "run_experiments.py", "experiments/lin/var0.16"])
p2 = subprocess.Popen(["python", "run_experiments.py", "experiments/lin/var0.64"])
p3 = subprocess.Popen(["python", "run_experiments.py", "experiments/sin/var0.16"])
p4 = subprocess.Popen(["python", "run_experiments.py", "experiments/sin/var0.64"])

p1.wait()
p2.wait()
p3.wait()
p4.wait()