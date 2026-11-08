"""Runs every simulation of the paper (121 runs in jobs.txt), then Table II, the figures, and a report of your results
next to the paper values (my_results.txt). Works on Windows, macOS and Linux.
Usage (from the sim folder):  python run_all.py [parallel_runs]     default 4
Finished runs (existing .json) are skipped, so the script can be stopped and restarted."""
import os, sys, subprocess, shlex
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 4
for d in ("results", "results_nu", "results_st3", "results_w7", "logs", os.path.join("..", "paper", "fig")):
    os.makedirs(d, exist_ok=True)
jobs = []
for line in open("jobs.txt"):
    line = line.strip()
    if not line: continue
    cmd, out = line.rsplit("|", 1)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    parts = shlex.split(cmd)
    while "=" in parts[0]:                      # leading VAR=value (e.g. CKAN_W=7)
        k, v = parts.pop(0).split("=", 1); env[k] = v
    if parts[0] == "python3": parts[0] = sys.executable
    jobs.append((parts, out, env))
todo = [j for j in jobs if not os.path.exists(j[1])]
print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already done, {len(todo)} to run with {NW} in parallel", flush=True)
def run(job):
    parts, out, env = job
    log = os.path.join("logs", os.path.basename(out).replace(".json", ".log"))
    with open(log, "w") as f:
        r = subprocess.run(parts, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(("done  " if r.returncode == 0 and os.path.exists(out) else "FAILED") + f"  {out}   (log: {log})", flush=True)
with ThreadPoolExecutor(NW) as ex:
    list(ex.map(run, todo))
missing = [o for _, o, _ in jobs if not os.path.exists(o)]
if missing:
    print(f"{len(missing)} runs missing, see logs/; rerun this script to retry"); sys.exit(1)
py = sys.executable
subprocess.run([py, "summary.py"], check=True)                    # Table II -> ../paper/numbers_smib.tex
for s in ("paper_figs_smib.py", "fig_robust.py", "fig_d011_paper.py"):   # Figs. 3-6 -> ../paper/fig
    subprocess.run([py, s], check=True)
subprocess.run([py, "report.py"], check=True)                     # my_results.txt: your numbers next to the paper values
print("Done. Your results are in sim/my_results.txt and the figures in paper/fig.")
