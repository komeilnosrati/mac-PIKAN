#!/bin/bash
# Runs every simulation of the paper (121 runs in jobs.txt), then Table II, the numbers in the text, the figures,
# and a report of your results next to the paper values (my_results.txt).
# Usage (from the sim folder):  bash run_all.sh [parallel_runs]     default 4
# Finished runs (existing .json) are skipped, so the script can be stopped and restarted.
cd "$(dirname "$0")"; NW=${1:-4}; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results results_nu results_st3 results_w7 logs ../paper/fig
worker() { while IFS= read -r line; do out="${line##*|}"; cmd="${line%|*}"
  [ -f "$out" ] && continue; mkdir "$out.lock" 2>/dev/null || continue
  echo "run: $cmd"; bash -c "$cmd" > "logs/$(basename "$out" .json).log" 2>&1; rmdir "$out.lock"; done < jobs.txt; }
for i in $(seq $NW); do worker & done; wait
python3 summary.py              # Table II -> ../paper/numbers_smib.tex
python3 paper_figs_smib.py      # Figs. 3 and 4
python3 fig_robust.py           # Fig. 5
python3 fig_d011_paper.py       # Fig. 6
python3 report.py               # my_results.txt: your numbers next to the paper values
