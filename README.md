# mac-PIKAN: Modal-Augmented Chebyshev Physics-Informed Kolmogorov–Arnold Networks for Power Systems

Code and results for the paper

> K. Nosrati, M. Ghorbani, A. Tepljakov, E. Petlenkov, and J. Belikov
> "Modal-Augmented Chebyshev Physics-Informed Kolmogorov–Arnold Networks for Power Systems,"
> submitted to the IEEE PES General Meeting 2027.

**Paper:** the arXiv version will be linked here.

mac-PIKAN adds the closed-form small-signal response of the linearized swing equations to a Chebyshev
physics-informed Kolmogorov–Arnold network (c-PIKAN):

$$\hat\delta(P,t) = \delta_{\mathrm{lin}}(P,t) + (t/T)^2 N_\theta(P,t),$$

so both initial conditions, $\delta(0)$ and $\omega(0)$, hold exactly, and the network only learns the nonlinear
correction. The models are compared on the single-machine infinite-bus (SMIB) benchmark of
Misyris, Venzke, and Chatzivasileiadis, "Physics-informed neural networks for power systems," IEEE PES GM 2020,
with the same training data, collocation points, and optimizer as their
[original code](https://github.com/gmisy/Physics-Informed-Neural-Networks-for-Power-Systems).

## Models

| Model | Network | Parameters | `--arch` |
|---|---|---|---|
| PINN (Misyris et al.) | MLP, 5 hidden layers × 10 nodes, tanh | 481 | `mlp` |
| bs-PIKAN | B-spline KAN, 1 hidden layer × 5 nodes, grid 10, cubic | 225 | `bkan` |
| c-PIKAN | Chebyshev KAN, 2 hidden layers × 8 nodes, degree K = 5 | 528 | `ckan` |
| ma-PINN (ablation) | modal-augmented trial solution with the MLP | 481 | `tsmlp` |
| **mac-PIKAN** (proposed) | modal-augmented trial solution with the c-PIKAN | 528 | `tsckan` |

## Installation

Python 3.10 or newer. In a new environment (for example with conda):

```bash
conda create -n macpikan python=3.12
conda activate macpikan
pip install -r requirements.txt
```

The code runs on the CPU; no GPU is needed.

## Reproducing the paper

**Tables and figures from the stored results (about one minute).**
The results of all 121 runs used in the paper are included in `sim/results*/`. From the `sim` folder:

```bash
python summary.py            # Table II -> paper/numbers_smib.tex
python paper_figs_smib.py    # Figs. 3 and 4
python fig_robust.py         # Fig. 5
python fig_d011_paper.py     # Fig. 6
python report.py             # sim/my_results.txt: every number in the paper, recomputed, next to the paper value
```

**All simulations from scratch.** Move or delete the `sim/results*` folders, then, from the `sim` folder:

```bash
python run_all.py 6          # 6 runs in parallel
```

`run_all.py` (or `bash run_all.sh 6` on Linux and macOS) runs the 121 simulations listed in `sim/jobs.txt`, one CPU
thread each, then all the steps above. Finished runs are skipped, so the script can be stopped and restarted.
The runs take about 42 CPU-hours in total, about 7 to 8 hours with six in parallel on an 8-core laptop;
a bs-PIKAN run takes up to about 4 hours, a PINN run one to two minutes.

**A single run**, for example mac-PIKAN with seed 0:

```bash
python repro.py inference --arch tsckan --seed 0
```

| Option | Meaning | Default |
|---|---|---|
| `--arch` | `mlp`, `bkan`, `ckan`, `tsmlp`, `tsckan` (see Models) | `mlp` |
| `--seed` | seed of the network initialization | 0 |
| `--nu` | number of training data $N_u$ | 40 |
| `--nf` | number of collocation points $N_f$ | 8000 |
| `--d` | damping $d$; other values recompute the reference data | 0.15 |
| `--out` | output folder | `results` |

Set `CKAN_W=7` for the smaller mac-PIKAN (2 × 7, 420 parameters). Each run writes a `.json` file (errors,
iterations, training time) and a `.npz` file (predicted and reference trajectories, loss history).

## Simulations

| Folder | Content | Used in |
|---|---|---|
| `sim/results/` | five models × five seeds; PINN with $N_f$ = 80 000 | Table II, Figs. 3 and 4 |
| `sim/results_nu/` | $N_u$ = 10, 20, 80, 160; PINN and mac-PIKAN × five seeds | Fig. 5(a) |
| `sim/results_st3/` | $d$ = 0.11, 0.12, 0.13, 0.14, 0.2; PINN and mac-PIKAN × five seeds | Figs. 5(b) and 6 |
| `sim/results_w7/` | mac-PIKAN 2 × 7 (420 parameters) × five seeds | Sec. VI-C, training cost |

## Files

| File | Content |
|---|---|
| `sim/repro.py` | benchmark, models, trial solution, training (one run) |
| `sim/kan.py` | Chebyshev and B-spline KAN layers |
| `sim/jobs.txt` | the 121 runs of the paper |
| `sim/run_all.py`, `sim/run_all.sh` | run everything |
| `sim/summary.py` | Table II and summary numbers |
| `sim/text_numbers.py` | every number written in the text, recomputed |
| `sim/report.py` | `my_results.txt`: your results next to the paper values |
| `sim/paper_figs_smib.py`, `sim/fig_robust.py`, `sim/fig_d011_paper.py` | Figs. 3–6 |
| `sim/data/` | reference data of Misyris et al. |
| `paper/` | figures and Table II written by the scripts (`paper/fig/`, `paper/numbers_smib.tex`) |

## Reproducibility

The results in the paper were obtained on an AMD Ryzen 9 5900HS (8 cores, 32 GB RAM) with Windows 11, Python 3.12, and
the package versions in `requirements.txt`, one thread per run and six runs in parallel.
On a given computer, a run with a given seed is deterministic with one thread (`OMP_NUM_THREADS=1`, set by
`run_all.py`; set it yourself for single runs). On another computer, or with other library versions, the
floating-point rounding changes, and L-BFGS-B can follow a different path, so single runs can differ from the stored
ones; the comparison over five seeds is the result of the paper. Training times depend on the computer.
The PINN weights are initialized in PyTorch, so their random values differ from the TensorFlow initialization of the
original code.

## Citation

If you use this code, please cite the paper. The BibTeX entry will be added here with the arXiv version.

## License

MIT (see `LICENSE`). The reference data in `sim/data/` and the original PINN setting are from
[gmisy/Physics-Informed-Neural-Networks-for-Power-Systems](https://github.com/gmisy/Physics-Informed-Neural-Networks-for-Power-Systems),
MIT License (see `NOTICE`).
