"""Recomputes every number written directly in the text of the paper (Section VI) from the saved runs and prints it
next to the value in the paper. The numbers in Table II come from summary.py.
Run from the sim folder after the simulations:  python text_numbers.py"""
import glob, json, numpy as np
from repro import omega_numdiff

rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
relv = lambda a, b: 100 * np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)
W = lambda U: omega_numdiff(U, 0.1)
A5 = ["mlp", "bkan", "ckan", "tsmlp", "tsckan"]


def errs(files, freq=False):
    """relative L2 error (%) of each run over the whole test grid, angle or frequency"""
    v = []
    for f in files:
        D = np.load(f); E, Pr = D["exact"], D["pred"]
        if freq: E, Pr = W(E), W(Pr)
        v.append(100 * rel(Pr, E))
    return np.array(v)


base = lambda a: sorted(glob.glob("results/inf_nf8000_s?.npz" if a == "mlp" else f"results/inf_{a}_nf8000_s?.npz"))
nuf = lambda nu, a: base(a) if nu == 40 else sorted(glob.glob(
    f"results_nu/nu{nu}_inf_nf8000_s?.npz" if a == "mlp" else f"results_nu/nu{nu}_inf_{a}_nf8000_s?.npz"))
df = lambda d, a: base(a) if d == 0.15 else sorted(glob.glob(f"results_st3/st_d{d:g}_T20_P0.18_{a}_s?.npz"))
times = lambda a: np.mean([json.load(open(f.replace(".npz", ".json")))["train_s"] for f in base(a)])
evals = lambda a: np.mean([(lambda j: j["train_s"] / j["nfev"])(json.load(open(f.replace(".npz", ".json")))) for f in base(a)])


def show(what, val, paper):
    print(f"{what:<66s} {val:<24s} paper: {paper}")


print("--- A. Accuracy (Table II, Fig. 3)")
p, m = errs(base("mlp")), errs(base("tsckan"))
show("mac-PIKAN better than PINN on every seed", str(bool(np.all(m < p))), "True")
show("mac-PIKAN worst seed (%)", f"{m.max():.2f}", "0.39")
rms = []
for a in A5:
    for f in base(a):
        z = np.load(f); P, t, Xu = z["P"].ravel(), z["t"].ravel(), z["Xu"]
        i = [int(np.argmin(abs(P - q))) for q in Xu[:, 0]]; k = [int(np.argmin(abs(t - q))) for q in Xu[:, 1]]
        rms.append(np.sqrt(np.mean((z["pred"][i, k] - z["exact"][i, k]) ** 2)))
show("largest RMS error at the training data, all models and seeds (rad)", f"{max(rms):.1e}", "below 1e-3")
show("bs-PIKAN / mac-PIKAN training time (times)", f"{times('bkan') / times('tsckan'):.1f}", "6.7")
D = np.load("results/inf_nf8000_s0.npz"); T = np.load("results/inf_tsckan_nf8000_s0.npz"); e = relv(D["pred"], D["exact"])
show("PINN trajectory errors, seed 0, min to max (%)", f"{e.min():.2f} to {e.max():.2f}", "0.08 to 3.70")
ib, iw = int(e.argmin()), int(e.argmax())
show("P of the lowest and highest PINN error, seed 0 (p.u.)", f"{D['P'].ravel()[ib]:.3f} and {D['P'].ravel()[iw]:.3f}",
     "0.180 and 0.163")
show("worst PINN trajectory: max angle error PINN, mac-PIKAN (rad)",
     f"{np.abs(D['pred'][iw] - D['exact'][iw]).max():.3f}, {np.abs(T['pred'][iw] - T['exact'][iw]).max():.3f}",
     "0.064, within 0.003")

print("--- B. Initial conditions (Fig. 4)")
show("training data at t = 0, of N_u = 40", str(int((D["Xu"][:, 1] == 0).sum())), "8")
l0 = {a: np.median([np.load(f)["loss"][0] for f in base(a)]) for a in A5}
show("initial loss, trial solutions (ma-PINN, mac-PIKAN)", f"{l0['tsmlp']:.3f}, {l0['tsckan']:.3f}", "about 1e-2")
show("initial loss, other models (PINN, bs-PIKAN, c-PIKAN)", f"{l0['mlp']:.2f}, {l0['bkan']:.2f}, {l0['ckan']:.2f}", "about 0.6")
S = json.load(open("results/summary.json")) if glob.glob("results/summary.json") else None
if S: show("largest initial deviation epsilon in the benchmark", f"{S['linear']['epshi']:.2f}", "1.02")

print("--- C. Robustness and training cost (Figs. 5 and 6)")
NU = [10, 20, 40, 80, 160]
r = np.array([errs(nuf(n, "mlp")).mean() for n in NU]) / np.array([errs(nuf(n, "tsckan")).mean() for n in NU])
show("PINN / mac-PIKAN over N_u = 10 ... 160 (times)", f"{r.min():.1f} to {r.max():.1f}", "3 to 9")
DD = [0.11, 0.12, 0.13, 0.14, 0.15, 0.2]
r = np.array([errs(df(d, "mlp")).mean() for d in DD]) / np.array([errs(df(d, "tsckan")).mean() for d in DD])
show("PINN / mac-PIKAN over d = 0.11 ... 0.2 (times)", f"{r.min():.1f} to {r.max():.1f}", "3 to 7")
a, b = errs(df(0.12, "mlp")), errs(df(0.12, "tsckan"))
show("d = 0.12: seeds on which mac-PIKAN is better", f"{int(np.sum(b < a))} of {len(a)}", "4 of 5")
pm = errs(df(0.11, "mlp")); s = int(np.argsort(pm)[len(pm) // 2])
A_ = np.load(f"results_st3/st_d0.11_T20_P0.18_mlp_s{s}.npz"); B_ = np.load(f"results_st3/st_d0.11_T20_P0.18_tsckan_s{s}.npz")
E, P = A_["exact"], A_["P"].ravel()
i1, i2 = int(relv(A_["pred"], E).argmax()), int(relv(B_["pred"], E).argmax())
show(f"d = 0.11, median PINN seed ({s}): P of the two hardest trajectories", f"{P[i1]:.3f} and {P[i2]:.3f}", "0.175 and 0.178")
show("  hardest for PINN: angle error PINN vs mac-PIKAN (%)",
     f"{relv(A_['pred'][i1], E[i1]):.2f} vs {relv(B_['pred'][i1], E[i1]):.2f}", "5.30 vs 0.65")
show("  hardest for mac-PIKAN: PINN / mac-PIKAN angle error (times)",
     f"{relv(A_['pred'][i2], E[i2]) / relv(B_['pred'][i2], E[i2]):.1f}", "2.6")
show("cost of one evaluation, bs-PIKAN / c-PIKAN (times)", f"{evals('bkan') / evals('ckan'):.1f}", "4.4")
show("mac-PIKAN / PINN training time (times)", f"{times('tsckan') / times('mlp'):.1f}", "about 8")
j = json.load(open("results/inf_nf80000_s0.json"))
show("PINN with N_f = 80 000: training time (s), error (%)", f"{j['train_s']:.0f}, {100 * j['error_u']:.2f}", "1176, 1.00")
f7 = sorted(glob.glob("results_w7/inf_tsckan_nf8000_s?.npz"))
t7 = np.mean([json.load(open(f.replace(".npz", ".json")))["train_s"] for f in f7])
show("mac-PIKAN 2x7 (420 parameters): error (%), training time (s)", f"{errs(f7).mean():.2f}, {t7:.0f}",
     "0.32, about half of mac-PIKAN")
print("(training times depend on the machine; the paper values come from the run described in Section V)")
