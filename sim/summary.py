"""Table II and the summary numbers of the paper from the saved runs.
    python summary.py   -> ../paper/numbers_smib.tex (LaTeX macros and table rows), results/summary.json, results/err_over_P.npz"""
import glob, json, numpy as np
from repro import omega_numdiff

A = ["mlp", "bkan", "ckan", "tsmlp", "tsckan"]
pat = lambda a, nf: f"results/inf_nf{nf}_s*.json" if a == "mlp" else f"results/inf_{a}_nf{nf}_s*.json"
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
S = {}
for a in A:
    J = [json.load(open(f)) for f in sorted(glob.glob(pat(a, 8000)))]
    e, ew, w0, tb, tw = [], [], [], [], []
    for f in sorted(glob.glob(pat(a, 8000).replace(".json", ".npz"))):
        d = np.load(f); U, E = d["pred"], d["exact"]
        tr = np.linalg.norm(U - E, axis=1) / np.linalg.norm(E, axis=1); tb.append(tr.min()); tw.append(tr.max())
        e.append(rel(U, E)); ew.append(rel(omega_numdiff(U, 0.1), omega_numdiff(E, 0.1)))
        w0.append(np.abs((U[:, 1] - U[:, 0]) - (E[:, 1] - E[:, 0])).max() / 0.1)
    j80 = glob.glob(pat(a, 80000).replace("_s*", "_s0"))
    S[a] = dict(n=len(J), params=J[0].get("params", 481), err=100 * np.mean(e), err_std=100 * np.std(e),
                err_min=100 * np.min(e), err_max=100 * np.max(e), err_w=100 * np.mean(ew), err_w_std=100 * np.std(ew),
                w0=float(np.mean(w0)), tbest=100 * float(np.mean(tb)), tworst=100 * float(np.mean(tw)), nit=float(np.mean([j["nit"] for j in J])), nfev=float(np.mean([j["nfev"] for j in J])),
                time=float(np.mean([j["train_s"] for j in J])),
                ms_eval=1000 * float(np.mean([j["train_s"] / j["nfev"] for j in J])),
                err80=100 * json.load(open(j80[0]))["error_u"] if j80 else None)
# small-signal (modal) response alone, eq. (9): no network, no training
d = np.load("results/inf_nf8000_s0.npz"); E, Pg, tg = d["exact"], d["P"].ravel(), d["t"].ravel()
m_, d_, B_, d0_, w0_ = 0.4, 0.15, 0.2, 0.1, 0.1
ds = np.arcsin(Pg / B_); Kp = B_ * np.cos(ds); sg = -d_ / (2 * m_); wd = np.sqrt(Kp / m_ - sg ** 2); D0 = d0_ - ds
LIN = ds[:, None] + np.exp(sg * tg)[None] * (D0[:, None] * np.cos(wd[:, None] * tg) + ((w0_ - sg * D0) / wd)[:, None] * np.sin(wd[:, None] * tg))
LINF = dict(err=100 * rel(LIN, E), err_w=100 * rel(omega_numdiff(LIN, 0.1), omega_numdiff(E, 0.1)),
            w0=float(np.abs((LIN[:, 1] - LIN[:, 0]) - (E[:, 1] - E[:, 0])).max() / 0.1),
            tbest=100 * float((np.linalg.norm(LIN - E, axis=1) / np.linalg.norm(E, axis=1)).min()),
            tworst=100 * float((np.linalg.norm(LIN - E, axis=1) / np.linalg.norm(E, axis=1)).max()),
            D0min=float(-D0.max()), D0max=float(-D0.min()), margin=float((Kp / m_ - sg ** 2).min()),
            epslo=float(np.sqrt(D0 ** 2 + w0_ ** 2).min()), epshi=float(np.sqrt(D0 ** 2 + w0_ ** 2).max()),
            elo=float(np.abs(E - LIN).max(1)[np.argmin(np.sqrt(D0 ** 2 + w0_ ** 2))]),
            ehi=float(np.abs(E - LIN).max(1)[np.argmax(np.sqrt(D0 ** 2 + w0_ ** 2))]))
# Proposition 1 check: at fixed P, shrink the initial deviation epsilon along the benchmark direction (exact ODE, no network);
# the slope of log(max error) over log(epsilon) is 2
from scipy.integrate import solve_ivp
def ce_slope(Pv):
    dsv = np.arcsin(Pv / B_); Kv = B_ * np.cos(dsv); wdv = np.sqrt(Kv / m_ - sg ** 2); tt = np.linspace(0, 20, 2001)
    u = np.array([d0_ - dsv, w0_]); u /= np.linalg.norm(u); res = []
    for ep in (1e-3, 1e-2):
        a0, b0 = ep * u
        so = solve_ivp(lambda _t, y: [y[1], (Pv - d_ * y[1] - B_ * np.sin(y[0])) / m_], (0, 20), [dsv + a0, b0],
                       t_eval=tt, rtol=1e-11, atol=1e-13)
        li = dsv + np.exp(sg * tt) * (a0 * np.cos(wdv * tt) + (b0 - sg * a0) / wdv * np.sin(wdv * tt))
        res.append(np.abs(so.y[0] - li).max())
    return res[0] / 1e-3 ** 2, np.log(res[1] / res[0]) / np.log(10)
PROP = dict(ce_lo=ce_slope(0.08)[0], ce_hi=ce_slope(0.18)[0], slope=min(ce_slope(p)[1] for p in (0.08, 0.13, 0.18)))
print("prop", PROP)

# per-seed comparison and error over P (mean over seeds), PINN vs mac-PIKAN
relr = lambda a, b: np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
def per(a):
    out = []
    for f in sorted(glob.glob(pat(a, 8000).replace(".json", ".npz"))):
        z = np.load(f); out.append((rel(z["pred"], z["exact"]), relr(z["pred"], z["exact"])))
    return out
pm, pt = per("mlp"), per("tsckan")
wins = sum(b[0] < a[0] for a, b in zip(pm, pt))
em, et = 100 * np.mean([x[1] for x in pm], 0), 100 * np.mean([x[1] for x in pt], 0)
ctr = (Pg >= 0.12) & (Pg <= 0.14); edg = (Pg <= 0.10) | (Pg >= 0.16)
BAND = dict(wins=int(wins), n=len(pm), ctr_mlp=float(em[ctr].mean()), ctr_ts=float(et[ctr].mean()),
            edg_mlp=float(em[edg].mean()), edg_ts=float(et[edg].mean()), ts_lo=float(et.min()), ts_hi=float(et.max()))
BAND["ctr_gain"] = BAND["ctr_mlp"] / BAND["ctr_ts"]; BAND["edg_gain"] = BAND["edg_mlp"] / BAND["edg_ts"]
np.savez("results/err_over_P.npz", P=Pg, mlp=em, tsckan=et, **{k: 100 * np.mean([x[1] for x in per(k)], 0) for k in ["bkan", "ckan", "tsmlp"]}, lin=100 * relr(LIN, E))
print("linear", LINF); print("band", BAND)
json.dump(dict(models=S, linear=LINF, band=BAND), open("results/summary.json", "w"), indent=1)
for a in A:
    s = S[a]
    print(f"{a:7s} n={s['n']} p={s['params']:4d} err {s['err']:.2f} +- {s['err_std']:.2f} ({s['err_min']:.2f}-{s['err_max']:.2f}) "
          f"w {s['err_w']:.2f} +- {s['err_w_std']:.2f} w0 {s['w0']:.4f} its {s['nit']:.0f} t {s['time']:.0f}s "
          f"{s['ms_eval']:.1f} ms/eval err80 {s['err80']}")

# ------------------------------------------------------------------ LaTeX macros and the rows of Table II
N = dict(mlp="MLP", bkan="BK", ckan="CK", tsmlp="TP", tsckan="TS")
L = ["% generated by summary.py -- do not edit"]
f2 = lambda v: f"{v:.2f}"
for a, n in N.items():
    s = S[a]
    L += [rf"\newcommand{{\e{n}}}{{{f2(s['err'])}}}", rf"\newcommand{{\s{n}}}{{{f2(s['err_std'])}}}",
          rf"\newcommand{{\w{n}}}{{{f2(s['err_w'])}}}", rf"\newcommand{{\t{n}}}{{{s['time']:.0f}}}",
          rf"\newcommand{{\z{n}}}{{{s['w0']:.3f}}}", rf"\newcommand{{\m{n}}}{{{s['ms_eval']:.0f}}}",
          rf"\newcommand{{\lo{n}}}{{{f2(s['err_min'])}}}", rf"\newcommand{{\hi{n}}}{{{f2(s['err_max'])}}}",
          rf"\newcommand{{\tbest{n}}}{{{f2(s['tbest'])}}}", rf"\newcommand{{\tworst{n}}}{{{f2(s['tworst'])}}}",
          rf"\newcommand{{\ei{n}}}{{{f2(s['err80']) if s['err80'] is not None else '--'}}}"]
r = S["mlp"]["err"] / S["tsckan"]["err"]; rw = S["mlp"]["err_w"] / S["tsckan"]["err_w"]
L += [rf"\newcommand{{\ERRMLP}}{{{f2(S['mlp']['err'])}}}", rf"\newcommand{{\ERRTS}}{{{f2(S['tsckan']['err'])}}}",
      rf"\newcommand{{\ERWMLP}}{{{f2(S['mlp']['err_w'])}}}", rf"\newcommand{{\ERWTS}}{{{f2(S['tsckan']['err_w'])}}}",
      rf"\newcommand{{\RATIO}}{{{r:.1f}}}", rf"\newcommand{{\RATIOW}}{{{rw:.1f}}}",
      rf"\newcommand{{\ERRMLPEIGHTY}}{{{f2(S['mlp']['err80'])}}}",
      rf"\newcommand{{\TB}}{{{S['bkan']['time'] / S['tsckan']['time']:.1f}}}",
      rf"\newcommand{{\TP}}{{{S['tsckan']['time'] / S['mlp']['time']:.1f}}}"]
L += [rf"\newcommand{{\eLIN}}{{{f2(LINF['err'])}}}", rf"\newcommand{{\wLIN}}{{{f2(LINF['err_w'])}}}",
      rf"\newcommand{{\CELO}}{{{PROP['ce_lo']:.2f}}}", rf"\newcommand{{\CEHI}}{{{PROP['ce_hi']:.2f}}}",
      rf"\newcommand{{\PSLOPE}}{{{PROP['slope']:.2f}}}",
      rf"\newcommand{{\EPSLO}}{{{LINF['epslo']:.2f}}}", rf"\newcommand{{\EPSHI}}{{{LINF['epshi']:.2f}}}",
      rf"\newcommand{{\ELINLO}}{{{LINF['elo']:.3f}}}", rf"\newcommand{{\ELINHI}}{{{LINF['ehi']:.2f}}}",
      rf"\newcommand{{\DLO}}{{{LINF['D0min']:.2f}}}", rf"\newcommand{{\DHI}}{{{LINF['D0max']:.2f}}}",
      rf"\newcommand{{\WINS}}{{{BAND['wins']}}}", rf"\newcommand{{\NSEED}}{{{BAND['n']}}}",
      rf"\newcommand{{\CTRMLP}}{{{f2(BAND['ctr_mlp'])}}}", rf"\newcommand{{\CTRTS}}{{{f2(BAND['ctr_ts'])}}}",
      rf"\newcommand{{\CTRGAIN}}{{{BAND['ctr_gain']:.1f}}}", rf"\newcommand{{\EDGGAIN}}{{{BAND['edg_gain']:.1f}}}",
      rf"\newcommand{{\TSLO}}{{{f2(BAND['ts_lo'])}}}", rf"\newcommand{{\TSHI}}{{{f2(BAND['ts_hi'])}}}"]
tie = lambda key: [a for a in S if f"{S[a][key]:.2f}" == f"{min(S[b][key] for b in S):.2f}"]
best_tb, best_tw = tie("tbest"), tie("tworst")
best_a = min(S, key=lambda a: S[a]["err"]); best_w = min(S, key=lambda a: S[a]["err_w"])
best_8 = min((a for a in S if S[a]["err80"] is not None), key=lambda a: S[a]["err80"]) if any(S[a]["err80"] for a in S) else None
best_z = [a for a in S if f"{S[a]['w0']:.3f}" == f"{min(S[b]['w0'] for b in S):.3f}"]
lab = dict(mlp=r"PINN~\cite{misyris2020pinn}", bkan=r"bs-PIKAN~\cite{shuai2025pikan}", ckan="c-PIKAN", tsmlp="ma-PINN (ablation)",
           tsckan=r"\rowcolor{black!6}mac-PIKAN (proposed)")
B = lambda cond, x: rf"\best{{{x}}}" if cond else x
rows = []
for a in A:
    s = S[a]
    rows.append(f"{lab[a]} & {s['params']} & "
                + B(a == best_a, rf"{f2(s['err'])}\,$\pm$\,{f2(s['err_std'])}") + " & "
                + B(a == best_w, rf"{f2(s['err_w'])}\,$\pm$\,{f2(s['err_w_std'])}") + " & "
                + B(a in best_tb, f2(s['tbest'])) + " & " + B(a in best_tw, f2(s['tworst'])) + " & "
                + B(a in best_z, f"{s['w0']:.3f}") + f" & {s['time']:.0f} \\\\")
rows.insert(0, rf"Small-signal model \eqref{{eq:smiblin}} & 0 & {f2(LINF['err'])} & {f2(LINF['err_w'])} & {f2(LINF['tbest'])} & {f2(LINF['tworst'])} & {LINF['w0']:.3f} & -- \\ \midrule")
L.append(r"\newcommand{\TABLEROWS}{" + "\n".join(rows) + "}")
open("../paper/numbers_smib.tex", "w").write("\n".join(L) + "\n")
print("wrote ../paper/numbers_smib.tex")
