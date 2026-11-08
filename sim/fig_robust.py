"""Fig. 5: angle error of PINN and mac-PIKAN over the number of training data N_u and over the damping d,
all other settings as in Misyris et al.     python fig_robust.py  -> ../paper/fig/fig_robust_a/b (.pdf and .png)"""
import glob, os
src = open("paper_figs_smib.py").read().split("# ------------------------------------------------------------------ Fig. 3")[0]
exec(src)
MM = [m for m in M if m[0] in ("mlp", "tsckan")]
def stats(files, k=0):
    v = []
    for f_ in files:
        D = np.load(f_); E, Pr = D["exact"], D["pred"]
        if k: E, Pr = omega_numdiff(E, 0.1), omega_numdiff(Pr, 0.1)
        v.append(100 * rel(Pr, E))
    return np.mean(v), np.min(v), np.max(v)
base = lambda a: sorted(glob.glob(f"results/inf_nf8000_s?.npz" if a == "mlp" else f"results/inf_{a}_nf8000_s?.npz"))
nuf = lambda nu, a: base(a) if nu == 40 else sorted(glob.glob(f"results_nu/nu{nu}_inf_nf8000_s?.npz" if a == "mlp" else f"results_nu/nu{nu}_inf_{a}_nf8000_s?.npz"))
df = lambda d, a: base(a) if abs(d - 0.15) < 1e-9 else sorted(glob.glob(f"results_st3/st_d{d:g}_T20_P0.18_{a}_s?.npz"))
NU = [10, 20, 40, 80, 160]; DD = [d for d in (0.11, 0.12, 0.13, 0.14, 0.15, 0.2) if len(df(d, "tsckan")) == 5 and len(df(d, "mlp")) == 5]
LEG = 0.02
fig, axs, H = layout(1); ax = axs[0]
for p, (X, F, xl) in enumerate(((NU, nuf, r"training data $N_u$"), (DD, df, r"damping $d$ (p.u.)"))):
    a_ = ax[p]; R = {}
    for k, lab, col, ls, lw in MM:
        S = np.array([stats(F(x, k)) for x in X]); R[k] = S[:, 0]
        XP = X if p == 0 else list(range(len(X)))          # damping: one equally spaced position per tested value
        a_.fill_between(XP, S[:, 1], S[:, 2], color=col, alpha=0.16, lw=0, zorder=1)
        a_.plot(XP, S[:, 0], color=col, ls=ls, lw=lw, marker=MK[k], ms=2.4, zorder=3)
    a_.axvline(40 if p == 0 else X.index(0.15), color="0.55", lw=0.5, ls=":", zorder=1)
    a_.set_yscale("log"); a_.set_ylim(0.06, 30); a_.set_yticks([0.1, 0.3, 1, 3, 10, 30]); a_.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    a_.minorticks_off()
    if p == 0:
        a_.set_xscale("log"); a_.set_xticks(NU); a_.set_xticklabels([str(n) for n in NU]); a_.set_xlim(8.5, 190)
    else:
        a_.set_xticks(range(len(DD))); a_.set_xticklabels([f"{d:g}" for d in DD]); a_.set_xlim(-0.4, len(DD) - 0.6)
    a_.set_xlabel(xl, labelpad=1); a_.set_ylabel(r"angle error $\varepsilon$ (%)", labelpad=1); style(a_)
    a_.minorticks_off()
legend_in(ax[0], [LH(col, ls, lab) for k, lab, col, ls, _ in MM], "upper right")
save_cols(fig, "fig_robust", H); save(fig, "fig_robust", tight=False)
print("d values:", DD)
for k in ("mlp", "tsckan"):
    print(k, "Nu:", [round(stats(nuf(n, k))[0], 2) for n in NU], "d:", [round(stats(df(d, k))[0], 2) for d in DD],
          "freq d:", [round(stats(df(d, k), 1)[0], 2) for d in DD])
