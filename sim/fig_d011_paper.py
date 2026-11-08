"""Fig. 6: trajectories at d = 0.11, all other settings as in Misyris et al., for the seed with the median PINN error
(chosen automatically): hardest trajectory for PINN and for mac-PIKAN.
    python fig_d011_paper.py  -> ../paper/fig/fig_d011_a/b (.pdf and .png)"""
src = open("paper_figs_smib.py").read().split("# ------------------------------------------------------------------ Fig. 3")[0]
exec(src)
LEG, ROWGAP, TTL = 0.02, 0.16, 0.06
_rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
_pe = [_rel(np.load(f"results_st3/st_d0.11_T20_P0.18_mlp_s{s}.npz")["pred"], np.load(f"results_st3/st_d0.11_T20_P0.18_mlp_s{s}.npz")["exact"]) for s in range(5)]
SEED = int(np.argsort(_pe)[len(_pe) // 2])          # seed with the median PINN error
A = np.load(f"results_st3/st_d0.11_T20_P0.18_mlp_s{SEED}.npz"); B = np.load(f"results_st3/st_d0.11_T20_P0.18_tsckan_s{SEED}.npz")
E, P, t = A["exact"], A["P"].ravel(), A["t"].ravel()
relv = lambda a, b: 100 * np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)
IW = [int(np.argmax(relv(A["pred"], E))), int(np.argmax(relv(B["pred"], E)))]
TT = ["hardest for PINN", "hardest for mac-PIKAN"]
EXC = "#c8c8c8"; (_, _, CP, LSP, _), (_, _, CM, _, _) = M[0], M[4]
W = lambda U: omega_numdiff(U, 0.1)
fig, axs, H = layout(2)
for c, i in enumerate(IW):
    for r, (fn, lab) in enumerate(((lambda U: U, r"$\delta$ (rad)"), (W, r"$\omega$ (rad/s)"))):
        ax = axs[r, c]; ye, yp, ym = fn(E)[i], fn(A["pred"])[i], fn(B["pred"])[i]
        ax.plot(t, ye, color=EXC, lw=2.6, solid_capstyle="round", zorder=1)
        ax.plot(t, ym, color=CM, lw=1.0, zorder=3)
        ax.plot(t, yp, color=CP, ls=LSP, lw=1.0, zorder=4)
        ax.set_xlim(0, 20); ax.set_xticks([0, 5, 10, 15, 20])
        lo, hi = ye.min(), ye.max(); ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))
        ax.set_ylabel(lab, labelpad=1)
        if r == 1: ax.set_xlabel("$t$ (s)", labelpad=1)
        else: ax.set_xticklabels([]); ax.set_yticks([0, 0.5, 1, 1.5])
        style(ax)
    print(f"fig_d011 ({'ab'[c]}): {TT[c]}, P = {P[i]:.3f}, seed {SEED}")
legend_in(axs[0, 0], [Line2D([], [], color=EXC, lw=2.6, label="exact"), LH(CP, LSP, "PINN"), LH(CM, "-", "mac-PIKAN (proposed)")], "lower right")
save_cols(fig, "fig_d011", H); save(fig, "fig_d011", tight=False)
