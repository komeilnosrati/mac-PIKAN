"""Figs. 3 and 4 of the paper: trajectories for seed 0, training loss and angle error over P.
    python paper_figs_smib.py     -> ../paper/fig/fig_traj_a/b and fig_lossm_a/b (.pdf and .png)
The first part (style, colors, layout) is shared by fig_robust.py and fig_d011_paper.py."""
import os, glob, json
import numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from repro import omega_numdiff

for f in glob.glob("/usr/share/fonts/truetype/liberation/Liberation*-*.ttf"):   # Times-like font on Linux
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 7.5,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.linewidth": 0.5, "axes.edgecolor": "0.25",
    "xtick.color": "0.25", "ytick.color": "0.25", "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "0.88", "grid.linewidth": 0.4, "axes.axisbelow": True, "pdf.fonttype": 42})
# "box" style of the paper: framed panels, ticks inside, boxed legend; Times text and STIX math.
BOX = os.environ.get("FIGSTYLE", "box") == "box"     # FIGSTYLE=clean gives open axes instead
if BOX:
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "0.0",
        "axes.linewidth": 0.6, "xtick.color": "0.0", "ytick.color": "0.0", "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "grid.color": "0.85", "grid.linewidth": 0.5,
        "legend.frameon": True, "legend.edgecolor": "0.0", "legend.fancybox": False, "legend.framealpha": 1.0})
OUT = os.environ.get("FIGOUT", "../paper/fig"); TW, CW = 7.16, 3.5; NF, SEED = 8000, 0
M = [("mlp", "PINN", "#0072B2", (0, (4, 1.6)), 1.0), ("bkan", "bs-PIKAN", "#E69F00", (0, (3.2, 1.2, 1, 1.2)), 1.0),
     ("ckan", "c-PIKAN", "#CC3311", (0, (1.2, 1.0)), 1.1), ("tsmlp", "ma-PINN", "#009E73", (0, (5, 1.5, 1.2, 1.5)), 1.0),
     ("tsckan", "mac-PIKAN (proposed)", "#7B3294", "-", 1.25)]
if BOX:   # soft colors of the box style; the proposed model keeps a strong color
    M = [("mlp", "PINN", "#5B9BD5", (0, (4, 1.6)), 1.2), ("bkan", "bs-PIKAN", "#F4C430", (0, (3.2, 1.2, 1, 1.2)), 1.2),
         ("ckan", "c-PIKAN", "#F08080", (0, (1.2, 1.0)), 1.3), ("tsmlp", "ma-PINN", "#66CDAA", (0, (5, 1.5, 1.2, 1.5)), 1.2),
         ("tsckan", "mac-PIKAN (proposed)", "#7B3294", "-", 1.4)]
MK = dict(mlp="o", bkan="s", ckan="^", tsmlp="D", tsckan="v")
f = lambda a, s=SEED, nf=NF: f"results/inf_nf{nf}_s{s}.npz" if a == "mlp" else f"results/inf_{a}_nf{nf}_s{s}.npz"
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)


def save_cols(fig, name, H, ncols=2):
    """save each column of a two-column figure as its own file (name_a, name_b) for subfigures in the paper"""
    from matplotlib.transforms import Bbox
    for c in range(ncols):
        x0 = c * (PW + GAPX); bb = Bbox([[x0, 0], [x0 + LM + PW + 0.12, H]])
        for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
            fig.savefig(f"{OUT}/{name}_{'ab'[c]}.{ext}", bbox_inches=bb, **kw)
    print("wrote", name + "_a, " + name + "_b")

def save(fig, name, tight=True):
    kw = dict(bbox_inches="tight", pad_inches=0.015) if tight else {}
    fig.savefig(f"{OUT}/{name}.pdf", **kw)
    fig.savefig(f"{OUT}/{name}.png", dpi=300, **kw); plt.close(fig); print("wrote", name)


# ------------------------------------------------------------------ common layout for Figs. 3-6
# Identical panel size, margins, gaps, fonts, line widths, and legend band (all in inches).
PW, PH = 1.33, 0.82            # panel width and height
LM, GAPX = 0.50, 0.60          # left margin, horizontal gap (= left margin, so each column can be saved as its own panel)
LEG, TTL, XLB, ROWGAP = 0.30, 0.06, 0.32, 0.40   # no panel titles: (a), (b) are subfigure captions in the paper
FS_T, FS_L, FS_K, FS_N = 6.8, 6.8, 6.0, 5.3   # title, axis label, tick, number sizes

def layout(nrows):
    W = LM + 2 * PW + GAPX + 0.2
    H = LEG + TTL + nrows * PH + (nrows - 1) * ROWGAP + XLB
    fig = plt.figure(figsize=(W, H)); axs = []
    for r in range(nrows):
        y = H - LEG - TTL - (r + 1) * PH - r * ROWGAP
        axs.append([fig.add_axes([(LM + c * (PW + GAPX)) / W, y / H, PW / W, PH / H]) for c in range(2)])
    return fig, np.array(axs), H

def style(ax):
    ax.grid(True); ax.tick_params(labelsize=FS_K); ax.xaxis.label.set_size(FS_L); ax.yaxis.label.set_size(FS_L)

def legend(fig, handles, H):
    lg = fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0 - 0.02 / H), fontsize=6.3,
               handlelength=2.8, columnspacing=1.0, borderaxespad=0.0, **(dict(borderpad=0.35, edgecolor="0.0") if BOX else {}))
    if BOX: lg.get_frame().set_linewidth(0.6)

def legend_in(ax, handles, loc, ncol=1, **kw):
    kw = dict(dict(handlelength=2.6, columnspacing=0.9), **kw)
    lg = ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=kw.pop("fontsize", FS_N + 0.6),
                   borderaxespad=0.35, frameon=True, **dict(dict(handletextpad=0.5, borderpad=0.35, labelspacing=0.25), **kw))
    lg.get_frame().set_linewidth(0.5); lg.get_frame().set_edgecolor("0.0"); lg.set_zorder(10)
    return lg

LH = lambda col, ls, lab: Line2D([], [], color=col, ls=ls, lw=1.3, label=lab)

# ------------------------------------------------------------------ Fig. 3: trajectories (layout of Fig. 3 of Misyris et al.)
# Seed 0 for every model (a single run, as in Misyris et al.): trajectories with the lowest and highest PINN error.
D = {k: np.load(f(k, 0)) for k, *_ in M}
E, P, t = D["mlp"]["exact"], D["mlp"]["P"].ravel(), D["mlp"]["t"].ravel()
eu = np.array([rel(D["mlp"]["pred"][i], E[i]) for i in range(len(P))])
cols = [int(eu.argmin()), int(eu.argmax())]
M2 = [m for m in M if m[0] in ("mlp", "tsckan")]          # PINN and mac-PIKAN
LEG0 = LEG; LEG = 0.02; EXC = "#c8c8c8"                     # exact: thick light grey
GAPS = [0.05, 0.05, 0.05]                                   # gaps below rows 1-3 (inches); x10^-3 is written inside the panels
W_ = LM + 2 * PW + GAPX + 0.2; H = LEG + TTL + 4 * PH + sum(GAPS) + XLB
fig = plt.figure(figsize=(W_, H)); axs = np.empty((4, 2), dtype=object); y = H - LEG - TTL
for r in range(4):
    y -= PH
    for c in range(2): axs[r, c] = fig.add_axes([(LM + c * (PW + GAPX)) / W_, y / H, PW / W_, PH / H])
    if r < 3: y -= GAPS[r]
W = lambda U: omega_numdiff(U, 0.1)
for c, i in enumerate(cols):
    for q, (g, nm, un, sc) in enumerate(((lambda U: U[i], r"\delta", "rad", 1.0), (lambda U: W(U)[i], r"\omega", "rad/s", 1.0))):
        ye = g(E); ax = axs[q, c]; ae = axs[2 + q, c]
        ax.plot(t, ye, color=EXC, lw=2.6, solid_capstyle="round", zorder=1)
        for j, (k, lab, col, ls, lw) in enumerate(M2):
            yp = g(D[k]["pred"]); ax.plot(t, yp, color=col, ls=ls, lw=1.0, zorder=4 if k == "mlp" else 3)
            e = sc * (yp - ye); ae.fill_between(t, 0, e, color=col, alpha=0.18, lw=0, zorder=2)
            ae.plot(t, e, color=col, ls=ls, lw=0.9, zorder=4 if k == "mlp" else 3)
        lo, hi = ye.min(), ye.max(); ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.1 * (hi - lo))
        ae.axhline(0, color="0.5", lw=0.5); m = np.abs(ae.get_ylim()).max(); ae.set_ylim(-m, 1.3 * m)   # headroom for x10^-3
        ax.set_ylabel(rf"${nm}$ ({un})", labelpad=1); ae.set_ylabel(rf"$\hat{nm}-{nm}$ ({un})", labelpad=1)
        fm = matplotlib.ticker.ScalarFormatter(useMathText=True); fm.set_powerlimits((-3, -3)); ae.yaxis.set_major_formatter(fm)
        ae.yaxis.get_offset_text().set_visible(False)
        ae.text(0.03, 0.95, r"$\times10^{-3}$", transform=ae.transAxes, ha="left", va="top", fontsize=FS_K,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8), zorder=6)
    for r in range(4):
        a = axs[r, c]; a.set_xlim(0, 20); a.set_xticks([0, 5, 10, 15, 20]); style(a); a.tick_params(labelbottom=(r == 3))
    axs[3, c].set_xlabel("$t$ (s)", labelpad=1)
    print(f"fig_traj ({'ab'[c]}): {'lowest' if c == 0 else 'highest'} PINN error, P = {P[i]:.3f}")
legend_in(axs[0, 0], [Line2D([], [], color=EXC, lw=2.6, label="exact")] + [LH(col, ls, lab) for k, lab, col, ls, _ in M2], "lower right")
save_cols(fig, "fig_traj", H); save(fig, "fig_traj", tight=False); LEG = LEG0

# ------------------------------------------------------------------ Fig. 4: training loss and angle error over P
LEG0 = LEG; LEG = 0.02
fig, axs, H = layout(1); ax = axs[0]
for k, lab, col, ls, lw in M:                      # best loss so far (hides line-search trials), median of five seeds
    Ls = [np.minimum.accumulate(np.load(f(k, s_))["loss"]) for s_ in range(5)]; n = max(len(L) for L in Ls)
    A = np.array([np.concatenate([L, np.full(n - len(L), L[-1])]) for L in Ls])   # converged runs keep their final loss
    nm = int(np.median([len(L) for L in Ls])); x = np.arange(1, nm + 1); y = np.median(A, 0)[:nm]
    ax[0].plot(x, y, color=col, ls=ls, lw=lw, zorder=3)
    ax[0].plot(x[-1], y[-1], "o", color=col, ms=2.6, mec="white", mew=0.4, zorder=4)   # median number of evaluations to convergence
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("L-BFGS-B evaluation", labelpad=1)
ax[0].set_ylabel("loss", labelpad=1); ax[0].set_xticks([1, 10, 100, 1000]); ax[0].set_xticklabels(["1", "10", "100", "1000"]); ax[0].set_xlim(0.8, 9000)
ax[0].set_yticks([1, 1e-3, 1e-6]); ax[0].minorticks_off(); style(ax[0])
ax[0].set_ylim(1e-8, 3); ax[0].set_yticks([1, 1e-2, 1e-4, 1e-6, 1e-8])
for k, lab, col, ls, lw in M:                      # mean of five seeds; band = range of the seeds for PINN and mac-PIKAN
    Rs = []
    for s_ in range(5):
        D_ = np.load(f(k, s_)); Rs.append(100 * np.linalg.norm(D_["pred"] - D_["exact"], axis=1) / np.linalg.norm(D_["exact"], axis=1))
    Rs = np.array(Rs); Pp = D_["P"].ravel()
    if k in ("mlp", "tsckan"): ax[1].fill_between(Pp, Rs.min(0), Rs.max(0), color=col, alpha=0.16, lw=0, zorder=0)
    ax[1].plot(Pp, Rs.mean(0), color=col, ls=ls, lw=lw)
ax[1].set_yscale("log"); ax[1].set_xlim(0.08, 0.18); ax[1].set_ylim(0.1, 100); ax[1].set_xticks([0.08, 0.13, 0.18])
ax[1].minorticks_off(); ax[1].set_yticks([0.1, 1, 10, 100]); ax[1].set_yticklabels(["0.1", "1", "10", "100"]); ax[1].set_xlabel("$P$ (p.u.)", labelpad=1); ax[1].set_ylabel(r"angle error $\varepsilon$ (%)", labelpad=1); style(ax[1])
legend_in(ax[1], [LH(col, ls, lab.replace(" (proposed)", "")) for k, lab, col, ls, _ in M], "upper center", ncol=2,
          handlelength=1.5, columnspacing=0.5, handletextpad=0.3, fontsize=FS_N, labelspacing=0.12, borderpad=0.25)
save_cols(fig, "fig_lossm", H); save(fig, "fig_lossm", tight=False); LEG = LEG0
