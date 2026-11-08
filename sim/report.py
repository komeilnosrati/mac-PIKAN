"""Writes my_results.txt: Table II, every number in the text and the training details from YOUR run, next to the values
in the paper. Run from the sim folder after run_all.py:  python report.py"""
import os, re, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
HDR = ["model", "parameters", "angle error (%)", "frequency error (%)", "best traj. (%)", "worst traj. (%)",
       "max omega(0) error (rad/s)", "training time (s)"]


def clean(x):
    x = re.sub(r"\\eqref\{[^}]*\}", "(9)", x); x = re.sub(r"~?\\cite\{[^}]*\}", "", x)
    x = re.sub(r"\\best\{([^}]*)\}", r"\1", x); x = x.replace("\\rowcolor{black!6}", "")
    x = x.replace("\\,$\\pm$\\,", " +- ").replace("--", "-").replace("\\midrule", "")
    return x.strip()


def rows(tex):
    return [[clean(c) for c in r.split("&")] for r in tex.split("\\\\") if "&" in r and "Para-" not in r and "meters" not in r]


def table(R):
    W = [max(len(str(r[i])) for r in R) for i in range(len(R[0]))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, W)) for r in R)


macro = re.compile(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}")
mine = open(os.path.join("..", "paper", "numbers_smib.tex"), encoding="utf-8").read()
MY = dict(macro.findall(mine))
k = mine.index("\\newcommand{\\TABLEROWS}{"); MYROWS = mine[k + len("\\newcommand{\\TABLEROWS}{"):].rstrip()[:-1]

# values printed in the paper (Table II and the numbers from summary.py)
PAPER = {
    'eMLP': '1.59', 'sMLP': '0.71', 'wMLP': '5.26', 'tMLP': '82', 'zMLP': '0.024', 'mMLP': '47', 'loMLP': '0.42',
    'hiMLP': '2.52', 'tbestMLP': '0.07', 'tworstMLP': '2.26', 'eiMLP': '1.00', 'eBK': '0.86', 'sBK': '0.41',
    'wBK': '3.74', 'tBK': '4368', 'zBK': '0.024', 'mBK': '611', 'loBK': '0.18', 'hiBK': '1.46', 'tbestBK': '0.02',
    'tworstBK': '1.53', 'eiBK': '--', 'eCK': '3.14', 'sCK': '1.30', 'wCK': '13.58', 'tCK': '634', 'zCK': '0.098',
    'mCK': '138', 'loCK': '0.73', 'hiCK': '4.43', 'tbestCK': '0.02', 'tworstCK': '5.28', 'eiCK': '--', 'eTP': '0.97',
    'sTP': '0.34', 'wTP': '2.77', 'tTP': '89', 'zTP': '0.001', 'mTP': '53', 'loTP': '0.44', 'hiTP': '1.45',
    'tbestTP': '0.09', 'tworstTP': '1.54', 'eiTP': '--', 'eTS': '0.29', 'sTS': '0.06', 'wTS': '0.99', 'tTS': '656',
    'zTS': '0.001', 'mTS': '158', 'loTS': '0.23', 'hiTS': '0.39', 'tbestTS': '0.04', 'tworstTS': '0.54',
    'eiTS': '--', 'ERRMLP': '1.59', 'ERRTS': '0.29', 'ERWMLP': '5.26', 'ERWTS': '0.99', 'RATIO': '5.5',
    'RATIOW': '5.3', 'ERRMLPEIGHTY': '1.00', 'TB': '6.7', 'TP': '8.0', 'eLIN': '6.39', 'wLIN': '15.26',
    'CELO': '0.07', 'CEHI': '0.42', 'PSLOPE': '2.00', 'EPSLO': '0.33', 'EPSHI': '1.02', 'ELINLO': '0.006',
    'ELINHI': '0.28', 'DLO': '0.31', 'DHI': '1.02', 'WINS': '5', 'NSEED': '5', 'CTRMLP': '1.31', 'CTRTS': '0.23',
    'CTRGAIN': '5.6', 'EDGGAIN': '4.0', 'TSLO': '0.04', 'TSHI': '0.44'}
PAPERROWS = r"""
Small-signal model \eqref{eq:smiblin} & 0 & 6.39 & 15.26 & 0.59 & 12.36 & 0.009 & -- \\
\midrule
PINN~\cite{misyris2020pinn} & 481 & 1.59\,$\pm$\,0.71 & 5.26\,$\pm$\,2.30 & 0.07 & 2.26 & 0.024 & 82 \\
bs-PIKAN~\cite{shuai2025pikan} & 225 & 0.86\,$\pm$\,0.41 & 3.74\,$\pm$\,1.79 & \best{0.02} & 1.53 & 0.024 & 4368 \\
c-PIKAN & 528 & 3.14\,$\pm$\,1.30 & 13.58\,$\pm$\,5.52 & \best{0.02} & 5.28 & 0.098 & 634 \\
ma-PINN (ablation) & 481 & 0.97\,$\pm$\,0.34 & 2.77\,$\pm$\,0.69 & 0.09 & 1.54 & \best{0.001} & 89 \\
\rowcolor{black!6}mac-PIKAN (proposed) & 528 & \best{0.29\,$\pm$\,0.06} & \best{0.99\,$\pm$\,0.24} & 0.04 & \best{0.54} & \best{0.001} & 656 \\
"""

out = ["=" * 100 + "\nTABLE II (your run)\n" + "=" * 100, table([HDR] + rows(MYROWS)),
       "\nTABLE II (paper)\n" + "-" * 100, table([HDR] + rows(PAPERROWS)),
       "\n" + "=" * 100 + "\nVALUES FROM summary.py (numbers_smib.tex): yours and paper\n" + "=" * 100]
for kk, v in PAPER.items():
    if kk in ("best", "TABLEROWS"): continue
    yv = MY.get(kk, "missing")
    out.append(f"{kk:<14s} yours {yv:<10s} paper {v:<10s}" + ("" if yv == v else "   <- differs"))
out.append("\n" + "=" * 100 + "\nNUMBERS WRITTEN DIRECTLY IN THE TEXT (text_numbers.py)\n" + "=" * 100)
out.append(subprocess.run([sys.executable, "text_numbers.py"], capture_output=True, text=True).stdout)
out.append("=" * 100 + "\nTRAINING DETAILS (summary.py): iterations, training time, cost per loss evaluation\n" + "=" * 100)
s = subprocess.run([sys.executable, "summary.py"], capture_output=True, text=True).stdout
out.append("\n".join(l for l in s.splitlines() if l.split()[:1] and l.split()[0] in ("mlp", "bkan", "ckan", "tsmlp", "tsckan")))
out.append("  (mlp = PINN, bkan = bs-PIKAN, ckan = c-PIKAN, tsmlp = ma-PINN, tsckan = mac-PIKAN; its = iterations, "
           "t = training time, ms/eval = cost of one loss and gradient evaluation; err80 = PINN with N_f = 80 000)")
open("my_results.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("wrote sim/my_results.txt")
