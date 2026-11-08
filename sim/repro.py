"""
Single-machine infinite-bus (SMIB) benchmark of Misyris, Venzke and Chatzivasileiadis, "Physics-informed neural
networks for power systems", IEEE PES GM 2020, and the models compared in the paper.

The PINN is a PyTorch port of the original TensorFlow 1.x code
(https://github.com/gmisy/Physics-Informed-Neural-Networks-for-Power-Systems) and keeps:
  * the data file, input normalization, network size (5 x 10, tanh), Xavier truncated-normal weights, zero biases;
  * the loss MSE_u + MSE_f with the residual f = m*u_tt + d*u_t + B*sin(u) - P;
  * the sampling of training data and collocation points with numpy seed 1234 (pyDOE 0.3.8 classic LHS);
  * the optimizer: scipy L-BFGS-B with the original options (maxiter = maxfun = 50000, maxcor = maxls = 50, ...).
Only the values of the random initial weights differ, since the TensorFlow random stream cannot be reproduced in PyTorch.

Models (--arch): mlp = PINN, bkan = bs-PIKAN, ckan = c-PIKAN, tsmlp = ma-PINN, tsckan = mac-PIKAN.

  python repro.py inference --arch tsckan --seed 0          # mac-PIKAN, seed 0
  python repro.py inference --arch mlp --nf 80000           # PINN with 80 000 collocation points
  python repro.py inference --arch tsckan --d 0.11          # lower damping (reference data recomputed)
"""
import argparse, json, os, sys, time
import numpy as np, scipy.io, torch
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")   # swingEquation_inference.mat from the public repository of Misyris et al.
torch.set_default_dtype(torch.float32)
sys.path.insert(0, HERE)
from kan import ChebyKAN, BSplineKAN


class KANNet(torch.nn.Module):
    """c-PIKAN [2, W, W, 1] (W = 8, or CKAN_W) with Chebyshev degree 5, or bs-PIKAN [2, 5, 1] with grid 10;
    same input normalization as the PINN"""
    def __init__(self, kind, lb, ub):
        super().__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32); self.ub = torch.tensor(ub, dtype=torch.float32)
        self.body = ChebyKAN([2, int(os.environ.get("CKAN_W", 8)), int(os.environ.get("CKAN_W", 8)), 1], degree=5, norm=False) if kind == "ckan" else BSplineKAN([2, 5, 1], grid=10, k=3)

    def forward(self, x, t):
        return self.body(2.0 * (torch.cat([x, t], 1) - self.lb) / (self.ub - self.lb) - 1.0)


class TrialSolution(torch.nn.Module):
    """modal-augmented trial solution: delta = delta_lin(P, t) + (t/T)^2 N(P, t), where delta_lin is the closed-form
    small-signal response of m d'' + d d' + B sin d = P about delta_s = asin(P/B) with delta(0) = delta0 and
    delta'(0) = omega0. Since (t/T)^2 N and its time derivative vanish at t = 0, both initial conditions hold exactly."""
    def __init__(self, inner, m=0.4, d=0.15, B=0.2, d0=0.1, w0=0.1, T=20.0):
        super().__init__(); self.inner = inner; self.c = (m, d, B, d0, w0, T)

    def forward(self, x, t):
        m, d, B, d0, w0, T = self.c
        ds = torch.asin(torch.clamp(x / B, -1, 1)); wn2 = B * torch.cos(ds) / m; sg = -d / (2 * m)
        wd = torch.sqrt(torch.clamp(wn2 - sg ** 2, min=1e-6)); e0 = d0 - ds
        lin = ds + torch.exp(sg * t) * (e0 * torch.cos(wd * t) + (w0 - sg * e0) / wd * torch.sin(wd * t))
        return lin + (t / T) ** 2 * self.inner(x, t)


def make_net(arch, layers, lb, ub, d=0.15, T=20.0):
    base = Net(layers, lb, ub) if arch in ("mlp", "tsmlp") else KANNet("ckan" if "ckan" in arch else "bkan", lb, ub)
    return TrialSolution(base, d=d, T=T) if arch.startswith("ts") else base


def lhs_classic(n, samples):
    """pyDOE 0.3.8 lhs(n, samples) with criterion=None (_lhsclassic), as used by the original code"""
    cut = np.linspace(0, 1, samples + 1)
    u = np.random.rand(samples, n)
    a, b = cut[:samples], cut[1:samples + 1]
    rd = np.zeros_like(u)
    for j in range(n):
        rd[:, j] = u[:, j] * (b - a) + a
    H = np.zeros_like(rd)
    for j in range(n):
        order = np.random.permutation(range(samples))
        H[:, j] = rd[order, j]
    return H


class Net(torch.nn.Module):
    """PINN of Misyris et al.: MLP with tanh, inputs (P, t) scaled to [-1, 1]"""
    def __init__(self, layers, lb, ub):
        super().__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32); self.ub = torch.tensor(ub, dtype=torch.float32)
        self.W = torch.nn.ParameterList(); self.b = torch.nn.ParameterList()
        for i in range(len(layers) - 1):
            std = np.sqrt(2 / (layers[i] + layers[i + 1]))
            w = torch.empty(layers[i], layers[i + 1]); torch.nn.init.trunc_normal_(w, std=std, a=-2 * std, b=2 * std)
            self.W.append(torch.nn.Parameter(w)); self.b.append(torch.nn.Parameter(torch.zeros(1, layers[i + 1])))

    def forward(self, x, t):
        H = 2.0 * (torch.cat([x, t], 1) - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.W[:-1], self.b[:-1]):
            H = torch.tanh(H @ W + b)
        return H @ self.W[-1] + self.b[-1]


def residual(net, x, t, m, d, nu=0.2):
    """swing-equation residual f = m u_tt + d u_t + B sin(u) - P (nu = B = 0.2), by automatic differentiation"""
    t = t.clone().requires_grad_(True)
    u = net(x, t)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
    return m * u_tt + d * u_t + nu * torch.sin(u) - x


def lbfgs(params, closure, opts, log=None):
    """scipy L-BFGS-B over a list of torch parameters (as tf.contrib.opt.ScipyOptimizerInterface in the original code)"""
    shapes = [p.shape for p in params]; sizes = [p.numel() for p in params]

    def setx(x):
        i = 0
        for p, s, n in zip(params, shapes, sizes):
            p.data = torch.tensor(x[i:i + n], dtype=torch.float32).reshape(s); i += n

    def fg(x):
        setx(x)
        for p in params: p.grad = None
        L = closure(); L.backward()
        g = np.concatenate([p.grad.detach().numpy().ravel().astype(np.float64) for p in params])
        if log is not None: log.append(float(L))
        return float(L), g

    x0 = np.concatenate([p.detach().numpy().ravel().astype(np.float64) for p in params])
    r = minimize(fg, x0, jac=True, method="L-BFGS-B", options=opts)
    setx(r.x); return r


def omega_numdiff(U, h):
    """frequency by forward differences of the angle, omega = (delta(t+h) - delta(t))/h with h = 0.1 s, as in Misyris et al."""
    W = np.empty_like(U); W[..., :-1] = (U[..., 1:] - U[..., :-1]) / h; W[..., -1] = W[..., -2]
    return W


def inference(a):
    np.random.seed(1234); torch.manual_seed(1234)
    nu, N_u, N_f, layers = 0.2, a.nu, a.nf, [2, 10, 10, 10, 10, 10, 1]
    data = scipy.io.loadmat(os.path.join(DATA, "swingEquation_inference.mat"))
    t = data["t"].flatten()[:, None]; x = data["x"].flatten()[:, None]; Exact = np.real(data["usol"]).T
    stress = (a.d != 0.15) or (a.T != 20.0) or (a.pmax != 0.18)
    if stress:   # changed setting: reference data recomputed on the same 100 x 201 grid, so the training data draw is identical
        x = np.linspace(0.08, a.pmax, 100)[:, None]; t = np.linspace(0, a.T, 201)[:, None]
        Exact = swing_data(0.4, a.d, x.ravel(), t.ravel()).T
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])); u_star = Exact.flatten()[:, None]
    lb = np.array([0.08, 0.0]); ub = np.array([float(x[-1, 0]), float(t[-1, 0])])
    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T)); uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1])); uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:])); uu3 = Exact[:, -1:]
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub - lb) * lhs_classic(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]; u_train = u_train[idx, :]

    torch.manual_seed(a.seed)
    net = make_net(a.arch, layers, lb, ub, d=a.d, T=float(ub[1]))
    f32 = lambda z: torch.tensor(z, dtype=torch.float32)
    xu, tu, uu = f32(X_u_train[:, :1]), f32(X_u_train[:, 1:]), f32(u_train)
    xf, tf_ = f32(X_f_train[:, :1]), f32(X_f_train[:, 1:])
    closure = lambda: (((uu - net(xu, tu)) ** 2).mean() if N_u > 0 else 0.0) + (residual(net, xf, tf_, 0.4, a.d, nu) ** 2).mean()
    hist = []; t0 = time.time()
    r = lbfgs(list(net.parameters()), closure, dict(maxiter=50000, maxfun=50000, maxcor=50, maxls=50, gtol=1e-8,
                                                    eps=1e-8, ftol=1e-15), hist)
    train_s = time.time() - t0
    with torch.no_grad():
        t0 = time.time(); u_pred = net(f32(X_star[:, :1]), f32(X_star[:, 1:])).numpy(); pred_s = time.time() - t0
    err = float(np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2))
    U_pred = u_pred.reshape(len(t), len(x)).T                       # (P, t)
    E = Exact.T
    per = np.linalg.norm(U_pred - E, axis=1) / np.linalg.norm(E, axis=1)
    print(f"[inference {a.arch} params={sum(p.numel() for p in net.parameters())} Nf={N_f} seed={a.seed}] L-BFGS its {r.nit}, evals {r.nfev}, {r.message}; train {train_s:.0f} s; "
          f"error u {err:.3e}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = f"inf_{a.arch}_nf{N_f}_s{a.seed}" if a.arch != "mlp" else f"inf_nf{N_f}_s{a.seed}"
    if stress: tag = f"st_d{a.d:g}_T{a.T:g}_P{a.pmax:g}_{a.arch}_s{a.seed}"
    if N_u != 40: tag = f"nu{N_u}_" + tag
    np.savez(os.path.join(a.out, tag + ".npz"), pred=U_pred, exact=E, P=x.ravel(), t=t.ravel(), Xu=X_u_train, loss=np.array(hist))
    json.dump(dict(arch=a.arch, params=sum(p.numel() for p in net.parameters()), nf=N_f, seed=a.seed, error_u=err, nit=int(r.nit), nfev=int(r.nfev), train_s=train_s, predict_s=pred_s,
                   best_traj=float(per.min()), worst_traj=float(per.max()), P_best=float(x.ravel()[per.argmin()]),
                   P_worst=float(x.ravel()[per.argmax()]), msg=str(r.message)),
              open(os.path.join(a.out, tag + ".json"), "w"), indent=1)


def swing_data(m, d, x, t):
    """reference trajectories of m u'' + d u' + 0.2 sin(u) = P for every P in x, u(0) = u'(0) = 0.1, on the time grid t"""
    U = np.zeros((len(x), len(t)))
    for i, p in enumerate(x):
        s = solve_ivp(lambda tt, y: [y[1], (p - d * y[1] - 0.2 * np.sin(y[0])) / m], [0, t[-1]], [0.1, 0.1], t_eval=t,
                      rtol=1e-10, atol=1e-12)
        U[i] = s.y[0]
    return U


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["inference"])
    ap.add_argument("--nf", type=int, default=8000)
    ap.add_argument("--arch", default="mlp", choices=["mlp", "bkan", "ckan", "tsmlp", "tsckan"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--d", type=float, default=0.15); ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--pmax", type=float, default=0.18)
    ap.add_argument("--nu", type=int, default=40)
    a = ap.parse_args()
    inference(a)
