"""
Kolmogorov-Arnold network (KAN) layers used in the paper.

Chebyshev KAN (c-PIKAN), following the formulation summarized in Toscano et al. (2025),
"From PINNs to PIKANs: Recent advances in physics-informed machine learning", Mach. Learn. Comput. Sci. Eng. 1:15:

    phi(z) = sum_{n=0..K} c_n T_n(tanh z),   T_n Chebyshev polynomials of the first kind, K = degree

B-spline KAN (bs-PIKAN), as in Liu et al. (2024) and Shuai and Li (2025).

Each layer maps in_dim -> out_dim with one learnable univariate function per edge, y_j = sum_i phi_ij(x_i).
"""
import torch
import torch.nn as nn


class ChebyKANLayer(nn.Module):
    """one Chebyshev KAN layer with (degree + 1) coefficients per edge"""
    def __init__(self, in_dim, out_dim, degree=5, norm=True):
        super().__init__()
        self.degree = degree
        self.coef = nn.Parameter(torch.empty(in_dim, out_dim, degree + 1))
        nn.init.normal_(self.coef, mean=0.0, std=1.0 / (in_dim * (degree + 1)))
        self.norm = nn.LayerNorm(out_dim) if norm else nn.Identity()

    def forward(self, x):                                  # (..., in_dim)
        z = torch.tanh(x)
        T = [torch.ones_like(z), z]
        for _ in range(2, self.degree + 1):
            T.append(2 * z * T[-1] - T[-2])
        T = torch.stack(T[: self.degree + 1], -1)          # (..., in_dim, d+1)
        y = torch.einsum("...id,iod->...o", T, self.coef)
        return self.norm(y)


class ChebyKAN(nn.Module):
    """stack of Chebyshev KAN layers; the paper uses norm=False (no layer normalization between layers)"""
    def __init__(self, dims, degree=5, norm=True):
        super().__init__()
        self.layers = nn.ModuleList([ChebyKANLayer(a, b, degree, norm=(norm and i < len(dims) - 2))
                                     for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class BSplineKANLayer(nn.Module):
    """B-spline KAN layer: phi(x) = w_b * silu(x) + w_s * sum_s c_s B_s(x), with B-splines of degree k (cubic for k = 3)
    on a uniform grid with G intervals on [-1, 1]; per edge G + k coefficients, one base weight and one scale"""
    def __init__(self, in_dim, out_dim, grid=5, k=3, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_dim, self.out_dim, self.k = in_dim, out_dim, k
        h = (grid_range[1] - grid_range[0]) / grid
        g = torch.arange(-k, grid + k + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("grid", g.expand(in_dim, -1).contiguous())          # (in, G+2k+1)
        self.base_w = nn.Parameter(torch.empty(out_dim, in_dim))
        self.spline_w = nn.Parameter(torch.randn(out_dim, in_dim, grid + k) * 0.1 / (in_dim ** 0.5))
        self.spline_scale = nn.Parameter(torch.ones(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_w, a=5 ** 0.5)

    def basis(self, x):                                    # x (..., in) -> (..., in, G+k)
        g = self.grid; x = x.unsqueeze(-1)
        B = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            B = (x - g[:, :-(p + 1)]) / (g[:, p:-1] - g[:, :-(p + 1)]) * B[..., :-1] + \
                (g[:, p + 1:] - x) / (g[:, p + 1:] - g[:, 1:-p]) * B[..., 1:]
        return B

    def forward(self, x):
        base = torch.nn.functional.silu(x) @ self.base_w.T
        spl = torch.einsum("...is,ois->...o", self.basis(x), self.spline_w * self.spline_scale.unsqueeze(-1))
        return base + spl


class BSplineKAN(nn.Module):
    """stack of B-spline KAN layers"""
    def __init__(self, dims, grid=5, k=3):
        super().__init__()
        self.layers = nn.ModuleList([BSplineKANLayer(a, b, grid, k) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
