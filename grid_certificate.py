"""Grid-based semiconcavity certificate on the torus T^2 (comparison baseline for the Monte Carlo
certificate of analysis/eval_all.py::mc_cert, which is the one reported in the paper).

At tau = TAU_CERT the candidate V(x, y, tau) is evaluated on an N x N grid of the fundamental
domain [-1, 1]^2; the 2 x 2 Hessian is formed from second differences of forward evaluations
(step = grid spacing), and its largest eigenvalue is compared with the semiconcavity bound
C* = 1/(dv d).  Probes at d <= 3 eps and the (dilated) neighbourhood of concave creases
(lambda_min < -3 C*) are masked; a probe is flagged when lambda_max > 2 C*.

Usage: certificate(fn, tau) with fn(X, Y, tau) -> V on the meshgrid arrays X, Y.
"""
import numpy as np

L, EPS, VE, VP = 2.0, 0.08, 1.0, 1.2
DV = VP - VE
TAU_CERT, N_GRID = 4.0, 241


def wrap(x):
    return (x + L / 2) % L - L / 2


def V_star_np(X, Y, tau):
    d = np.hypot(wrap(X), wrap(Y))
    return np.minimum(tau, np.maximum(d - EPS, 0.0) / DV)


def grid(N=N_GRID):
    xs = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(xs, xs)
    return xs, X, Y, xs[1] - xs[0]


def dilate(mask):
    out = mask.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out |= np.roll(np.roll(mask, dx, 0), dy, 1)
    return out


def certificate(fn, tau=TAU_CERT, N=N_GRID):
    xs, X, Y, h = grid(N)
    d = np.hypot(X, Y)
    ev = lambda sx, sy: fn(X + sx, Y + sy, tau)
    V = ev(0, 0)
    Vxx = (ev(h, 0) - 2 * V + ev(-h, 0)) / h**2
    Vyy = (ev(0, h) - 2 * V + ev(0, -h)) / h**2
    Vxy = (ev(h, h) - ev(h, -h) - ev(-h, h) + ev(-h, -h)) / (4 * h**2)
    tr, disc = 0.5 * (Vxx + Vyy), np.sqrt(0.25 * (Vxx - Vyy)**2 + Vxy**2)
    lmax, lmin = tr + disc, tr - disc
    Cs = 1.0 / (DV * np.maximum(d, 1e-9))
    mask = (d <= 3 * EPS) | dilate(lmin < -3 * Cs)
    ratio = lmax / Cs
    flagged = (ratio > 2.0) & ~mask
    return dict(V=V, ratio=ratio, mask=mask, flagged=flagged, d=d,
                pct=100.0 * flagged.sum() / (~mask).sum(),
                maxratio=float(ratio[~mask].max()))
