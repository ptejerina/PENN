"""Shared setup of the analysis scripts: repository root, data root, trainer module, constants.

ROOT is the repository root (gpu_critic.py and grid_certificate.py are imported from there).
B is the data root, where runs_fullhorizon/ and fold_evolution/ are expected; it defaults to ROOT
and can be overridden with the environment variable P2_BUNDLE when the checkpoints live elsewhere.
"""
import os, sys, torch, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.environ.get('P2_BUNDLE', ROOT)
sys.path.insert(0, ROOT)
import gpu_critic as gc
import grid_certificate as se
torch.set_num_threads(int(os.environ.get('P2_THREADS', 4)))
EPS, DV = 0.08, 0.2


def vstar(dist, tau):
    """Analytic value V*(d, tau) = min(tau, (d - eps)_+ / dv)."""
    return np.minimum(tau, np.maximum(dist - EPS, 0) / DV)


def fn_of(fwd):
    """Wrap a torch forward fwd(points, tau) into fn(X, Y, tau) on meshgrid arrays (for grid_certificate)."""
    def f(X, Y, tau):
        with torch.no_grad():
            p = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32)
            return fwd(p, torch.full((p.shape[0],), float(tau))).numpy().reshape(X.shape)
    return f


NAMES = {'smooth': 'Fourier-feature', 'rawsoft': 'soft-periodic', 'min': 'min-of-charts', 'folded': 'geometric'}
