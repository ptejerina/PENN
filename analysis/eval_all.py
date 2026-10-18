"""Uniform scoring of every checkpoint of the campaign (4 configurations x 3 geometries x 10 seeds)
with one set of definitions.  Runs are read from <repo>/runs_fullhorizon/{S1,T2,T4}/<geom>_<config>_s<seed>/<ckpt>
(every seed found on disk, up to P2_SEEDS, default 10) and the output is results_<geom>_FH10.json.

  value RMS vs V* on 2e5 uniform states, tau ~ U(0.05, 8), d > eps; share within 5 %:
      |V - V*| < 0.05 max(V*, 1)
  capture band d in (eps, 1.5 eps], tau ~ U(0.5, 7.5): mean ||grad V|| (analytic 1/dv = 5) and MSR
  mean-squared residual (MSR): protocol region (d > 1.5 eps, tau in (0.05, 8)) and
      interior (d > 3 eps, 0.5 < tau < 7.5); 16k points each
  seam gap: max |V(x_1 = L/2 - delta) - V(x_1 = -L/2 + delta)| over 2000 random (rest, tau), delta = 1e-3, 1e-2
  Monte Carlo semiconcavity certificate at tau = 4: 4e4 probes, 8 random unit directions (1 on S^1),
      h = 0.02, lambda_max ~ max_j S_h; mask d <= 3 eps and concave creases (min_j S_h < -3 C*),
      C* = 1/(dv d); flag lambda_max / C* > 2; also applied to V* and to a planted sawtooth
  grid certificate (T^2 only, 241^2, grid_certificate.certificate) for comparison
  selector: share of states where the min-of-charts NN selects the identity image
  best validation loss and best residual term over training (from the checkpoint history)

Usage (from this directory, checkpoints in ../runs_fullhorizon or $P2_BUNDLE/runs_fullhorizon):
  python3 eval_all.py circle; python3 eval_all.py torus; python3 eval_all.py torus4
"""
import os, re, sys, json, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import B, gc, se, EPS, DV
torch.set_num_threads(int(os.environ.get('P2_THREADS', 2)))

# ---- the campaign: runs_fullhorizon/{S1,T2,T4}/<geom>_<config>_s<seed>/<ckpt>
RF = B + '/runs_fullhorizon'; NS = int(os.environ.get('P2_SEEDS', 10))
SUB = {'circle': 'S1', 'torus': 'T2', 'torus4': 'T4'}
CK = {'smooth': 'gpu_{g}.pt', 'rawsoft': 'gpu_{g}_rawsoft.pt', 'min': 'gpu_{g}_charts.pt', 'folded': 'gpu_{g}_charts1.pt'}
runs = {}
for geom in ('circle', 'torus', 'torus4'):
    runs[geom] = {}
    for arm in ('smooth', 'rawsoft', 'min', 'folded'):
        paths = [f'{RF}/{SUB[geom]}/{geom}_{arm}_s{s}/' + CK[arm].format(g=geom) for s in range(NS)]
        runs[geom][arm] = [p for p in paths if os.path.exists(p)]
GEOM = {'torus': gc.Torus, 'circle': gc.Circle, 'torus4': gc.Torus4}

def load(geom, arm, path):
    G = GEOM[geom](); D = G.features(G.sample_pos(2, 'cpu'), torch.zeros(2)).shape[1]
    if arm == 'smooth': m = gc.MLP(D, 128); fwd = lambda d, t: m(G.features(d, t)); charts = False
    elif arm == 'rawsoft': m = gc.RawSoftCritic(G, 128); fwd = lambda d, t: m.forward_torus(d, t); charts = True
    else: m = gc.ChartsCritic(G, D, 128, single=(arm == 'folded')); fwd = lambda d, t: m.forward_torus(d, t); charts = True
    ck = torch.load(path, map_location='cpu'); m.load_state_dict(ck['model']); m.eval()
    return G, m, fwd, charts, ck

def chunked(fwd, d, t, n=20000):
    with torch.no_grad():
        return torch.cat([fwd(d[i:i + n], t[i:i + n]) for i in range(0, d.shape[0], n)])

def score(geom, arm, path):
    G, m, fwd, charts, ck = load(geom, arm, path)
    L, dim = G.L, G.d
    out = {'geom': geom, 'arm': arm, 'path': path.replace(B + '/', ''), 'iter': ck['iter'],
           'final_val': ck['hist']['val'][-1], 'final_hji': ck['hist']['hji'][-1]}
    # value error
    torch.manual_seed(7)
    d = G.sample_pos(200000, 'cpu'); dist = G.dist(d); mk = dist > EPS; d, dist = d[mk], dist[mk]
    tau = torch.rand(d.shape[0]) * 7.95 + 0.05
    V = chunked(fwd, d, tau)
    true = torch.minimum(tau, (dist - EPS).clamp_min(0) / DV); err = (V - true).abs()
    acc = err < torch.maximum(0.05 * true, torch.tensor(0.05))
    out['value_rms'] = float((err ** 2).mean().sqrt()); out['mean_true'] = float(true.mean())
    out['frac_within_5pct'] = float(acc.float().mean()); out['max_abs_err'] = float(err.max())
    hor = true >= tau - 1e-6
    out['bias_horizon'] = float((V - true)[hor].mean()); out['bias_cone'] = float((V - true)[~hor].mean())
    out['rms_horizon'] = float(((V - true)[hor] ** 2).mean().sqrt()); out['rms_cone'] = float(((V - true)[~hor] ** 2).mean().sqrt())
    if arm == 'min':
        with torch.no_grad():
            sel = torch.cat([m.active_chart(d[i:i + 20000], tau[i:i + 20000]) for i in range(0, d.shape[0], 20000)])
        out['sel_identity_frac'] = float((sel == 0).float().mean())
        out['sel_hist'] = torch.bincount(sel, minlength=2 ** dim).tolist()
    out['best_val'] = float(min(ck['hist']['val'])); out['best_hji'] = float(min(ck['hist']['hji']))
    # capture band d in (eps, 1.5 eps], tau ~ U(0.5, 7.5): mean gradient norm (analytic 5) and MSR
    torch.manual_seed(99)
    db = G.sample_pos(400000, 'cpu'); dbd = G.dist(db); mb = (dbd > EPS) & (dbd <= 1.5 * EPS); db = db[mb]
    tb = torch.rand(db.shape[0]) * 7.0 + 0.5
    gbs, rbs = [], []
    for i in range(0, db.shape[0], 4000):
        Vv, gd, gt = gc.eval_V_and_grads(m, G, charts, db[i:i + 4000].clone(), tb[i:i + 4000].clone())
        rbs.append(G.residual_terms(Vv, gd, gt, 1.0, 1.2).detach()); gbs.append(gd.detach().norm(dim=-1))
    out['grad_band'] = float(torch.cat(gbs).mean()); out['msr_band'] = float((torch.cat(rbs) ** 2).mean()); out['n_band'] = int(db.shape[0])
    # residuals
    def msr(n, lo, tlo, thi, seed=4242):
        torch.manual_seed(seed)
        dd = G.sample_pos(8 * n, 'cpu'); mm = G.dist(dd) > lo * EPS; d2 = dd[mm][:n]
        t2 = torch.rand(d2.shape[0]) * (thi - tlo) + tlo
        rs = []
        for i in range(0, d2.shape[0], 4000):
            Vv, gd, gt = gc.eval_V_and_grads(m, G, charts, d2[i:i + 4000].clone(), t2[i:i + 4000].clone())
            rs.append(G.residual_terms(Vv, gd, gt, 1.0, 1.2).detach())
        r = torch.cat(rs)
        return float((r ** 2).mean()), float(r.abs().max())
    out['msr_protocol'], out['maxres_protocol'] = msr(16384, 1.5, 0.05, 8.0)
    out['msr_interior'], out['maxres_interior'] = msr(16384, 3.0, 0.5, 7.5)
    # seam gap
    torch.manual_seed(11)
    rest = torch.rand(2000, dim - 1) * 1.9 - 0.95; taus = torch.rand(2000) * 7.8 + 0.1
    gaps = {}
    for delta in (1e-3, 1e-2):
        xp = torch.cat([torch.full((2000, 1), L / 2 - delta), rest], 1)
        xm = torch.cat([torch.full((2000, 1), -L / 2 + delta), rest], 1)
        with torch.no_grad():
            gaps[str(delta)] = float((fwd(xp, taus) - fwd(xm, taus)).abs().max())
    out['seam_gap'] = gaps
    # Monte Carlo certificate
    def wrap(x): return (x + L / 2) % L - L / 2
    def V_model(x, tau):
        p = torch.tensor(wrap(x), dtype=torch.float32)
        return chunked(fwd, p, torch.full((p.shape[0],), float(tau))).numpy()
    def V_star(x, tau):
        dd = np.linalg.norm(wrap(x), axis=-1); return np.minimum(tau, np.maximum(dd - EPS, 0) / DV)
    def V_saw(x, tau):
        u = (x[:, 0] / 0.5) % 1.0; return 1.25 * (1.0 - np.abs(2.0 * u - 1.0))
    def mc_cert(fn, tau=4.0, n_probe=40000, n_dir=8, h=0.02, seed=123):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-L / 2, L / 2, size=(n_probe, dim)); dd = np.linalg.norm(x, axis=-1)
        keep = dd > 3 * EPS; x, dd = x[keep], dd[keep]
        if dim == 1: u = np.ones((1, 1))
        else: u = rng.normal(size=(n_dir, dim)); u /= np.linalg.norm(u, axis=-1, keepdims=True)
        V0 = fn(x, tau); S = np.empty((u.shape[0], x.shape[0]))
        for i in range(u.shape[0]):
            S[i] = (fn(x + h * u[i], tau) - 2 * V0 + fn(x - h * u[i], tau)) / h ** 2
        lmax, lmin = S.max(0), S.min(0); Cs = 1.0 / (DV * dd)
        mask = lmin < -3 * Cs; ratio = lmax / Cs; flagged = (ratio > 2.0) & ~mask
        return dict(pct=float(100.0 * flagged.sum() / max(1, (~mask).sum())), maxratio=float(ratio[~mask].max()),
                    n_probes=int((~mask).sum()))
    out['cert_mc'] = mc_cert(V_model); out['cert_mc_vstar'] = mc_cert(V_star); out['cert_mc_saw'] = mc_cert(V_saw)
    if geom == 'torus':
        def fn2(X, Y, tau):
            p = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32)
            return chunked(fwd, p, torch.full((p.shape[0],), float(tau))).numpy().reshape(X.shape)
        c = se.certificate(fn2, tau=4.0); out['cert_grid'] = dict(pct=float(c['pct']), maxratio=float(c['maxratio']))
    return out

if __name__ == '__main__':
    sel = sys.argv[1:]  # optional: geom names
    res = []
    for geom in runs:
        if sel and geom not in sel: continue
        for arm in runs[geom]:
            for i, p in enumerate(runs[geom][arm]):
                o = score(geom, arm, p); ms = re.search(r'_s(\d+)/', p); o['seed'] = int(ms.group(1)) if ms else i; res.append(o)
                print(f"{geom:7s} {arm:8s} s{o['seed']}  vRMS={o['value_rms']:.4f}  5%={100*o['frac_within_5pct']:.1f}  "
                      f"msr_int={o['msr_interior']:.2e}  seam={o['seam_gap']['0.001']:.1e}  cert={o['cert_mc']['pct']:.2f}%/{o['cert_mc']['maxratio']:.1f}"
                      + (f"  grid={o['cert_grid']['pct']:.2f}%" if 'cert_grid' in o else '')
                      + (f"  sel0={o['sel_identity_frac']:.3f}" if 'sel_identity_frac' in o else ''), flush=True)
    tag = '_'.join(sel) if sel else 'all'
    suffix = '_FH10'
    json.dump(res, open(f'results_{tag}{suffix}.json', 'w'), indent=1)
