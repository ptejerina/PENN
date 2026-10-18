"""Selector statistics of every min-of-charts run on distance-limited states (V* < tau): share of the
identity image and the dominant image when it is not the identity."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from eval_all import runs, load
out = {}
for geom in ('circle', 'torus', 'torus4'):
    for p in runs[geom]['min']:
        G, m, fwd, ch, _ = load(geom, 'min', p); torch.manual_seed(5)
        d = G.sample_pos(60000, 'cpu'); dist = G.dist(d); tau = torch.rand(d.shape[0]) * 7.95 + 0.05
        true = torch.minimum(tau, (dist - EPS).clamp_min(0) / DV); dl = (dist > EPS) & (true < tau - 1e-6)
        d, tau = d[dl], tau[dl]
        with torch.no_grad(): sel = torch.cat([m.active_chart(d[i:i+20000], tau[i:i+20000]) for i in range(0, d.shape[0], 20000)])
        h = torch.bincount(sel, minlength=2 ** G.d).float(); h /= h.sum()
        seed = int(p.split('_s')[-1].split('/')[0]); dom = int(h.argmax())
        out[f'{geom}|{seed}'] = dict(identity=float(h[0]), dominant=dom, dominant_share=float(h[dom]), hist=h.tolist())
        print(f'{geom:7s} s{seed}  identity {h[0]:.3f}  dominant chart {dom} ({h[dom]:.3f})', flush=True)
json.dump(out, open('selector10.json', 'w'), indent=1)
