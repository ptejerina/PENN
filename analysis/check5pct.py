"""Re-evaluate the 'within 5 %' share of every run with (a) the criterion used in eval_all.py, |V-V*| < 0.05 max(V*, 1),
and (b) the purely relative criterion |V-V*| / V* < 0.05, on the same 2e5 states (same seed), and compare with results_*_FH10.json.
Also records the share of states with V* < 1, where the absolute floor is the binding tolerance."""
import os, re, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_all import runs, load, chunked, EPS, DV
out = {}
for geom in ('circle', 'torus', 'torus4'):
    R = {(o['arm'], o['seed']): o for o in json.load(open(f'results_{geom}_FH10.json'))}
    for arm in runs[geom]:
        for p in runs[geom][arm]:
            seed = int(re.search(r'_s(\d+)/', p).group(1))
            G, m, fwd, ch, ck = load(geom, arm, p)
            torch.manual_seed(7)
            d = G.sample_pos(200000, 'cpu'); dist = G.dist(d); mk = dist > EPS; d, dist = d[mk], dist[mk]
            tau = torch.rand(d.shape[0]) * 7.95 + 0.05
            V = chunked(fwd, d, tau); true = torch.minimum(tau, (dist - EPS).clamp_min(0) / DV); err = (V - true).abs()
            mix = float((err < torch.maximum(0.05 * true, torch.tensor(0.05))).float().mean())
            rel = float((err / true < 0.05).float().mean())
            floor_active = float((true < 1.0).float().mean())          # states where the absolute floor 0.05 is the binding one
            stored = R[(arm, seed)]['frac_within_5pct']
            out[f'{geom}|{arm}|{seed}'] = dict(mixed=mix, relative=rel, stored=stored, floor_share=floor_active, min_true=float(true.min()))
            print(f'{geom:7s} {arm:8s} s{seed}  stored {100*stored:6.2f}  recomputed(mixed) {100*mix:6.2f}  purely relative {100*rel:6.2f}   (V*<1 on {100*floor_active:.1f}% of states)', flush=True)
json.dump(out, open('check5pct.json', 'w'), indent=1)
