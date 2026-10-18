"""Score only the checkpoints not yet in results_<geom>_FH10.json and merge them in (same scorer as eval_all.py)."""
import os, re, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_all import runs, score
for geom in sys.argv[1:]:
    fn = f'results_{geom}_FH10.json'; res = json.load(open(fn)); have = {(o['arm'], o['seed']) for o in res}
    for arm in runs[geom]:
        for p in runs[geom][arm]:
            seed = int(re.search(r'_s(\d+)/', p).group(1))
            if (arm, seed) in have: continue
            o = score(geom, arm, p); o['seed'] = seed; res.append(o)
            print(f"{geom:7s} {arm:8s} s{seed}  vRMS={o['value_rms']:.4f}  5%={100*o['frac_within_5pct']:.1f}  msr_int={o['msr_interior']:.2e}  grad_band={o['grad_band']:.2f}  cert={o['cert_mc']['pct']:.2f}%/{o['cert_mc']['maxratio']:.1f}", flush=True)
    res.sort(key=lambda o: (o['arm'], o['seed'])); json.dump(res, open(fn, 'w'), indent=1); print(geom, 'now', len(res), 'runs')
