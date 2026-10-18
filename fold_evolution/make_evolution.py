"""Training evolution of the Fourier-feature NN on S^1 (seed 0, 1e5 iterations, snapshots every 500 its plus
log-spaced early ones), from the snapshots written by gpu_critic_snap.py into <run>/snapshots/snap_*.pt.
Outputs in <run>/: fold_evolution.gif (all snapshots, V(Delta, tau=TAU) against V*, with the validation-loss
curve and a marker), fig_fold_evolution.pdf (static 2 x 4 panel at selected iterations; the paper's version is
drawn by the notebook from evolution_profiles.npz) and evolution_profiles.npz (every profile).
Usage: python3 make_evolution.py [run_dir]      (EVO_SEL=0,100,... selects the iterations of the static panel)"""
import sys, os, glob, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))   # repository root
import gpu_critic as gc   # the snapshot trainer gpu_critic_snap.py defines the same networks and geometries
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'circle_smooth_s0_1e5')
TAU = 3.0; EPS, DV = 0.08, 0.2
G = gc.Circle(); D = G.features(G.sample_pos(2, 'cpu'), torch.zeros(2)).shape[1]
m = gc.MLP(D, 128); m.eval()
x = np.linspace(-0.999, 0.999, 1201); d = np.abs(x); Vs = np.minimum(TAU, np.maximum(d - EPS, 0) / DV)
pt = torch.tensor(x[:, None], dtype=torch.float32); tt = torch.full((len(x),), TAU)
files = sorted(glob.glob(f'{RUN}/snapshots/snap_*.pt')); its, prof, lrs = [], [], []
for f in files:
    ck = torch.load(f, map_location='cpu'); m.load_state_dict(ck['model'])
    with torch.no_grad(): v = m(G.features(pt, tt)).numpy()
    its.append(int(ck['iter'])); prof.append(v); lrs.append(float(ck['lr']))
its = np.array(its); prof = np.array(prof)
hist = json.load(open(f'{RUN}/gpu_circle.json'))
np.savez(f'{RUN}/evolution_profiles.npz', x=x, its=its, prof=prof, Vs=Vs, hist_iter=hist['iter'], hist_val=hist['val'], tau=TAU)
print(len(its), 'snapshots; iterations', its[:12], '...', its[-3:])

# ---- gif
plt.rcParams.update({'font.size': 13})
frames = []; os.makedirs(f'{RUN}/frames', exist_ok=True)
hit, hval = np.array(hist['iter']), np.array(hist['val'])
for k, (it, v) in enumerate(zip(its, prof)):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    a.plot(x, Vs, 'k', lw=3.5, label='analytic $V^*$'); vv = v.copy(); vv[d <= EPS] = np.nan
    a.plot(x, vv, color='#c0392b', lw=2, label='Fourier-feature NN')
    a.axvspan(-EPS, EPS, color='k', alpha=0.15); a.set_xlim(-1, 1); a.set_ylim(-1.5, 4.0)
    a.set_xlabel('displacement $\\Delta$'); a.set_ylabel(f'$V(\\Delta,\\tau={TAU:g})$'); a.legend(loc='upper left', frameon=False)
    a.set_title(f'iteration {it:>7d}', loc='left')
    b.semilogy(hit, hval, color='0.4', lw=1.5); jj = np.searchsorted(hit, it); jj = min(jj, len(hit) - 1)
    b.plot(hit[jj], hval[jj], 'o', color='#c0392b', ms=8); b.set_xlim(0, its.max()); b.set_ylim(1e-5, 10)
    b.set_xlabel('iteration'); b.set_ylabel('validation loss'); b.grid(alpha=0.3, which='both')
    fig.tight_layout(); fn = f'{RUN}/frames/f{k:04d}.png'; fig.savefig(fn, dpi=80); plt.close(fig); frames.append(fn)
ims = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in frames]
# hold the first and last frames longer; early (log-spaced) frames slower than the regular ones
dur = [120] * len(ims); dur[0] = 1500; dur[-1] = 2500
for k, it in enumerate(its):
    if it <= 7000: dur[k] = 250
ims[0].save(f'{RUN}/fold_evolution.gif', save_all=True, append_images=ims[1:], duration=dur, loop=0, optimize=False)
print('gif:', f'{RUN}/fold_evolution.gif', len(ims), 'frames')

# ---- static figure: 2 x 4 panels at selected iterations
SEL = [int(s) for s in os.environ.get('EVO_SEL', '0,100,500,2000,5000,20000,50000,100000').split(',')]
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13.5, 'axes.labelsize': 13})
fig, ax = plt.subplots(2, 4, figsize=(17, 7.2), sharex=True, sharey=True)
for a, it in zip(ax.ravel(), SEL):
    k = int(np.argmin(np.abs(its - it))); v = prof[k].copy(); v[d <= EPS] = np.nan
    a.plot(x, Vs, 'k', lw=3, zorder=1); a.plot(x, v, color='#c0392b', lw=1.8, zorder=2)
    a.axvspan(-EPS, EPS, color='k', alpha=0.15); a.set_title(f'iteration {its[k]}', loc='left')
    a.set_xlim(-1, 1); a.set_ylim(-1.5, 4.0)
for a in ax[1]: a.set_xlabel('displacement $\\Delta$')
for a in ax[:, 0]: a.set_ylabel(f'$V(\\Delta,\\tau={TAU:g})$')
fig.tight_layout(); fig.savefig(f'{RUN}/fig_fold_evolution.pdf'); print('ok static figure', SEL)
