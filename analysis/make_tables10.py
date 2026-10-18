"""Tables of the campaign from results_<geom>_FH10.json: plain text (tables10.txt) and LaTeX bodies (tables10.tex).
Main table: best validation loss | RMS against V* (scientific notation) | ||grad V|| in the capture band | share within 5 %.
Supplementary table: interior MSR | certificate flagged % | certificate max ratio, with V* and the planted sawtooth as references.
Entries: median [min, max] over seeds.  Also prints the per-seed selector and within-5% summaries used in the text.
The complete LaTeX tables of the paper (with captions) are table_campaign10_rms.tex and table_cert10_sci.tex."""
import json, math, numpy as np
import os
def _load(g): return json.load(open(f'results_{g}_FH10.json' if os.path.exists(f'results_{g}_FH10.json') else f'results_{g}.json'))
R = _load('circle') + _load('torus') + _load('torus4')
ARMS = ['smooth', 'rawsoft', 'min', 'folded']
LAB = {'smooth': 'Fourier-feature', 'rawsoft': 'soft-periodic', 'min': 'min-of-charts', 'folded': 'geometric'}
GEOMS = [('circle', 'circle $S^{1}$, $5\\times10^{5}$ iterations', 'circle S^1'), ('torus', 'torus $\\mathbb{T}^{2}$, $10^{6}$ iterations', 'torus T^2'),
         ('torus4', 'torus $\\mathbb{T}^{4}$, $10^{6}$ iterations', 'torus T^4')]
def rows(geom, arm): return sorted([o for o in R if o['geom'] == geom and o['arm'] == arm], key=lambda o: o['seed'])
def get(geom, arm, key):
    vals = []
    for o in rows(geom, arm):
        v = o
        for k in key.split('|'): v = v[k]
        vals.append(v)
    return np.array(vals, float)
def _sci1(x, digits=1):
    e = int(math.floor(math.log10(x))); return f'{x / 10 ** e:.{digits}f}\\times10^{{{e}}}'
def sci_tex(vals, digits=1):
    """median [lo, hi] with a common exponent; each number gets its own exponent when the range spans
    more than two decades, and no exponent is printed when the median is O(1)."""
    med, lo, hi = np.median(vals), vals.min(), vals.max(); e = int(math.floor(math.log10(med)))
    if hi / 10 ** e >= 100 or lo / 10 ** e < 0.1:
        return f'${_sci1(med)}\\ [{_sci1(lo)},\\,{_sci1(hi)}]$'
    if e == 0:
        return f'${med:.2f}\\ [{lo:.2f},{hi:.2f}]$'
    f = lambda x: f'{x / 10 ** e:.{digits}f}'
    return f'${f(med)}\\ [{f(lo)},{f(hi)}]\\times10^{{{e}}}$'
def fix_tex(vals, digits):
    med, lo, hi = np.median(vals), vals.min(), vals.max(); f = lambda x: f'{x:.{digits}f}'
    return f'${f(med)}\\ [{f(lo)},{f(hi)}]$'
def pct_tex(vals, digits=0):
    med, lo, hi = 100 * np.median(vals), 100 * vals.min(), 100 * vals.max(); f = lambda x: f'{x:.{digits}f}'
    return f'${f(med)}\\ [{f(lo)},{f(hi)}]\\%$' if f(lo) != f(hi) else f'${f(hi)}\\%$'
def sci_txt(vals):
    med, lo, hi = np.median(vals), vals.min(), vals.max(); return f'{med:.1e} [{lo:.1e},{hi:.1e}]'
def fix_txt(vals, digits):
    med, lo, hi = np.median(vals), vals.min(), vals.max(); return f'{med:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}]'
def pct_txt(vals):
    med, lo, hi = 100 * np.median(vals), 100 * vals.min(), 100 * vals.max(); return f'{med:.1f} [{lo:.1f},{hi:.1f}]'

txt, tex = [], []
txt.append('Full-horizon campaign, all seeds on disk.  Entries: median [min, max] over seeds.')
txt.append('best val loss = min over training of the validation loss (all terms, fixed validation set);')
txt.append('RMS = root-mean-square error against the analytic V* on 2e5 uniform states (tau ~ U(0.05, 8), d > eps);')
txt.append('|grad V| at capture = mean gradient norm in d in (eps, 1.5 eps], tau ~ U(0.5, 7.5) (analytic 5);')
txt.append('within 5% = share of 2e5 uniform states (tau ~ U(0.05, 8), d > eps) with |V - V*| < max(0.05 V*, 0.05).')
hdr = f"{'geometry':10s} {'configuration':16s} {'n':>2s}  {'best val loss':>24s}  {'RMS vs V*':>24s}  {'|grad V| capture':>18s}  {'within 5% [%]':>20s}"
txt.append(''); txt.append(hdr); txt.append('-' * len(hdr))
tex.append('%%%%% TABLE campaign (full horizon, all seeds): best val loss | RMS vs V* | grad at capture | within 5%')
for geom, title, short in GEOMS:
    tex.append(f'\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\')
    for arm in ARMS:
        bv = get(geom, arm, 'best_val'); mse = get(geom, arm, 'value_rms'); gb = get(geom, arm, 'grad_band'); w = get(geom, arm, 'frac_within_5pct')
        n = len(bv)
        vd = 2 if mse.min() > 0.1 else (3 if mse.min() > 0.01 else 4)
        txt.append(f'{short:10s} {LAB[arm]:16s} {n:2d}  {sci_txt(bv):>24s}  {fix_txt(mse, vd):>24s}  {fix_txt(gb, 2):>18s}  {pct_txt(w):>20s}')
        tex.append(f'{LAB[arm]} ({n}) & {sci_tex(bv)} & {sci_tex(mse)} & {fix_tex(gb, 2)} & {pct_tex(w)} \\\\')
    txt.append(''); tex.append('\\midrule')
# supplementary: value RMS and certificate
txt.append(''); txt.append('Supplementary (same seeds): interior MSR (d > 3 eps, 0.5 < tau < 7.5), value RMS vs V*, MC certificate (flagged %, max ratio)')
hdr2 = f"{'geometry':10s} {'configuration':16s}  {'interior MSR':>24s}  {'cert flagged %':>20s}  {'cert max ratio':>22s}"
txt.append(hdr2); txt.append('-' * len(hdr2))
tex.append('%%%%% TABLE certificate (full horizon, all seeds)')
for geom, title, short in GEOMS:
    vs = get(geom, 'smooth', 'cert_mc_vstar|pct')[0]; vr = get(geom, 'smooth', 'cert_mc_vstar|maxratio')[0]
    ss = get(geom, 'smooth', 'cert_mc_saw|pct')[0]; sr = get(geom, 'smooth', 'cert_mc_saw|maxratio')[0]
    tex.append(f'\\multicolumn{{4}}{{l}}{{\\emph{{{title.split(",")[0]}}}}} \\\\')
    tex.append(f'$V^{{*}}$ (analytic) & --- & ${vs:.1f}$ & ${vr:.1f}$ \\\\'); tex.append(f'planted sawtooth & --- & ${ss:.1f}$ & ${sr:.0f}$ \\\\')
    txt.append(f'{short:10s} {"V* analytic":16s}  {"":>24s}  {vs:>20.1f}  {vr:>22.1f}')
    txt.append(f'{short:10s} {"planted sawtooth":16s}  {"":>24s}  {ss:>20.1f}  {sr:>22.0f}')
    for arm in ARMS:
        msr = get(geom, arm, 'msr_interior'); v = get(geom, arm, 'value_rms'); p = get(geom, arm, 'cert_mc|pct'); r = get(geom, arm, 'cert_mc|maxratio')
        vd = 2 if v.min() > 0.1 else (3 if v.min() > 0.01 else 4); rd = 0 if r.min() > 20 else 1
        txt.append(f'{short:10s} {LAB[arm]:16s}  {sci_txt(msr):>24s}  {fix_txt(p, 1):>20s}  {fix_txt(r, rd):>22s}')
        tex.append(f'{LAB[arm]} ({len(v)}) & {sci_tex(msr)} & {fix_tex(p, 1)} & {fix_tex(r, rd)} \\\\')
    txt.append(''); tex.append('\\midrule')
txt.append('Per-seed min-of-charts selector (identity-image share on uniform states; relabelled runs marked *):')
for geom, _, short in GEOMS:
    line = f'{short:10s} '
    for o in rows(geom, 'min'):
        flag = '*' if (o['frac_within_5pct'] < 0.999 or o['cert_mc']['maxratio'] > 10) else ''
        line += f"s{o['seed']}:{o['sel_identity_frac']:.2f}{flag} "
    txt.append(line)
txt.append('')
txt.append('Per-seed Fourier-feature within-5% share:')
for geom, _, short in GEOMS:
    txt.append(f'{short:10s} ' + ' '.join(f"s{o['seed']}:{100*o['frac_within_5pct']:.0f}" for o in rows(geom, 'smooth')))
open('tables10.txt', 'w').write('\n'.join(txt)); open('tables10.tex', 'w').write('\n'.join(tex))
print('\n'.join(txt)); print(); print('\n'.join(tex))
