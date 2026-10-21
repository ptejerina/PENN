# PENN — Pursuit-Evasion Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2610.27193-b31b1b.svg)](https://arxiv.org/abs/2610.27193)

Pursuit-Evasion Neural Networks (PENNs) are physics-informed networks that learn the viscosity solution,
and avoid the almost-everywhere spurious solutions, of a differential game described by a
Hamilton–Jacobi–Isaacs equation.  This repository holds the code, results and figures of

> R. Jimenez, P. Tejerina-Pérez, L. Sarieddine, P. Tarancón-Álvarez, M. Traforetti, P. Solé-Vilaró, 
> O. Ignatenko, N. Tkach, P. Protopapas, 
> *Low Residuals, Wrong Value: Learning the Viscosity Solution of Hamilton–Jacobi–Isaacs Equations with PINNs*
> arXiv:2610.27193 (2026), submitted to the Journal of Computational Physics.

A physics-informed neural network (PINN) is trained on the residual of the Hamilton–Jacobi–Isaacs
equation of a fixed-horizon pursuit–evasion game on the circle $S^1$ and the flat tori $\mathbb{T}^2$,
$\mathbb{T}^4$, where the value function is known in closed form.  Four parameterisations of the value
(Fourier-feature, soft-periodic, min-of-charts, geometric) are trained with ten seeds on each geometry,
120 runs in total, and compared with the analytic viscosity solution.  The repository also contains the
mesh-free semiconcavity test ("certificate") that detects folded almost-everywhere solutions from
forward evaluations of the network alone.

## Layout

```
PENN/
├── gpu_critic.py            networks, geometries, loss and trainer (one file)
├── grid_certificate.py      grid-based certificate on T^2 (comparison baseline)
├── launch/                  how the 120 runs are launched
├── runs_fullhorizon/        where the trained networks go (not versioned)
├── analysis/                evaluation, tables and the figure notebook
├── figs/                    the figures of the paper (PDF)
├── fold_evolution/          the fold-formation experiment (own README)
└── paper/                   bibliography
```

### Top level

| File / folder | What it is |
| --- | --- |
| `gpu_critic.py` | The trainer of the campaign: the three geometries ($S^1$, $\mathbb{T}^2$, $\mathbb{T}^4$), the four network configurations, the PINN loss and the Adam loop with checkpointing. One file, command line below. |
| `grid_certificate.py` | The grid-based semiconcavity certificate on $\mathbb{T}^2$ ($241^2$ grid, finite-difference Hessian), used as a comparison baseline for the Monte Carlo certificate of `analysis/eval_all.py`. |
| `launch/train_run.sh` | Trains one run with the exact flags of the paper: `bash launch/train_run.sh <geom> <config> <seed>`. |
| `launch/slurm_campaign.sbatch` | The whole campaign as a 120-task Slurm array (calls `train_run.sh`). |
| `runs_fullhorizon/` | Destination of the trained networks, `{S1,T2,T4}/<geom>_<config>_s<seed>/`; not versioned, see "Checkpoints". Its README describes the layout. |
| `figs/` | The figures of the paper as PDF, written by the notebook. |
| `fold_evolution/` | Snapshot trainer, profiles and gif of the fold-formation run on $S^1$; see its README. |
| `paper/references.bib` | Bibliography of the paper. |
| `requirements.txt` | Python dependencies. |

### `analysis/`

| File | What it does |
| --- | --- |
| `common.py` | Locates the repository root (or `$P2_BUNDLE`), imports the trainer and the certificate module, defines the constants $\varepsilon$, $\Delta v$ and $V^*$. |
| `eval_all.py` | Scores every checkpoint with one set of definitions (value RMS, within-5 % share, capture-band gradient, residuals, seam gap, Monte Carlo certificate, chart selector) and writes `results_<geom>_FH10.json`. |
| `score_new.py` | Scores only the checkpoints missing from the results files and merges them in. |
| `selector10.py` | Chart-selector statistics of every min-of-charts run on distance-limited states, written to `selector10.json`. |
| `check5pct.py` | The within-5 % share under the paper's criterion, $\lvert V - V^*\rvert < 0.05\max(V^*, 1)$, and under a purely relative one, written to `check5pct.json`. |
| `make_tables10.py` | The table bodies `tables10.txt` and `tables10.tex` from the results files. |
| `table_campaign10_rms.tex` | Table 1 of the paper, complete with caption. |
| `table_cert10_sci.tex` | The certificate table of the paper, complete with caption. |
| `results_{circle,torus,torus4}_FH10.json` | The evaluation of all 120 runs; every number in the paper derives from these. |
| `selector10.json`, `check5pct.json` | Outputs of the two scripts above for the paper's runs. |
| `paper_figures.ipynb` | Every figure of the paper, one panel per cell, assembled and written to `../figs/`. |

Configuration names in the code: `smooth` = Fourier-feature, `rawsoft` = soft-periodic, `min` =
min-of-charts, `folded` (`--charts_single`) = geometric.  Geometries: `circle`, `torus`, `torus4`.

## Installation

Python ≥ 3.10 with PyTorch, NumPy, Matplotlib, Pillow and Jupyter:

```
pip install -r requirements.txt
```

Training runs on CPU, CUDA or Apple-silicon MPS (chosen automatically); evaluation and figures are
CPU only.

## Training

One run with the exact flags of the paper (width 128, batch 2048, Adam with a single cosine
learning-rate cycle $3\times10^{-3} \to 5\times10^{-5}$, $\tau \sim U(0, 8)$, boundary weight 5):

```
bash launch/train_run.sh torus min 3        # <geom> <config> <seed>
```

writes `runs_fullhorizon/T2/torus_min_s3/` with `cmd.txt`, `train.log`, the checkpoint
`gpu_torus_charts.pt` and its loss history `gpu_torus_charts.json`.  The budget is $5\times10^5$
iterations on $S^1$ and $10^6$ on the tori; a run takes between one hour (S^1) and about sixteen hours
(min-of-charts on $\mathbb{T}^4$, $2^4$ network evaluations per point) on an NVIDIA L40S.  The whole
campaign is the Slurm array `launch/slurm_campaign.sbatch` (120 tasks; adapt the module lines).
The trainer itself is

```
python gpu_critic.py --geom {circle,torus,torus4} [--raw_soft | --charts | --charts_single] \
       --iters N --width 128 --bc_margin 1.0 --restarts 1 --tau_min 0.0 --seed S
```

(no flag = Fourier-feature).  The training details are in Appendix A of the paper.

## Evaluation and tables

From `analysis/`, with the checkpoints in `../runs_fullhorizon/` (or `P2_BUNDLE=/path/to/root`):

```
python eval_all.py circle; python eval_all.py torus; python eval_all.py torus4
python selector10.py          # min-of-charts selector statistics
python check5pct.py           # within-5 % share: paper criterion vs purely relative
python make_tables10.py       # -> tables10.txt, tables10.tex
```

`eval_all.py` computes, for every checkpoint: best validation loss, value RMS against $V^*$ on
$2\times10^5$ uniform states, the share within 5 % ($|V - V^*| < 0.05\max(V^*, 1)$), the mean gradient
norm in the capture band, the interior mean-squared residual, the seam gap, the Monte Carlo
semiconcavity certificate at $\tau = 4$ (also for $V^*$ and a planted sawtooth), and the chart
selector of the min-of-charts runs.  About 15 minutes on four CPU cores for the 120 runs.  The results
of the paper's runs are versioned (`results_*_FH10.json`), so the tables and the figures that do not
need the networks (losses, census) regenerate without any training.

## Figures

`analysis/paper_figures.ipynb` draws every figure of the paper: one cell per panel with its tweakable
parameters in `P[...]`, and one assemble cell per figure writing the PDF into `../figs/`.  Global
knobs (font sizes, journal style, palette, style of the analytic $V^*$) are in the first two code
cells.  Run it from `analysis/` (`jupyter nbconvert --execute paper_figures.ipynb --to notebook`,
or interactively).  Most panels evaluate the trained networks and need the checkpoints; the
losses, census and fold-evolution sections only need the versioned files.

## Checkpoints

The trained networks (120 checkpoints with weights, optimiser state and loss history, a few hundred
MB in total) and the 217 training snapshots of the fold-evolution run are not in this repository.
They are available from the corresponding author on request and will be deposited with a DOI on
acceptance; the result files in `analysis/` contain their complete evaluation.

## Two pitfalls that silently invalidate this kind of training

Both should be checked before any campaign.

1. On some PyTorch/MPS versions the backward pass of a final `Linear(width, 1)` layer returns a
   gradient with respect to the layer *input* that is about $30\times$ too small through a
   two-hidden-layer $\tanh$ network, while the forward pass is correct.  Any loss involving input
   gradients then trains silently wrong.  `gpu_critic.py` applies the final layer in vector form;
   compare autograd with finite differences on the target backend before trusting a run.
2. The default `Linear` initialisation (per-layer scale about half of the $\mathcal N(0, 1.5/\mathrm{fan\_in})$
   used here) starts the network in the trivial basin $V \equiv 0$, from which the loss does not
   escape.  A short run that checks that the loss leaves that basin exposes it.

## Citation

```bibtex
@article{penn2026,
  title   = {Low Residuals, Wrong Value: Learning the Viscosity Solution of
             Hamilton--Jacobi--Isaacs Equations with PINNs},
  author  = {Jimenez, Raul and Ignatenko, Oleksii and Protopapas, Pavlos and Sarieddine, Leonid
             and Sol{\'e}-Vilar{\'o}, Pau and Taranc{\'o}n-{\'A}lvarez, Pedro
             and Tejerina-P{\'e}rez, Pablo and Tkach, Nazar and Traforetti, Marisol},
  journal = {submitted to Journal of Computational Physics},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
