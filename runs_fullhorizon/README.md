# runs_fullhorizon/

Trained networks live here, one folder per run, and are not versioned (see the main README,
"Checkpoints"):

```
runs_fullhorizon/
  S1/<geom>_<config>_s<seed>/     circle,  5e5 iterations
  T2/<geom>_<config>_s<seed>/     torus,   1e6 iterations
  T4/<geom>_<config>_s<seed>/     torus4,  1e6 iterations
```

with `geom` in `circle, torus, torus4`, `config` in `smooth, rawsoft, min, folded` and `seed`
in `0 ... 9`.  Each folder holds `cmd.txt` (the exact command), `train.log`, the checkpoint
`gpu_<geom>[_rawsoft|_charts|_charts1].pt` (model and optimiser state, iteration, loss history)
and `gpu_<geom>[...].json` (the same history in plain text).  `launch/train_run.sh` writes
this layout; `analysis/eval_all.py` and `analysis/paper_figures.ipynb` read it.
