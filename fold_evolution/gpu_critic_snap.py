"""
GPU long-run honest HJI critic (PyTorch: Apple-Silicon MPS / CUDA / CPU).

Purpose (Appendix "GPU long-run protocol" of the paper): decide whether the
torus and Mobius residual levels asymptote above the kink-free (disk/ball)
levels, at budgets the sandbox cannot reach (>= 1e6 iterations, width sweeps),
and run the min-of-charts structural control.

Fidelity to the NumPy pipeline:
  * identical sampling measures (uniform positions with the same margins and
    capture-core exclusion; tau ~ U(0.05, tau_max); tau_max = 8 torus, 4 else)
  * identical loss structure: HJI residual + terminal BC + capture BC with
    lam_term = lam_cap = 5 (no V*, no u*, nothing analytic -- honesty contract
    unchanged)
  * NEW, and better: derivatives of V w.r.t. inputs are EXACT (autograd),
    replacing the central finite differences of the NumPy code; the O(h^2)
    probe and its ridge ambiguity disappear.
  * --charts: V = min_m V_m over the 4 nearest torus images (2 orientation
    charts on the Mobius strip). Architecture-level kink handling; the loss
    is unchanged. If charts reach ~1e-4 quickly while the smooth critic
    stalls at 10x budget, the kink account is confirmed; if both stall, it
    is refuted.

Usage examples (Apple Silicon: install torch>=2.2, then MPS is automatic):
  python3 gpu_critic.py --geom torus  --iters 1000000 --width 128
  python3 gpu_critic.py --geom torus  --iters 1000000 --width 128 --charts
  python3 gpu_critic.py --geom mobius --iters 1000000 --width 128
  python3 gpu_critic.py --geom disk --iters 500000; ... --geom ball ...
  python3 gpu_critic.py --geom torus --resume gpu_torus.pt   (continue)

Output: checkpoint gpu_<geom>[_charts].pt + gpu_<geom>[_charts].json with the
loss trace, periodic fixed-batch validation (seed 999), and protocol-matched
gradient diagnostics -- drop-in comparable with geom_summary.json.
Expected wall-clock on an M-series GPU: ~1-3 h per 1e6 iterations at batch
2048 (vs ~4 days CPU-NumPy at the paper's batch sizes).
"""
import argparse, json, math, os, time
import torch

torch.manual_seed(0)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------- geometries
class Torus:
    name, d, tau_max, L = "torus", 2, 8.0, 2.0
    def sample_pos(self, n, dev):
        # displacement coordinates, uniform with the NumPy margins
        return (torch.rand(n, 2, device=dev) * (self.L - 0.04)
                - (self.L / 2 - 0.02))
    def dist(self, x):                       # x = Delta
        return x.norm(dim=-1)
    def features(self, x, tau, n_freqs=8):
        # periodic Fourier features per component + polynomial tau features
        k = torch.arange(1, n_freqs + 1, device=x.device) * (2 * math.pi / self.L)
        ang = x.unsqueeze(-1) * k                       # (N, 2, K)
        f = torch.cat([torch.sin(ang).flatten(1), torch.cos(ang).flatten(1)], 1)
        tt = (tau / self.tau_max).unsqueeze(-1)
        return torch.cat([f, tt, tt**2, torch.sin(math.pi*tt), torch.cos(math.pi*tt)], 1)
    def residual_terms(self, V, gx, gt, vE, vP):
        # displacement form: dV/dtau = 1 - (vP - vE) |grad V|
        return gt - 1.0 + (vP - vE) * gx.norm(dim=-1)


class Circle:
    """S^1 (1D torus), displacement form -- the M6 second-ground-truth
    geometry: analytic V* = min(tau, (|delta|_min-image - eps)/(vP - vE))
    with the same antipodal ridge structure as T^2 (at delta = +-L/2), and
    independently checkable by a classical monotone grid solve (1D+time)."""
    name, d, tau_max, L = "circle", 1, 8.0, 2.0
    def sample_pos(self, n, dev):
        return (torch.rand(n, 1, device=dev) * (self.L - 0.04)
                - (self.L / 2 - 0.02))
    def dist(self, x):
        return x.norm(dim=-1)
    def features(self, x, tau, n_freqs=8):
        k = torch.arange(1, n_freqs + 1, device=x.device) * (2 * math.pi / self.L)
        ang = x.unsqueeze(-1) * k
        f = torch.cat([torch.sin(ang).flatten(1), torch.cos(ang).flatten(1)], 1)
        tt = (tau / self.tau_max).unsqueeze(-1)
        return torch.cat([f, tt, tt**2, torch.sin(math.pi*tt), torch.cos(math.pi*tt)], 1)
    residual_terms = None  # set below (same displacement form as Torus)
Circle.residual_terms = Torus.residual_terms


class Torus4:
    """T^4 = [0,2)^4, displacement form.  Same game, closed form and bound as
    T^2: V* = min(tau, (d - eps)/dv) with d the minimum-image distance.
    2^4 = 16 nearest images for the charts critic.  Periodic features:
    8 harmonics x 4 components x (sin, cos) = 64, plus 4 time features."""
    name, d, tau_max, L = "torus4", 4, 8.0, 2.0
    def sample_pos(self, n, dev):
        return (torch.rand(n, 4, device=dev) * (self.L - 0.04)
                - (self.L / 2 - 0.02))
    def dist(self, x):
        return x.norm(dim=-1)
    def features(self, x, tau, n_freqs=8):
        k = torch.arange(1, n_freqs + 1, device=x.device) * (2 * math.pi / self.L)
        ang = x.unsqueeze(-1) * k                       # (N, 4, K)
        f = torch.cat([torch.sin(ang).flatten(1), torch.cos(ang).flatten(1)], 1)
        tt = (tau / self.tau_max).unsqueeze(-1)
        return torch.cat([f, tt, tt**2, torch.sin(math.pi*tt), torch.cos(math.pi*tt)], 1)
    residual_terms = Torus.residual_terms


class BoundedBase:
    """Disk / Ball / Mobius: V(x_E, x_P, tau); residual
       dV/dtau = 1 + vE |grad_E V| - vP |grad_P V| (uncaptured region)."""
    def residual_terms(self, V, gE, gP, gt, vE, vP):
        return gt - 1.0 - vE * gE.norm(dim=-1) + vP * gP.norm(dim=-1)


class Disk(BoundedBase):
    name, d, tau_max, R = "disk", 2, 4.0, 1.0
    def sample_pos(self, n, dev):
        th = torch.rand(n, device=dev) * 2 * math.pi
        r = torch.sqrt(torch.rand(n, device=dev)) * (self.R - 1e-3)
        return torch.stack([r * th.cos(), r * th.sin()], -1)
    def dist(self, xE, xP): return (xE - xP).norm(dim=-1)
    def features(self, xE, xP, tau):
        diff = xE - xP
        tt = (tau / self.tau_max).unsqueeze(-1)
        return torch.cat([xE, xP, xE**2, xP**2, diff,
                          diff.norm(dim=-1, keepdim=True),
                          xE.norm(dim=-1, keepdim=True),
                          xP.norm(dim=-1, keepdim=True),
                          tt, tt**2, torch.sin(math.pi*tt), torch.cos(math.pi*tt)], 1)


class Ball(BoundedBase):
    name, d, tau_max, R = "ball", 3, 4.0, 1.0
    def sample_pos(self, n, dev):
        v = torch.randn(n, 3, device=dev)
        v = v / v.norm(dim=-1, keepdim=True)
        r = torch.rand(n, device=dev).pow(1/3) * (self.R - 1e-3)
        return v * r.unsqueeze(-1)
    dist = Disk.dist
    features = Disk.features


class Mobius(BoundedBase):
    name, d, tau_max, L, w = "mobius", 2, 4.0, 2.0, 0.5
    def sample_pos(self, n, dev):
        u = torch.rand(n, device=dev) * self.L - self.L / 2
        v = torch.rand(n, device=dev) * 2 * self.w - self.w
        return torch.stack([u, v], -1)
    def dist(self, xE, xP):
        du = (xE[:, 0] - xP[:, 0] + self.L/2) % self.L - self.L/2
        d0 = torch.stack([du, xE[:, 1] - xP[:, 1]], -1).norm(dim=-1)
        duf = (xE[:, 0] - xP[:, 0] + self.L) % (2*self.L) - self.L  # flip image
        d1 = torch.stack([duf, xE[:, 1] + xP[:, 1]], -1).norm(dim=-1)
        return torch.minimum(d0, d1)
    def features(self, xE, xP, tau, n_freqs=4):
        k = torch.arange(1, n_freqs + 1, device=xE.device) * (2*math.pi/self.L)
        def per(u):
            a = u.unsqueeze(-1) * k
            return torch.cat([a.sin(), a.cos()], -1)
        tt = (tau / self.tau_max).unsqueeze(-1)
        return torch.cat([per(xE[:, 0]), per(xP[:, 0]),
                          per(xE[:, 0] - xP[:, 0]),
                          xE[:, 1:2], xP[:, 1:2], xE[:, 1:2]**2, xP[:, 1:2]**2,
                          (xE[:, 1:2] - xP[:, 1:2]),
                          tt, tt**2, torch.sin(math.pi*tt), torch.cos(math.pi*tt)], 1)


GEOMS = {"torus": Torus, "circle": Circle, "torus4": Torus4,
         "disk": Disk, "ball": Ball, "mobius": Mobius}
DISPLACEMENT_GEOMS = ("torus", "circle", "torus4")   # V(Delta, tau) form


# ------------------------------------------------------------------- networks
class MLP(torch.nn.Module):
    def __init__(self, D, width):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 1))
        # match honest_critic_geom.init_params: W ~ N(0, 1.5/fan_in), b = 0.
        # torch's default (Kaiming-uniform, ~2x smaller std) leaves V ~ 0 and
        # the critic trapped in the trivial residual = -1 basin (smoke-tested).
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, torch.nn.Linear):
                    m.weight.normal_(0.0, math.sqrt(1.5 / m.in_features))
                    m.bias.zero_()
    def forward(self, phi):
        # final layer applied in vector form: torch 2.12 MPS computes a wrong
        # input-gradient in the backward of Linear(width, 1) (max elementwise
        # error ~9e-2 vs analytic; verified against CPU and FD on this machine),
        # which silently kills dV/dx.  h @ w is exact on MPS.
        h = self.net[:-1](phi)
        return h @ self.net[-1].weight.squeeze(0) + self.net[-1].bias[0]


class ChartsCritic(torch.nn.Module):
    """Torus: V = min over the 4 nearest images m of a smooth V(psi(d_m), tau),
       d_m = delta - L*(sign-selected shift).  psi is NON-periodic (raw
       components, squares, norm): periodic features are blind to a period-L
       shift, which made every chart of the first implementation compute the
       IDENTICAL function -- the min collapsed and the 'active chart' was
       float noise (caught post-run when the active-chart/winding test was
       attempted; that arm is superseded).  With non-periodic per-chart
       features the min is real and exact, so the antipodal ridge needs no
       smooth approximation.  Mobius (not yet trained): two orientation
       charts."""
    def __init__(self, G, D, width, single=False, reduce="min"):
        super().__init__()
        self.G = G
        # reduce: "min" = min-of-charts (semiconcave, admissible); "max" =
        # max-of-charts falsification arm (identical kink capacity, inadmissible
        # convex sign -- the sign-experiment impostor).
        self.reduce = reduce
        # single=True: ablation arm -- identical non-periodic features on the
        # wrapped displacement, WITHOUT the min over images.  Disentangles
        # featurisation from min-structure: under the kink account this arm
        # must fail at the antipodal ridge (a smooth function of
        # non-periodic features cannot be periodic across it).
        self.single = single
        if G.name in DISPLACEMENT_GEOMS:
            # d (dim) + d^2 (dim) + |d| (1) + tau features (4)
            self.n_ext = 2 * G.d + 5
            self.net = MLP(self.n_ext, width)  # shared weights across images
        else:
            self.net0, self.net1 = MLP(D, width), MLP(D, width)
    def ext_features(self, d, tau):
        tt = (tau / self.G.tau_max).unsqueeze(-1)
        return torch.cat([d, d**2, d.norm(dim=-1, keepdim=True),
                          tt, tt**2, torch.sin(math.pi*tt),
                          torch.cos(math.pi*tt)], 1)
    def _image_shifts(self, delta):
        """The 2^dim nearest-image shifts, sign-selected per axis."""
        L = self.G.L
        s = torch.sign(delta)
        zero = torch.zeros_like(delta)
        if self.G.d == 1:
            return [zero, -L*s]
        if self.G.d == 2:
            return [zero,
                    torch.stack([-L*s[:, 0], zero[:, 1]], -1),
                    torch.stack([zero[:, 0], -L*s[:, 1]], -1),
                    -L*s]
        # general d: all 2^d sign-selected image shifts (axis 0 varies slowest)
        import itertools
        out = []
        for mask in itertools.product((0, 1), repeat=self.G.d):
            m = torch.tensor(mask, dtype=delta.dtype, device=delta.device)
            out.append(-L * s * m)
        return out
    def forward_torus(self, delta, tau):
        if self.single:
            return self.net(self.ext_features(delta, tau))
        outs = [self.net(self.ext_features(delta + sh, tau))
                for sh in self._image_shifts(delta)]
        stacked = torch.stack(outs, 0)
        return (stacked.max(0).values if self.reduce == "max"
                else stacked.min(0).values)
    def active_chart(self, delta, tau):
        """Post-hoc homotopy selector m*(delta, tau)."""
        outs = [self.net(self.ext_features(delta + sh, tau))
                for sh in self._image_shifts(delta)]
        stacked = torch.stack(outs, 0)
        return (stacked.argmax(0) if self.reduce == "max"
                else stacked.argmin(0))


class RawSoftCritic(torch.nn.Module):
    """'Naive' periodic arm: a plain MLP on the RAW displacement (no Fourier
    features, no charts, no distance feature), restricted to the fundamental
    domain [-L/2, L/2)^d, with periodicity imposed SOFTLY through a loss term
    that matches the network's VALUES on opposite faces of the domain,
    (N(x | x_i = +L/2) - N(x | x_i = -L/2))^2.  Values only: gradients are free
    to jump across the seam, which is exactly where the ridge kink of V* lies in
    displacement coordinates.  forward_torus wraps its input into the fundamental
    domain, so the critic is a genuine (possibly discontinuous) function on the
    torus; the penalty is what closes the seam."""
    def __init__(self, G, width):
        super().__init__()
        self.G = G
        self.net = MLP(G.d + 4, width)
    def feats(self, x, tau):
        tt = (tau / self.G.tau_max).unsqueeze(-1)
        return torch.cat([x, tt, tt**2, torch.sin(math.pi*tt), torch.cos(math.pi*tt)], 1)
    def raw(self, delta, tau):                    # no wrapping: used on the seam faces
        return self.net(self.feats(delta, tau))
    def forward_torus(self, delta, tau):
        L = self.G.L
        w = torch.remainder(delta + L / 2, L) - L / 2
        return self.raw(w, tau)


def seam_pairs(G, n, dev):
    """n pairs of points on opposite faces x_i = +L/2 / x_i = -L/2 (random axis i,
    identical other coordinates, random tau): the periodic identifications."""
    x = G.sample_pos(n, dev)
    ax = torch.randint(0, G.d, (n,), device=dev)
    idx = torch.arange(n, device=dev)
    xp, xm = x.clone(), x.clone()
    xp[idx, ax] = G.L / 2
    xm[idx, ax] = -G.L / 2
    tau = torch.rand(n, device=dev) * G.tau_max
    return xp, xm, tau


def eval_V_and_grads(model, G, charts, *args):
    """Exact input-gradients via autograd; args differ per geometry."""
    if G.name in DISPLACEMENT_GEOMS:
        delta, tau = args
        delta = delta.requires_grad_(True); tau = tau.requires_grad_(True)
        V = (model.forward_torus(delta, tau) if charts
             else model(G.features(delta, tau)))
        gd, gt = torch.autograd.grad(V.sum(), (delta, tau), create_graph=True)
        return V, gd, gt
    xE, xP, tau = args
    xE = xE.requires_grad_(True); xP = xP.requires_grad_(True)
    tau = tau.requires_grad_(True)
    V = model(G.features(xE, xP, tau))
    gE, gP, gt = torch.autograd.grad(V.sum(), (xE, xP, tau), create_graph=True)
    return V, gE, gP, gt


def batch(G, n_res, n_bc, dev, eps, gen=None, margin=1.5, tau_min=0.05):
    """gen != None  ->  fully deterministic batch (positions AND taus):
    the entire generation runs under a saved-and-restored RNG state.

    margin: residual points are sampled at dist > margin*eps.  The released
    NumPy protocol uses 1.5 (an FD-stencil clearance).  With exact autograd
    the annulus (eps, 1.5 eps) need not be excluded -- and must not be at
    long budgets: the v1 1e6-iter torus runs found a degenerate minimiser
    (V ~ tau with a cliff at the capture core) hiding a mean-squared
    residual of ~4e3 inside the unsampled annulus while scoring ~3e-8 on
    the sampled region (gpu_v1_diagnostics.json).  margin=1.0 closes the
    corridor and removes that minimiser."""
    if gen is not None:
        st_cpu = torch.random.get_rng_state()
        st_dev = (torch.cuda.get_rng_state() if dev.type == "cuda"
                  else torch.mps.get_rng_state() if dev.type == "mps" else None)
        torch.manual_seed(int(gen.initial_seed()))
        try:
            return batch(G, n_res, n_bc, dev, eps, gen=None, margin=margin,
                         tau_min=tau_min)
        finally:
            torch.random.set_rng_state(st_cpu)
            if dev.type == "cuda": torch.cuda.set_rng_state(st_dev)
            elif dev.type == "mps": torch.mps.set_rng_state(st_dev)
    def upos(n):
        return G.sample_pos(n, dev)
    if G.name in DISPLACEMENT_GEOMS:
        d = upos(2 * n_res)
        d = d[G.dist(d) > margin * eps][:n_res]
        tau = torch.rand(d.shape[0], device=dev) * (G.tau_max - tau_min) + tau_min
        dbc = upos(n_bc); dcap = upos(n_bc)
        dcap = dcap / dcap.norm(dim=-1, keepdim=True).clamp_min(1e-9) \
               * (torch.rand(n_bc, 1, device=dev) * eps)
        tcap = torch.rand(n_bc, device=dev) * G.tau_max
        return (d, tau), (dbc,), (dcap, tcap)
    xE, xP = upos(2 * n_res), upos(2 * n_res)
    m = G.dist(xE, xP) > margin * eps
    xE, xP = xE[m][:n_res], xP[m][:n_res]
    tau = torch.rand(xE.shape[0], device=dev) * (G.tau_max - tau_min) + tau_min
    xEb, xPb = upos(n_bc), upos(n_bc)
    xEc = upos(n_bc)
    xPc = xEc + torch.randn_like(xEc) * (eps / 3)
    tc = torch.rand(n_bc, device=dev) * G.tau_max
    return (xE, xP, tau), (xEb, xPb), (xEc, xPc, tc)


def loss_fn(model, G, charts, res, bc, cap, vE, vP, lam=5.0):
    if G.name in DISPLACEMENT_GEOMS:
        V, gd, gt = eval_V_and_grads(model, G, charts, *res)
        r = G.residual_terms(V, gd, gt, vE, vP)
        Vb = (model.forward_torus(bc[0], torch.zeros(bc[0].shape[0], device=bc[0].device))
              if charts else model(G.features(bc[0], torch.zeros(bc[0].shape[0], device=bc[0].device))))
        Vc = (model.forward_torus(*cap) if charts else model(G.features(*cap)))
        gnorm = gd.norm(dim=-1).mean()
    else:
        V, gE, gP, gt = eval_V_and_grads(model, G, charts, *res)
        r = G.residual_terms(V, gE, gP, gt, vE, vP)
        zb = torch.zeros(bc[0].shape[0], device=bc[0].device)
        Vb = model(G.features(bc[0], bc[1], zb))
        Vc = model(G.features(*cap))
        gnorm = gE.norm(dim=-1).mean()
    l_hji, l_t, l_c = (r**2).mean(), (Vb**2).mean(), (Vc**2).mean()
    total = l_hji + lam*l_t + lam*l_c
    if isinstance(model, RawSoftCritic):          # soft periodicity: match values on opposite faces
        xp, xm, tp = seam_pairs(G, bc[0].shape[0], bc[0].device)
        l_per = ((model.raw(xp, tp) - model.raw(xm, tp))**2).mean()
        total = total + lam * l_per
    return total, l_hji, l_t, l_c, gnorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True, choices=list(GEOMS))
    ap.add_argument("--iters", type=int, default=1_000_000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--charts", action="store_true")
    ap.add_argument("--charts_single", action="store_true",
                    help="ablation: charts features, no min (see ChartsCritic)")
    ap.add_argument("--charts_max", action="store_true",
                    help="max-of-charts falsification arm (inadmissible convex "
                         "sign; sign experiment). Trains gpu_<geom>_chartsmax.pt")
    ap.add_argument("--raw_soft", action="store_true",
                    help="'naive' arm: plain MLP on the raw displacement with a "
                         "soft (loss-level, values-only) periodicity penalty; "
                         "trains gpu_<geom>_rawsoft.pt")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--restarts", type=int, default=4,
                    help="cosine warm-restart cycles over the run")
    ap.add_argument("--bc_margin", type=float, default=1.5,
                    help="residual sampled at dist > bc_margin*eps; 1.5 = "
                         "released NumPy protocol, 1.0 closes the annulus "
                         "(see batch() docstring)")
    ap.add_argument("--tau_min", type=float, default=0.05,
                    help="lower bound of the residual tau measure, tau ~ "
                         "U(tau_min, tau_max); 0.05 = campaign protocol, 0.0 "
                         "closes the tau-corridor next to the terminal anchor "
                         "(corridor-closed max-of-charts run)")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--snap_every", type=int, default=0,
                    help="save a model snapshot every N iterations (plus a log-spaced set of early ones and the initialisation) into --snap_dir")
    ap.add_argument("--snap_dir", type=str, default="snapshots")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed (init + sampling stream); the fixed "
                         "validation batch (seed 999) is unaffected")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = device(); print("device:", dev, " seed:", args.seed)
    G = GEOMS[args.geom]()
    vE, vP, eps = 1.0, 1.2, 0.08 if args.geom in DISPLACEMENT_GEOMS else 0.1

    D = G.features(*(
        (G.sample_pos(2, dev), torch.zeros(2, device=dev)) if args.geom in DISPLACEMENT_GEOMS
        else (G.sample_pos(2, dev), G.sample_pos(2, dev), torch.zeros(2, device=dev))
    )).shape[1]
    if args.raw_soft:
        assert args.geom in DISPLACEMENT_GEOMS, "raw_soft needs displacement form"
        model = RawSoftCritic(G, args.width).to(dev)
        args.charts = True          # loss path through model.forward_torus
    elif args.charts or args.charts_single or args.charts_max:
        assert args.geom in DISPLACEMENT_GEOMS, \
            "charts implemented for displacement-form geometries"
        model = ChartsCritic(G, D, args.width, single=args.charts_single,
                             reduce="max" if args.charts_max else "min").to(dev)
        args.charts = True          # downstream loss path is the same
    else:
        model = MLP(D, args.width).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    it0, hist = 0, {"iter": [], "loss": [], "hji": [], "val": []}
    tag = f"gpu_{args.geom}" + ("_rawsoft" if args.raw_soft
                                else "_chartsmax" if args.charts_max
                                else "_charts1" if args.charts_single
                                else "_charts" if args.charts else "")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        it0, hist = ck["iter"], ck["hist"]; print("resumed at", it0)

    cyc = args.iters // args.restarts
    t0 = time.time()
    SNAP_EARLY = {1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000}
    def snap(it):
        os.makedirs(args.snap_dir, exist_ok=True)
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "iter": it,
                    "lr": opt.param_groups[0]["lr"]}, os.path.join(args.snap_dir, f"snap_{it:07d}.pt"))
    if args.snap_every and it0 == 0: snap(0)          # initialisation, before any update
    for it in range(it0, args.iters):
        p = (it % cyc) / cyc
        for g in opt.param_groups:
            g["lr"] = 5e-5 + 0.5*(args.lr - 5e-5)*(1 + math.cos(math.pi*p))
        res, bc, cap = batch(G, args.batch, args.batch // 4, dev, eps,
                             margin=args.bc_margin, tau_min=args.tau_min)
        loss, l_hji, l_t, l_c, gn = loss_fn(model, G, args.charts, res, bc, cap, vE, vP)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if args.snap_every and ((it + 1) % args.snap_every == 0 or (it + 1) in SNAP_EARLY or it + 1 == args.iters): snap(it + 1)
        if it % 2000 == 0 or it == args.iters - 1:
            with torch.no_grad():
                pass
            gv = torch.Generator(device="cpu").manual_seed(999)
            vres, vbc, vcap = batch(G, 8192, 2048, dev, eps, gen=gv,
                                    margin=args.bc_margin, tau_min=args.tau_min)
            vloss, vhji, *_ = loss_fn(model, G, args.charts, vres, vbc, vcap, vE, vP)
            hist["iter"].append(it); hist["loss"].append(float(loss))
            hist["hji"].append(float(l_hji)); hist["val"].append(float(vloss))
            print(f"it {it:>8d}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"loss {loss:.3e}  hji {l_hji:.3e}  VAL {vloss:.3e}  "
                  f"<|gE|> {gn:.3f}  [{time.time()-t0:.0f}s]")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "iter": it + 1, "hist": hist}, tag + ".pt")
            json.dump(hist, open(tag + ".json", "w"))
    print("done:", tag)


if __name__ == "__main__":
    main()
