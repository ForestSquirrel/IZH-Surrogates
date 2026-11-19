"""
gain.py -- synaptic gain of the Schulte surrogate network, predicted without simulating the
network (the self-consistent single-neuron rate of ivanov2021/gain.py, per population).

Open loop: every neuron keeps its in-edges (weights, delays, STP, receptors) and its input
streams; each recurrent in-edge from population p is driven by an independent Poisson train at
rate nu_p. The gain multiplies the surrogate's EXCITATORY synaptic conductances (recurrent and
input), g_E on E cells and g_I on I cells (run.py '<view>@xe<g_E>:<g_I>'). Scaling all synaptic
input instead (inhibition too) leaves the I cells' balance unchanged: they saturate below the
target's rates and no gain matches them.
  1. target: the stable fixed point nu* of  r_p(nu) = nu_p  for the six populations (damped
     iteration in log rate)
  2. surrogate at the fixed rates nu*: g_E solves  sum_{E cells} r_i(g_E) = sum_{E cells} nu*_p(i),
     g_I likewise on the I cells (in open loop the two types do not interact)

    $PY gain.py [--view best] [--no-noise] [--runs 10] [--trials 20]
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import spec as S
import tasks
import circuit as CI
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec

POPS = list(CI.IDX)
NCH = 2 * S.N_CH
GRID = np.geomspace(0.7, 2.0, 24)


def open_loop(c):
    src = np.where(c.src < c.N, c.N + NCH + c.src, c.src).astype(np.int32)
    return LS.Circuit(N=c.N, nin=NCH + c.N, types=c.types, src=src, post=c.post, w=c.w, dly=c.dly, rec=c.rec,
                      U=c.U, D=c.D, F=c.F, stp=c.stp, meta=dict(c.meta, open_loop=True))


def setup(runs, trials):
    circs, streams, V0 = [], [], []
    for r in range(runs):
        rng = np.random.default_rng(3000 + r)
        circs.append(open_loop(CI.build(rng)))
        V0.append(rng.uniform(*S.V0, CI.N))
        inp, _, T = tasks.make(np.random.default_rng(4000 + r), trials)
        streams.append(inp)
    return circs, streams, np.stack(V0), T


def events(streams, nu, T, seed=5):
    """stream spikes + independent Poisson trains (rate nu[pop of j]) on channel NCH + j"""
    rng = np.random.default_rng(seed)
    rate = np.zeros(CI.N)
    for p, ids in CI.IDX.items():
        rate[ids] = nu[p]
    out = []
    for t, ch in streams:
        m = rng.poisson(rate * T / 1000.0)
        tp = rng.uniform(0.0, T, m.sum())
        out.append((np.concatenate([t, tp]), np.concatenate([ch, NCH + np.arange(CI.N).repeat(m)])))
    return LS.events(out, S.DT, NCH + CI.N)


def pop_rates(counts, T):
    return {p: float(counts[..., ids].mean() / (T / 1000.0)) for p, ids in CI.IDX.items()}


def sim(bindings, circs, Y0, ev, T, noise, seed=0):
    net = LS.stack(circs, bindings, CI.receptors(noise))
    K = len(circs)
    o = engine_gpu.simulate(net, np.arange(K), np.full(K, int(np.ceil(T / S.DT)), np.int64), Y0, ev, S.DT, Rec(), seed=seed)
    return o["counts"].astype(float)


def target_fixed_point(circs, streams, V0, T, noise, iters=20, tol=0.02, log=print):
    nu = {p: 10.0 for p in POPS}
    hist = []
    for k in range(iters):
        c = sim([CI.bind_target()] * len(circs), circs, CI.y0_target(V0), events(streams, nu, T), T, noise, seed=k)
        r = pop_rates(c, T)
        hist.append(dict(nu=dict(nu), r=r))
        err = max(abs(np.log(max(r[p], 1e-3)) - np.log(nu[p])) for p in POPS)
        log(f"  target iter {k}: " + " ".join(f"{p} {nu[p]:.1f}->{r[p]:.1f}" for p in POPS) + f"  max|dlog| {err:.3f}")
        nu = {p: float(np.exp(0.5 * np.log(nu[p]) + 0.5 * np.log(max(r[p], 1e-3)))) for p in POPS}
        if err < tol:
            break
    return nu, hist


def scale_exc(c, gE, gI):
    """circuit with its excitatory synapses (receptor 0: recurrent E and input) scaled by gE onto
    E cells and gI onto I cells"""
    f = np.where(CI.TYPES[c.post] > 0, gE, gI)
    w = np.where(c.rec == 0, c.w * f, c.w)
    return LS.Circuit(N=c.N, nin=c.nin, types=c.types, src=c.src, post=c.post, w=w, dly=c.dly, rec=c.rec,
                      U=c.U, D=c.D, F=c.F, stp=c.stp, meta=dict(c.meta, exc_gain=(gE, gI)))


def surrogate_gain(view, nu, circs, streams, V0, T, noise):
    g = CI.gauges(view)
    b = CI.bind_surrogate(g)
    R = len(circs)
    bnd, cc = [], []
    for G in GRID:
        bnd += [b] * R; cc += [scale_exc(c, G, G) for c in circs]
    Y0 = np.concatenate([CI.y0_surrogate(V0, g)] * len(GRID))
    ev = events(streams, nu, T)
    ptr, st, ch = ev
    evK = (np.concatenate([[0], np.cumsum(np.tile(np.diff(ptr), len(GRID)))]).astype(np.int64), np.tile(st, len(GRID)), np.tile(ch, len(GRID)))
    c = sim(bnd, cc, Y0, evK, T, noise).reshape(len(GRID), R, CI.N)
    hz = c.mean(1) / (T / 1000.0)                                    # (G, N)
    out = {}
    for t, m in (("E", CI.TYPES > 0), ("I", CI.TYPES < 0)):
        want = sum(nu[p] * len(ids) for p, ids in CI.IDX.items() if (S.PTYPE[p] == 0) == (t == "E")) / m.sum()
        got = hz[:, m].mean(1)
        gg = None
        for i in range(1, len(GRID)):
            if got[i - 1] < want <= got[i]:
                f = (np.log(want) - np.log(max(got[i - 1], 1e-6))) / (np.log(got[i]) - np.log(max(got[i - 1], 1e-6)))
                gg = float(np.exp(np.log(GRID[i - 1]) + f * (np.log(GRID[i]) - np.log(GRID[i - 1]))))
                break
        out[t] = dict(g=gg, want_hz=float(want), curve_hz=got.tolist())
    return out


def predict(views=("best",), noise=True, runs=10, trials=20, log=print):
    """-> {view: prediction}; the target fixed point is shared by the views"""
    circs, streams, V0, T = setup(runs, trials)
    t0 = time.time()
    nu, hist = target_fixed_point(circs, streams, V0, T, noise, log=log)
    out = {}
    for v in views:
        sg = surrogate_gain(v, nu, circs, streams, V0, T, noise)
        out[v] = dict(view=v, noise=noise, runs=runs, trials=trials, nu=nu, history=hist, grid=GRID.tolist(),
                      g_E=sg["E"]["g"], g_I=sg["I"]["g"], detail=sg, wall_s=time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="best,mean")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "gain.json"))
    a = ap.parse_args()
    noise = not a.no_noise
    js = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for v, p in predict(a.views.split(","), noise, a.runs, a.trials).items():
        k = f"{v}|{'noise' if noise else 'nonoise'}"
        js[k] = p
        json.dump(js, open(a.out, "w"), indent=1)
        print(f"{k}: nu* " + " ".join(f"{q} {p['nu'][q]:.1f}" for q in POPS) + f" | g_E {p['g_E']} g_I {p['g_I']}  ({p['wall_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
