"""
bench_integration.py -- neuron-integration cost of a liquid, without the network: the 1000 liquid
neurons of a condition (its binding and step) with no synapses and no input, GPU wall per 250 ms
sample. This is the cost the surrogate replaces; the LSM wall time adds spike delivery and the
per-step work of the network, which depend on synapses, signal propagation and the code base.
Cached in data/integration_cost.json (keyed by liquid binding and dt).

    $PY bench_integration.py [--conditions lif,hh@fastin,...] [--n 2000]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import spec as S
import circuit as CI
import run as R
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec

CACHE = os.path.join(HERE, "data", "integration_cost.json")


def key(cond):
    base = cond.partition("@")[0]
    return f"{base}|dt{R.dt_for(cond):g}"


def bench(cond, n=2000, reps=3):
    base = cond.partition("@")[0]
    dt = R.dt_for(cond)
    N = S.N_LIQ
    z = np.zeros(0)
    empty = LS.Circuit(N=N, nin=1, types=np.ones(N, np.int8), src=z.astype(np.int32), post=z.astype(np.int32), w=z,
                       dly=z.astype(np.int32), rec=z.astype(np.int32), U=z, D=z, F=z, stp=z.astype(np.int32), meta={})
    b, _ = R.binding(base, N, dt)
    net = LS.stack([empty], [b], CI.receptors())
    ev = (np.zeros(n + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int32))
    ns = int(round(S.T_SAMPLE / dt))
    walls = []
    for r in range(reps + 1):                       # first call compiles
        o = engine_gpu.simulate(net, np.zeros(n, np.int32), np.full(n, ns, np.int64), R.y0_for(base, n, N), ev, dt, Rec())
        if r:
            walls.append(o["compute"]["wall_s"] / n * 1e3)
    return dict(ms_per_sample=float(np.median(walls)), reps=walls, substeps=o["compute"]["model_substeps"], dt=dt, n=n)


def cost(cond, compute=True):
    """ms per sample of neuron integration for the liquid of cond (cached)"""
    js = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    k = key(cond)
    if k not in js and compute:
        js[k] = bench(cond)
        json.dump(js, open(CACHE, "w"), indent=1)
    return js.get(k, {}).get("ms_per_sample")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="lif,hh@fastin,ml_c1@fastin,adex_adapting@fastin,"
                                            "sur:hh@fastin,sur:ml_c1@fastin,sur:adex_adapting@fastin")
    a = ap.parse_args()
    for c in a.conditions.split(","):
        print(f"{c:28s} {cost(c):6.3f} ms / sample")


if __name__ == "__main__":
    main()
