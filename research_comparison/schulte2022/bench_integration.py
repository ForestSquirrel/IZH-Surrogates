"""
bench_integration.py -- neuron-integration cost of the Schulte circuit, without the network: the
560 neurons of a model (HH target, or the surrogate binding of a posterior view) with no synapses,
no input and no noise, GPU wall per simulated second of one circuit. This is the cost the
surrogate replaces; the full run adds spike delivery, conductance noise and the per-step work of
the network. Cached in data/integration_cost.json.

    $PY bench_integration.py [--models target,best,mean] [--n 200]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import spec as S
import circuit as CI
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec

CACHE = os.path.join(HERE, "data", "integration_cost.json")
T_SIM = 1000.0                                        # ms simulated per instance


def bench(model, n=200, reps=3):
    N = CI.N
    z = np.zeros(0)
    empty = LS.Circuit(N=N, nin=1, types=CI.TYPES, src=z.astype(np.int32), post=z.astype(np.int32), w=z,
                       dly=z.astype(np.int32), rec=z.astype(np.int32), U=z, D=z, F=z, stp=z.astype(np.int32), meta={})
    V0 = np.random.default_rng(0).uniform(*S.V0, (n, N))
    if model == "target":
        b, Y0 = CI.bind_target(), CI.y0_target(V0)
    else:
        g = CI.gauges(model)
        b, Y0 = CI.bind_surrogate(g), CI.y0_surrogate(V0, g)
    net = LS.stack([empty] * n, [b] * n, CI.receptors(noise=False))
    ev = (np.zeros(n + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int32))
    ns = int(round(T_SIM / S.DT))
    walls = []
    for r in range(reps + 1):                         # first call compiles
        o = engine_gpu.simulate(net, np.arange(n), np.full(n, ns, np.int64), Y0, ev, S.DT, Rec())
        if r:
            walls.append(o["compute"]["wall_s"] / n * 1e3)
    return dict(ms_per_s=float(np.median(walls)), reps=walls, substeps=o["compute"]["model_substeps"], dt=S.DT, n=n)


def cost(model, compute=True):
    """ms of GPU wall per simulated second of one circuit's neuron integration (cached)"""
    model = model.partition("@")[0]
    js = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if model not in js and compute:
        js[model] = bench(model)
        json.dump(js, open(CACHE, "w"), indent=1)
    return js.get(model, {}).get("ms_per_s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="target,best,mean")
    a = ap.parse_args()
    for m in a.models.split(","):
        print(f"{m}: {cost(m):.3f} ms / simulated s / circuit", flush=True)


if __name__ == "__main__":
    main()
