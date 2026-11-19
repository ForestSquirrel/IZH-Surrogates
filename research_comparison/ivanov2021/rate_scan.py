"""
rate_scan.py -- liquid / input-neuron rates vs liquid weight w per model (sweep-range finder).

    $PY rate_scan.py --models lif,hh,ml_c1,adex_adapting --n 200   # -> data/rate_scan_<ds>.json
"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R
import spec as S
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec
import circuit as CI, encode as EN

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="mnist")
ap.add_argument("--models", default="lif,hh,ml_c1,adex_adapting,sur:hh,sur:ml_c1,sur:adex_adapting,sur*:hh,sur*:ml_c1,sur*:adex_adapting")
ap.add_argument("--w", default="0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8")
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
data, y, nin = R.load(a.dataset)
res = {}
for m in a.models.split(","):
    dt = S.DT_LIF if m == "lif" else 0.1
    res[m] = {}
    line = []
    for w in map(float, a.w.split(",")):
        b, pulse = R.binding(m, nin + S.N_LIQ, dt)
        circ = CI.build(nin, w, dt, np.random.default_rng(100), pulse=pulse)
        net = LS.stack([circ], [b], CI.receptors())
        rng = np.random.default_rng(1)
        ev = [EN.mnist_events(data[i], rng) if a.dataset == "mnist" else EN.nmnist_events(*data[i]) for i in range(a.n)]
        o = engine_gpu.simulate(net, np.zeros(a.n, np.int32), np.full(a.n, int(round(S.T_SAMPLE / dt)), np.int64),
                                R.y0_for(m, a.n, circ.N), LS.events(ev, dt, nin), dt, Rec())
        c = o["counts"] / (S.T_SAMPLE / 1000)
        rl, ri = float(c[:, nin:].mean()), float(c[:, :nin].mean())
        res[m][w] = dict(liquid_hz=rl, input_hz=ri, silent=float((c[:, nin:].sum(0) == 0).mean()))
        line.append(f"{w:g}:{rl:.1f}")
    print(f"{m:20s} input {ri:5.1f} Hz | liquid Hz by w  " + "  ".join(line), flush=True)
json.dump(res, open(os.path.join(HERE, "data", f"rate_scan_{a.dataset}.json"), "w"), indent=1)
