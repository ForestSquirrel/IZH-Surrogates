"""
input_speed.py -- smoke test: time-scale the input neurons (speed s, like the gauge tau) so a
continuous model's input layer can follow Ivanov's pulse code.

Speed s: the neuron integrates s ms of its own time per ms of network time (zoo.params with
dt * s: same RK4 step in model time, s-fold more substeps). For each s the 1 ms pulse
threshold is re-measured and pulses are 2 x thr. Rate vs pixel probability p (Bernoulli pulse
each ms, 2 s) is compared with the paper's LIF input neuron, E[ISI] = 3 + (1-p)^3 / p ms.

    $PY input_speed.py            # -> stdout table, data/input_speed.json
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from lsmkit.neurons import zoo, MODELS

DT = 0.1
CAL = json.load(open(os.path.join(HERE, "data", "calib.json")))
PS = (0.02, 0.1, 0.3, 0.6, 1.0)
SPEEDS = (1, 2, 4, 8, 16, 32)


def lif_rate(p):
    return 1000.0 / (3 + (1 - p) ** 3 / p)


def sim(key, s, amp, pulses_ms, T):
    st = MODELS[zoo.SPECS[key]["model"]].step
    P = zoo.params(key, 1, DT * s)[0]
    y = np.array(CAL[key]["rest"], np.float32)
    on = np.zeros(int(T / DT) + 2, bool)
    for t in pulses_ms:                       # 1 ms pulse starting the step after the event
        a = int(round((t + 1) / DT)); on[a:a + int(round(1 / DT))] = True
    n = 0
    for i in range(int(T / DT)):
        n += st(y, P, np.float32(amp if on[i] else 0.0))
    return n


def thr(key, s):
    lo, hi = 0.0, 4 * CAL[key]["thr"] * 50
    for _ in range(30):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if sim(key, s, m, [0.0], 50.0) >= 1 else (m, hi)
    return hi


def main():
    rng = np.random.default_rng(0)
    T = 2000.0
    trains = {p: np.nonzero(rng.random(int(T)) < p)[0].astype(float) for p in PS}
    out = {"lif": {str(p): lif_rate(p) for p in PS}}
    print("p:                 " + "  ".join(f"{p:>6g}" for p in PS))
    print("LIF (paper)        " + "  ".join(f"{lif_rate(p):6.1f}" for p in PS))
    for key in zoo.SPECS:
        out[key] = {}
        for s in SPEEDS:
            th = thr(key, s)
            r = [sim(key, s, 2 * th, trains[p], T) / (T / 1000) for p in PS]
            err = np.mean([abs(np.log(max(x, 1e-3) / lif_rate(p))) for x, p in zip(r, PS)])
            out[key][s] = dict(thr=th, rates=r, mean_abs_log_err=float(err))
            print(f"{key:14s} s={s:<3d}  " + "  ".join(f"{x:6.1f}" for x in r) + f"   |log err| {err:.2f}  thr {th:.3g}", flush=True)
    json.dump(out, open(os.path.join(HERE, "data", "input_speed.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
