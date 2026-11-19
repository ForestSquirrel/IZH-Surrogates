"""
test_zoo.py -- convergence of the zoo network neurons (float32, RK4 substeps inside dt = 0.1
ms) against the pipeline's own float64 right-hand sides integrated with RK4 at h/4.

Probes, each from the cell's rest (1 s at I = 0): DC at 2 x i_rh for 500 ms (spike count
within 1, mean ISI within 1 %), and the Ivanov input pulse (1 ms of 320 x i_rh; spike count
in the following 50 ms equal).

    $PY lsmkit/tests/test_zoo.py          (from research_comparison)
"""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PF = os.path.dirname(ROOT)
for p in (ROOT, os.path.join(PF, "pipeline")):
    sys.path.insert(0, p)
from lsmkit.neurons import zoo, MODELS

DT = 0.1


def ref_fn(key):
    if key == "hh":
        from models_offamily import hh_rhs
        return hh_rhs, 4
    if key == "ml_c1":
        from models_offamily import morris_lecar_c1_rhs
        return morris_lecar_c1_rhs, 2
    from lsmkit.zoo_targets import adex_rhs
    return adex_rhs, 2


def ref_run(key, y, I_of_t, T, h):
    """float64 RK4; returns spike times and final state (spike rules as zoo.py)"""
    rhs, d = ref_fn(key)
    s = zoo.SPECS[key]
    pr = np.zeros(4)
    k = [np.zeros(d) for _ in range(4)]
    y = np.array(y[:d], float)
    up, sp = False, []
    for i in range(int(round(T / h))):
        t = i * h
        I = I_of_t(t)
        rhs(t, y, pr, I, k[0]); rhs(t, y + h / 2 * k[0], pr, I, k[1])
        rhs(t, y + h / 2 * k[1], pr, I, k[2]); rhs(t, y + h * k[2], pr, I, k[3])
        y = y + h / 6 * (k[0] + 2 * k[1] + 2 * k[2] + k[3])
        if key.startswith("adex"):
            if y[0] >= 30.0:
                y[0] = s["Vr"]; y[1] += s["b"]; sp.append(t + h)
        else:
            if y[0] >= s["v_th"]:
                if not up:
                    sp.append(t + h)
                up = True
            else:
                up = False
    return np.array(sp), y


def net_run(key, y, I_of_t, T):
    s = zoo.SPECS[key]
    step = MODELS[s["model"]].step
    P = zoo.params(key, 1, DT)[0]
    y = np.array(list(y) + [0.0] * (MODELS[s["model"]].DIM - len(y)), np.float32)
    sp = []
    for i in range(int(round(T / DT))):
        if step(y, P, np.float32(I_of_t(i * DT))):
            sp.append((i + 1) * DT)
    return np.array(sp)


def irh(key):
    return json.load(open(os.path.join(ROOT, "ivanov2021", "data", "zoo_posteriors", f"{key}.json")))["i_rh_target"]


def check(key):
    s = zoo.SPECS[key]
    h = s["h"] / 4
    _, rest = ref_run(key, s["y0"], lambda t: 0.0, 1000.0, h)
    i = irh(key)
    dc = lambda t: 2 * i
    r1, _ = ref_run(key, rest, dc, 500.0, h)
    n1 = net_run(key, rest, dc, 500.0)
    pulse = lambda t: 320 * i if t < 1.0 else 0.0
    r2, _ = ref_run(key, rest, pulse, 51.0, h)
    n2 = net_run(key, rest, pulse, 51.0)
    isi = lambda a: np.diff(a).mean() if len(a) > 2 else np.nan
    print(f"{key:14s} i_rh {i:8.3f}  DC: ref {len(r1)} spk ISI {isi(r1):7.3f} | net {len(n1)} spk ISI "
          f"{isi(n1):7.3f}  | pulse: ref {len(r2)} net {len(n2)} spikes")
    assert abs(len(r1) - len(n1)) <= 1
    assert abs(isi(n1) / isi(r1) - 1) < 0.01
    assert len(r2) == len(n2)
    return rest


if __name__ == "__main__":
    for key in zoo.SPECS:
        check(key)
    print("ok")
