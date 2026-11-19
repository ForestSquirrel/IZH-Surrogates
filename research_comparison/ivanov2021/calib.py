"""
calib.py -- per-model units for the Ivanov liquid (zoo targets and their surrogates).

Ivanov's currents are unitless per-ms increments of a LIF with threshold 20; a synaptic event
(kernel exp(-t/1 ms)) and the input pulse (1 ms) are brief, so the unit that carries over
between neuron models is the 1 ms square-pulse threshold from rest (thr), not the DC
rheobase (HH: thr = 1.1 i_rh; ML C1 22.7; AdEx 24.9). A model receives I = u * thr / 20.
Input neurons get a pulse of 2 thr (one spike per stimulus event for every model; the
paper's 5 x makes AdEx burst); the LIF keeps the paper's 100 (= 5 x 20).

Surrogates are drop-in replacements: the target's current through the gauge (I / k_I,
time t / tau), i.e. iscale_sur = iscale_target / k_I, same pulse, rest of the surrogate.

    $PY calib.py         # -> data/calib.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
PF = os.path.dirname(RC)
sys.path.insert(0, RC)
sys.path.insert(0, os.path.join(RC, "lsmkit", "tests"))
from lsmkit.neurons import zoo, izh, MODELS
import test_zoo as TZ

ZOO = ("hh", "ml_c1", "adex_adapting")
DT = 0.1
OUT = os.path.join(HERE, "data", "calib.json")


def zoo_gauge(key):
    js = json.load(open(os.path.join(HERE, "data", "zoo_posteriors", f"{key}.json")))   # data input, see README
    b = js["best"]
    return dict(theta=list(b["theta"]), tau=float(b["tau"]), k_I=float(b["k_I"]),
                d2=float(b["d2"]), i_rh=float(js["i_rh_target"]), T_cycle=float(js["T_cycle_target"]))


def pulse_threshold(run, hi):
    lo = 0.0
    for _ in range(40):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if run(m) >= 1 else (m, hi)
    return hi


def izh_run(g, y0, amp_target_units, T=51.0):
    P = izh.params(1, g["theta"], DT, g["tau"])[0]
    y = np.array(y0, np.float32)
    n = 0
    for i in range(int(round(T / DT))):
        I = amp_target_units / g["k_I"] if i * DT < 1.0 else 0.0
        n += izh.step(y, P, np.float32(I))
    return n


def main():
    out = {}
    for key in ZOO:
        s = zoo.SPECS[key]
        _, rest = TZ.ref_run(key, s["y0"], lambda t: 0.0, 1000.0, s["h"] / 4)
        rest = [float(v) for v in rest] + [0.0] * (MODELS[s["model"]].DIM - len(rest))
        g = zoo_gauge(key)
        thr = pulse_threshold(lambda a: len(TZ.net_run(key, rest, lambda t: a if t < 1.0 else 0.0, 51.0)),
                              400 * g["i_rh"])
        r = izh.rest(g["theta"], 0.0)
        y0s = list(r) if r is not None else [-65.0, -65.0 * g["theta"][1]]
        thr_s = pulse_threshold(lambda a: izh_run(g, y0s, a), 400 * g["i_rh"])
        n_t = len(TZ.net_run(key, rest, lambda t: 2 * thr if t < 1.0 else 0.0, 51.0))
        n_s = izh_run(g, y0s, 2 * thr)
        out[key] = dict(rest=rest, thr=thr, iscale=thr / 20.0, gauge=g, sur_rest=y0s,
                        sur_thr_target_units=thr_s, spikes_per_pulse=dict(target=n_t, surrogate=n_s))
        print(f"{key:14s} thr {thr:9.3f}  surrogate thr {thr_s:9.3f} (target units)  "
              f"pulse 2 thr -> target {n_t}, surrogate {n_s} spikes; tau {g['tau']:.3g} k_I {g['k_I']:.3g}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
