"""
sd_check.py -- single-cell integration window: HH-DP target vs its Izhikevich surrogate.

Hypothesis (README, results): the surrogate integrates over a longer window, which would
explain more memory and less XOR in the network. Three probes, each cell from rest, in
paper units (ms, pA, nS, mV; the surrogate through its gauge and voltage map, exactly as in
the network engine):
  1. strength-duration: the smallest square-pulse current that fires the cell, for widths
     0.1-100 ms -> rheobase (long-pulse plateau) and chronaxie (width at 2 x rheobase);
  2. PSP: response to one excitatory conductance event (exp, tau 3 ms, E = 0 mV) of a
     sub-threshold size -> peak, time to peak, half-width, 1/e decay;
  3. coincidence window: smallest conductance of TWO identical events that fires the cell,
     as a function of their separation; window = separation at which it is 1.5 x the
     zero-separation value.

    PY=python
    $PY sd_check.py [--view best]          # -> data/sd_check.json, stdout table
"""
import os, sys, json, argparse
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import circuit as CI
import targets as TG
from lsmkit.neurons import hh_dp as H, izh

DT = 0.01        # ms


def make_sim(step):
    @njit
    def sim(y0, P, va, vb, iscale, T, I0, I1, Iamp, g_times, g_amp, tau_s, E_s, rec):
        """square current Iamp on [I0, I1) plus exp-conductance events of size g_amp at
        g_times (E_s, tau_s); returns (spiked, V trace if rec)"""
        y = y0.copy()
        n = int(T / DT)
        g = np.float32(0.0)
        dec = np.float32(np.exp(-DT / tau_s))
        out = np.zeros(n if rec else 1, np.float32)
        k = 0
        for s in range(n):
            t = s * DT
            while k < len(g_times) and g_times[k] <= t:
                g += g_amp
                k += 1
            V = va * y[0] + vb
            I = g * (E_s - V)
            if I0 <= t < I1:
                I += Iamp
            if step(y, P, np.float32(I * iscale)):
                return 1, out
            if rec:
                out[s] = va * y[0] + vb
            g *= dec
        return 0, out
    return sim


SIM = {"hh": make_sim(H.step), "izh": make_sim(izh.step)}


def cells(view):
    """-> {name: dict(sim, y0, P, va, vb, iscale)} for E/I targets and surrogates"""
    out = {}
    g = CI.gauges(view)
    for t, gM in (("E", H.G_M_E), ("I", H.G_M_I)):
        P = H.params(1, DT, gM)[0]
        y0 = TG.rest_state(gM).astype(np.float32)
        out[f"HH-{t}"] = dict(sim=SIM["hh"], y0=y0, P=P, va=np.float32(1), vb=np.float32(0),
                              iscale=np.float32(1))
        gg = g[t]
        r = izh.rest(gg.theta, 0.0)
        out[f"sur-{t}"] = dict(sim=SIM["izh"], y0=np.array(r, np.float32),
                               P=izh.params(1, gg.theta, DT, gg.tau)[0],
                               va=np.float32(gg.va), vb=np.float32(gg.vb),
                               iscale=np.float32(1.0 / gg.k_I))
    return out


def fires(c, T, I0=0., I1=0., Iamp=0., gt=(), ga=0.):
    s, _ = c["sim"](c["y0"], c["P"], c["va"], c["vb"], c["iscale"], T, I0, I1, Iamp,
                    np.asarray(gt, np.float64), np.float32(ga), np.float32(3.0),
                    np.float32(0.0), False)
    return bool(s)


def bisect(f, lo, hi, n=30):
    if not f(hi):
        return np.nan
    for _ in range(n):
        m = np.sqrt(lo * hi)
        lo, hi = (lo, m) if f(m) else (m, hi)
    return hi


def strength_duration(c, widths):
    t0 = 5.0
    return np.array([bisect(lambda a: fires(c, t0 + w + 60.0, t0, t0 + w, a), 1.0, 1e6)
                     for w in widths])


def psp(c, g_amp):
    s, v = c["sim"](c["y0"], c["P"], c["va"], c["vb"], c["iscale"], 100.0, 0., 0., 0.,
                    np.array([5.0]), np.float32(g_amp), np.float32(3.0), np.float32(0.0), True)
    v0 = v[int(4.0 / DT)]
    d = v - v0
    k = int(np.argmax(d))
    pk = float(d[k])
    above = np.nonzero(d >= pk / 2)[0]
    after = np.nonzero(d[k:] <= pk / np.e)[0]
    return dict(spiked=bool(s), peak_mV=pk, t_peak_ms=float(k * DT - 5.0),
                half_width_ms=float((above[-1] - above[0]) * DT),
                decay_1e_ms=float(after[0] * DT) if len(after) else np.nan)


def coincidence(c, seps):
    g0 = [bisect(lambda a: fires(c, 120.0, gt=(5.0, 5.0 + s), ga=a), 0.01, 3000.0) for s in seps]
    g0 = np.array(g0)
    rel = g0 / g0[0]
    above = np.nonzero(rel >= 1.5)[0]
    return g0, float(seps[above[0]]) if len(above) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="best")
    a = ap.parse_args()
    widths = np.logspace(-1, 2, 25)
    seps = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50])
    res = {}
    for name, c in cells(a.view).items():
        sd = strength_duration(c, widths)
        rh = sd[-1]
        ch = float(np.interp(2 * rh, sd[::-1], widths[::-1]))
        g1 = bisect(lambda x: fires(c, 120.0, gt=(5.0,), ga=x), 0.01, 3000.0)
        p = psp(c, 0.3 * g1)
        gc, win = coincidence(c, seps)
        res[name] = dict(sd_pA=sd.tolist(), rheobase_pA=float(rh), chronaxie_ms=ch,
                         g_single_nS=float(g1), psp_at_0p3=p, g_pair_nS=gc.tolist(),
                         coinc_window_ms=win)
        print(f"{name:6s} rheobase {rh:7.1f} pA  chronaxie {ch:6.2f} ms  single-EPSC thr "
              f"{g1:7.1f} nS  PSP(0.3 thr) peak {p['peak_mV']:5.2f} mV t_pk {p['t_peak_ms']:5.2f} "
              f"hw {p['half_width_ms']:5.2f} decay {p['decay_1e_ms']:5.2f} ms  pair window {win} ms",
              flush=True)
    json.dump(dict(view=a.view, widths_ms=widths.tolist(), seps_ms=seps.tolist(), cells=res),
              open(os.path.join(HERE, "data", f"sd_check_{a.view}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
