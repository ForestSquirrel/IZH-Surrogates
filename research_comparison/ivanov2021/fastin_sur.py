"""
fastin_sur.py -- surrogate input layer for '@fastin': the time-scaled zoo input neuron
(dy/dt = s f(y, I), spec.FASTIN_SPEED) is a target of its own, so it gets its own pipeline
posterior. The surrogate network 'sur*:<key>@fastin' then has a surrogate liquid AND a
surrogate input layer.

Input pulse (target units, delivered through the gauge, I / k_I in time t / tau): RATE-MATCHED.
The surrogate needs ~25x the target's brief-pulse charge (HH x4: 1 ms threshold 2.53 vs 63.9), so
neither the target's pulse (drop-in: single pulses never fire, dim pixels lost) nor 2 x its own
threshold (self-calibrated: fused pulses = 20 x rheobase DC, no saturation, rates 3-5x the
target's) reproduces the input code. fit_pulse() picks the one amplitude whose rate-vs-pixel
curve best matches the target input layer on the same stimuli (a tuning step, reported as such).

    $PY fastin_sur.py hh        # posterior -> data/fastin_posteriors/hh_s4.json; pulse fit -> hh_s4_pulse.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
PF = os.path.dirname(RC)
sys.path.insert(0, RC); sys.path.insert(0, HERE)
import spec as S
from lsmkit import spec as LS
from lsmkit.neurons import izh

DT = 0.1
OUT = os.path.join(HERE, "data", "fastin_posteriors")


def _path(key):
    return os.path.join(OUT, f"{key}_s{S.FASTIN_SPEED[key]}.json")


def fast_spec(key):
    """the pipeline's V3Spec of target key, s-fold faster: rhs x s, step and bootstrap windows / s"""
    sys.path.insert(0, os.path.join(PF, "pipeline"))
    import dataclasses
    from numba import njit
    from lsmkit import zoo_targets as Z
    sp = Z.spec(key)
    s = float(S.FASTIN_SPEED[key])
    base = sp.rhs

    @njit(cache=False)
    def rhs_fast(t, y, params, I, dydt):
        base(t, y, params, I, dydt)
        for i in range(len(dydt)):
            dydt[i] *= s
    return dataclasses.replace(sp, rhs=rhs_fast, h=sp.h / s, boot_t_trans=sp.boot_t_trans / s,
                               boot_t_win=sp.boot_t_win / s, boot_t_disc=sp.boot_t_disc / s,
                               name=f"{sp.name} x{int(s)} (fastin)")


def fast_reset_target(key):
    """AdEx (pipeline reset path, models_offamily.adex_rhs + v3 reset kernel): rhs x s; the reset
    map is instantaneous and unchanged. The reset path integrates at the pipeline's fixed DT
    (0.01), i.e. the s-fold faster AdEx at an effective 0.01 s model-ms step; checked against the
    gauge-equivariant prediction tau / s after the draw"""
    sys.path.insert(0, os.path.join(PF, "pipeline"))
    from numba import njit
    from v3_an_zoo import reset_targets
    name = {"adex_adapting": "AdEx adapting"}[key]
    hits = [t for t in reset_targets() if t[0].lower() == name.lower()]
    assert len(hits) == 1, [t[0] for t in reset_targets()]
    lbl, base, pr, rheo_hi = hits[0]
    s = float(S.FASTIN_SPEED[key])

    @njit(cache=False)
    def rhs_fast(t, y, params, I, dydt):
        base(t, y, params, I, dydt)
        for i in range(len(dydt)):
            dydt[i] *= s
    return f"{lbl} x{int(s)} (fastin)", rhs_fast, pr, rheo_hi


def draw(key):
    from lsmkit import infer
    vm = dict(va=1.0, vb=0.0, note="current synapses: unused")
    info = dict(zoo_key=key, speed=S.FASTIN_SPEED[key])
    if key.startswith("adex"):
        name, rhs, pr, rheo_hi = fast_reset_target(key)
        return infer.run("reset", (rhs, pr), _path(key), vm, name=name, rheo_hi=rheo_hi, target_info=info)
    sp = fast_spec(key)
    return infer.run("continuous", sp, _path(key), vm, name=sp.name, target_info=info)


def gauge(key):
    js = json.load(open(_path(key)))
    b = js["best"]
    return dict(theta=list(b["theta"]), tau=float(b["tau"]), k_I=float(b["k_I"]), d2=float(b["d2"]),
                i_rh=float(js["i_rh_target"]), T_cycle=float(js["T_cycle_target"]))


def _izh_run(g, y0, amp, T=51.0):
    P = izh.params(1, g["theta"], DT, g["tau"])[0]
    y = np.array(y0, np.float32)
    n = 0
    for i in range(int(round(T / DT))):
        n += izh.step(y, P, np.float32(amp / g["k_I"] if i * DT < 1.0 else 0.0))
    return n


def calib(key):
    """-> dict(gauge, rest, thr_sur, pulse) for the surrogate input neuron (target units)"""
    g = gauge(key)
    r = izh.rest(g["theta"], 0.0)
    y0 = list(r) if r is not None else [-65.0, -65.0 * g["theta"][1]]
    lo, hi = 0.0, 400 * g["i_rh"]
    for _ in range(40):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if _izh_run(g, y0, m) >= 1 else (m, hi)
    return dict(gauge=g, rest=[float(v) for v in y0], thr_sur=hi, pulse=max(2 * hi, 2 * g["i_rh"]),
                spikes_per_pulse=_izh_run(g, y0, max(2 * hi, 2 * g["i_rh"])))


BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


def _rates(b, pulse, y0, data, ids, dt):
    import circuit as CI, run as R
    from lsmkit import engine_gpu
    from lsmkit.engine_cpu import Rec
    nin = b.P.shape[0]
    net = LS.stack([CI.build_inputs_only(nin, dt, pulse)], [b], CI.receptors())
    evs = LS.events([R.stimulus(data, i, "mnist") for i in ids], dt, nin)
    o = engine_gpu.simulate(net, np.zeros(len(ids), np.int32), np.full(len(ids), int(round(S.T_SAMPLE / dt)), np.int64),
                            y0, evs, dt, Rec())
    return o["counts"].astype(float) / (S.T_SAMPLE / 1000)


def _curve(r, pix):
    return np.array([r[(pix > lo) & (pix <= hi)].mean() for lo, hi in zip(BINS[:-1], BINS[1:])]), \
        np.array([((pix > lo) & (pix <= hi)).sum() for lo, hi in zip(BINS[:-1], BINS[1:])])


def fit_pulse(key, n=50, dt=DT):
    """rate-matched pulse: minimise the count-weighted squared error of the mean rate per pixel
    bin (p > 0; silent pixels are silent in both) between surrogate and target input layers"""
    import run as R
    data, _, nin = R.load("mnist")
    ids = np.arange(n)
    pix = data[ids].astype(float) / 255
    bt, pt = R.binding(key, nin, dt)
    bt, pt = R.fastin(key, bt, nin, dt)
    ct, w = _curve(_rates(bt, pt, R.y0_for(key, n, nin), data, ids, dt), pix)
    c = calib(key)
    g = c["gauge"]
    b = _binding(g, nin, dt)
    y0s = np.broadcast_to(np.asarray(c["rest"], np.float32), (n, nin, 2)).copy()
    loss = lambda A: float((w * (_curve(_rates(b, A, y0s, data, ids, dt), pix)[0] - ct) ** 2).sum() / w.sum())
    grid = np.geomspace(2.0, 2 * c["thr_sur"], 48)
    L = [loss(A) for A in grid]
    j = int(np.argmin(L))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    for _ in range(25):                              # golden section on the bracket
        a = hi - 0.618 * (hi - lo); bb = lo + 0.618 * (hi - lo)
        lo, hi = (lo, bb) if loss(a) < loss(bb) else (a, hi)
    A = 0.5 * (lo + hi)
    cs = _curve(_rates(b, A, y0s, data, ids, dt), pix)[0]
    out = dict(pulse=A, rmse_hz=loss(A) ** 0.5, bins=list(BINS), target_curve=ct.tolist(), sur_curve=cs.tolist(),
               n_per_bin=w.tolist(), target_pulse=pt * float(bt.iscale[0]), thr_sur=c["thr_sur"], n_samples=n,
               grid=grid.tolist(), grid_loss=L)
    json.dump(out, open(_path(key).replace(".json", "_pulse.json"), "w"), indent=1)
    return out


def _binding(g, nin, dt):
    f = np.float32
    return LS.Binding(model="izh", P=izh.params(nin, g["theta"], dt, g["tau"]), tref=np.zeros(nin, np.int32),
                      clamp=np.zeros(nin, np.int32), bias=np.zeros(nin, f),
                      iscale=np.full(nin, 1.0 / g["k_I"], f), va=np.ones(nin, f), vb=np.zeros(nin, f), dim=2,
                      meta=dict(model="sur fastin", gauge=g))


def input_binding(key, nin, dt):
    """-> (Binding, pulse) of the surrogate input layer: rate-matched pulse (target units),
    iscale 1 / k_I"""
    pulse = json.load(open(_path(key).replace(".json", "_pulse.json")))["pulse"]
    b = _binding(gauge(key), nin, dt)
    b.meta.update(model=f"sur fastin {key}", pulse_target_units=pulse, pulse_rule="rate-matched")
    return b, pulse


def _input_binding_selfcal(key, nin, dt):
    """(unused) self-calibrated pulse max(2 thr_sur, 2 i_rh): rates 3-5x the target's"""
    c = calib(key)
    g = c["gauge"]
    f = np.float32
    b = LS.Binding(model="izh", P=izh.params(nin, g["theta"], dt, g["tau"]), tref=np.zeros(nin, np.int32),
                   clamp=np.zeros(nin, np.int32), bias=np.zeros(nin, f),
                   iscale=np.full(nin, 1.0 / g["k_I"], f), va=np.ones(nin, f), vb=np.zeros(nin, f), dim=2,
                   meta=dict(model=f"sur fastin {key}", gauge=g, pulse_target_units=c["pulse"]))
    return b, c["pulse"]


def y0(key, n, N):
    return np.broadcast_to(np.asarray(calib(key)["rest"], np.float32), (n, N, 2)).copy()


if __name__ == "__main__":
    key = sys.argv[1]
    if not os.path.exists(_path(key)):
        draw(key)
    c = calib(key)
    g = c["gauge"]
    pf = _path(key).replace(".json", "_pulse.json")
    fp = json.load(open(pf)) if os.path.exists(pf) else fit_pulse(key)
    print(f"rate-matched pulse {fp['pulse']:.3f} target units (target pulse {fp['target_pulse']:.3f}, thr_sur {fp['thr_sur']:.2f}); "
          f"rmse {fp['rmse_hz']:.1f} Hz")
    for lo, hi, a, bq, nn in zip(BINS[:-1], BINS[1:], fp["target_curve"], fp["sur_curve"], fp["n_per_bin"]):
        print(f"   p ({lo:.1f}, {hi:.1f}]  target {a:6.1f} Hz  surrogate {bq:6.1f} Hz   n {nn}")
    print(f"{key} x{S.FASTIN_SPEED[key]}: tau {g['tau']:.4g} k_I {g['k_I']:.4g} d2 {g['d2']:.3g} theta {np.round(g['theta'], 4)}; "
          f"thr_sur {c['thr_sur']:.3f}, pulse {c['pulse']:.3f} (target units) -> {c['spikes_per_pulse']} spike(s)")
