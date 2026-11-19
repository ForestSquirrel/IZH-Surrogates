"""
traces.py -- record a voltage trace under the reference step.

The pipeline integrates to produce features and spike counts, never a trace, so this is a
small faithful re-implementation of its integrator: the same RK4, the same fixed step, the
same initial condition, and for reset models the same upward-crossing reset
(`v_prev < V_TH <= v` then `v -> params[2]`, `u -> u + params[3]`, exactly as
`v3_core.count_in_window_uncapped` applies it). The right-hand sides are the pipeline's own
njit functions, called directly.

Only for figures. Nothing scored anywhere uses this path.

    t, v = under_step(spec_or_rhs, ...)   # time in the cell's own cycle times
"""
import numpy as np

import common as K                                            # noqa: F401  (sets sys.path)
import v3_config as C


def _rk4(rhs, t, y, params, I, h, k1, k2, k3, k4, ytmp, out):
    n = y.size
    rhs(t, y, params, I, k1)
    for j in range(n):
        ytmp[j] = y[j] + 0.5 * h * k1[j]
    rhs(t + 0.5 * h, ytmp, params, I, k2)
    for j in range(n):
        ytmp[j] = y[j] + 0.5 * h * k2[j]
    rhs(t + 0.5 * h, ytmp, params, I, k3)
    for j in range(n):
        ytmp[j] = y[j] + h * k3[j]
    rhs(t + h, ytmp, params, I, k4)
    for j in range(n):
        out[j] = y[j] + (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])


def integrate(rhs, params, y0, I, h, t_trans, t_rec, reset=None, v_th=None,
              max_pts=200_000):
    """(t, v) over [0, t_rec] after discarding t_trans, with v = y[0].

    reset : None for a continuous model, or (reset_v, reset_du) for one that jumps.
    v_th  : the crossing that triggers the reset (reset models) -- unused otherwise.
    """
    y = np.asarray(y0, dtype=np.float64).copy()
    params = np.ascontiguousarray(np.asarray(params, dtype=np.float64))
    n = y.size
    k1, k2, k3, k4 = (np.empty(n) for _ in range(4))
    ytmp, ynxt = np.empty(n), np.empty(n)

    n_tr = int(t_trans / h)
    n_rec = int(t_rec / h)
    stride = max(1, n_rec // max_pts)
    ts, vs = [], []
    t = 0.0
    for i in range(n_tr + n_rec):
        v_prev = y[0]
        _rk4(rhs, t, y, params, I, h, k1, k2, k3, k4, ytmp, ynxt)
        y[:] = ynxt
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > C.DIVERGE_BOUND:
            break
        if reset is not None and v_prev < v_th <= y[0]:
            # record the peak before the jump, so the trace shows a spike rather than a
            # discontinuity at threshold -- the dynamics are unchanged, only the drawing
            if i >= n_tr and (i - n_tr) % stride == 0:
                ts.append(t)
                vs.append(max(v_th, y[0]))
            y[0] = reset[0]
            y[1] = y[1] + reset[1]
        t += h
        if i >= n_tr and (i - n_tr) % stride == 0:
            ts.append(t)
            vs.append(y[0])
    t0 = n_tr * h
    return np.asarray(ts) - t0, np.asarray(vs)


def run_stim(rhs, params, y0, stim_id, sp, h, t_end, v_detect, reset=None,
             v_th=None, max_pts=200_000):
    """(t, v, I, spike_times) under one of the pipeline's own stimuli.

    `stim_id`/`sp` are passed straight to `v3_core.stim_v3`, so the drive is the one the
    protocol actually applies, not a redrawing of it. Spikes are upward crossings of
    `v_detect`, which is the model's own declared threshold.
    """
    from v3_core import stim_v3
    y = np.asarray(y0, dtype=np.float64).copy()
    params = np.ascontiguousarray(np.asarray(params, dtype=np.float64))
    n = y.size
    k1, k2, k3, k4 = (np.empty(n) for _ in range(4))
    ytmp, ynxt = np.empty(n), np.empty(n)
    n_steps = int(t_end / h)
    stride = max(1, n_steps // max_pts)
    ts, vs, Is, spikes = [], [], [], []
    t = 0.0
    for i in range(n_steps):
        I = stim_v3(stim_id, t, sp)
        v_prev = y[0]
        _rk4(rhs, t, y, params, I, h, k1, k2, k3, k4, ytmp, ynxt)
        y[:] = ynxt
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > C.DIVERGE_BOUND:
            break
        if v_prev < v_detect <= y[0]:
            spikes.append(t)
            if reset is not None:
                ts.append(t); vs.append(max(v_detect, y[0])); Is.append(I)
                y[0] = reset[0]
                y[1] = y[1] + reset[1]
        t += h
        if i % stride == 0:
            ts.append(t); vs.append(y[0]); Is.append(I)
    return (np.asarray(ts), np.asarray(vs), np.asarray(Is), np.asarray(spikes))


def target_trace(slug, cycles=4.0, trans_cycles=6.0, gamma=None, normalize=True):
    """The target model under I = gamma * i_rh.

    normalize=True returns time in the cell's own cycle times; False returns milliseconds,
    which is what shows that the three models live on different time scales at all.
    """
    import targets as T
    import common as K
    import os
    gamma = C.GAMMA_STEP if gamma is None else gamma
    label, kind, payload, rh = None, None, None, None
    for s, lab, k, p, r in T.study1():
        if s == slug:
            label, kind, payload, rh = lab, k, p, r
    if kind is None:
        raise KeyError(slug)
    d = K.jread(os.path.join(K.TARGETS, f"{slug}.json"))
    tc, i_rh = d["T_cycle"], d["i_rh"]
    if kind == "continuous":
        sp = payload
        t, v = integrate(sp.rhs, sp.params, sp.y0, gamma * i_rh, sp.h,
                         trans_cycles * tc, cycles * tc)
    else:
        rhs, pr = payload
        y0 = np.array([C.V_REST, pr[1] * C.V_REST])
        t, v = integrate(rhs, pr, y0, gamma * i_rh, C.DT,
                         trans_cycles * tc, cycles * tc,
                         reset=(pr[2], pr[3]), v_th=C.V_TH)
    return (t / tc if normalize else t), v, dict(label=label, T_cycle=tc, i_rh=i_rh,
                                                 kind=kind)


def izh_trace(theta, i_rh, tc, cycles=4.0, trans_cycles=6.0, gamma=None,
              normalize=True, t_abs=None):
    """An Izhikevich surrogate under the same drive, in ITS own cycle times.

    Both axes are each cell's own T_cycle, which is what the gauge fixes: tau carries
    surrogate time to target time, so two cells that are equivalent under the gauge lie on
    top of each other on this axis without any fitting of the time base.
    """
    from v3_models import izh_rhs_v3
    gamma = C.GAMMA_STEP if gamma is None else gamma
    th = np.asarray(theta, dtype=np.float64)
    y0 = np.array([C.V_REST, th[1] * C.V_REST])
    t_rec = cycles * tc if t_abs is None else float(t_abs)
    t, v = integrate(izh_rhs_v3, th, y0, gamma * i_rh, C.DT,
                     trans_cycles * tc, t_rec,
                     reset=(th[2], th[3]), v_th=C.V_TH)
    return (t / tc if normalize else t), v
