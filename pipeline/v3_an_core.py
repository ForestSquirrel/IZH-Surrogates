"""
v3_an_core.py -- reset-free, dimension-agnostic core for the v3 schema.

The v3 kernels in v3_core.py / v3_protocols.py have three things baked in that a
foreign model does not share:

    STATE_DIM = 2            HH has 4 states, Wilson 2, Morris-Lecar 2, HR 3
    V_TH = 30.0              every model has its own spike threshold
    v <- c, u <- u + d       only Izhikevich resets; the rest are bounded
                             oscillators that repolarize on their own

So the reset-free path is a parallel implementation rather than a flag. This
mirrors src_any_neuron's split for v2, and the reason is the same one stated
there: a reset model integrated on the reset-free path DIVERGES, and a
reset-free model integrated on the reset path is silently wrong, so the two must
not share a kernel with a runtime switch.

WHAT IS PRESERVED FROM v3 (this is the part that matters):

  * B1's uncapped counters. `integrate_an` returns (n_stored, n_total, invalid);
    counts come from n_total and never touch a buffer. A foreign model driven
    hard produces just as many spikes as an Izhikevich one, and §7's cap defect
    was never Izhikevich-specific.
  * B7's geometry. Every window is a multiple of the cell's own T_cycle
    (P8': of L_inf). Nothing here contains an absolute duration.
  * Spec 5.2's sentinel discipline: a non-outcome gates to SENTINEL, never to a
    bound or a fake zero.
"""
import numpy as np
from numba import njit

from v3_config import (DIVERGE_BOUND, DT, NAN_SENTINEL, SENT_CUT,
                       STIM_STEP, STIM_RAMP_HOLD, STIM_PULSE_PAIR,
                       STIM_HYPER_REL, STIM_HYST, STIM_TWO_PHASE,
                       STIM_PARAMS_LEN)
from v3_core import stim_v3


@njit(cache=True)
def _rk4_n(rhs, t, y, params, I, h, ynxt, k1, k2, k3, k4, ytmp):
    """RK4 over a state vector of ANY length -- the only structural difference
    from v3_core._rk4, which unrolls STATE_DIM = 2."""
    n = y.shape[0]
    h2 = h * 0.5
    rhs(t, y, params, I, k1)
    for j in range(n):
        ytmp[j] = y[j] + h2 * k1[j]
    rhs(t + h2, ytmp, params, I, k2)
    for j in range(n):
        ytmp[j] = y[j] + h2 * k2[j]
    rhs(t + h2, ytmp, params, I, k3)
    for j in range(n):
        ytmp[j] = y[j] + h * k3[j]
    rhs(t + h, ytmp, params, I, k4)
    for j in range(n):
        ynxt[j] = y[j] + (h / 6.0) * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j])


@njit(cache=True)
def integrate_an(rhs, stim_id, sp, i_scale, y0, t0, t1, h, v_th,
                 spike_times, params):
    """Reset-free integrate + collect. Returns (n_stored, n_total, invalid).

    n_total is UNCAPPED (B1). Callers needing spike TIMES must check
    n_stored == n_total; callers needing a COUNT never have to.
    """
    n = y0.shape[0]
    y = np.empty(n)
    for j in range(n):
        y[j] = y0[j]
    k1 = np.empty(n); k2 = np.empty(n); k3 = np.empty(n); k4 = np.empty(n)
    ytmp = np.empty(n); ynxt = np.empty(n)

    cap = len(spike_times)
    n_stored = 0
    n_total = 0
    t = t0
    n_steps = int((t1 - t0) / h) + 2
    for _ in range(n_steps):
        if t >= t1:
            break
        I = i_scale * stim_v3(stim_id, t, sp)
        v_prev = y[0]
        _rk4_n(rhs, t, y, params, I, h, ynxt, k1, k2, k3, k4, ytmp)
        for j in range(n):
            y[j] = ynxt[j]
        amax = 0.0
        for j in range(n):
            av = y[j] if y[j] >= 0.0 else -y[j]
            if av > amax:
                amax = av
        if amax > DIVERGE_BOUND:
            return n_stored, n_total, 1
        if v_prev < v_th and y[0] >= v_th:
            n_total += 1
            if n_stored < cap:
                spike_times[n_stored] = t
                n_stored += 1
        t += h
    return n_stored, n_total, 0


@njit(cache=True)
def count_an(rhs, stim_id, sp, i_scale, y0, t0, t1, h, v_th, t_lo, t_hi, params):
    """Count spikes in [t_lo, t_hi) storing NONE -- exact at any buffer size."""
    n = y0.shape[0]
    y = np.empty(n)
    for j in range(n):
        y[j] = y0[j]
    k1 = np.empty(n); k2 = np.empty(n); k3 = np.empty(n); k4 = np.empty(n)
    ytmp = np.empty(n); ynxt = np.empty(n)
    c = 0
    t = t0
    n_steps = int((t1 - t0) / h) + 2
    for _ in range(n_steps):
        if t >= t1:
            break
        I = i_scale * stim_v3(stim_id, t, sp)
        v_prev = y[0]
        _rk4_n(rhs, t, y, params, I, h, ynxt, k1, k2, k3, k4, ytmp)
        for j in range(n):
            y[j] = ynxt[j]
        amax = 0.0
        for j in range(n):
            av = y[j] if y[j] >= 0.0 else -y[j]
            if av > amax:
                amax = av
        if amax > DIVERGE_BOUND:
            return -1
        if v_prev < v_th and y[0] >= v_th:
            if t_lo <= t < t_hi:
                c += 1
        t += h
    return c


@njit(cache=True)
def latency_an(rhs, stim_id, sp, i_scale, y0, t0, t1, h, v_th, t_on, params):
    """Time from t_on to the first spike at or after it. -1 none, -2 diverged.

    No buffer, so uncapped by construction -- the same property the Izhikevich
    P8' relies on.
    """
    n = y0.shape[0]
    y = np.empty(n)
    for j in range(n):
        y[j] = y0[j]
    k1 = np.empty(n); k2 = np.empty(n); k3 = np.empty(n); k4 = np.empty(n)
    ytmp = np.empty(n); ynxt = np.empty(n)
    t = t0
    n_steps = int((t1 - t0) / h) + 2
    for _ in range(n_steps):
        if t >= t1:
            break
        I = i_scale * stim_v3(stim_id, t, sp)
        v_prev = y[0]
        _rk4_n(rhs, t, y, params, I, h, ynxt, k1, k2, k3, k4, ytmp)
        for j in range(n):
            y[j] = ynxt[j]
        amax = 0.0
        for j in range(n):
            av = y[j] if y[j] >= 0.0 else -y[j]
            if av > amax:
                amax = av
        if amax > DIVERGE_BOUND:
            return -2.0
        if v_prev < v_th and y[0] >= v_th and t >= t_on:
            return t - t_on
        t += h
    return -1.0


def make_sp(*vals):
    sp = np.zeros(STIM_PARAMS_LEN)
    for i, v in enumerate(vals):
        sp[i] = v
    return sp
