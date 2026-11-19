"""
v3_core.py -- RK4 integrator, stimuli, and the uncapped-counter contract.

Two things differ from v2's integrator and both are decisions from the spec:

B1  UNCAPPED COUNTERS. `integrate_v3` returns (n_stored, n_total, invalid).
    `n_total` counts every spike that occurred; `n_stored` counts those that fit
    in the caller's buffer. Features that need a COUNT read `n_total` and are
    exact at any buffer size; features that need spike TIMES read the buffer and
    must be gated on n_stored == n_total. ss32.1 is the cautionary case: no buffer
    size suffices for alpha_acc, because the ratio of two counts keeps moving as
    long as EITHER can saturate.

    The closed cap-exposed list is {P1-tail, P4, P5-alpha_acc, P7, P10}
    (ss32.2, ss33.4, ss33.9) and every member of it now reads a counter.

    Consequence for P4 (B2): BOTH delta_star and rho_res read the uncapped count
    -- ss33.9 measured rho_res at up to 1508 sigma when only delta_star was repaired.

    `spike_times` length is read with len() at call time, so the buffer size is a
    RUNTIME property. In v2 MAX_SPIKES was a module global folded into
    @njit(cache=True), which is the exact shape of the stale-cache bug ss12/ss20
    document. v3 cannot have that bug.
"""
import numpy as np
from numba import njit

from v3_config import (
    STATE_DIM, V_TH, DIVERGE_BOUND, DT, V_REST,
    STIM_STEP, STIM_RAMP_HOLD, STIM_PULSE_PAIR, STIM_HYPER_REL,
    STIM_HYST, STIM_TWO_PHASE,
)


# ============================================================ stimuli
@njit(cache=True)
def stim_v3(stim_id, t, sp):
    """All shape parameters arrive already expressed in absolute ms by the
    caller, which is where the T_cycle / L_inf multiplication happens. Nothing
    in this file knows about a fixed duration."""
    if stim_id == STIM_STEP:
        # sp = [t_on, amp]
        return sp[1] if t >= sp[0] else 0.0

    elif stim_id == STIM_RAMP_HOLD:
        # sp = [t_on, t_ramp_end, amp_final]  -- linear ramp then HOLD (spec 3.3)
        if t < sp[0]:
            return 0.0
        if t >= sp[1]:
            return sp[2]
        return sp[2] * (t - sp[0]) / (sp[1] - sp[0])

    elif stim_id == STIM_PULSE_PAIR:
        # sp = [t1, t2, width, amp]
        if sp[0] <= t < sp[0] + sp[2]:
            return sp[3]
        if sp[1] <= t < sp[1] + sp[2]:
            return sp[3]
        return 0.0

    elif stim_id == STIM_HYPER_REL:
        # sp = [t_on, t_off, amp_neg]
        return sp[2] if sp[0] <= t < sp[1] else 0.0

    elif stim_id == STIM_HYST:
        # sp = [t_on, t_peak, t_end, amp_max] -- triangular up/down
        if t < sp[0] or t >= sp[2]:
            return 0.0
        if t < sp[1]:
            return sp[3] * (t - sp[0]) / (sp[1] - sp[0])
        return sp[3] * (sp[2] - t) / (sp[2] - sp[1])

    elif stim_id == STIM_TWO_PHASE:
        # sp = [t_cond, w_cond, amp_cond, t_test, t_test_end, amp_test]  (P8')
        if sp[2] != 0.0 and sp[0] <= t < sp[0] + sp[1]:
            return sp[2]
        if sp[3] <= t < sp[4]:
            return sp[5]
        return 0.0

    return 0.0


# ============================================================ RK4
@njit(cache=True)
def _rk4(rhs, t, y, params, I, h, ynxt, k1, k2, k3, k4, ytmp):
    h2 = h * 0.5
    rhs(t, y, params, I, k1)
    for j in range(STATE_DIM):
        ytmp[j] = y[j] + h2 * k1[j]
    rhs(t + h2, ytmp, params, I, k2)
    for j in range(STATE_DIM):
        ytmp[j] = y[j] + h2 * k2[j]
    rhs(t + h2, ytmp, params, I, k3)
    for j in range(STATE_DIM):
        ytmp[j] = y[j] + h * k3[j]
    rhs(t + h, ytmp, params, I, k4)
    for j in range(STATE_DIM):
        ynxt[j] = y[j] + (h / 6.0) * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j])


@njit(cache=True)
def integrate_v3(rhs, stim_id, stim_params, scale,
                 y, t0, t1, h, spike_times, reset_v, reset_du, params):
    """Returns (n_stored, n_total, invalid).

    n_total is the UNCAPPED spike count; n_stored is how many times fit in the
    buffer. They differ exactly when the buffer bound; callers that need times
    must check n_stored == n_total, callers that need a count never care.
    """
    k1 = np.empty(STATE_DIM); k2 = np.empty(STATE_DIM)
    k3 = np.empty(STATE_DIM); k4 = np.empty(STATE_DIM)
    ytmp = np.empty(STATE_DIM); ynxt = np.empty(STATE_DIM)

    cap = len(spike_times)
    n_stored = 0
    n_total = 0
    invalid = 0
    t = t0
    n_steps = int((t1 - t0) / h) + 2

    for _ in range(n_steps):
        if t >= t1:
            break
        I = scale * stim_v3(stim_id, t, stim_params)
        v_prev = y[0]
        _rk4(rhs, t, y, params, I, h, ynxt, k1, k2, k3, k4, ytmp)
        for j in range(STATE_DIM):
            y[j] = ynxt[j]

        amax = 0.0
        for j in range(STATE_DIM):
            av = y[j] if y[j] >= 0.0 else -y[j]
            if av > amax:
                amax = av
        if amax > DIVERGE_BOUND:
            invalid = 1
            break

        if v_prev < V_TH and y[0] >= V_TH:
            n_total += 1
            if n_stored < cap:
                spike_times[n_stored] = t
                n_stored += 1
            y[0] = reset_v
            y[1] = y[1] + reset_du
        t += h

    return n_stored, n_total, invalid


@njit(cache=True)
def count_in_window(spike_times, n, t_lo, t_hi):
    c = 0
    for i in range(n):
        if t_lo <= spike_times[i] < t_hi:
            c += 1
    return c


@njit(cache=True)
def count_in_window_uncapped(rhs, stim_id, stim_params, scale, params,
                             t0, t1, h, t_lo, t_hi):
    """Count spikes in [t_lo, t_hi) WITHOUT storing any -- exact at any size.

    This is the kernel every member of the cap-exposed list routes through when
    it needs a count rather than times (B1). No buffer is involved at all, so
    there is no size at which the answer changes.
    """
    k1 = np.empty(STATE_DIM); k2 = np.empty(STATE_DIM)
    k3 = np.empty(STATE_DIM); k4 = np.empty(STATE_DIM)
    ytmp = np.empty(STATE_DIM); ynxt = np.empty(STATE_DIM)
    y = np.empty(STATE_DIM)
    y[0] = V_REST
    y[1] = params[1] * V_REST
    reset_v = params[2]; reset_du = params[3]
    c = 0
    t = t0
    n_steps = int((t1 - t0) / h) + 2
    for _ in range(n_steps):
        if t >= t1:
            break
        I = scale * stim_v3(stim_id, t, stim_params)
        v_prev = y[0]
        _rk4(rhs, t, y, params, I, h, ynxt, k1, k2, k3, k4, ytmp)
        for j in range(STATE_DIM):
            y[j] = ynxt[j]
        amax = 0.0
        for j in range(STATE_DIM):
            av = y[j] if y[j] >= 0.0 else -y[j]
            if av > amax:
                amax = av
        if amax > DIVERGE_BOUND:
            return -1
        if v_prev < V_TH and y[0] >= V_TH:
            if t_lo <= t < t_hi:
                c += 1
            y[0] = reset_v
            y[1] = y[1] + reset_du
        t += h
    return c


def alloc_buffers(n_spikes=None):
    """Runtime-sized buffers. Nothing here is folded into a numba signature."""
    from v3_config import SPIKE_BUFFER_MIN, N_FI, DELTA_GRID_N
    n = SPIKE_BUFFER_MIN if n_spikes is None else int(n_spikes)
    return {
        "spike_times": np.empty(n, dtype=np.float64),
        "isi":         np.empty(n, dtype=np.float64),
        "fi":          np.empty(N_FI, dtype=np.float64),
        "resp":        np.empty(DELTA_GRID_N, dtype=np.float64),
        "scratch":     np.empty(1, dtype=np.float64),
    }
