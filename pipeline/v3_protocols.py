"""
v3_protocols.py -- the v3 protocol suite.

Every window below is a multiple of T_cycle (P8': of L_inf). The `geom` argument
carries the per-cell absolute durations computed once from the landmarks, so no
protocol ever sees a constant.

Cap discipline (B1/B2), stated per protocol because it is the thing most likely
to be silently undone by a later edit:

  P1   tail  -> needs TIMES. Gated on n_stored == n_total; a bound record is a
                sentinel, never a truncated ISI sequence.
  P2/P3      -> rates from UNCAPPED counts.
  P4   both  -> delta_star AND rho_res read the uncapped count (B2). ss33.9
                measured rho_res at up to 1508 sigma when only delta_star was fixed.
  P5   alpha_acc -> UNCAPPED counts on both arms. ss32.1: no buffer size suffices,
                because a ratio of two counts keeps moving while either saturates.
  P7   eta   -> i_off from the UNCAPPED last spike, tracked as a running scalar
                so it never touches the buffer. n == 0 -> SENTINEL, not 0.0 (ss23.2).
  P10  nu_spont -> UNCAPPED count.
"""
import math

import numpy as np
from numba import njit

from v3_config import (
    STATE_DIM, V_REST, DT, NAN_SENTINEL, SENT_CUT, I_RH_MIN, V_TH,
    STIM_STEP, STIM_RAMP_HOLD, STIM_PULSE_PAIR, STIM_HYPER_REL, STIM_HYST,
    STIM_TWO_PHASE, STIM_PARAMS_LEN,
    GAMMA_PULSE, GAMMA_STEP, RAMP_GAMMA_F, GAMMA_HYPER_MAX, REBOUND_DEPTH_N,
    N_FI, DELTA_GRID_N, HYST_GAMMA_MAX,
    N_DELAY_REC, REC_MIN_PTS, REC_TOL_DT, GAMMA_LINF,
    REC_W_COND, REC_T_SEARCH,
    P_MAX, PERIOD_TOL, PERIOD_TAIL,
)
from v3_core import (integrate_v3, count_in_window,
                     count_in_window_uncapped, _rk4, stim_v3)
from v3_landmarks import isi_period

FI_G = np.array([1.02, 1.05, 1.10, 1.20, 1.50, 2.00, 3.00, 4.00])


@njit(cache=True)
def burst_ratio(isi, start, count, p):
    """max/min ISI over ONE cycle of length p. Ported from src/features.py.

    NOT the coefficient of variation. `rho` is a burst ratio: for a period-p
    cell it contrasts the long and short ISIs WITHIN one cycle, which is what
    makes it a bursting descriptor rather than a dispersion one. v3 first
    shipped a CV here and it correlated with v2's rho at only +0.12.
    """
    if p <= 0 or count < p:
        return 1.0
    mn = isi[start]
    mx = isi[start]
    for i in range(p):
        v = isi[start + i]
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    if mn <= 0.0:
        return 1.0
    return mx / mn


@njit(cache=True)
def adaptation_index(isi, n_isi):
    """A = (mean of last <=5 ISIs - isi[0]) / isi[0], over ALL ISIs.

    Over the FULL train, not the settled tail: A exists to compare the first ISI
    against the last, so restricting it to the settled window removes the very
    contrast it measures.
    """
    if n_isi < 2:
        return NAN_SENTINEL
    first = isi[0]
    if first <= 0.0:
        return NAN_SENTINEL
    k = n_isi if n_isi <= 5 else 5
    tot = 0.0
    for i in range(n_isi - k, n_isi):
        tot += isi[i]
    return (tot / k - first) / first


# ---------------------------------------------------------------- P1
@njit(cache=True)
def p1_reference(rhs, params, i_rh, g_trans, g_win, g_disc,
                 spike_times, isi_buf):
    """(p, rho, p_tr, A, ell, mean_isi, unused, n_isi, ok).

    ok == 0 means the record was buffer-bound and every TIME-derived value here
    is a sentinel. Counts are not affected -- they come from n_total.
    """
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = g_trans
    sp[1] = GAMMA_STEP * i_rh
    y = np.empty(STATE_DIM)
    y[0] = V_REST
    y[1] = params[1] * V_REST
    t_end = g_trans + g_win
    ns, nt, inv = integrate_v3(rhs, STIM_STEP, sp, 1.0, y, 0.0, t_end, DT,
                               spike_times, params[2], params[3], params)
    if inv:
        return (NAN_SENTINEL, NAN_SENTINEL, NAN_SENTINEL, NAN_SENTINEL, NAN_SENTINEL, NAN_SENTINEL, NAN_SENTINEL, 0.0, 0.0)
    if ns < nt:
        # buffer-bound: every TIME-derived value here is a sentinel (B1/B3).
        return (NAN_SENTINEL, NAN_SENTINEL, NAN_SENTINEL, NAN_SENTINEL, NAN_SENTINEL, NAN_SENTINEL, NAN_SENTINEL, 0.0, 0.0)

    t_ss = g_trans + g_disc
    # transient period p_tr, on the discard window
    n_tr = 0
    for i in range(1, ns):
        if spike_times[i] < t_ss:
            isi_buf[n_tr] = spike_times[i] - spike_times[i - 1]
            n_tr += 1
    # p_tr is 0 for "aperiodic", which is a MEASUREMENT, not a missing value.
    # Sentinelling it here put ~25% of cells into a mask pattern they do not
    # belong in.
    p_tr = float(isi_period(isi_buf, n_tr, P_MAX, PERIOD_TOL)) if n_tr >= 1 else 0.0

    i0 = 0
    while i0 < ns and spike_times[i0] < t_ss:
        i0 += 1
    n_isi = 0
    for i in range(i0 + 1, ns):
        isi_buf[n_isi] = spike_times[i] - spike_times[i - 1]
        n_isi += 1

    # ell: first-spike latency after onset, normalized by T_cycle (= g_win/k)
    ell = NAN_SENTINEL
    for i in range(ns):
        if spike_times[i] >= g_trans:
            ell = spike_times[i] - g_trans
            break

    if n_isi < 2:
        # fires, but not enough to characterize a cycle
        return (NAN_SENTINEL, NAN_SENTINEL, p_tr, NAN_SENTINEL, ell,
                NAN_SENTINEL, 0.0, float(n_isi), 1.0)

    k = n_isi if n_isi < PERIOD_TAIL else PERIOD_TAIL
    off = n_isi - k
    p = float(isi_period(isi_buf[off:], k, P_MAX, PERIOD_TOL))

    tot = 0.0
    for i in range(off, n_isi):
        tot += isi_buf[i]
    mean_isi = tot / k

    # rho is the BURST RATIO over one cycle, sentinel when there is no cycle
    rho = burst_ratio(isi_buf, off, k, int(p)) if p > 0.0 else NAN_SENTINEL

    # A: adaptation over the FULL train, gated to p == 1
    A = adaptation_index(isi_buf, n_isi) if p == 1.0 else NAN_SENTINEL

    # Pi is NOT computed here -- it is P3, a persistence RATIO N(2W)/N(W) run at
    # a doubled window. v3 first shipped a binary phasic flag here, which
    # correlated with v2's Pi at -0.004.
    return (p, rho, p_tr, A, ell, mean_isi, 0.0, float(n_isi), 1.0)


# ---------------------------------------------------------------- P2
@njit(cache=True)
def p2_fi(rhs, params, i_rh, g_trans, g_win, fi_out):
    """f-I curve from UNCAPPED counts (no buffer involved)."""
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = g_trans
    t_end = g_trans + g_win
    for j in range(len(FI_G)):
        sp[1] = FI_G[j] * i_rh
        c = count_in_window_uncapped(rhs, STIM_STEP, sp, 1.0, params,
                                     0.0, t_end, DT, g_trans, t_end)
        fi_out[j] = NAN_SENTINEL if c < 0 else float(c) / g_win


@njit(cache=True)
def beta_chi(fi_out, t_cycle):
    f0 = fi_out[0]; f3 = fi_out[3]
    beta = NAN_SENTINEL
    if f0 > 0.0 and f3 > 0.0 and f0 > SENT_CUT and f3 > SENT_CUT:
        beta = (math.log(f3) - math.log(f0)) / (math.log(0.20) - math.log(0.02))
    f_lo = fi_out[1]; f_hi = fi_out[5]
    chi = NAN_SENTINEL
    if f_hi > 0.0 and f_lo >= 0.0 and f_hi > SENT_CUT and f_lo > SENT_CUT:
        chi = f_lo / f_hi
    lam = NAN_SENTINEL
    if f_lo > 0.0 and f_hi > 0.0:
        lam = f_hi / f_lo
    return beta, chi, lam


# ---------------------------------------------------------------- P3
@njit(cache=True)
def p3_persistence(rhs, params, i_rh, g_trans, g_win, g_win_long):
    """Pi = N(2W)/N(W) at gamma = GAMMA_STEP. ~2 tonic, ~1 phasic.

    Both counts UNCAPPED (B1). ss7 is the cautionary case: once the buffer bound,
    BOTH counts saturated and Pi read exactly 1.0 -- the pipeline's definition of
    phasic -- on 50.3% of the atlas.
    """
    # not i_rh-gated: spec 5.1 gates Pi on neither rheobase nor L_inf
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = g_trans
    sp[1] = GAMMA_STEP * i_rh
    t_mid = g_trans + g_win
    t_end = g_trans + g_win_long
    n_full = count_in_window_uncapped(rhs, STIM_STEP, sp, 1.0, params,
                                      0.0, t_end, DT, g_trans, t_end)
    n_half = count_in_window_uncapped(rhs, STIM_STEP, sp, 1.0, params,
                                      0.0, t_end, DT, g_trans, t_mid)
    if n_full < 0 or n_half <= 0:
        return NAN_SENTINEL
    return float(n_full) / float(n_half)


# ---------------------------------------------------------------- P4
@njit(cache=True)
def p4_pulse_pair(rhs, params, i_rh, t_cycle, g_trans, g_win,
                  d_max, width, resp_out):
    """UNCAPPED counts per separation. Feeds BOTH delta_star and rho_res (B2)."""
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[2] = width
    sp[3] = GAMMA_PULSE * i_rh
    step = d_max / (DELTA_GRID_N - 1) if DELTA_GRID_N > 1 else d_max
    for di in range(DELTA_GRID_N):
        delta = di * step
        sp[0] = g_trans
        sp[1] = g_trans + delta
        t_end = sp[1] + width + g_win
        c = count_in_window_uncapped(rhs, STIM_PULSE_PAIR, sp, 1.0, params,
                                     0.0, t_end, DT, g_trans, t_end)
        resp_out[di] = NAN_SENTINEL if c < 0 else float(c)


@njit(cache=True)
def delta_star_rho_res(resp, t_cycle, d_max):
    """(delta_star, rho_res) from the pulse-pair response curve.

    delta_star is the STEP LOCATION, not the argmax (ss41.2).

    P4's response curve in this box has no peak. Censused over 246 cells and
    across pulse amplitude 1.2-5.0 and width 0.5-12 cycles, the peak PROMINENCE
    -- the height of the maximum above its immediate neighbours -- is 0.00 spikes
    at median AND p90, with 100% of cells within one spike, at every setting. The
    curve is a step onto a plateau. An argmax over that reports which index won a
    tie, not a property of the cell, and it produced a 68.7% point mass whose
    location moved with the grid rather than with the neurons.

    What the curve does carry is WHERE the step is: the smallest separation at
    which two pulses beat one. That is a paired-pulse recovery threshold rather
    than a resonant separation -- the feature is renamed in meaning, not in slot
    -- and it is bimodal (one population recovering inside a grid step, one
    needing about eight), which is the structure behind ss36.6's flip rate of
    0.28-0.32 that a pure tie-break could not have produced.

    Measured at half height between the baseline and the plateau, so it does not
    depend on a tie among equal counts.

    rho_res is UNCHANGED and deliberately so: it is a ratio of two curve values,
    not an argmax, so none of the above touches it. ss36.6 measured it as the
    schema's only single-parameter column, a pure `b` detector at 5.621.
    """
    step = d_max / (DELTA_GRID_N - 1) if DELTA_GRID_N > 1 else d_max

    # rho_res: peak over baseline-at-max-separation. v2's argmax_response_cpu
    # SKIPS sentinel entries rather than propagating them.
    bi = 0
    best = resp[0]
    for i in range(1, DELTA_GRID_N):
        if resp[i] > best and resp[i] > SENT_CUT:
            best = resp[i]
            bi = i
    last = resp[DELTA_GRID_N - 1]
    rho_res = best / last if (last > 0.0 and last > SENT_CUT) else NAN_SENTINEL

    if t_cycle <= 0.0:
        return NAN_SENTINEL, rho_res

    # plateau from the upper half of the grid, baseline from zero separation
    n_hi = 0
    acc = 0.0
    for i in range(DELTA_GRID_N // 2, DELTA_GRID_N):
        if resp[i] > SENT_CUT:
            acc += resp[i]
            n_hi += 1
    base = resp[0]
    if n_hi == 0 or base < SENT_CUT:
        return NAN_SENTINEL, rho_res
    plateau = acc / n_hi
    if plateau <= base:
        # no step up: two pulses never beat one, so there is no recovery
        # threshold to report. SENTINEL, not a bound (spec 5.2 rule 2).
        return NAN_SENTINEL, rho_res

    thr = base + 0.5 * (plateau - base)
    k = -1
    for i in range(DELTA_GRID_N):
        if resp[i] > SENT_CUT and resp[i] >= thr:
            k = i
            break
    if k < 0:
        return NAN_SENTINEL, rho_res
    return (k * step) / t_cycle, rho_res


# ---------------------------------------------------------------- P5
@njit(cache=True)
def p5_gamma_ramp(rhs, params, i_rh, g_trans, t_ramp, g_win):
    """gamma_ramp = I(t_first)/i_rh on ONE ramp over 0.1*T_cycle, HOLD RETAINED.

    The hold is not incidental (spec 3.3): ss30.2 established it is what makes the
    feature mean accommodation, and ss32.5 showed removing it yields 1/ell at
    r^2 = 0.98 -- a duplicate column, not a new one.
    """
    if i_rh < I_RH_MIN:
        return NAN_SENTINEL
    amp = RAMP_GAMMA_F * i_rh
    t_ramp_end = g_trans + t_ramp
    t_end = t_ramp_end + g_win

    k1 = np.empty(STATE_DIM); k2 = np.empty(STATE_DIM)
    k3 = np.empty(STATE_DIM); k4 = np.empty(STATE_DIM)
    ytmp = np.empty(STATE_DIM); ynxt = np.empty(STATE_DIM)
    y = np.empty(STATE_DIM)
    y[0] = V_REST
    y[1] = params[1] * V_REST
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = g_trans; sp[1] = t_ramp_end; sp[2] = amp
    t = 0.0
    n_steps = int(t_end / DT) + 2
    for _ in range(n_steps):
        if t >= t_end:
            break
        I = 0.0
        if t >= g_trans:
            I = amp if t >= t_ramp_end else amp * (t - g_trans) / t_ramp
        v_prev = y[0]
        _rk4(rhs, t, y, params, I, DT, ynxt, k1, k2, k3, k4, ytmp)
        for j in range(STATE_DIM):
            y[j] = ynxt[j]
        if y[0] != y[0] or y[0] > 1.0e6 or y[0] < -1.0e6:
            return NAN_SENTINEL
        if v_prev < V_TH and y[0] >= V_TH:
            if t <= g_trans:
                return 0.0
            if t >= t_ramp_end:
                return RAMP_GAMMA_F
            return (amp * (t - g_trans) / t_ramp) / i_rh
        t += DT
    return NAN_SENTINEL


@njit(cache=True)
def p5_alpha_acc(rhs, params, i_rh, g_trans, t_ramp, g_win):
    """Ramp-vs-step spike ratio, BOTH arms from uncapped counts (ss32.1).

    Not i_rh-gated (spec 5.1). It is a RATIO of two counts taken under the same
    amplitude, so it stays meaningful when that amplitude is ~0.
    """
    amp = RAMP_GAMMA_F * i_rh
    t_ramp_end = g_trans + t_ramp
    t_end = t_ramp_end + g_win
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = g_trans; sp[1] = t_ramp_end; sp[2] = amp
    n_ramp = count_in_window_uncapped(rhs, STIM_RAMP_HOLD, sp, 1.0, params,
                                      0.0, t_end, DT, g_trans, t_end)
    sp2 = np.zeros(STIM_PARAMS_LEN)
    sp2[0] = g_trans; sp2[1] = amp
    n_step = count_in_window_uncapped(rhs, STIM_STEP, sp2, 1.0, params,
                                      0.0, t_end, DT, g_trans, t_end)
    if n_ramp < 0 or n_step <= 0:
        return NAN_SENTINEL
    return float(n_ramp) / float(n_step)


# ---------------------------------------------------------------- P6
@njit(cache=True)
def p6_rebound(rhs, params, i_rh, g_trans, t_hyper, g_win, spike_times):
    """(N_reb, g_reb). Depth range 6->12 with resolution 12->24 (B13): ss33.5
    showed the range alone drops distinct values 12->7 and doubles the atom."""
    if i_rh < I_RH_MIN:
        return NAN_SENTINEL, NAN_SENTINEL
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = g_trans
    sp[1] = g_trans + t_hyper
    n_at = np.zeros(REBOUND_DEPTH_N)
    g_first = NAN_SENTINEL
    for j in range(REBOUND_DEPTH_N):
        g = GAMMA_HYPER_MAX * float(j + 1) / REBOUND_DEPTH_N
        sp[2] = -g * i_rh
        t_end = sp[1] + g_win
        c = count_in_window_uncapped(rhs, STIM_HYPER_REL, sp, 1.0, params,
                                     0.0, t_end, DT, sp[1], t_end)
        if c < 0:
            return NAN_SENTINEL, NAN_SENTINEL
        n_at[j] = float(c)
        if c > 0 and g_first < SENT_CUT:
            g_first = g
    n_reb = n_at[REBOUND_DEPTH_N - 1]
    return n_reb, g_first


# ---------------------------------------------------------------- P7
@njit(cache=True)
def p7_hysteresis(rhs, params, i_rh, t_total):
    """eta = (I_on - I_off)/i_rh over a triangular quasi-static ramp.

    I_on  = current at the FIRST spike, during the UP ramp
    I_off = current at the LAST spike, during the DOWN ramp

    The two halves are tracked separately, which is the whole content of the
    feature. v3 first kept i_off as a running value over every spike including
    up-ramp ones, so a cell that fires only while the current rises reported a
    negative width; eta correlated with v2 at +0.13.

    Both currents come from running scalars, so no buffer is involved and the
    cap cannot truncate the last spike (B1 -- ss27's defect).

    n == 0 -> SENTINEL, not 0.0 (spec 5.2 rule 2, ss23.2). v2 returned 0.0 here,
    which is a fake zero: a cell that never fires has no hysteresis width, it
    does not have a width of zero. Same for a cell that fires only on one half.
    """
    if i_rh < I_RH_MIN:
        return NAN_SENTINEL
    half = t_total * 0.5
    gmax = HYST_GAMMA_MAX

    k1 = np.empty(STATE_DIM); k2 = np.empty(STATE_DIM)
    k3 = np.empty(STATE_DIM); k4 = np.empty(STATE_DIM)
    ytmp = np.empty(STATE_DIM); ynxt = np.empty(STATE_DIM)
    y = np.empty(STATE_DIM)
    y[0] = V_REST
    y[1] = params[1] * V_REST
    t = 0.0
    i_on = NAN_SENTINEL
    i_off = NAN_SENTINEL
    n = 0
    n_steps = int(t_total / DT) + 2
    for _ in range(n_steps):
        if t >= t_total:
            break
        if t <= half:
            frac = t / half
        else:
            frac = (t_total - t) / half
        I = frac * gmax * i_rh
        v_prev = y[0]
        _rk4(rhs, t, y, params, I, DT, ynxt, k1, k2, k3, k4, ytmp)
        for j in range(STATE_DIM):
            y[j] = ynxt[j]
        if y[0] != y[0] or y[0] > 1.0e6 or y[0] < -1.0e6:
            return NAN_SENTINEL
        if v_prev < V_TH and y[0] >= V_TH:
            n += 1
            if t < half:
                if i_on < SENT_CUT:
                    i_on = I           # FIRST up-ramp spike
            else:
                i_off = I              # LAST down-ramp spike
            y[0] = params[2]
            y[1] = y[1] + params[3]
        t += DT
    if n == 0 or i_on < SENT_CUT or i_off < SENT_CUT:
        return NAN_SENTINEL
    return (i_on - i_off) / i_rh


# ---------------------------------------------------------------- P8'
@njit(cache=True)
def _latency_two_phase(rhs, params, amp_cond, w_cond, amp_test, t_cond, delta,
                       t_search):
    """Latency from test-step onset to the first spike after it. -1 = no spike.

    ss38.1: the test MUST be a sustained step near onset (1.2*i_rh). At 3*i_rh
    L_inf is 0.87 ms and latency has no dynamic range in which a deficit could
    appear -- that arm gave 9.5% definedness. No buffer is touched, so this is
    uncapped by construction.
    """
    k1 = np.empty(STATE_DIM); k2 = np.empty(STATE_DIM)
    k3 = np.empty(STATE_DIM); k4 = np.empty(STATE_DIM)
    ytmp = np.empty(STATE_DIM); ynxt = np.empty(STATE_DIM)
    y = np.empty(STATE_DIM)
    y[0] = V_REST
    y[1] = params[1] * V_REST
    t_on = t_cond + delta
    t_end = t_on + t_search
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = t_cond; sp[1] = w_cond; sp[2] = amp_cond
    sp[3] = t_on;   sp[4] = t_end;  sp[5] = amp_test
    t = 0.0
    n_steps = int(t_end / DT) + 2
    for _ in range(n_steps):
        if t >= t_end:
            break
        I = stim_v3(STIM_TWO_PHASE, t, sp)
        v_prev = y[0]
        _rk4(rhs, t, y, params, I, DT, ynxt, k1, k2, k3, k4, ytmp)
        for j in range(STATE_DIM):
            y[j] = ynxt[j]
        if y[0] != y[0] or y[0] > 1.0e6 or y[0] < -1.0e6:
            return -2.0
        if v_prev < V_TH and y[0] >= V_TH:
            y[0] = params[2]
            y[1] = y[1] + params[3]
            if t >= t_on:
                return t - t_on
        t += DT
    return -1.0


@njit(cache=True)
def p8_a_rec(rhs, params, i_rh, l_inf, t_cond, delays_rel):
    """A_rec -- signed post-spike excitability change (B10).

    Geometry entirely in L_inf units (ss39.1): the delay grid, the search window
    and the conditioning width all scale with the cell, which is what removed
    91% of the tau leak while `a` rose. The value is normalized by L_inf.

    THE FIT IS ON ln|D|, AND THE SIGN IS CARRIED (ss38.1). `c` reaches -25 mV, so
    the reset frequently leaves the cell MORE depolarized than rest: 91.4% of
    this box comes back FACILITATED, not depressed. A fit that assumed a positive
    deficit discarded three quarters of the atlas for violating an assumption it
    never made. Definedness 24% -> 67.8% once the sign was carried.
    """
    if i_rh < I_RH_MIN or l_inf < SENT_CUT or l_inf <= 0.0:
        return NAN_SENTINEL
    amp_c = GAMMA_PULSE * i_rh
    amp_t = GAMMA_LINF * i_rh
    w_c = REC_W_COND * l_inf
    if w_c < 5.0 * DT:
        w_c = 5.0 * DT
    t_sr = REC_T_SEARCH * l_inf

    # the conditioning pulse has to evoke a spike, else there is no deficit to
    # recover from and the feature is undefined rather than zero
    nc = _latency_two_phase(rhs, params, amp_c, w_c, 0.0, t_cond, 0.0, w_c + t_sr)
    if nc < 0.0:
        return NAN_SENTINEL

    li = _latency_two_phase(rhs, params, 0.0, w_c, amp_t, t_cond,
                            delays_rel[N_DELAY_REC - 1] * l_inf, t_sr)
    if li <= 0.0:
        return NAN_SENTINEL

    tol = REC_TOL_DT * DT
    sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0
    m = 0
    sgn = 1.0
    for k in range(N_DELAY_REC):
        dl = delays_rel[k] * l_inf
        L = _latency_two_phase(rhs, params, amp_c, w_c, amp_t, t_cond, dl, t_sr)
        if L <= 0.0:
            continue
        D = L - li
        ad = D if D >= 0.0 else -D
        if ad <= tol:
            continue
        if m == 0:
            sgn = 1.0 if D >= 0.0 else -1.0
        x = dl / li
        yv = np.log(ad)
        sx += x; sy += yv; sxx += x * x; sxy += x * yv
        m += 1
    if m < REC_MIN_PTS:
        return NAN_SENTINEL
    den = m * sxx - sx * sx
    if den <= 0.0:
        return NAN_SENTINEL
    slope = (m * sxy - sx * sy) / den
    if slope >= 0.0:
        return NAN_SENTINEL
    icpt = (sy - slope * sx) / m
    return sgn * np.exp(icpt) / li


# ---------------------------------------------------------------- P10
@njit(cache=True)
def p10_spontaneous(rhs, params, g_trans, g_win):
    """UNCAPPED count at I = 0. ss33.2 measured up to 152 sigma when capped.

    Not gated on i_rh: a cell with no rheobase can still fire spontaneously, and
    that is exactly the information nu_spont exists to carry (the I_RH_MIN note).
    """
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = g_trans
    sp[1] = 0.0
    t_end = g_trans + g_win
    c = count_in_window_uncapped(rhs, STIM_STEP, sp, 1.0, params,
                                 0.0, t_end, DT, g_trans, t_end)
    if c < 0:
        return NAN_SENTINEL
    return float(c)
