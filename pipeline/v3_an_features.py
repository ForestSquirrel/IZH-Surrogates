"""
v3_an_features.py -- the v3 18-column schema for an arbitrary reset-free model.

Mirrors v3_landmarks.py + v3_protocols.py + v3_features.py function for
function, so the two paths are auditable against each other. Every window is a
multiple of the cell's own T_cycle (P8': of L_inf), so a v3 model can be
conditioned on a foreign target without either side knowing an absolute duration.

Scale recovery, unchanged in form from §2 but with v3's anchor (B7):

    k_I = i_rh^model    / i_rh^Izh(theta)
    tau = T_cycle^model / T_cycle^Izh(theta)

`t_ref` does not appear. It was v2's anchor and §33.2 found it contaminated on
91% of cells; T_cycle replaces it everywhere.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

import v3_config as C
from v3_config import (NAN_SENTINEL, SENT_CUT, DT, I_RH_MIN, N_FI, DELTA_GRID_N,
                       P_MAX, PERIOD_TOL, PERIOD_TAIL, GAMMA_STEP, GAMMA_PULSE,
                       RAMP_GAMMA_F, GAMMA_HYPER_MAX, REBOUND_DEPTH_N,
                       HYST_GAMMA_MAX, GAMMA_LINF, K_TRANS, K_DISCARD, K_WIN,
                       K_WIN_LONG, K_DELTA_MAX, K_PULSE_WIDTH, K_WIN_PULSE,
                       K_RAMP, K_HYPER, N_DELAY_REC, REC_W_COND, REC_T_SEARCH,
                       REC_MIN_PTS, REC_TOL_DT, REC_D_LO, REC_D_HI,
                       RHEO_SCAN_N, RHEO_BISECT_ITERS, N_MIN_SPIKES,
                       BOOT_T_TRANS, BOOT_T_WIN, BOOT_T_DISC, LINF_SEARCH_K,
                       GATE_K_DT)
from v3_an_core import integrate_an, count_an, latency_an, make_sp
from v3_landmarks import isi_period
from v3_protocols import burst_ratio, adaptation_index

FI_G = np.array([1.02, 1.05, 1.10, 1.20, 1.50, 2.00, 3.00, 4.00])
DELAYS_REL = np.geomspace(REC_D_LO, REC_D_HI, N_DELAY_REC)


@dataclass
class V3Spec:
    """A foreign model, in the terms v3's protocols need.

    `rheo_lo/hi` are in the MODEL's own current units, which is why they are per
    model: v3's Izhikevich bracket [0, 200] is meaningless for Hodgkin-Huxley.
    `h` is the model's integration step -- §34's resolution gate is stated in
    units of DT, so a model with its own h carries its own floor.
    """
    rhs: object
    y0: np.ndarray
    params: np.ndarray = field(default_factory=lambda: np.zeros(4))
    v_th: float = 0.0
    rheo_lo: float = 0.0
    rheo_hi: float = 200.0
    h: float = DT
    # PASS-0 bootstrap window, in the MODEL's time units. The bootstrap is the
    # one place v3 permits an absolute duration (it exists to measure the
    # landmarks every other window is then expressed in), so it is the one place
    # that must be per-model. v3's 500 is tuned to Izhikevich; a slow burster
    # like Hindmarsh-Rose fits too few cycles in it and the settled tail comes
    # back with < 2 ISIs, which reads as "no anchor" rather than "window too
    # short". NeuronSpec carries t_win/t_discard for exactly this reason.
    boot_t_trans: float = BOOT_T_TRANS
    boot_t_win: float = BOOT_T_WIN
    boot_t_disc: float = BOOT_T_DISC
    name: str = "model"


# ------------------------------------------------------------------ landmarks
def _fires(spec, amp, t_trans, t_win, st):
    sp = make_sp(t_trans, amp)
    ns, nt, inv = integrate_an(spec.rhs, C.STIM_STEP, sp, 1.0, spec.y0, 0.0,
                               t_trans + t_win, spec.h, spec.v_th, st, spec.params)
    if inv:
        return -1
    if ns < nt:
        return 1
    c = 0
    for i in range(ns):
        if st[i] >= t_trans:
            c += 1
    return 1 if c >= N_MIN_SPIKES else 0


def rheobase_an(spec, t_trans, t_win, st):
    """Scan for a firing bracket, then bisect -- the §9.4 shape, so a Class-2
    bounded firing band is bracketed rather than missed. No I_RH_MIN gate here;
    spec 5.1 puts that on the rheobase-normalized columns."""
    lo, hi, prev, found = spec.rheo_lo, spec.rheo_hi, spec.rheo_lo, 0
    for i in range(1, RHEO_SCAN_N + 1):
        fr = i / RHEO_SCAN_N
        amp = spec.rheo_lo + (spec.rheo_hi - spec.rheo_lo) * fr ** 3
        if _fires(spec, amp, t_trans, t_win, st) == 1:
            lo, hi, found = prev, amp, 1
            break
        prev = amp
    if not found:
        return NAN_SENTINEL
    for _ in range(RHEO_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if _fires(spec, mid, t_trans, t_win, st) == 1:
            hi = mid
        else:
            lo = mid
    return hi


def t_cycle_an(spec, i_rh, t_trans, t_win, t_disc, st, isi):
    """(T_cycle, p_hat, capped). Phase-invariant: p * mean settled ISI."""
    sp = make_sp(t_trans, GAMMA_STEP * i_rh)
    ns, nt, inv = integrate_an(spec.rhs, C.STIM_STEP, sp, 1.0, spec.y0, 0.0,
                               t_trans + t_win, spec.h, spec.v_th, st, spec.params)
    if inv:
        return NAN_SENTINEL, 0, 0
    if ns < nt:
        return NAN_SENTINEL, 0, 1
    t_ss = t_trans + t_disc
    i0 = 0
    while i0 < ns and st[i0] < t_ss:
        i0 += 1
    n_isi = 0
    for i in range(i0 + 1, ns):
        isi[n_isi] = st[i] - st[i - 1]
        n_isi += 1
    if n_isi < 2:
        return NAN_SENTINEL, 0, 0
    k = min(n_isi, PERIOD_TAIL)
    off = n_isi - k
    p = isi_period(isi[off:], k, P_MAX, PERIOD_TOL)
    mean_isi = float(np.mean(isi[off:n_isi]))
    return (p if p > 0 else 1.0) * mean_isi, p, 0


def l_inf_an(spec, i_rh, t_search, t_on):
    sp = make_sp(t_on, GAMMA_LINF * i_rh)
    L = latency_an(spec.rhs, C.STIM_STEP, sp, 1.0, spec.y0, 0.0, t_on + t_search,
                   spec.h, spec.v_th, t_on, spec.params)
    return NAN_SENTINEL if L <= 0.0 else L


def _rheo_expand(spec, bt, bw, st, n_expand, out):
    """Rheobase with a DOUBLING ceiling -- the bracket failure is loud, so retry."""
    hi = float(spec.rheo_hi)
    saved = spec.rheo_hi
    try:
        for _ in range(max(1, n_expand)):
            spec.rheo_hi = hi
            v = rheobase_an(spec, bt, bw, st)
            if v > SENT_CUT:
                out["rheo_hi"] = hi
                return v
            hi *= 2.0
    finally:
        spec.rheo_hi = saved
    out["rheo_hi"] = hi
    return NAN_SENTINEL


def landmarks_an(spec, st, isi, auto_boot=True):
    """PASS 0 as a fixed-point iteration, exactly as v3_landmarks.landmarks.

    `auto_boot` is ON by default here, unlike the Izhikevich path: a foreign
    model's time scale is unknown by definition, so the spec's boot_t_win is a
    seed. The criterion is CONVERGENCE, not the cycle ratio -- a starved window
    under-measures T_cycle, so the ratio looks healthiest exactly when the window
    is worst (§50).
    """
    from v3_config import BOOT_MAX_RETRY, RHEO_MAX_EXPAND
    out = {"i_rh": NAN_SENTINEL, "T_cycle": NAN_SENTINEL, "L_inf": NAN_SENTINEL,
           "p_hat": 0, "capped": 0, "boot_t_win": float(spec.boot_t_win),
           "boot_retries": 0, "boot_cycles": 0.0, "rheo_hi": float(spec.rheo_hi)}
    bt, bw, bd = spec.boot_t_trans, spec.boot_t_win, spec.boot_t_disc
    tc = NAN_SENTINEL; p = 0; capped = 0; tc_prev = -1.0; n_sent = 0
    for attempt in range(BOOT_MAX_RETRY + 1):
        i_rh = _rheo_expand(spec, bt, bw, st, RHEO_MAX_EXPAND if auto_boot else 1, out)
        if i_rh < SENT_CUT:
            return out
        out["i_rh"] = i_rh
        tc, p, capped = t_cycle_an(spec, i_rh, bt, bw, bd, st, isi)
        out["boot_t_win"] = bw
        out["boot_cycles"] = bw / tc if tc > 0.0 else 0.0
        if capped or tc < SENT_CUT or tc <= 0.0:
            # SENTINEL anchor -> widen, don't conclude (see v3_landmarks)
            if (not auto_boot) or capped or attempt == BOOT_MAX_RETRY:
                out["capped"] = capped
                return out
            # 4x per retry and at most SENT_RETRY of them: 8x compounding over
            # the full budget is a 4096x window, which does not terminate in
            # useful time on a slow model. A cell that still has no settled cycle
            # after 16x almost certainly has none.
            if n_sent >= 2:
                out["capped"] = capped
                return out
            n_sent += 1
            bt, bw, bd = bt * 4.0, bw * 4.0, bd * 4.0
            out["boot_retries"] = attempt + 1
            tc_prev = -1.0
            continue
        if not auto_boot or attempt == BOOT_MAX_RETRY:
            break
        if tc_prev > 0.0 and abs(tc - tc_prev) <= 0.02 * tc_prev:
            break
        tc_prev = tc
        bw = K_WIN * tc
        bt = bd = K_TRANS * tc
        out["boot_retries"] = attempt + 1
    i_rh = rheobase_an(spec, K_TRANS * tc, K_WIN * tc, st)
    if i_rh < SENT_CUT:
        out["i_rh"] = NAN_SENTINEL
        return out
    out["i_rh"] = i_rh
    tc2, p2, cap2 = t_cycle_an(spec, i_rh, K_TRANS * tc, K_WIN * tc,
                               K_DISCARD * tc, st, isi)
    if cap2:
        out["capped"] = 1
        return out
    if tc2 > SENT_CUT and tc2 > 0.0:
        tc, p = tc2, p2
    out["T_cycle"], out["p_hat"] = tc, p
    l0 = l_inf_an(spec, i_rh, bw, K_TRANS * tc)
    if l0 > SENT_CUT:
        l1 = l_inf_an(spec, i_rh, LINF_SEARCH_K * l0, K_TRANS * tc)
        out["L_inf"] = l1 if l1 > SENT_CUT else l0
    return out


# ------------------------------------------------------------------- features
def compute_features_an(spec, buf_n=1 << 16):
    """The v3 18-column vector for a foreign model. Returns (features, info)."""
    st = np.empty(buf_n); isi = np.empty(buf_n)
    out = np.full(C.N_COLS, NAN_SENTINEL)
    lm = landmarks_an(spec, st, isi)
    info = dict(lm); info["gated"] = 0; info["reason"] = ""
    I = {n: i for i, n in enumerate(C.FEATURE_NAMES)}
    i_rh, tc, linf = lm["i_rh"], lm["T_cycle"], lm["L_inf"]

    if lm["capped"]:
        info["gated"] = 1; info["reason"] = "anchor on a bound record"
        return out, info
    if tc > SENT_CUT and tc < GATE_K_DT * spec.h:
        info["gated"] = 1; info["reason"] = "T_cycle < 10*h"
        return out, info

    tc_or = tc if tc > SENT_CUT else spec.boot_t_win / K_WIN
    g_tr, g_wn = K_TRANS * tc_or, K_WIN * tc_or
    c = count_an(spec.rhs, C.STIM_STEP, make_sp(g_tr, 0.0), 1.0, spec.y0, 0.0,
                 g_tr + g_wn, spec.h, spec.v_th, g_tr, g_tr + g_wn, spec.params)
    out[I["nu_spont"]] = NAN_SENTINEL if c < 0 else float(c)

    if i_rh < SENT_CUT or tc < SENT_CUT:
        info["reason"] = info["reason"] or "no rheobase / no anchor"
        return out, info

    trans, disc, win = K_TRANS*tc, K_DISCARD*tc, K_WIN*tc
    win_long, d_max = K_WIN_LONG*tc, K_DELTA_MAX*tc
    width, win_pulse = K_PULSE_WIDTH*tc, K_WIN_PULSE*tc
    ramp, hyper = K_RAMP*tc, K_HYPER*tc
    rh_ok = i_rh >= I_RH_MIN

    # ---- P1 ----
    sp = make_sp(trans, GAMMA_STEP * i_rh)
    ns, nt, inv = integrate_an(spec.rhs, C.STIM_STEP, sp, 1.0, spec.y0, 0.0,
                               trans + win, spec.h, spec.v_th, st, spec.params)
    if not inv and ns == nt and ns >= 2:
        t_ss = trans + disc
        n_tr = 0
        for k in range(1, ns):
            if st[k] < t_ss:
                isi[n_tr] = st[k] - st[k-1]; n_tr += 1
        out[I["p_tr"]] = float(isi_period(isi, n_tr, P_MAX, PERIOD_TOL)) if n_tr >= 1 else 0.0
        i0 = 0
        while i0 < ns and st[i0] < t_ss:
            i0 += 1
        n_isi = 0
        for k in range(i0 + 1, ns):
            isi[n_isi] = st[k] - st[k-1]; n_isi += 1
        for k in range(ns):
            if st[k] >= trans:
                out[I["ell"]] = (st[k] - trans) / tc
                break
        if n_isi >= 2:
            kk = min(n_isi, PERIOD_TAIL); off = n_isi - kk
            p = isi_period(isi[off:], kk, P_MAX, PERIOD_TOL)
            out[I["p"]] = float(p)
            if p > 0:
                out[I["rho"]] = burst_ratio(isi, off, kk, p)
            if p == 1:
                out[I["A"]] = adaptation_index(isi, n_isi)

    # ---- P2 ----
    fi = np.full(N_FI, NAN_SENTINEL)
    for j in range(N_FI):
        cn = count_an(spec.rhs, C.STIM_STEP, make_sp(trans, FI_G[j]*i_rh), 1.0,
                      spec.y0, 0.0, trans+win, spec.h, spec.v_th,
                      trans, trans+win, spec.params)
        fi[j] = NAN_SENTINEL if cn < 0 else cn / win
    f0, f3, f_lo, f_hi = fi[0], fi[3], fi[1], fi[5]
    if f0 > 0 and f3 > 0:
        out[I["beta"]] = (math.log(f3)-math.log(f0))/(math.log(0.20)-math.log(0.02))
    if f_hi > 0 and f_lo > 0:
        out[I["chi"]] = f_lo/f_hi
        out[I["lam"]] = f_hi/f_lo
    if f_hi > 0:
        out[I["fbar"]] = tc * f_hi

    # ---- P3 ----
    nf = count_an(spec.rhs, C.STIM_STEP, make_sp(trans, GAMMA_STEP*i_rh), 1.0,
                  spec.y0, 0.0, trans+win_long, spec.h, spec.v_th,
                  trans, trans+win_long, spec.params)
    nh = count_an(spec.rhs, C.STIM_STEP, make_sp(trans, GAMMA_STEP*i_rh), 1.0,
                  spec.y0, 0.0, trans+win_long, spec.h, spec.v_th,
                  trans, trans+win, spec.params)
    if nf >= 0 and nh > 0:
        out[I["Pi"]] = nf/nh

    # ---- P4 ----
    resp = np.full(DELTA_GRID_N, NAN_SENTINEL)
    step = d_max/(DELTA_GRID_N-1) if DELTA_GRID_N > 1 else d_max
    for di in range(DELTA_GRID_N):
        t2 = trans + di*step
        t_end = t2 + width + win_pulse
        cn = count_an(spec.rhs, C.STIM_PULSE_PAIR,
                      make_sp(trans, t2, width, GAMMA_PULSE*i_rh), 1.0,
                      spec.y0, 0.0, t_end, spec.h, spec.v_th, trans, t_end,
                      spec.params)
        resp[di] = NAN_SENTINEL if cn < 0 else float(cn)
    from v3_protocols import delta_star_rho_res
    ds, rr = delta_star_rho_res(resp, tc, d_max)
    out[I["delta_star"]], out[I["rho_res"]] = ds, rr

    # ---- P5 gamma_ramp ----
    if rh_ok:
        amp = RAMP_GAMMA_F * i_rh
        t_re = trans + ramp
        L = latency_an(spec.rhs, C.STIM_RAMP_HOLD, make_sp(trans, t_re, amp), 1.0,
                       spec.y0, 0.0, t_re + win, spec.h, spec.v_th, 0.0, spec.params)
        if L > 0.0:
            tf = L
            if tf <= trans:
                out[I["gamma_ramp"]] = 0.0
            elif tf >= t_re:
                out[I["gamma_ramp"]] = RAMP_GAMMA_F
            else:
                out[I["gamma_ramp"]] = (amp*(tf-trans)/ramp)/i_rh

    # ---- P6 ----
    if rh_ok:
        g_first, n_deep, ok = NAN_SENTINEL, NAN_SENTINEL, True
        for j in range(REBOUND_DEPTH_N):
            gh = GAMMA_HYPER_MAX*(j+1)/REBOUND_DEPTH_N
            t_off = trans + hyper
            cn = count_an(spec.rhs, C.STIM_HYPER_REL,
                          make_sp(trans, t_off, -gh*i_rh), 1.0, spec.y0, 0.0,
                          t_off+win, spec.h, spec.v_th, t_off, t_off+win, spec.params)
            if cn < 0:
                ok = False; break
            if cn > 0 and g_first < SENT_CUT:
                g_first = gh
            if j == REBOUND_DEPTH_N-1:
                n_deep = float(cn)
        if ok:
            out[I["N_reb"]], out[I["g_reb"]] = n_deep, g_first

    # ---- P7 eta ----
    if rh_ok:
        out[I["eta"]] = _eta_an(spec, i_rh, win_long, st)

    # ---- P8' A_rec ----
    if rh_ok and linf > SENT_CUT:
        out[I["A_rec"]] = _a_rec_an(spec, i_rh, linf, trans)

    return out, info


def _eta_an(spec, i_rh, t_total, st):
    """i_on = first UP-ramp spike, i_off = last DOWN-ramp spike; one-sided -> SENTINEL."""
    half = t_total * 0.5
    sp = make_sp(0.0, half, t_total, HYST_GAMMA_MAX * i_rh)
    ns, nt, inv = integrate_an(spec.rhs, C.STIM_HYST, sp, 1.0, spec.y0, 0.0,
                               t_total, spec.h, spec.v_th, st, spec.params)
    if inv or nt == 0:
        return NAN_SENTINEL
    i_on = i_off = NAN_SENTINEL
    for k in range(ns):
        t = st[k]
        frac = t/half if t <= half else (t_total - t)/half
        I = frac * HYST_GAMMA_MAX * i_rh
        if t < half:
            if i_on < SENT_CUT:
                i_on = I
        else:
            i_off = I
    if i_on < SENT_CUT or i_off < SENT_CUT:
        return NAN_SENTINEL
    return (i_on - i_off) / i_rh


def _a_rec_an(spec, i_rh, l_inf, t_cond):
    """Signed post-spike excitability change; grid entirely in L_inf units."""
    amp_c, amp_t = GAMMA_PULSE*i_rh, GAMMA_LINF*i_rh
    w_c = max(REC_W_COND*l_inf, 5.0*spec.h)
    t_sr = REC_T_SEARCH*l_inf

    def lat(ac, delta):
        t_on = t_cond + delta
        sp = make_sp(t_cond, w_c, ac, t_on, t_on + t_sr, amp_t)
        return latency_an(spec.rhs, C.STIM_TWO_PHASE, sp, 1.0, spec.y0, 0.0,
                          t_on + t_sr, spec.h, spec.v_th, t_on, spec.params)

    if lat(amp_c, 0.0) < 0.0 and lat(0.0, 0.0) < 0.0:
        return NAN_SENTINEL
    li = lat(0.0, DELAYS_REL[-1]*l_inf)
    if li <= 0.0:
        return NAN_SENTINEL
    tol = REC_TOL_DT * spec.h
    xs, ys, sgn = [], [], 1.0
    for k in range(N_DELAY_REC):
        dl = DELAYS_REL[k]*l_inf
        L = lat(amp_c, dl)
        if L <= 0.0:
            continue
        D = L - li
        ad = abs(D)
        if ad <= tol:
            continue
        if not xs:
            sgn = 1.0 if D >= 0 else -1.0
        xs.append(dl/li); ys.append(math.log(ad))
    if len(xs) < REC_MIN_PTS:
        return NAN_SENTINEL
    x = np.array(xs); y = np.array(ys)
    den = len(x)*np.sum(x*x) - np.sum(x)**2
    if den <= 0:
        return NAN_SENTINEL
    slope = (len(x)*np.sum(x*y) - np.sum(x)*np.sum(y))/den
    if slope >= 0:
        return NAN_SENTINEL
    icpt = (np.sum(y) - slope*np.sum(x))/len(x)
    return float(sgn*math.exp(icpt)/li)
