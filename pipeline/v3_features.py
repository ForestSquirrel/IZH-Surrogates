"""
v3_features.py -- assemble the 19-active / 32-column v3 vector.

Order is FEATURE_NAMES (v3_config). Columns 19..31 are ANONYMOUS reserve
(spec 4.6): inhibition is gone from v3 entirely -- protocols, columns and
reserved slots -- because carrying three permanently-sentinel columns to
preserve a migration path for an out-of-scope capability puts noise in the very
mask channel ss5 is trying to keep interpretable.

Sentinel discipline (spec 5.2): a non-firing outcome gates to SENTINEL, never to
a bound or a fake zero. Both v2 instances are closed here -- s_crit by deletion,
eta by explicit gating in p7_hysteresis.
"""
import numpy as np
from numba import njit

from v3_config import (
    N_COLS, N_ACTIVE, FEATURE_NAMES, NAN_SENTINEL, SENT_CUT, DT, I_RH_MIN,
    K_TRANS, K_DISCARD, K_WIN, K_WIN_LONG, K_DELTA_MAX, K_PULSE_WIDTH,
    K_RAMP, K_HYPER, K_WIN_PULSE, GATE_K_DT, N_DELAY_REC, REC_D_LO, REC_D_HI,
    N_FI, DELTA_GRID_N, SPIKE_BUDGET_PER_CYCLE, SPIKE_BUFFER_MIN,
)
from v3_landmarks import landmarks
from v3_protocols import (
    p1_reference, p2_fi, beta_chi, p3_persistence, p4_pulse_pair,
    delta_star_rho_res,
    p5_gamma_ramp, p6_rebound, p7_hysteresis, p8_a_rec,
    p10_spontaneous,
)

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
DELAYS_REL = np.geomspace(REC_D_LO, REC_D_HI, N_DELAY_REC)


class Geometry:
    """Per-cell absolute durations, derived once from the landmarks.

    This class is the whole point of B7: it is the ONLY place a multiple becomes
    a duration, so 'no protocol contains an absolute time constant' is checkable
    by reading one object rather than auditing every kernel.
    """
    __slots__ = ("tc", "linf", "trans", "disc", "win", "win_long",
                 "d_max", "width", "ramp", "hyper", "win_pulse")

    def __init__(self, t_cycle, l_inf):
        self.tc = t_cycle
        self.linf = l_inf
        self.trans = K_TRANS * t_cycle
        self.disc = K_DISCARD * t_cycle
        self.win = K_WIN * t_cycle
        self.win_long = K_WIN_LONG * t_cycle
        self.d_max = K_DELTA_MAX * t_cycle
        self.width = K_PULSE_WIDTH * t_cycle
        self.ramp = K_RAMP * t_cycle
        self.hyper = K_HYPER * t_cycle
        self.win_pulse = K_WIN_PULSE * t_cycle


def buffer_size_for(t_cycle):
    """Per-cycle budget (spec 3.1) rather than a fixed MAX_SPIKES."""
    n = int(SPIKE_BUDGET_PER_CYCLE * K_WIN) + 64
    return max(n, SPIKE_BUFFER_MIN)


def compute_features_v3(rhs, params, bufs, out=None, lm=None,
                        rheo_hi=None, auto_boot=False):
    """One cell, PASS 0 + PASS 1. Returns (out, info).

    info carries the landmarks and the gate decision so the build can record why
    a cell was dropped rather than only that it was.
    """
    if out is None:
        out = np.full(N_COLS, NAN_SENTINEL, dtype=np.float64)
    else:
        out[:] = NAN_SENTINEL

    if lm is None:
        # rheo_hi is a RUNTIME argument, never a folded constant: v3's 200 is
        # tuned to Izhikevich, and a foreign reset model has its own current
        # units (AdEx's bursting regimes need ~300 pA and were reported as
        # "no rheobase" at 200). Default None keeps the atlas path untouched.
        kw = {"auto_boot": bool(auto_boot)}
        if rheo_hi is not None:
            kw["rheo_hi"] = float(rheo_hi)
        lm = landmarks(rhs, params, bufs, **kw)
    info = dict(lm)
    info["gated"] = 0
    info["reason"] = ""

    i_rh = lm["i_rh"]
    tc = lm["T_cycle"]
    linf = lm["L_inf"]

    # ---- B5: resolution gate, on a CAP-FREE screen (ss35.2) -----------------
    # ss35.2 is the reason `capped` is checked first: at the shipped buffer a
    # T_cycle < 5*DT screen finds 0 where 0.27% exists, because a bounded record
    # measures the anchor off transient ISIs that are too long -- the screen
    # pushes its own targets out of range.
    if lm["capped"]:
        info["gated"] = 1; info["reason"] = "anchor measured on a bound record"
        return out, info
    if tc > SENT_CUT and tc < GATE_K_DT * DT:
        info["gated"] = 1; info["reason"] = "T_cycle < 10*DT"
        return out, info

    # nu_spont is not gated on i_rh: a cell with no rheobase can still fire
    # spontaneously, and that is what the column exists to carry.
    boot_trans = K_TRANS * (tc if tc > SENT_CUT else 50.0)
    boot_win = K_WIN * (tc if tc > SENT_CUT else 50.0)
    out[IDX["nu_spont"]] = p10_spontaneous(rhs, params, boot_trans, boot_win)

    if i_rh < SENT_CUT or tc < SENT_CUT:
        info["reason"] = info["reason"] or "no rheobase / no anchor"
        return out, info

    # i_rh below I_RH_MIN is a MEASUREMENT (the cell fires at ~0 drive), not a
    # failure. It gates the rheobase-NORMALIZED columns only (spec 5.1); the
    # rest are computed as usual, at an amplitude of ~0.
    rh_ok = i_rh >= I_RH_MIN

    g = Geometry(tc, linf)

    # ---- P1 ---------------------------------------------------------------
    p, rho, p_tr, A, ell, mean_isi, _unused, n_isi, ok = p1_reference(
        rhs, params, i_rh, g.trans, g.win, g.disc,
        bufs["spike_times"], bufs["isi"])
    out[IDX["p"]] = p
    out[IDX["rho"]] = rho
    out[IDX["p_tr"]] = p_tr
    out[IDX["A"]] = A
    out[IDX["ell"]] = ell / tc if ell > SENT_CUT else NAN_SENTINEL
    info["p1_ok"] = ok
    info["mean_isi"] = mean_isi

    # ---- P2 ---------------------------------------------------------------
    p2_fi(rhs, params, i_rh, g.trans, g.win, bufs["fi"])
    beta, chi, lam = beta_chi(bufs["fi"], tc)
    out[IDX["beta"]] = beta
    out[IDX["chi"]] = chi
    out[IDX["lam"]] = lam

    # fbar = anchor * rate(gamma=2). v2 used t_ref; v3's anchor is T_cycle (B7).
    # It reads the P2 rate, not a P1 count -- same protocol amplitude, but this
    # is the definition that is on disk and the one the comparison is against.
    r2 = bufs["fi"][5]
    out[IDX["fbar"]] = tc * r2 if (r2 > 0.0 and r2 > SENT_CUT) else NAN_SENTINEL

    # ---- P3 -- Pi is a persistence RATIO, at a doubled window ----
    out[IDX["Pi"]] = p3_persistence(rhs, params, i_rh, g.trans, g.win,
                                    g.win_long)

    # ---- P4 (B2: both columns read the uncapped count) ---------------------
    p4_pulse_pair(rhs, params, i_rh, tc, g.trans, g.win_pulse, g.d_max,
                  g.width, bufs["resp"])
    d_star, rho_res = delta_star_rho_res(bufs["resp"], tc, g.d_max)
    out[IDX["delta_star"]] = d_star
    out[IDX["rho_res"]] = rho_res

    # ---- P5 ---------------------------------------------------------------
    out[IDX["gamma_ramp"]] = p5_gamma_ramp(rhs, params, i_rh, g.trans, g.ramp,
                                           g.win)

    # ---- P6 ---------------------------------------------------------------
    n_reb, g_reb = p6_rebound(rhs, params, i_rh, g.trans, g.hyper, g.win,
                              bufs["spike_times"])
    out[IDX["N_reb"]] = n_reb
    out[IDX["g_reb"]] = g_reb

    # ---- P7 ---------------------------------------------------------------
    # v2 runs the triangle over 2*T_WIN with no pre-settle; v3 keeps the shape
    # and makes the duration cell-relative (spec 3.2).
    out[IDX["eta"]] = p7_hysteresis(rhs, params, i_rh, g.win_long)

    # ---- P8' --------------------------------------------------------------
    out[IDX["A_rec"]] = p8_a_rec(rhs, params, i_rh, linf, g.trans, DELAYS_REL)

    return out, info


def mask_of(vec):
    """Definedness mask over the ACTIVE columns only."""
    return (np.asarray(vec)[:N_ACTIVE] > SENT_CUT).astype(np.int8)
