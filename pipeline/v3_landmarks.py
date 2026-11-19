"""
v3_landmarks.py -- i_rh, T_cycle, L_inf.  PASS 0 of the build.

These three are measured on ABSOLUTE bootstrap windows and are the only place
in v3 where an absolute duration appears. Everything downstream is expressed as
a multiple of what this file returns.

i_rh    scan-then-bisect (spec 2). v2 bisected alone, which assumes firing is
        MONOTONE in I; ss's rheobase-bisection note records that Class-2 cells
        with a bounded firing BAND silently report "no rheobase" under that
        assumption. The coarse scan finds a firing bracket first, so a band is
        bracketed rather than missed, and bisection then runs inside it.

T_cycle the universal anchor, replacing t_ref (B7). Phase-invariant: for a
        period-p cell the characteristic cycle is p times the mean settled ISI,
        so T_cycle is the same quantity whether the cell fires singlets or
        p-tuplets. t_ref (mean of last <=5 ISIs) is NOT phase-invariant and is
        demoted to a diagnostic.

        Read CAP-FREE (B3). ss33.2 raised t_ref contamination from <=1% to 91%
        of cells once the reference was genuinely uncapped, which is why the
        anchor cannot be measured off a bounded record.

L_inf   unconditioned first-spike latency under a sustained step at 1.2*i_rh.
        Sets P8''s sampling geometry (ss39.1). Median 15.4 ms on the v2 cohort.
"""
import numpy as np
from numba import njit

from v3_config import (
    STATE_DIM, V_REST, DT, NAN_SENTINEL, SENT_CUT, I_RH_MIN,
    I_RHEO_GUESS_LO, I_RHEO_GUESS_HI, RHEO_SCAN_N, RHEO_BISECT_ITERS,
    N_MIN_SPIKES, BOOT_T_TRANS, BOOT_T_WIN, BOOT_T_DISC,
    P_MAX, PERIOD_TOL, PERIOD_TAIL, GAMMA_LINF, LINF_SEARCH_K,
    STIM_STEP, STIM_PARAMS_LEN,
)
from v3_core import integrate_v3, count_in_window, _rk4


# ------------------------------------------------------------------ rheobase
@njit(cache=True)
def _fires_at(rhs, params, amp, spike_times, t_trans, t_win):
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = t_trans
    sp[1] = amp
    y = np.empty(STATE_DIM)
    y[0] = V_REST
    y[1] = params[1] * V_REST
    t_end = t_trans + t_win
    ns, nt, inv = integrate_v3(rhs, STIM_STEP, sp, 1.0, y, 0.0, t_end, DT,
                               spike_times, params[2], params[3], params)
    if inv:
        return -1
    # count in the steady window only -- an onset transient is not firing
    c = count_in_window(spike_times, ns, t_trans, t_end)
    if ns < nt:                      # buffer bound: certainly enough spikes
        return 1
    return 1 if c >= N_MIN_SPIKES else 0


@njit(cache=True)
def rheobase_v3(rhs, params, spike_times, t_trans, t_win,
                rheo_hi=I_RHEO_GUESS_HI):
    """Coarse scan for a firing bracket, then bisect inside it.

    The WINDOW IS AN ARGUMENT, and that is load-bearing (spec 3.2: "P0 ...
    windows cell-relative"). i_rh means "fires >= N_MIN_SPIKES in this window",
    so if P0 is measured on an absolute 500 ms window while the features observe
    K_WIN cycles, the two disagree about what threshold means. That mismatch
    showed up as beta defined on 41% of cells against v2's 100%: at gamma = 1.02
    a cell that fires 3 times in 500 ms fires 0 times in 68 ms.
    """
    lo = I_RHEO_GUESS_LO
    hi = rheo_hi

    # log-ish scan: find the FIRST amplitude that fires, keeping the last
    # non-firing amplitude below it. A bounded firing band is caught here.
    prev = lo
    found = 0
    for i in range(1, RHEO_SCAN_N + 1):
        amp = rheo_hi * (float(i) / RHEO_SCAN_N) ** 3
        f = _fires_at(rhs, params, amp, spike_times, t_trans, t_win)
        if f == 1:
            lo = prev
            hi = amp
            found = 1
            break
        prev = amp
    if found == 0:
        return NAN_SENTINEL          # caller expands the ceiling (rheobase_auto)

    for _ in range(RHEO_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        f = _fires_at(rhs, params, mid, spike_times, t_trans, t_win)
        if f == 1:
            hi = mid
        else:
            lo = mid
    # NO I_RH_MIN GATE HERE. A cell that fires at essentially zero current has a
    # rheobase of ~0; it does not have "no rheobase", and the two are different
    # facts. Gating here collapsed them and silently dropped every spontaneously
    # firing cell (the smoke's second failure mode). The gate belongs in the
    # feature layer, where spec 5.1 puts it -- and where nu_spont is explicitly
    # exempt from it, which is the whole point of keeping the distinction.
    return hi


# ------------------------------------------------------------------- period
@njit(cache=True)
def isi_period(isi, n_isi, p_max, tol):
    """Smallest p such that the settled ISI sequence repeats with period p."""
    if n_isi < 2:
        return 0
    for p in range(1, p_max + 1):
        if n_isi < 2 * p:
            break
        ok = True
        for i in range(n_isi - p):
            a = isi[i]; b = isi[i + p]
            m = 0.5 * (a + b)
            if m <= 0.0:
                ok = False
                break
            d = a - b
            if d < 0.0:
                d = -d
            if d / m > tol:
                ok = False
                break
        if ok:
            return p
    return 0


# ------------------------------------------------------------------ T_cycle
@njit(cache=True)
def t_cycle_win(rhs, params, i_rh, spike_times, isi_buf, gamma,
                t_trans, t_win, t_disc):
    """(T_cycle, p_hat, n_settled, capped) on an EXPLICIT window.

    `capped` is 1 if the buffer bound the record, in which case the anchor is
    NOT trustworthy (B3) and the caller must widen the buffer and retry rather
    than use the value.
    """
    # NO I_RH_MIN GATE. spec 5.1 gates only eta, N_reb, g_reb and gamma_ramp on
    # the rheobase; p, rho, A, beta, chi, ell, fbar and Pi are NOT gated on it.
    # A spontaneously firing cell has i_rh ~ 0, fires at ~0 drive, and therefore
    # HAS a characteristic cycle -- v2 measured one for it. Gating the anchor
    # here propagated to every column and cost ~24% of the atlas, which showed up
    # as v3 definedness 75.9% against v2's 100%.
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = t_trans
    sp[1] = gamma * i_rh
    y = np.empty(STATE_DIM)
    y[0] = V_REST
    y[1] = params[1] * V_REST
    t_end = t_trans + t_win
    ns, nt, inv = integrate_v3(rhs, STIM_STEP, sp, 1.0, y, 0.0, t_end, DT,
                               spike_times, params[2], params[3], params)
    if inv:
        return NAN_SENTINEL, 0, 0, 0
    if ns < nt:
        return NAN_SENTINEL, 0, 0, 1          # capped -> caller must retry

    t_ss = t_trans + t_disc
    i0 = 0
    while i0 < ns and spike_times[i0] < t_ss:
        i0 += 1
    n_isi = 0
    for i in range(i0 + 1, ns):
        isi_buf[n_isi] = spike_times[i] - spike_times[i - 1]
        n_isi += 1
    if n_isi < 2:
        return NAN_SENTINEL, 0, n_isi, 0

    k = n_isi if n_isi < PERIOD_TAIL else PERIOD_TAIL
    off = n_isi - k
    p = isi_period(isi_buf[off:], k, P_MAX, PERIOD_TOL)

    # phase-invariant cycle: p * mean settled ISI. For p == 0 (aperiodic) the
    # mean ISI is still the natural characteristic time, so the estimator is
    # smooth across the periodic/aperiodic boundary rather than falling back.
    tot = 0.0
    for i in range(off, n_isi):
        tot += isi_buf[i]
    mean_isi = tot / k
    pf = float(p) if p > 0 else 1.0
    return pf * mean_isi, p, n_isi, 0


# -------------------------------------------------------------------- L_inf
@njit(cache=True)
def l_inf_v3(rhs, params, i_rh, t_search):
    """First-spike latency under a sustained step at GAMMA_LINF*i_rh.

    No buffer: the first crossing is returned directly, so this is uncapped by
    construction.
    """
    if i_rh < I_RH_MIN:
        return NAN_SENTINEL
    k1 = np.empty(STATE_DIM); k2 = np.empty(STATE_DIM)
    k3 = np.empty(STATE_DIM); k4 = np.empty(STATE_DIM)
    ytmp = np.empty(STATE_DIM); ynxt = np.empty(STATE_DIM)
    y = np.empty(STATE_DIM)
    y[0] = V_REST
    y[1] = params[1] * V_REST
    amp = GAMMA_LINF * i_rh
    t_on = BOOT_T_TRANS
    t_end = t_on + t_search
    t = 0.0
    n_steps = int(t_end / DT) + 2
    for _ in range(n_steps):
        if t >= t_end:
            break
        I = amp if t >= t_on else 0.0
        v_prev = y[0]
        _rk4(rhs, t, y, params, I, DT, ynxt, k1, k2, k3, k4, ytmp)
        for j in range(STATE_DIM):
            y[j] = ynxt[j]
        if y[0] > 1.0e6 or y[0] < -1.0e6 or y[0] != y[0]:
            return NAN_SENTINEL
        if v_prev < 30.0 and y[0] >= 30.0:
            if t >= t_on:
                return t - t_on
            y[0] = params[2]
            y[1] = y[1] + params[3]
        t += DT
    return NAN_SENTINEL


@njit(cache=True)
def rheobase_seeded(rhs, params, spike_times, t_trans, t_win, seed,
                    rheo_hi=I_RHEO_GUESS_HI):
    """Round-2 rheobase, bracketed around round 1's answer instead of rescanned.

    THE FALLBACK IS THE WHOLE SAFETY ARGUMENT. A seeded search that cannot
    bracket must run the full scan and never report "no rheobase", because
    downstream that is indistinguishable from a cell that genuinely never fires.
    Without it this silently lost 12.6% of cells -- the ones whose round-2
    rheobase sits far above the round-1 seed, where the geometric walk runs out
    of range before reaching it (ss43).
    """
    if seed > 0.0:
        if _fires_at(rhs, params, seed, spike_times, t_trans, t_win) == 1:
            lo = seed
            hi = seed
            ok = 0
            for _ in range(20):
                lo = lo * 0.5
                if lo < 1.0e-9:
                    lo = 0.0
                    ok = 1
                    break
                if _fires_at(rhs, params, lo, spike_times, t_trans, t_win) != 1:
                    ok = 1
                    break
            if ok == 1:
                for _ in range(RHEO_BISECT_ITERS):
                    mid = 0.5 * (lo + hi)
                    if _fires_at(rhs, params, mid, spike_times, t_trans, t_win) == 1:
                        hi = mid
                    else:
                        lo = mid
                return hi
        else:
            hi = seed
            ok = 0
            for _ in range(20):
                hi = hi * 4.0
                if hi > rheo_hi:
                    break
                if _fires_at(rhs, params, hi, spike_times, t_trans, t_win) == 1:
                    ok = 1
                    break
            if ok == 1:
                lo = hi * 0.25
                for _ in range(RHEO_BISECT_ITERS):
                    mid = 0.5 * (lo + hi)
                    if _fires_at(rhs, params, mid, spike_times, t_trans, t_win) == 1:
                        hi = mid
                    else:
                        lo = mid
                return hi
    return rheobase_v3(rhs, params, spike_times, t_trans, t_win, rheo_hi)


def rheobase_auto(rhs, params, spike_times, t_trans, t_win, rheo_hi, n_expand):
    """Scan-then-bisect, DOUBLING the ceiling while nothing fires.

    The bracket failure is loud (SENTINEL), so it can simply be retried: a
    ceiling that is too low is indistinguishable from a cell that never fires,
    and the two are told apart by widening. Bounded, so a genuinely silent cell
    still terminates.
    """
    hi = float(rheo_hi)
    for _ in range(max(1, n_expand)):
        v = rheobase_v3(rhs, params, spike_times, t_trans, t_win, hi)
        if v > SENT_CUT:
            return v, hi
        hi *= 2.0
    return NAN_SENTINEL, hi


def landmarks(rhs, params, bufs, gamma_tc=2.0, rheo_hi=I_RHEO_GUESS_HI,
              auto_boot=False):
    """PASS 0 for one cell, as a fixed-point iteration.

    i_rh and T_cycle are mutually dependent: the rheobase is "fires >=
    N_MIN_SPIKES in a window", and in v3 the window is a multiple of T_cycle,
    which is itself measured at 2*i_rh. So PASS 0 iterates:

        i_rh(absolute bootstrap) -> T_cycle(bootstrap)
                                 -> i_rh(cell-relative) -> T_cycle(cell-relative)

    Two rounds is enough -- the bootstrap only has to land in the right decade
    for the second round's window to be the right shape. The alternative,
    measuring P0 on an absolute window and everything else on a cell-relative
    one, leaves the two disagreeing about what threshold means.

    SENTINEL fields mean 'undefined', never a bound or a fake zero (spec 5.2).

    `auto_boot` defaults to FALSE, and the asymmetry is deliberate (§50):

      * OFF for the atlas path, which preserves §7.2's contract exactly -- the
        shipped 10.25M atlas was built without it. It would barely matter
        anyway: the median atlas cell has 516 bootstrap cycles and only 0.20%
        have fewer than 3, which is the same order as B5's 0.82% gate. Those
        0.20% are a known, quantified limitation of the shipped atlas rather
        than a silent one, and `boot_cycles` is recomputable from the landmark
        sidecar for any of them.
      * ON for foreign targets, where the time scale is unknown by definition
        and a fixed 500-unit seed has no reason to be right.
    """
    from v3_config import (K_TRANS, K_WIN, K_DISCARD, BOOT_MIN_CYCLES,
                           BOOT_MAX_RETRY, RHEO_MAX_EXPAND)
    st = bufs["spike_times"]
    out = {"i_rh": NAN_SENTINEL, "T_cycle": NAN_SENTINEL, "p_hat": 0,
           "L_inf": NAN_SENTINEL, "capped": 0, "n_isi": 0,
           "T_cycle_prov": NAN_SENTINEL, "i_rh_prov": NAN_SENTINEL,
           "boot_cycles": 0.0, "boot_t_win": float(BOOT_T_WIN),
           "boot_retries": 0, "rheo_hi": float(rheo_hi)}

    # ---- round 1: bootstrap, purely to get the scale ----------------------
    # The bootstrap is a SEED, not a requirement (§50). Both failure modes are
    # detectable -- an unbracketed rheobase is SENTINEL, and a starved window
    # shows as boot_cycles = BOOT_T_WIN / T_cycle < 1 (§49.3: 0.98 where the
    # answer was exact, 0.22 where it was 2.2x wrong). So widen and re-measure
    # rather than require the caller to guess correctly.
    # boot_cycles = BOOT_T_WIN / T_cycle_prov CANNOT be the criterion: a starved
    # window UNDER-measures T_cycle, so the ratio looks healthiest exactly when
    # the window is worst (at tau=100 it read 5.03 while the anchor was 2.2x
    # wrong). The diagnostic is corrupted by the failure it is meant to detect --
    # §6's rule, and the reason this is a CONVERGENCE test instead.
    #
    # Fixed point: run PASS 0, then re-seed the bootstrap from the anchor it
    # produced and run again. A bootstrap that was long enough returns the same
    # anchor; one that was starved returns a different one, and the iteration
    # walks out to the right scale.
    bt, bw, bd = BOOT_T_TRANS, BOOT_T_WIN, BOOT_T_DISC
    n_exp = RHEO_MAX_EXPAND if auto_boot else 1
    tc = NAN_SENTINEL; p = 0; n_isi = 0; capped = 0
    tc_prev = -1.0; n_sent = 0
    for attempt in range(BOOT_MAX_RETRY + 1):
        i_rh, hi_used = rheobase_auto(rhs, params, st, bt, bw, rheo_hi, n_exp)
        out["rheo_hi"] = hi_used
        out["i_rh_prov"] = i_rh
        if i_rh < SENT_CUT:
            return out
        out["i_rh"] = i_rh
        tc, p, n_isi, capped = t_cycle_win(rhs, params, i_rh, st, bufs["isi"],
                                           gamma_tc, bt, bw, bd)
        out["T_cycle_prov"] = tc
        out["boot_t_win"] = bw
        out["boot_cycles"] = bw / tc if tc > 0.0 else 0.0
        if capped or tc < SENT_CUT or tc <= 0.0:
            # A SENTINEL anchor is itself a widening signal: a window too short to
            # hold 2 settled ISIs reports "no anchor", which is indistinguishable
            # from a cell that genuinely has none. Widen before concluding.
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
        # converged when re-seeding does not move the anchor
        if tc_prev > 0.0 and abs(tc - tc_prev) <= 0.02 * tc_prev:
            break
        tc_prev = tc
        bw = K_WIN * tc          # same size as round 2's own window
        bt = bd = K_TRANS * tc
        out["boot_retries"] = attempt + 1

    # ---- round 2: everything on the cell's own scale ----
    from v3_config import RHEO_SEED_ROUND2
    if RHEO_SEED_ROUND2:
        i_rh = rheobase_seeded(rhs, params, st, K_TRANS*tc, K_WIN*tc, i_rh,
                               out['rheo_hi'])
    else:
        i_rh = rheobase_v3(rhs, params, st, K_TRANS*tc, K_WIN*tc, out['rheo_hi'])
    if i_rh < SENT_CUT:
        out["i_rh"] = NAN_SENTINEL
        return out
    out["i_rh"] = i_rh

    tc2, p2, n_isi2, capped2 = t_cycle_win(
        rhs, params, i_rh, st, bufs["isi"], gamma_tc,
        K_TRANS * tc, K_WIN * tc, K_DISCARD * tc)
    if capped2:
        out["capped"] = 1
        return out
    if tc2 > SENT_CUT and tc2 > 0.0:
        tc, p, n_isi = tc2, p2, n_isi2
    out["T_cycle"] = tc
    out["p_hat"] = p
    out["n_isi"] = n_isi

    # L_inf sets its own scale, so it is calibrated the same way: a generous
    # absolute first pass, then the cell-relative window.
    l0 = l_inf_v3(rhs, params, i_rh, out['boot_t_win'])
    if l0 > SENT_CUT and l0 > 0.0:
        l1 = l_inf_v3(rhs, params, i_rh, LINF_SEARCH_K * l0)
        out["L_inf"] = l1 if (l1 > SENT_CUT and l1 > 0.0) else l0
    return out
