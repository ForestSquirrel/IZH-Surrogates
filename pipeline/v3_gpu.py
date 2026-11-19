"""
v3_gpu.py -- CUDA path for the v3 atlas build.

Mirrors v3_protocols.py / v3_landmarks.py function for function, so the two are
auditable against each other. Three things necessarily differ, and all three are
forced by the device:

1. THE RHS IS INLINED. Device code cannot portably take a function pointer, so
   Izhikevich is written directly into the RK4 stage instead of passed as a
   first-class njit callable. This is v2's arrangement too.

2. BUFFERS ARE COMPILE-TIME SIZED. `cuda.local.array` needs a constant extent, so
   v3's runtime-sized buffer becomes GPU_MAX_SPIKES here.

   This costs far less than it did in v2, and the reason is B1. v2 needed
   MAX_SPIKES-length local arrays because every count came from a stored train;
   v3 takes counts from uncapped accumulators that never touch a buffer, so the
   ONLY consumer of stored spike times is P1's ISI sequence. The buffer has to
   hold one reference train, not a worst case over ten protocols.

   The cap-free contract is preserved exactly: a bound record still reports
   n_stored < n_total and every time-derived value gates to SENTINEL rather than
   being computed from a truncated train.

3. PRECISION IS SELECTABLE. IZH_V3_GPU_ARCH = f32 (default) or f64. On a GeForce
   part FP64 runs at a small fraction of FP32, so f32 is the build arch and f64
   exists for the verification arm.

WARP BINNING IS MANDATORY, NOT AN OPTIMIZATION. Every window in v3 is a multiple
of the cell's own T_cycle, so two cells in the same warp can differ by orders of
magnitude in step count and the warp runs at its slowest lane. Cells must be
SORTED BY PROVISIONAL SCALE before the feature kernel. ss6.4 measured 6.3x for
random ordering in v2, where only some windows were cell-relative; v3 is fully
cell-relative, so the exposure is larger, not smaller. `build_features` does the
sort; `bench_binning` measures what it buys.
"""
import os
import math

import numpy as np
from numba import cuda, float32, float64, int32

import v3_config as C

GPU_ARCH = os.environ.get("IZH_V3_GPU_ARCH", "f32")
if GPU_ARCH not in ("f32", "f64"):
    raise ValueError(f"IZH_V3_GPU_ARCH must be f32 or f64, got {GPU_ARCH!r}")
REAL = float32 if GPU_ARCH == "f32" else float64
NP_REAL = np.float32 if GPU_ARCH == "f32" else np.float64

# 4096, matching the CPU path. Measured on 512 atlas cells, spikes produced in
# the ABSOLUTE bootstrap window (500 ms at 2*i_rh): p50 510, p90 1997, p99 3554,
# max 7893. At 1024 the buffer bound 30.5% of cells, the anchor reported
# `capped`, and T_cycle definedness fell to 71% against the CPU's 98.4%; at 4096
# it is 0.9% and 99.1%. Costs 5% wall clock.
#
# NOTE: this constant is folded into cuda.local.array, so changing it REQUIRES a
# fresh NUMBA_CACHE_DIR. Adding cache=True to the kernels without one served a
# stale kernel and made a 256-vs-1024 comparison come back byte-identical --
# ss12/ss20's failure mode, reproduced here in the GPU path.
GPU_MAX_SPIKES = int(os.environ.get("IZH_V3_GPU_MAX_SPIKES", 4096))

# --- constants lifted to module scope so the device code closes over floats ---
DT = C.DT
V_TH = C.V_TH
V_REST = C.V_REST
DIVERGE = C.DIVERGE_BOUND
SENT = C.NAN_SENTINEL
SENT_CUT = C.SENT_CUT
I_RH_MIN = C.I_RH_MIN
N_COLS = C.N_COLS
N_ACTIVE = C.N_ACTIVE
N_FI = C.N_FI
DELTA_GRID_N = C.DELTA_GRID_N
P_MAX = C.P_MAX
PERIOD_TOL = C.PERIOD_TOL
PERIOD_TAIL = C.PERIOD_TAIL
BOOT_T_TRANS = C.BOOT_T_TRANS
BOOT_T_WIN = C.BOOT_T_WIN
BOOT_T_DISC = C.BOOT_T_DISC
RHEO_SCAN_N = C.RHEO_SCAN_N
RHEO_BISECT_ITERS = C.RHEO_BISECT_ITERS
I_RHEO_HI = C.I_RHEO_GUESS_HI
N_MIN_SPIKES = C.N_MIN_SPIKES
K_TRANS = C.K_TRANS
K_DISCARD = C.K_DISCARD
K_WIN = C.K_WIN
K_WIN_LONG = C.K_WIN_LONG
K_DELTA_MAX = C.K_DELTA_MAX
K_PULSE_WIDTH = C.K_PULSE_WIDTH
K_WIN_PULSE = C.K_WIN_PULSE
K_RAMP = C.K_RAMP
K_HYPER = C.K_HYPER
GAMMA_PULSE = C.GAMMA_PULSE
GAMMA_STEP = C.GAMMA_STEP
RAMP_GAMMA_F = C.RAMP_GAMMA_F
GAMMA_HYPER_MAX = C.GAMMA_HYPER_MAX
REBOUND_DEPTH_N = C.REBOUND_DEPTH_N
HYST_GAMMA_MAX = C.HYST_GAMMA_MAX
GAMMA_LINF = C.GAMMA_LINF
LINF_SEARCH_K = C.LINF_SEARCH_K
GATE_K_DT = C.GATE_K_DT
N_DELAY_REC = C.N_DELAY_REC
REC_W_COND = C.REC_W_COND
REC_T_SEARCH = C.REC_T_SEARCH
REC_MIN_PTS = C.REC_MIN_PTS
REC_TOL_DT = C.REC_TOL_DT

FI_G = np.array([1.02, 1.05, 1.10, 1.20, 1.50, 2.00, 3.00, 4.00], dtype=np.float64)
DELAYS_REL = np.geomspace(C.REC_D_LO, C.REC_D_HI, C.N_DELAY_REC)

IDX = {n: i for i, n in enumerate(C.FEATURE_NAMES)}
I_p = IDX["p"]; I_rho = IDX["rho"]; I_ptr = IDX["p_tr"]; I_A = IDX["A"]
I_beta = IDX["beta"]; I_chi = IDX["chi"]; I_ds = IDX["delta_star"]
I_rr = IDX["rho_res"]; I_ell = IDX["ell"]; I_lam = IDX["lam"]; I_Pi = IDX["Pi"]
I_fbar = IDX["fbar"]; I_nreb = IDX["N_reb"]; I_arec = IDX["A_rec"]
I_eta = IDX["eta"]; I_gr = IDX["gamma_ramp"]; I_greb = IDX["g_reb"]
I_nu = IDX["nu_spont"]


# ==================================================================== stimuli
@cuda.jit(device=True, inline=True)
def _stim(sid, t, s0, s1, s2, s3, s4, s5):
    """Same six shapes as v3_core.stim_v3, flattened to scalars.

    Scalars rather than an array: a local array here would be per-thread global
    memory for something that lives in registers.
    """
    if sid == 0:                       # STEP        [t_on, amp]
        return s1 if t >= s0 else 0.0
    elif sid == 1:                     # RAMP+HOLD   [t_on, t_ramp_end, amp]
        if t < s0:
            return 0.0
        if t >= s1:
            return s2
        return s2 * (t - s0) / (s1 - s0)
    elif sid == 2:                     # PULSE PAIR  [t1, t2, width, amp]
        if s0 <= t < s0 + s2:
            return s3
        if s1 <= t < s1 + s2:
            return s3
        return 0.0
    elif sid == 3:                     # HYPER REL   [t_on, t_off, amp]
        return s2 if s0 <= t < s1 else 0.0
    elif sid == 4:                     # HYST        [t_on, t_peak, t_end, amp]
        if t < s0 or t >= s2:
            return 0.0
        if t < s1:
            return s3 * (t - s0) / (s1 - s0)
        return s3 * (s2 - t) / (s2 - s1)
    elif sid == 5:                     # TWO PHASE
        if s2 != 0.0 and s0 <= t < s0 + s1:
            return s2
        if s3 <= t < s4:
            return s5
        return 0.0
    return 0.0


# ======================================================= integrate + collect
@cuda.jit(device=True)
def _integrate(a, b, c, d, sid, s0, s1, s2, s3, s4, s5,
               t0, t1, spike_times):
    """RK4 with the Izhikevich RHS inlined. Returns (n_stored, n_total, invalid).

    n_total is UNCAPPED (B1). A caller that needs TIMES must check
    n_stored == n_total; a caller that needs a COUNT never has to.
    """
    v = V_REST
    u = b * V_REST
    t = t0
    n_stored = 0
    n_total = 0
    n_steps = int((t1 - t0) / DT) + 2
    h2 = DT * 0.5
    for _ in range(n_steps):
        if t >= t1:
            break
        v_prev = v

        I1 = _stim(sid, t, s0, s1, s2, s3, s4, s5)
        k1v = 0.04 * v * v + 5.0 * v + 140.0 - u + I1
        k1u = a * (b * v - u)

        vt = v + h2 * k1v
        ut = u + h2 * k1u
        I2 = _stim(sid, t + h2, s0, s1, s2, s3, s4, s5)
        k2v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I2
        k2u = a * (b * vt - ut)

        vt = v + h2 * k2v
        ut = u + h2 * k2u
        k3v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I2
        k3u = a * (b * vt - ut)

        vt = v + DT * k3v
        ut = u + DT * k3u
        I4 = _stim(sid, t + DT, s0, s1, s2, s3, s4, s5)
        k4v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I4
        k4u = a * (b * vt - ut)

        v = v + (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        u = u + (DT / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        t += DT

        if not (v == v) or v > DIVERGE or v < -DIVERGE:
            return n_stored, n_total, 1
        if v_prev < V_TH and v >= V_TH:
            n_total += 1
            if n_stored < GPU_MAX_SPIKES:
                spike_times[n_stored] = t
                n_stored += 1
            v = c
            u = u + d
    return n_stored, n_total, 0


@cuda.jit(device=True)
def _count_win(a, b, c, d, sid, s0, s1, s2, s3, s4, s5,
               t0, t1, t_lo, t_hi):
    """Count spikes in [t_lo, t_hi) storing NONE. Exact at any buffer size."""
    v = V_REST
    u = b * V_REST
    t = t0
    cnt = 0
    n_steps = int((t1 - t0) / DT) + 2
    h2 = DT * 0.5
    for _ in range(n_steps):
        if t >= t1:
            break
        v_prev = v
        I1 = _stim(sid, t, s0, s1, s2, s3, s4, s5)
        k1v = 0.04 * v * v + 5.0 * v + 140.0 - u + I1
        k1u = a * (b * v - u)
        vt = v + h2 * k1v
        ut = u + h2 * k1u
        I2 = _stim(sid, t + h2, s0, s1, s2, s3, s4, s5)
        k2v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I2
        k2u = a * (b * vt - ut)
        vt = v + h2 * k2v
        ut = u + h2 * k2u
        k3v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I2
        k3u = a * (b * vt - ut)
        vt = v + DT * k3v
        ut = u + DT * k3u
        I4 = _stim(sid, t + DT, s0, s1, s2, s3, s4, s5)
        k4v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I4
        k4u = a * (b * vt - ut)
        v = v + (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        u = u + (DT / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        t += DT
        if not (v == v) or v > DIVERGE or v < -DIVERGE:
            return -1
        if v_prev < V_TH and v >= V_TH:
            if t_lo <= t < t_hi:
                cnt += 1
            v = c
            u = u + d
    return cnt


@cuda.jit(device=True)
def _first_latency(a, b, c, d, amp_cond, w_cond, amp_test, t_cond, delta,
                   t_search):
    """Latency from test-step onset to the first spike after it. No buffer."""
    v = V_REST
    u = b * V_REST
    t_on = t_cond + delta
    t_end = t_on + t_search
    t = 0.0
    h2 = DT * 0.5
    n_steps = int(t_end / DT) + 2
    for _ in range(n_steps):
        if t >= t_end:
            break
        v_prev = v
        I1 = _stim(5, t, t_cond, w_cond, amp_cond, t_on, t_end, amp_test)
        k1v = 0.04 * v * v + 5.0 * v + 140.0 - u + I1
        k1u = a * (b * v - u)
        vt = v + h2 * k1v
        ut = u + h2 * k1u
        I2 = _stim(5, t + h2, t_cond, w_cond, amp_cond, t_on, t_end, amp_test)
        k2v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I2
        k2u = a * (b * vt - ut)
        vt = v + h2 * k2v
        ut = u + h2 * k2u
        k3v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I2
        k3u = a * (b * vt - ut)
        vt = v + DT * k3v
        ut = u + DT * k3u
        I4 = _stim(5, t + DT, t_cond, w_cond, amp_cond, t_on, t_end, amp_test)
        k4v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I4
        k4u = a * (b * vt - ut)
        v = v + (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        u = u + (DT / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        t += DT
        if not (v == v) or v > DIVERGE or v < -DIVERGE:
            return -2.0
        if v_prev < V_TH and v >= V_TH:
            v = c
            u = u + d
            if t >= t_on:
                return t - t_on
    return -1.0


# ================================================================= landmarks
@cuda.jit(device=True)
def _fires_at(a, b, c, d, amp, t_trans, t_win, spike_times):
    ns, nt, inv = _integrate(a, b, c, d, 0, t_trans, amp, 0.0, 0.0, 0.0, 0.0,
                             0.0, t_trans + t_win, spike_times)
    if inv == 1:
        return -1
    if ns < nt:
        return 1
    cnt = 0
    for i in range(ns):
        if spike_times[i] >= t_trans:
            cnt += 1
    return 1 if cnt >= N_MIN_SPIKES else 0


@cuda.jit(device=True)
def _rheobase(a, b, c, d, t_trans, t_win, spike_times):
    """Scan for a firing bracket, then bisect. No I_RH_MIN gate -- i_rh ~ 0 is a
    measurement (the cell fires at ~0 drive), and spec 5.1 puts the gate on the
    rheobase-normalized columns, not on the landmark."""
    lo = 0.0
    hi = I_RHEO_HI
    prev = 0.0
    found = 0
    for i in range(1, RHEO_SCAN_N + 1):
        fr = float(i) / RHEO_SCAN_N
        amp = I_RHEO_HI * fr * fr * fr
        f = _fires_at(a, b, c, d, amp, t_trans, t_win, spike_times)
        if f == 1:
            lo = prev
            hi = amp
            found = 1
            break
        prev = amp
    if found == 0:
        return SENT
    for _ in range(RHEO_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        f = _fires_at(a, b, c, d, mid, t_trans, t_win, spike_times)
        if f == 1:
            hi = mid
        else:
            lo = mid
    return hi


@cuda.jit(device=True)
def _rheobase_seeded(a, b, c, d, t_trans, t_win, seed, spike_times):
    """Round 2, bracketed around round 1. Falls back to the full scan (ss43)."""
    if seed > 0.0:
        if _fires_at(a, b, c, d, seed, t_trans, t_win, spike_times) == 1:
            lo = seed
            hi = seed
            ok = 0
            for _ in range(20):
                lo = lo * 0.5
                if lo < 1.0e-9:
                    lo = 0.0
                    ok = 1
                    break
                if _fires_at(a, b, c, d, lo, t_trans, t_win, spike_times) != 1:
                    ok = 1
                    break
            if ok == 1:
                for _ in range(RHEO_BISECT_ITERS):
                    mid = 0.5 * (lo + hi)
                    if _fires_at(a, b, c, d, mid, t_trans, t_win, spike_times) == 1:
                        hi = mid
                    else:
                        lo = mid
                return hi
        else:
            hi = seed
            ok = 0
            for _ in range(20):
                hi = hi * 4.0
                if hi > I_RHEO_HI:
                    break
                if _fires_at(a, b, c, d, hi, t_trans, t_win, spike_times) == 1:
                    ok = 1
                    break
            if ok == 1:
                lo = hi * 0.25
                for _ in range(RHEO_BISECT_ITERS):
                    mid = 0.5 * (lo + hi)
                    if _fires_at(a, b, c, d, mid, t_trans, t_win, spike_times) == 1:
                        hi = mid
                    else:
                        lo = mid
                return hi
    return _rheobase(a, b, c, d, t_trans, t_win, spike_times)


@cuda.jit(device=True)
def _isi_period(isi, start, count):
    if count < 2:
        return 0
    for p in range(1, P_MAX + 1):
        if count < 2 * p:
            break
        ok = 1
        for i in range(count - p):
            x = isi[start + i]
            y = isi[start + i + p]
            m = 0.5 * (x + y)
            if m <= 0.0:
                ok = 0
                break
            dd = x - y
            if dd < 0.0:
                dd = -dd
            if dd / m > PERIOD_TOL:
                ok = 0
                break
        if ok == 1:
            return p
    return 0


@cuda.jit(device=True)
def _t_cycle(a, b, c, d, i_rh, t_trans, t_win, t_disc, spike_times, isi):
    """(T_cycle, p_hat, capped). Phase-invariant: p * mean settled ISI.

    Not i_rh-gated (spec 5.1): a spontaneously firing cell has i_rh ~ 0 and still
    has a characteristic cycle.
    """
    ns, nt, inv = _integrate(a, b, c, d, 0, t_trans, GAMMA_STEP * i_rh,
                             0.0, 0.0, 0.0, 0.0, 0.0, t_trans + t_win,
                             spike_times)
    if inv == 1:
        return SENT, 0, 0
    if ns < nt:
        return SENT, 0, 1                     # bound record -> anchor untrusted
    t_ss = t_trans + t_disc
    i0 = 0
    while i0 < ns and spike_times[i0] < t_ss:
        i0 += 1
    n_isi = 0
    for i in range(i0 + 1, ns):
        isi[n_isi] = spike_times[i] - spike_times[i - 1]
        n_isi += 1
    if n_isi < 2:
        return SENT, 0, 0
    k = n_isi if n_isi < PERIOD_TAIL else PERIOD_TAIL
    off = n_isi - k
    p = _isi_period(isi, off, k)
    tot = 0.0
    for i in range(off, n_isi):
        tot += isi[i]
    mean_isi = tot / k
    pf = float(p) if p > 0 else 1.0
    return pf * mean_isi, p, 0


@cuda.jit(device=True)
def _l_inf(a, b, c, d, i_rh, t_search):
    if i_rh < I_RH_MIN:
        return SENT
    L = _first_latency(a, b, c, d, 0.0, 1.0, GAMMA_LINF * i_rh,
                       BOOT_T_TRANS, 0.0, t_search)
    if L <= 0.0:
        return SENT
    return L


# ================================================================= protocols
@cuda.jit(device=True)
def _burst_ratio(isi, start, count, p):
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


@cuda.jit(device=True)
def _p5_gamma_ramp(a, b, c, d, i_rh, g_trans, t_ramp, g_win):
    """I(t_first)/i_rh on ONE ramp over K_RAMP cycles, HOLD RETAINED."""
    if i_rh < I_RH_MIN:
        return SENT
    amp = RAMP_GAMMA_F * i_rh
    t_re = g_trans + t_ramp
    t_end = t_re + g_win
    v = V_REST
    u = b * V_REST
    t = 0.0
    h2 = DT * 0.5
    n_steps = int(t_end / DT) + 2
    for _ in range(n_steps):
        if t >= t_end:
            break
        v_prev = v
        I1 = _stim(1, t, g_trans, t_re, amp, 0.0, 0.0, 0.0)
        k1v = 0.04 * v * v + 5.0 * v + 140.0 - u + I1
        k1u = a * (b * v - u)
        vt = v + h2 * k1v
        ut = u + h2 * k1u
        I2 = _stim(1, t + h2, g_trans, t_re, amp, 0.0, 0.0, 0.0)
        k2v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I2
        k2u = a * (b * vt - ut)
        vt = v + h2 * k2v
        ut = u + h2 * k2u
        k3v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I2
        k3u = a * (b * vt - ut)
        vt = v + DT * k3v
        ut = u + DT * k3u
        I4 = _stim(1, t + DT, g_trans, t_re, amp, 0.0, 0.0, 0.0)
        k4v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I4
        k4u = a * (b * vt - ut)
        v = v + (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        u = u + (DT / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        t += DT
        if not (v == v) or v > DIVERGE or v < -DIVERGE:
            return SENT
        if v_prev < V_TH and v >= V_TH:
            if t <= g_trans:
                return 0.0
            if t >= t_re:
                return RAMP_GAMMA_F
            return (amp * (t - g_trans) / t_ramp) / i_rh
    return SENT


@cuda.jit(device=True)
def _p7_eta(a, b, c, d, i_rh, t_total):
    """i_on = FIRST up-ramp spike, i_off = LAST down-ramp spike. Running scalars,
    so the cap cannot truncate the last spike. n == 0 or a one-sided train ->
    SENTINEL, not 0.0 (spec 5.2 rule 2)."""
    if i_rh < I_RH_MIN:
        return SENT
    half = t_total * 0.5
    gmax = HYST_GAMMA_MAX
    v = V_REST
    u = b * V_REST
    t = 0.0
    h2 = DT * 0.5
    i_on = SENT
    i_off = SENT
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
        v_prev = v
        k1v = 0.04 * v * v + 5.0 * v + 140.0 - u + I
        k1u = a * (b * v - u)
        vt = v + h2 * k1v
        ut = u + h2 * k1u
        k2v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I
        k2u = a * (b * vt - ut)
        vt = v + h2 * k2v
        ut = u + h2 * k2u
        k3v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I
        k3u = a * (b * vt - ut)
        vt = v + DT * k3v
        ut = u + DT * k3u
        k4v = 0.04 * vt * vt + 5.0 * vt + 140.0 - ut + I
        k4u = a * (b * vt - ut)
        v = v + (DT / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        u = u + (DT / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        t += DT
        if not (v == v) or v > DIVERGE or v < -DIVERGE:
            return SENT
        if v_prev < V_TH and v >= V_TH:
            n += 1
            if t < half:
                if i_on < SENT_CUT:
                    i_on = I
            else:
                i_off = I
            v = c
            u = u + d
    if n == 0 or i_on < SENT_CUT or i_off < SENT_CUT:
        return SENT
    return (i_on - i_off) / i_rh


@cuda.jit(device=True)
def _p8_a_rec(a, b, c, d, i_rh, l_inf, t_cond, delays):
    """Signed post-spike excitability change, geometry entirely in L_inf units.

    Fit is on ln|D| with the sign carried: 91.4% of this box comes back
    FACILITATED, not depressed, because `c` reaches -25 mV and the v-reset beats
    the u-jump (ss38.1).
    """
    if i_rh < I_RH_MIN or l_inf < SENT_CUT or l_inf <= 0.0:
        return SENT
    amp_c = GAMMA_PULSE * i_rh
    amp_t = GAMMA_LINF * i_rh
    w_c = REC_W_COND * l_inf
    if w_c < 5.0 * DT:
        w_c = 5.0 * DT
    t_sr = REC_T_SEARCH * l_inf

    nc = _first_latency(a, b, c, d, amp_c, w_c, 0.0, t_cond, 0.0, w_c + t_sr)
    if nc < 0.0:
        return SENT
    li = _first_latency(a, b, c, d, 0.0, w_c, amp_t, t_cond,
                        delays[N_DELAY_REC - 1] * l_inf, t_sr)
    if li <= 0.0:
        return SENT
    tol = REC_TOL_DT * DT
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    m = 0
    sgn = 1.0
    for k in range(N_DELAY_REC):
        dl = delays[k] * l_inf
        L = _first_latency(a, b, c, d, amp_c, w_c, amp_t, t_cond, dl, t_sr)
        if L <= 0.0:
            continue
        D = L - li
        ad = D if D >= 0.0 else -D
        if ad <= tol:
            continue
        if m == 0:
            sgn = 1.0 if D >= 0.0 else -1.0
        x = dl / li
        yv = math.log(ad)
        sx += x
        sy += yv
        sxx += x * x
        sxy += x * yv
        m += 1
    if m < REC_MIN_PTS:
        return SENT
    den = m * sxx - sx * sx
    if den <= 0.0:
        return SENT
    slope = (m * sxy - sx * sy) / den
    if slope >= 0.0:
        return SENT
    icpt = (sy - slope * sx) / m
    return sgn * math.exp(icpt) / li


# =================================================================== kernels
@cuda.jit(cache=True)
def landmark_kernel(params, out_lm):
    """PASS 0. out_lm[i] = (i_rh, T_cycle, L_inf, p_hat, capped).

    Cheap and uniform enough to run unsorted; its output is what the sort key
    for PASS 1 comes from.
    """
    i = cuda.grid(1)
    if i >= params.shape[0]:
        return
    a = params[i, 0]; b = params[i, 1]; c = params[i, 2]; d = params[i, 3]
    st = cuda.local.array(GPU_MAX_SPIKES, dtype=REAL)
    isi = cuda.local.array(GPU_MAX_SPIKES, dtype=REAL)

    out_lm[i, 0] = SENT; out_lm[i, 1] = SENT; out_lm[i, 2] = SENT
    out_lm[i, 3] = 0.0;  out_lm[i, 4] = 0.0

    i_rh = _rheobase(a, b, c, d, BOOT_T_TRANS, BOOT_T_WIN, st)
    if i_rh < SENT_CUT:
        return
    tc, p, capped = _t_cycle(a, b, c, d, i_rh, BOOT_T_TRANS, BOOT_T_WIN,
                             BOOT_T_DISC, st, isi)
    if capped == 1 or tc < SENT_CUT or tc <= 0.0:
        out_lm[i, 0] = i_rh
        out_lm[i, 4] = float(capped)
        return

    # round 2 -- i_rh and T_cycle on the cell's own scale, so P0 and the feature
    # protocols agree about what threshold means
    i_rh = _rheobase_seeded(a, b, c, d, K_TRANS * tc, K_WIN * tc, i_rh, st)
    if i_rh < SENT_CUT:
        return
    tc2, p2, cap2 = _t_cycle(a, b, c, d, i_rh, K_TRANS * tc, K_WIN * tc,
                             K_DISCARD * tc, st, isi)
    if cap2 == 1:
        out_lm[i, 0] = i_rh
        out_lm[i, 4] = 1.0
        return
    if tc2 > SENT_CUT and tc2 > 0.0:
        tc = tc2
        p = p2
    l0 = _l_inf(a, b, c, d, i_rh, BOOT_T_WIN)
    li = SENT
    if l0 > SENT_CUT:
        l1 = _l_inf(a, b, c, d, i_rh, LINF_SEARCH_K * l0)
        li = l1 if l1 > SENT_CUT else l0
    out_lm[i, 0] = i_rh
    out_lm[i, 1] = tc
    out_lm[i, 2] = li
    out_lm[i, 3] = float(p)
    out_lm[i, 4] = 0.0


@cuda.jit(cache=True)
def feature_kernel(params, lm, fi_g, delays, out):
    """PASS 1. Consumes PASS 0's landmarks; writes N_COLS per cell.

    Launch on cells SORTED BY lm[:,1] (T_cycle) -- see the module docstring.
    """
    i = cuda.grid(1)
    if i >= params.shape[0]:
        return
    for j in range(N_COLS):
        out[i, j] = SENT

    a = params[i, 0]; b = params[i, 1]; c = params[i, 2]; d = params[i, 3]
    i_rh = lm[i, 0]
    tc = lm[i, 1]
    linf = lm[i, 2]
    capped = lm[i, 4]

    if capped != 0.0:
        return
    if tc > SENT_CUT and tc < GATE_K_DT * DT:
        return                                   # B5 resolution gate

    st = cuda.local.array(GPU_MAX_SPIKES, dtype=REAL)
    isi = cuda.local.array(GPU_MAX_SPIKES, dtype=REAL)
    fi = cuda.local.array(N_FI, dtype=REAL)
    resp = cuda.local.array(DELTA_GRID_N, dtype=REAL)

    tc_or = tc if tc > SENT_CUT else 50.0
    g_tr = K_TRANS * tc_or
    g_wn = K_WIN * tc_or

    # nu_spont is NOT i_rh-gated
    nsp = _count_win(a, b, c, d, 0, g_tr, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, g_tr + g_wn, g_tr, g_tr + g_wn)
    out[i, I_nu] = SENT if nsp < 0 else float(nsp)

    if i_rh < SENT_CUT or tc < SENT_CUT:
        return

    trans = K_TRANS * tc
    disc = K_DISCARD * tc
    win = K_WIN * tc
    win_long = K_WIN_LONG * tc
    d_max = K_DELTA_MAX * tc
    width = K_PULSE_WIDTH * tc
    win_pulse = K_WIN_PULSE * tc
    ramp = K_RAMP * tc
    hyper = K_HYPER * tc

    # ---- P1 ----
    ns, nt, inv = _integrate(a, b, c, d, 0, trans, GAMMA_STEP * i_rh,
                             0.0, 0.0, 0.0, 0.0, 0.0, trans + win, st)
    if inv == 0 and ns == nt:
        t_ss = trans + disc
        n_tr = 0
        for k in range(1, ns):
            if st[k] < t_ss:
                isi[n_tr] = st[k] - st[k - 1]
                n_tr += 1
        out[i, I_ptr] = float(_isi_period(isi, 0, n_tr)) if n_tr >= 1 else 0.0

        i0 = 0
        while i0 < ns and st[i0] < t_ss:
            i0 += 1
        n_isi = 0
        for k in range(i0 + 1, ns):
            isi[n_isi] = st[k] - st[k - 1]
            n_isi += 1

        for k in range(ns):
            if st[k] >= trans:
                out[i, I_ell] = (st[k] - trans) / tc
                break

        if n_isi >= 2:
            kk = n_isi if n_isi < PERIOD_TAIL else PERIOD_TAIL
            off = n_isi - kk
            p = _isi_period(isi, off, kk)
            out[i, I_p] = float(p)
            if p > 0:
                out[i, I_rho] = _burst_ratio(isi, off, kk, p)
            if p == 1:
                first = isi[0]
                if first > 0.0:
                    k5 = n_isi if n_isi <= 5 else 5
                    tt = 0.0
                    for k in range(n_isi - k5, n_isi):
                        tt += isi[k]
                    out[i, I_A] = (tt / k5 - first) / first

    # ---- P2 ----
    for j in range(N_FI):
        cn = _count_win(a, b, c, d, 0, trans, fi_g[j] * i_rh, 0.0, 0.0, 0.0,
                        0.0, 0.0, trans + win, trans, trans + win)
        fi[j] = SENT if cn < 0 else float(cn) / win
    f0 = fi[0]; f3 = fi[3]; f_lo = fi[1]; f_hi = fi[5]
    if f0 > 0.0 and f3 > 0.0:
        out[i, I_beta] = (math.log(f3) - math.log(f0)) / (math.log(0.20) - math.log(0.02))
    if f_hi > 0.0 and f_lo > 0.0:
        out[i, I_chi] = f_lo / f_hi
        out[i, I_lam] = f_hi / f_lo
    if f_hi > 0.0:
        out[i, I_fbar] = tc * f_hi

    # ---- P3: Pi = N(2W)/N(W), both counts uncapped ----
    n_full = _count_win(a, b, c, d, 0, trans, GAMMA_STEP * i_rh, 0.0, 0.0, 0.0,
                        0.0, 0.0, trans + win_long, trans, trans + win_long)
    n_half = _count_win(a, b, c, d, 0, trans, GAMMA_STEP * i_rh, 0.0, 0.0, 0.0,
                        0.0, 0.0, trans + win_long, trans, trans + win)
    if n_full >= 0 and n_half > 0:
        out[i, I_Pi] = float(n_full) / float(n_half)

    # ---- P4: uncapped counts feed BOTH delta_star and rho_res (B2) ----
    step = d_max / (DELTA_GRID_N - 1) if DELTA_GRID_N > 1 else d_max
    for di in range(DELTA_GRID_N):
        dl = di * step
        t2 = trans + dl
        t_end = t2 + width + win_pulse
        cn = _count_win(a, b, c, d, 2, trans, t2, width, GAMMA_PULSE * i_rh,
                        0.0, 0.0, 0.0, t_end, trans, t_end)
        resp[di] = SENT if cn < 0 else float(cn)
    bi = 0
    best = resp[0]
    for k in range(1, DELTA_GRID_N):
        if resp[k] > best and resp[k] > SENT_CUT:
            best = resp[k]
            bi = k
    last = resp[DELTA_GRID_N - 1]
    if last > 0.0 and last > SENT_CUT:
        out[i, I_rr] = best / last
    # delta_star = STEP LOCATION at half height, not the argmax (ss41.2)
    n_hi = 0
    acc = 0.0
    for k in range(DELTA_GRID_N // 2, DELTA_GRID_N):
        if resp[k] > SENT_CUT:
            acc += resp[k]
            n_hi += 1
    base = resp[0]
    if n_hi > 0 and base > SENT_CUT:
        plateau = acc / n_hi
        if plateau > base:
            thr = base + 0.5 * (plateau - base)
            for k in range(DELTA_GRID_N):
                if resp[k] > SENT_CUT and resp[k] >= thr:
                    out[i, I_ds] = (k * step) / tc
                    break

    # ---- P5 / P6 / P7 / P8' ----
    out[i, I_gr] = _p5_gamma_ramp(a, b, c, d, i_rh, trans, ramp, win)

    if i_rh >= I_RH_MIN:
        g_first = SENT
        n_deep = SENT
        okr = 1
        for j in range(REBOUND_DEPTH_N):
            gh = GAMMA_HYPER_MAX * float(j + 1) / REBOUND_DEPTH_N
            t_off = trans + hyper
            cn = _count_win(a, b, c, d, 3, trans, t_off, -gh * i_rh, 0.0, 0.0,
                            0.0, 0.0, t_off + win, t_off, t_off + win)
            if cn < 0:
                okr = 0
                break
            if cn > 0 and g_first < SENT_CUT:
                g_first = gh
            if j == REBOUND_DEPTH_N - 1:
                n_deep = float(cn)
        if okr == 1:
            out[i, I_nreb] = n_deep
            out[i, I_greb] = g_first

    out[i, I_eta] = _p7_eta(a, b, c, d, i_rh, win_long)
    out[i, I_arec] = _p8_a_rec(a, b, c, d, i_rh, linf, trans, delays)


# ============================================================== host drivers
def _launch(kernel, n, args, tpb=64):
    blocks = (n + tpb - 1) // tpb
    kernel[blocks, tpb](*args)
    cuda.synchronize()


def landmarks_gpu(theta, tpb=64):
    """PASS 0 for a batch. Returns (n, 5): i_rh, T_cycle, L_inf, p_hat, capped."""
    th = np.ascontiguousarray(theta, dtype=NP_REAL)
    d_th = cuda.to_device(th)
    d_lm = cuda.device_array((th.shape[0], 5), dtype=NP_REAL)
    _launch(landmark_kernel, th.shape[0], (d_th, d_lm), tpb)
    return d_lm.copy_to_host()


def features_gpu(theta, lm=None, tpb=64, sort=True):
    """PASS 1. Returns (features (n, N_COLS), landmarks (n, 5)).

    `sort=True` bins cells by provisional T_cycle before launching. Every v3
    window is a multiple of the cell's own cycle, so an unsorted warp runs at its
    slowest lane; ss6.4 measured 6.3x for random ordering when only SOME windows
    were cell-relative. Results are unsorted back to input order, so sorting is
    invisible to the caller except in the wall clock.
    """
    th = np.ascontiguousarray(theta, dtype=NP_REAL)
    n = th.shape[0]
    if lm is None:
        lm = landmarks_gpu(th, tpb)
    order = np.argsort(lm[:, 1], kind="stable") if sort else np.arange(n)
    inv = np.argsort(order, kind="stable")

    d_th = cuda.to_device(np.ascontiguousarray(th[order]))
    d_lm = cuda.to_device(np.ascontiguousarray(lm[order]))
    d_fi = cuda.to_device(FI_G.astype(NP_REAL))
    d_dl = cuda.to_device(DELAYS_REL.astype(NP_REAL))
    d_out = cuda.device_array((n, N_COLS), dtype=NP_REAL)
    _launch(feature_kernel, n, (d_th, d_lm, d_fi, d_dl, d_out), tpb)
    out = d_out.copy_to_host()[inv]
    return out, lm


def bench_binning(theta, repeats=1, tpb=64):
    """Measure what warp binning buys on THIS box, rather than inheriting ss6.4."""
    import time
    th = np.ascontiguousarray(theta, dtype=NP_REAL)
    lm = landmarks_gpu(th, tpb)
    res = {}
    for tag, srt in (("unsorted", False), ("sorted", True)):
        t0 = time.time()
        for _ in range(repeats):
            features_gpu(th, lm=lm, tpb=tpb, sort=srt)
        res[tag] = (time.time() - t0) / repeats
    res["speedup"] = res["unsorted"] / max(res["sorted"], 1e-12)
    return res
