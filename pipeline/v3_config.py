"""
v3_config.py -- Atlas v3 constants.

Rules this file enforces, from the build spec:

  * NO ABSOLUTE TIME CONSTANT MAY REACH A PROTOCOL (spec 3.1). Every duration
    below is a MULTIPLE of a per-cell landmark -- `T_cycle` for the firing
    protocols, `L_inf` for P8'. The only absolute durations are the PASS-0
    bootstrap windows, which exist solely to measure the landmarks that all
    other windows are then expressed in.
  * DT is env-parameterized (B6) but stays 0.01. Setting IZH_V3_DT MUST also set
    its own NUMBA_CACHE_DIR: DT is folded into @njit(cache=True) and the cache
    key does not see globals (spec 6, ss12/ss20).
  * MAX_SPIKES is NOT here. In v3 the spike buffer is a runtime array whose
    length the kernels read with len(); counts that features actually need are
    accumulated in uncapped counters that never touch the buffer (B1). That
    removes the folded-buffer-size class of cache bug entirely.
"""
import os as _os

# ------------------------------------------------------------------ numerics
DT            = float(_os.environ.get("IZH_V3_DT", 0.01))
V_TH          = 30.0
DIVERGE_BOUND = 1.0e6
STATE_DIM     = 2
V_REST        = -65.0

NAN_SENTINEL  = -9999.0
SENT_CUT      = -9000.0

# ------------------------------------------------------------- schema layout
N_COLS        = 32     # on-disk width; surplus is ANONYMOUS reserve (spec 4.6)
N_ACTIVE      = 18           # 19 minus alpha_acc (ss41.1)

# alpha_acc is DROPPED (ss41.1). Its spread was the slope grid, not accommodation:
# v2 averaged n_ramp/n_step over 6 ramps whose DURATIONS differ, so each trial
# integrated for a different total time -- the same duration artifact ss32.5 caught
# in the no-hold s_cut. On a single ramp the IQR is 0.018 against v2's 0.610.
# Its share of the corrected totals is 3.5-5.5% on every parameter, and those
# derivatives measure the SIX-SLOPE version, so ~5% is an upper bound on what the
# single-ramp column could carry. gamma_ramp measures accommodation directly on
# the same protocol, continuously, at 656 distinct values.
FEATURE_NAMES = [
    "p", "rho", "p_tr", "A", "beta", "chi", "delta_star", "rho_res",
    "ell", "lam", "Pi", "fbar", "N_reb", "A_rec", "eta",
    "gamma_ramp", "g_reb", "nu_spont",
]
assert len(FEATURE_NAMES) == N_ACTIVE

# ------------------------------------------------- PASS 0 bootstrap (absolute)
# The ONLY absolute windows in v3. They measure i_rh / T_cycle / L_inf; every
# other duration is a multiple of what they return.
BOOT_T_TRANS  = 50.0
BOOT_T_WIN    = 500.0
BOOT_T_DISC   = 50.0

# --- bootstrap adequacy (ss49/ss50) ----------------------------------------
# BOOT_T_WIN and I_RHEO_GUESS_HI are the LAST absolute constants in an otherwise
# fully cell-relative pipeline, and ss49.3 showed they wall the (tau, k_I) plane
# at tau >~ 10 and kappa <~ 0.1.
#
# The bracket failure is loud (no rheobase -> SENTINEL). The WINDOW failure is
# NOT: at tau = 100 the pipeline returned 17/18 features, no gate fired, and the
# anchor was 2.2x wrong. That is the same shape as ss47.3's overlap guard and
# ss46's wrong-model w1 -- plausible numbers, no error raised.
#
# It is, however, detectable from quantities PASS 0 already produces:
# boot_cycles = BOOT_T_WIN / T_cycle was 0.98 where the answer was exact and
# 0.22 where it was 2.2x wrong. So the bootstrap is treated as a SEED, not a
# requirement: measure the ratio, and if it is short, rescale by the provisional
# anchor and re-measure.
BOOT_MIN_CYCLES = 3.0        # widen the bootstrap below this many cycles
BOOT_MAX_RETRY  = 4          # bounded: each retry rescales by the provisional
RHEO_MAX_EXPAND = 6          # doublings of the rheobase ceiling before giving up

# ------------------------------------------------------------------ landmarks
I_RH_MIN         = 1.0e-3
I_RHEO_GUESS_LO  = 0.0
I_RHEO_GUESS_HI  = 200.0     # must match v2 (src/config.py). At 100 the smoke
                             # lost every cell whose rheobase exceeds it -- 3 of
                             # the first 6 dropped cells had i_rh 102-166
RHEO_SCAN_N      = 24        # scan-then-bisect (spec 2): coarse scan first, so a
                             # Class-2 bounded firing band is bracketed, not missed
RHEO_BISECT_ITERS= int(_os.environ.get("IZH_V3_RHEO_BIS", 40))         # STAYS AT 40. Cutting to 20 was tested (ss43) and
                             # rejected on the FEATURE comparison, not on i_rh:
                             # i_rh itself is fine (abs err 2e-05 vs I_RH_MIN
                             # 1e-3, 0.0% gate flips at any depth down to 8), but
                             # `eta` then changes by >1% on 16.0% of cells and by
                             # >10% on 4.6%, because it is a threshold-crossing
                             # quantity and a 2e-05 shift in i_rh flips which
                             # spike is last. Saves ~7 h on a one-time 31 h
                             # build; not worth perturbing the schema's most
                             # fragile column (ss25.4). Set IZH_V3_RHEO_BIS=20 to
                             # take the trade deliberately.
N_MIN_SPIKES     = 3
RHEO_SEED_ROUND2 = int(_os.environ.get("IZH_V3_RHEO_SEED", 0))          # PASS 0 round 2 brackets around round 1's answer
                             # instead of re-scanning from scratch. 1.87x fewer
                             # integrations, 0.0% gate flips, 0.0% definedness
                             # disagreements -- but ONLY with the fallback: a
                             # seeded search that fails to bracket must run the
                             # full scan, never report "no rheobase", because
                             # downstream that is indistinguishable from a cell
                             # that genuinely never fires. Without the fallback
                             # it silently lost 12.6% of cells.

P_MAX            = 8
PERIOD_TOL       = 0.05
PERIOD_TAIL      = 40

GAMMA_LINF       = 1.2       # test-step amplitude for L_inf (spec 3.4)
LINF_SEARCH_K    = 10.0      # search window, in L_inf units, after calibration

# ------------------------------------------------ cell-relative geometry (B7)
# every entry is a MULTIPLE of T_cycle
K_TRANS       = 4.0          # pre-stimulus settle
K_DISCARD     = 4.0          # post-onset transient (feeds p_tr only)
K_WIN         = 100.0        # steady-state window. Chosen on statistical
                             # adequacy, not on reproducing v2: 100 cycles minus
                             # K_DISCARD leaves ~96 settled ISIs, 2.4x
                             # PERIOD_TAIL, so period detection and rho both get
                             # a full tail with margin. Cost is 0.73x v2 --
                             # cell-relative windows are a CUT, because the prior
                             # is dominated by fast cells (ss6.4).
                             #
                             # For the record of what B7 is fixing: v2's fixed
                             # 500 ms window gave the MEDIAN cell 741 cycles and
                             # the slowest 2.57 -- a 288x spread in how much
                             # evidence each cell contributed.
K_WIN_LONG    = 200.0        # persistence / hysteresis -- exactly 2*K_WIN, which
                             # Pi = N(2W)/N(W) requires by construction
K_DELTA_MAX   = 296.0        # P4 max pulse separation. Set so the grid STEP in
                             # cycle units matches v2 at the atlas median:
                             # v2's 200 ms / 19 = 10.5 ms = 15.6 cycles at
                             # T_cycle 0.675, and 296/19 = 15.6. delta_star is
                             # reported in cycles, so it is the step in CYCLES
                             # that has to be preserved, not the span in ms.
K_PULSE_WIDTH = 3.0          # P4 / conditioning pulse width. v2's 2 ms at the
                             # atlas median T_cycle of 0.675 ms IS ~3 cycles.
                             # v3 first shipped 0.16, calibrated against a
                             # preset that is not typical of the atlas, which
                             # made the pulse 0.1 ms -- far too little charge to
                             # evoke a spike at 3*i_rh, and rho_res collapsed to
                             # 7% defined against v2's 99.7%.
K_WIN_PULSE   = 10.0         # P4 post-pulse observation window. Swept over
                             # 4/10/30/60/100 and the features are IDENTICAL to
                             # 16 significant digits at every value: P4 holds
                             # baseline current at 0, so there is no tonic
                             # firing for a longer window to accumulate. The
                             # short value is chosen purely for cost.
K_RAMP        = 0.1 * K_WIN  # P5 single ramp. THE SPEC SAYS "0.1*T_cycle" AND
                             # THAT IS A RAIL: at the atlas median T_cycle of
                             # 0.654 ms it makes a 0.065 ms ramp, and every cell
                             # returns gamma_ramp = RAMP_GAMMA_F because no spike
                             # fits inside the ramp. What ss33.6/ss37 actually
                             # measured was 0.1*t_win -- a tenth of the OBSERVATION
                             # WINDOW, not a tenth of one cycle. That relationship
                             # is what is preserved here.
K_HYPER       = 4.0          # P6 hyperpolarizing duration -- magnitude unchanged

SPIKE_BUDGET_PER_CYCLE = 4   # buffer sizing: k*K_WIN cycles worth, plus margin
SPIKE_BUFFER_MIN       = 4096

# ------------------------------------------------------------------ amplitudes
GAMMA_PULSE      = 3.0       # P4 pulse amplitude, rheobase units
GAMMA_STEP       = 2.0       # P1 reference step
RAMP_GAMMA_F     = 2.0       # P5 ramp final amplitude
GAMMA_HYPER_MAX  = 12.0      # B13: was 6
REBOUND_DEPTH_N  = 24        # B13: must scale with the range
FI_GAMMAS        = (1.02, 1.05, 1.10, 1.20, 1.50, 2.00, 3.00, 4.00)
N_FI             = len(FI_GAMMAS)
DELTA_GRID_N     = 20
HYST_GAMMA_MAX   = 3.0       # v2's GAMMA_HYST_MAX

# ------------------------------------------------------------------- P8' (B10)
N_DELAY_REC   = 5
REC_D_LO      = 0.13         # delay grid in L_inf units: geomspace(lo, hi, n).
REC_D_HI      = 13.0         # calibrated so the median cell (L_inf = 15.4 ms)
REC_W_COND    = 0.13         # sees the ss38 absolute protocol unchanged
REC_T_SEARCH  = 10.0
REC_MIN_PTS   = 3
REC_TOL_DT    = 3.0          # ss38.1: 3*DT, absorbs the 1-step boundary offset

# --------------------------------------------------------------- resolution gate
GATE_K_DT     = 10.0         # B5: drop cells with T_cycle < 10*DT

# ------------------------------------------------------------------- stim ids
STIM_STEP        = 0
STIM_RAMP_HOLD   = 1
STIM_PULSE_PAIR  = 2
STIM_HYPER_REL   = 3
STIM_HYST        = 4
STIM_TWO_PHASE   = 5
STIM_PARAMS_LEN  = 8
