"""
sweeps.py — parameterised out-of-family targets, ~10 configurations per family.

Single-point targets are anecdotes. HR at one parameter set produced a dramatic
result (posterior collapse onto near-zero rheobase, support 4/256) that could be
the general behaviour of the count-fixed checkpoint out-of-family, or could be
one unlucky point on a boundary — and with n=1 there is no way to tell. Sweeping
each family turns every model into a distribution, which is the same reason the
2396-cell prior cohort settled questions the 13 canonical presets could not.

Every `rhs` here reads its constants from `params` rather than baking them in, so
a sweep is a real sweep and not four hand-written variants.

Two conventions, because there are two extraction paths:

  RESET (AdEx)   `src/` pipeline: params[2] = reset value of y[0], params[3] =
                 increment to y[1], y0 = [-65, params[1]*(-65)], spike at V_TH=30.
                 Only params[0] is free, so AdEx sweeps the pair that defines its
                 regimes anyway (V_reset, b) — Naud's taxonomy axis.
  RESET-FREE     `src_any_neuron`: params is forwarded untouched to rhs, so all
                 four slots are available.
"""

import numpy as np
from numba import njit


# ===========================================================================
# RESET family — AdEx, through the src/ Izhikevich pipeline
# ===========================================================================

@njit(cache=True)
def adex_rhs(t, y, params, I, dydt):
    """AdEx. y = (V, w); reset V -> params[2], w -> w + params[3].

    The exponential argument is clipped at 10. Not cosmetic: the pipeline's
    integrator tests DIVERGE_BOUND (1e4) BEFORE the threshold crossing, so an
    unclipped AdEx upstroke (V -> 5e5 in one 0.01 ms step) aborts the run as
    "diverged" and the spike at V_TH=30 is never registered. Clipping caps
    dV/dt at ~2200 mV/ms — the upstroke still crosses within a few steps.
    Dynamics above V_TH are discarded by the reset, so nothing is lost."""
    V = y[0]
    w = y[1]
    x = (V - (-50.0)) / 2.0
    if x > 10.0:
        x = 10.0
    # Conductances scaled down 4x from the Brette-Gerstner values (C 200->50,
    # g_L 10->2.5, a 2->0.5) so AdEx rheobase (~g_L*(V_T-E_L) = 37 pA, up to
    # ~120 with strong adaptation) lands INSIDE the pipeline's rheobase bisection
    # bracket [I_RHEO_GUESS_LO, I_RHEO_GUESS_HI] = [0, 200]. At the original
    # scale the adapting/bursting regimes need >200 pA to reach the 4-spike
    # sustained-firing criterion, P0 finds no rheobase, and every feature comes
    # back SENTINEL (measured: n_defined = 1 of 22). Scaling current and
    # conductance together leaves the dynamics identical up to units.
    dydt[0] = (-2.5 * (V - (-65.0)) + 2.5 * (2.0 * np.exp(x)) - w + I) / 50.0
    dydt[1] = (0.5 * (V - (-65.0)) - w) / 300.0


def adex_sweep():
    """(V_reset, b) grid. b is the spike-triggered adaptation jump and V_reset the
    reset voltage; together they select tonic / adapting / initial-burst /
    bursting in AdEx."""
    out = []
    for vr in (-65.0, -58.0, -51.0, -46.0):
        for b in (0.0, 15.0, 30.0):     # b scaled with the conductances
            out.append((f"AdEx Vr={vr:.0f} b={b:.0f}",
                        np.array([0.0, 0.0, vr, b])))
    return out[:10]


# ===========================================================================
# RESET-FREE family — through src_any_neuron
# ===========================================================================

@njit(cache=False)
def fhn_rhs(t, y, params, I, dydt):
    """FitzHugh-Nagumo. params = (a, b, eps, -)."""
    v = y[0]; w = y[1]
    dydt[0] = v - v * v * v / 3.0 - w + I
    dydt[1] = params[2] * (v + params[0] - params[1] * w)


@njit(cache=False)
def morris_lecar_rhs(t, y, params, I, dydt):
    """Morris-Lecar. params = (phi, g_Ca, V3, V4).

    V3/V4 set the K-activation midpoint and slope, which is what moves the model
    between Class 1 (SNIC, V3~12) and Class 2 (Hopf, V3~2) onset."""
    V = y[0]; N = y[1]
    m_inf = 0.5 * (1.0 + np.tanh((V - (-1.2)) / 18.0))
    w_inf = 0.5 * (1.0 + np.tanh((V - params[2]) / params[3]))
    tau = 1.0 / np.cosh((V - params[2]) / (2.0 * params[3]))
    dydt[0] = (I - params[1] * m_inf * (V - 120.0) - 8.0 * N * (V - (-84.0))
               - 2.0 * (V - (-60.0))) / 20.0
    dydt[1] = params[0] * (w_inf - N) * tau


@njit(cache=False)
def hh_rhs(t, y, params, I, dydt):
    """Hodgkin-Huxley. params = (g_Na, g_K, g_L, -)."""
    V = y[0]; m = y[1]; h = y[2]; n = y[3]
    dv = V + 40.0
    am = 1.0 if abs(dv) < 1e-7 else 0.1 * dv / (1.0 - np.exp(-dv / 10.0))
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    dn = V + 55.0
    an = 0.1 if abs(dn) < 1e-7 else 0.01 * dn / (1.0 - np.exp(-dn / 10.0))
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    dydt[0] = (I - params[0] * m * m * m * h * (V - 50.0)
               - params[1] * n * n * n * n * (V - (-77.0))
               - params[2] * (V - (-54.387)))
    dydt[1] = am * (1.0 - m) - bm * m
    dydt[2] = ah * (1.0 - h) - bh * h
    dydt[3] = an * (1.0 - n) - bn * n


@njit(cache=False)
def hr_rhs(t, y, params, I, dydt):
    """Hindmarsh-Rose. params = (r, s, x_r, -). r sets the slow-variable speed
    (bursting at r~0.005-0.02), s the adaptation strength."""
    x = y[0]; yy = y[1]; z = y[2]
    dydt[0] = yy - x * x * x + 3.0 * x * x - z + I
    dydt[1] = 1.0 - 5.0 * x * x - yy
    dydt[2] = params[0] * (params[1] * (x - params[2]) - z)


@njit(cache=False)
def wilson_rhs(t, y, params, I, dydt):
    """Wilson (1999) polynomial reduction of a neocortical neuron.
    y = (V, R, T, H), V in units of 100 mV, t in ms.
    params = (g_K, g_T, g_H, -)."""
    V = y[0]; R = y[1]; T = y[2]; H = y[3]
    g_Na = 17.81 + 47.71 * V + 32.63 * V * V
    R_inf = 1.24 + 3.7 * V + 3.2 * V * V
    T_inf = 4.205 + 11.6 * V + 8.0 * V * V
    dydt[0] = (-g_Na * (V - 0.48) - params[0] * R * (V + 0.95)
               - params[1] * T * (V - 1.2) - params[2] * H * (V + 0.95)
               + I) / 0.8
    dydt[1] = (-R + R_inf) / 1.9
    dydt[2] = (-T + T_inf) / 14.0
    dydt[3] = (-H + 3.0 * T) / 45.0


# --- sweep definitions -------------------------------------------------------
# Each entry: (family, rhs, y0, v_th, rheo_hi, h, t_win, t_discard, configs)
# configs: list of (label, params)

def _lin(a, b, n):
    return list(np.linspace(a, b, n))


def resetfree_sweeps():
    S = []
    S.append(dict(
        family="FitzHugh-Nagumo", rhs=fhn_rhs, y0=np.array([-1.2, -0.62]),
        v_th=1.0, rheo_hi=2.0, h=0.005, t_win=800.0, t_discard=100.0,
        configs=[(f"FHN eps={e:.3f}", np.array([0.7, 0.8, e, 0.0]))
                 for e in _lin(0.02, 0.20, 10)]))
    S.append(dict(
        family="Morris-Lecar", rhs=morris_lecar_rhs, y0=np.array([-60.0, 0.0]),
        v_th=10.0, rheo_hi=200.0, h=0.01, t_win=1000.0, t_discard=150.0,
        # V3 sweeps Class 2 (Hopf, ~2) -> Class 1 (SNIC, ~12)
        configs=[(f"ML V3={v3:.1f}", np.array([0.0667, 4.0, v3, 17.4]))
                 for v3 in _lin(2.0, 14.0, 10)]))
    S.append(dict(
        family="Hodgkin-Huxley", rhs=hh_rhs,
        y0=np.array([-65.0, 0.053, 0.596, 0.318]),
        # gK capped at 40: above that the model enters depolarization block and
        # stops firing at every current, which breaks the rheobase bisection
        # (it assumes firing exists above rheobase) and would hand the pipeline
        # a target it cannot probe. Verified by a bracket scan, not assumed.
        v_th=0.0, rheo_hi=30.0, h=0.005, t_win=500.0, t_discard=50.0,
        configs=[(f"HH gK={gk:.0f}", np.array([120.0, gk, 0.3, 0.0]))
                 for gk in _lin(24.0, 40.0, 10)]))
    S.append(dict(
        family="Hindmarsh-Rose", rhs=hr_rhs,
        y0=np.array([-1.6, 1.0 - 5.0 * 1.6 * 1.6, 0.0]),
        v_th=1.0, rheo_hi=5.0, h=0.01, t_win=2000.0, t_discard=200.0,
        configs=[(f"HR r={r:.4f}", np.array([r, 7.0, -1.6, 0.0]))
                 for r in _lin(0.004, 0.030, 10)]))
    S.append(dict(
        family="Wilson", rhs=wilson_rhs,
        y0=np.array([-0.7, 0.088, 0.562, 1.687]),
        # rheo_hi=1.5: Wilson blocks above ~1.5 in these units (bracket scan).
        v_th=0.0, rheo_hi=1.5, h=0.005, t_win=800.0, t_discard=100.0,
        configs=[(f"Wilson gT={gt:.2f}", np.array([26.0, gt, 13.0, 0.0]))
                 for gt in _lin(0.0, 2.25, 10)]))
    return S
