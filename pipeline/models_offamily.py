"""
models_offamily.py — out-of-family target neurons.

Every conclusion about the count fix currently rests on ONE out-of-family target
(Hindmarsh-Rose), and it points the opposite way from 2396 in-family cells:
in-family theta error fell 0.57x, while HR's posterior collapsed onto near-zero
rheobase (98.4% of draws below I_RH_MIN, support 4/256). n=1 cannot tell an
idiosyncratic failure from a general one, and out-of-family transfer is what the
pipeline exists to do.

Two extraction paths, both already in the repo — nothing new is integrated here:

  RESET models     -> the `src/` Izhikevich pipeline, which is generic in `rhs`.
                      Its protocols take `params[2]` as the reset value of y[0]
                      and `params[3]` as the increment to y[1], and hardcode the
                      initial condition `y = [-65, params[1]*(-65)]`. A model
                      that adopts those conventions (spike peak at V_TH = 30 mV,
                      E_L = -65 mV, `params[1] = 0` so the second variable starts
                      at 0) runs unmodified — and therefore also inherits
                      matched count-fixed extraction for free.
  RESET-FREE models-> `src_any_neuron`, which is dimension-agnostic and takes its
                      own v_th / rheobase bracket / window per model.

Why AdEx matters most: it is 2-D with a hard reset, i.e. structurally the closest
thing to Izhikevich in the set. It is the NEAR-family control. If a checkpoint
fails even on AdEx, the failure is not about exotic dynamics; if it handles AdEx
but not HR, the problem is specific to reset-free / higher-dimensional targets.
AdEx also spans tonic / adapting / initial-burst / bursting from one equation set
(Naud et al. 2008), so it supplies several regimes at fixed model family.
"""

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# RESET family — run through the src/ Izhikevich pipeline
#   params = [unused, 0.0, V_reset, b_increment]
#   y = [V, w],  y0 = [-65, 0],  spike declared at V >= 30
# ---------------------------------------------------------------------------

# AdEx constants (Brette & Gerstner). Baked into the closure because the
# pipeline's params vector is only 4 wide and two slots are spoken for by the
# reset convention. C in pF, g_L in nS, voltages in mV, tau in ms, current in pA
# -> dV/dt is mV/ms, matching the pipeline's time unit.
_ADEX = dict(C=200.0, g_L=10.0, E_L=-65.0, V_T=-50.0, Delta_T=2.0, tau_w=300.0,
             a=2.0)


@njit(cache=True)
def adex_rhs(t, y, params, I, dydt):
    """Adaptive exponential integrate-and-fire. y = (V, w).

    Reset (applied by the pipeline's integrator): V -> params[2], w -> w + params[3].
    The exponential term is clipped before it overflows: past ~0 mV the spike is
    declared anyway at V_TH = 30, so the unclipped divergence carries no
    information and only risks an inf inside RK4."""
    V = y[0]
    w = y[1]
    # Clip the exponential argument. Not cosmetic: the pipeline's integrator
    # checks DIVERGE_BOUND (1e4) BEFORE the threshold crossing, so an unclipped
    # AdEx spike (V -> 5e5 in one 0.01 ms step) aborts the run as "diverged"
    # before the crossing at V_TH = 30 is ever registered. x <= 10 caps dV/dt at
    # ~2200 mV/ms, i.e. ~22 mV per step: the upstroke still crosses 30 within a
    # few steps and is reset, while |V| never approaches the bound. Dynamics
    # above V_TH are discarded by the reset anyway, so nothing is lost.
    x = (V - (-50.0)) / 2.0
    if x > 10.0:
        x = 10.0
    expo = 2.0 * np.exp(x)          # Delta_T * exp((V - V_T)/Delta_T)
    dydt[0] = (-10.0 * (V - (-65.0)) + 10.0 * expo - w + I) / 200.0
    dydt[1] = (2.0 * (V - (-65.0)) - w) / 300.0


# (label, params=[.,0,V_reset,b]) — Naud-style regimes at fixed a/tau_w
ADEX_TARGETS = [
    ("AdEx tonic",          np.array([0.0, 0.0, -65.0,   0.0])),
    ("AdEx adapting",       np.array([0.0, 0.0, -65.0,  60.0])),
    ("AdEx initial burst",  np.array([0.0, 0.0, -51.0, 100.0])),
    ("AdEx bursting",       np.array([0.0, 0.0, -46.0, 120.0])),
]


# ---------------------------------------------------------------------------
# RESET-FREE family — run through src_any_neuron (each carries its own spec)
# ---------------------------------------------------------------------------

@njit(cache=False)
def fhn_rhs(t, y, params, I, dydt):
    """FitzHugh-Nagumo relaxation oscillator. y = (v, w), v is the potential."""
    v = y[0]; w = y[1]
    dydt[0] = v - v * v * v / 3.0 - w + I
    dydt[1] = 0.08 * (v + 0.7 - 0.8 * w)


@njit(cache=False)
def morris_lecar_c1_rhs(t, y, params, I, dydt):
    """Morris-Lecar, Class 1 (SNIC onset) parameter set. y = (V, N)."""
    V = y[0]; N = y[1]
    m_inf = 0.5 * (1.0 + np.tanh((V - (-1.2)) / 18.0))
    w_inf = 0.5 * (1.0 + np.tanh((V - 12.0) / 17.4))
    tau_w = 1.0 / np.cosh((V - 12.0) / (2.0 * 17.4))
    dydt[0] = (I - 4.0 * m_inf * (V - 120.0) - 8.0 * N * (V - (-84.0))
               - 2.0 * (V - (-60.0))) / 20.0
    dydt[1] = 0.0667 * (w_inf - N) * tau_w


@njit(cache=False)
def morris_lecar_c2_rhs(t, y, params, I, dydt):
    """Morris-Lecar, Class 2 (Hopf onset) parameter set. y = (V, N)."""
    V = y[0]; N = y[1]
    m_inf = 0.5 * (1.0 + np.tanh((V - (-1.2)) / 18.0))
    w_inf = 0.5 * (1.0 + np.tanh((V - 2.0) / 30.0))
    tau_w = 1.0 / np.cosh((V - 2.0) / (2.0 * 30.0))
    dydt[0] = (I - 4.4 * m_inf * (V - 120.0) - 8.0 * N * (V - (-84.0))
               - 2.0 * (V - (-60.0))) / 20.0
    dydt[1] = 0.04 * (w_inf - N) * tau_w


@njit(cache=False)
def hh_rhs(t, y, params, I, dydt):
    """Hodgkin-Huxley. y = (V, m, h, n). The reference conductance model."""
    V = y[0]; m = y[1]; h = y[2]; n = y[3]
    am = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0)) \
        if abs(V + 40.0) > 1e-7 else 1.0
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    an = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0)) \
        if abs(V + 55.0) > 1e-7 else 0.1
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    dydt[0] = (I - 120.0 * m * m * m * h * (V - 50.0)
               - 36.0 * n * n * n * n * (V - (-77.0))
               - 0.3 * (V - (-54.387))) / 1.0
    dydt[1] = am * (1.0 - m) - bm * m
    dydt[2] = ah * (1.0 - h) - bh * h
    dydt[3] = an * (1.0 - n) - bn * n


@njit(cache=False)
def hr_tonic_rhs(t, y, params, I, dydt):
    """Hindmarsh-Rose away from the doublet boundary (r larger -> faster slow
    variable, tonic-ish). Companion to the existing bursting HR."""
    x = y[0]; yy = y[1]; z = y[2]
    dydt[0] = yy - 1.0 * x * x * x + 3.0 * x * x - z + I
    dydt[1] = 1.0 - 5.0 * x * x - yy
    dydt[2] = 0.005 * (4.0 * (x - (-1.6)) - z)


def resetfree_specs():
    """NeuronSpec per reset-free target. v_th / rheobase bracket / window / step
    are per-model by design — an out-of-family neuron has its own units, and the
    harness reports feature definedness so a starved extraction is visible rather
    than silently scored."""
    from neuron_spec import NeuronSpec
    out = []
    out.append(("FitzHugh-Nagumo", NeuronSpec(
        rhs=fhn_rhs, y0=np.array([-1.2, -0.62]), v_th=1.0,
        rheo_lo=0.0, rheo_hi=2.0, rheo_neg_lo=-2.0,
        h=0.005, t_win=800.0, t_discard=100.0, noinh=True, name="fhn")))
    out.append(("Morris-Lecar C1", NeuronSpec(
        rhs=morris_lecar_c1_rhs, y0=np.array([-60.0, 0.0]), v_th=10.0,
        rheo_lo=0.0, rheo_hi=200.0, rheo_neg_lo=-200.0,
        h=0.01, t_win=1000.0, t_discard=150.0, noinh=True, name="ml_c1")))
    out.append(("Morris-Lecar C2", NeuronSpec(
        rhs=morris_lecar_c2_rhs, y0=np.array([-60.0, 0.0]), v_th=10.0,
        rheo_lo=0.0, rheo_hi=200.0, rheo_neg_lo=-200.0,
        h=0.01, t_win=1000.0, t_discard=150.0, noinh=True, name="ml_c2")))
    out.append(("Hodgkin-Huxley", NeuronSpec(
        rhs=hh_rhs, y0=np.array([-65.0, 0.053, 0.596, 0.318]), v_th=0.0,
        rheo_lo=0.0, rheo_hi=50.0, rheo_neg_lo=-50.0,
        h=0.005, t_win=500.0, t_discard=50.0, noinh=True, name="hh")))
    out.append(("HR tonic", NeuronSpec(
        rhs=hr_tonic_rhs, y0=np.array([-1.6, 1.0 - 5.0 * 1.6 * 1.6, 0.0]),
        v_th=1.0, rheo_lo=0.0, rheo_hi=5.0, rheo_neg_lo=-5.0,
        h=0.01, t_win=2000.0, t_discard=200.0, noinh=True, name="hr_tonic")))
    return out
