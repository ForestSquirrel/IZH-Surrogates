"""
hh_dp.py -- Destexhe-Pare (1999) Hodgkin-Huxley-type neuron as used by Schulte to Brinke
et al. 2022 (eq. 1-16, Table 1 code values; NESTML hh_cond_exp_destexhe).

    C dV/dt = -gL(V-EL) - gNa m^3 h (V-ENa) - gK n^4 (V-EK) - gM p (V-EM)
              - gE0 (V-Eex) - gI0 (V-Ein) + I
Units: mV, ms, pF, nS, pA. gE0 / gI0 are the MEANS of the intrinsic OU noise
conductances (12 / 57 nS): they are part of the neuron (its operating point) and are
included in the inference target; the zero-mean OU fluctuations are engine receptors
(spec.Receptors.sig) so that the surrogate receives them through the voltage map.
gM = 100 nS (E) / 0 (I).

Two implementations of one RHS:
  rhs_pipeline  float64 numba RHS rhs(t, y, params, I, dydt) for the pipeline's
                continuous endpoint, y = (V, m, h, n, p), params = (gM, gE0, gI0, 0)
  step / step_cuda   float32 network step: nsub RK4 substeps of hs, then the spike test
                of the NEST model (V > V_thresh and falling: a peak); refractoriness (3 ms,
                emission only) is applied by the engine. State y = (V, m, h, n, p),
                P row = (gM, gE0, gI0, hs, nsub, V_thresh).
"""
import math
import numpy as np
from numba import njit, cuda

NP = 6
DIM = 5
C_M, G_NA, G_K, G_L = 346.36, 17872.176, 3463.6, 15.5862
E_NA, E_K, E_L, E_M = 50.0, -90.0, -80.0, -80.0
E_EX, E_IN = 0.0, -75.0
V_T, V_S, R_P = -63.0, -10.0, 0.001
V_THRESH = -30.0
T_REF = 3.0
G_E0, G_I0 = 12.0, 57.0
G_M_E, G_M_I = 100.0, 0.0


@njit(cache=True)
def _xe64(x, k):
    """x / (exp(x/k) - 1), with its limit k at x = 0"""
    if abs(x) < 1e-6:
        return k - 0.5 * x
    return x / (np.exp(x / k) - 1.0)


@njit(cache=True)
def rhs_pipeline(t, y, params, I, dydt):
    V, m, h, n, p = y[0], y[1], y[2], y[3], y[4]
    gM, gE0, gI0 = params[0], params[1], params[2]
    vr = V - V_T
    am = 0.32 * _xe64(13.0 - vr, 4.0)
    bm = 0.28 * _xe64(vr - 40.0, 5.0)
    ah = 0.128 * np.exp(-(vr - V_S - 17.0) / 18.0)
    bh = 4.0 / (1.0 + np.exp(-(vr - V_S - 40.0) / 5.0))
    an = 0.032 * _xe64(15.0 - vr, 5.0)
    bn = 0.5 * np.exp((10.0 - vr) / 40.0)
    ap = R_P * _xe64(-(V + 30.0), 9.0)
    bp = R_P * _xe64(V + 30.0, 9.0)
    Iion = (G_L * (V - E_L) + G_NA * m * m * m * h * (V - E_NA) + G_K * n * n * n * n * (V - E_K)
            + gM * p * (V - E_M) + gE0 * (V - E_EX) + gI0 * (V - E_IN))
    dydt[0] = (I - Iion) / C_M
    dydt[1] = am * (1.0 - m) - bm * m
    dydt[2] = ah * (1.0 - h) - bh * h
    dydt[3] = an * (1.0 - n) - bn * n
    dydt[4] = ap * (1.0 - p) - bp * p


# ---------------------------------------------------------------- float32 network step
f = np.float32
_C = dict(C_M=f(C_M), G_NA=f(G_NA), G_K=f(G_K), G_L=f(G_L), E_NA=f(E_NA), E_K=f(E_K),
          E_L=f(E_L), E_M=f(E_M), E_EX=f(E_EX), E_IN=f(E_IN), V_T=f(V_T), V_S=f(V_S),
          R_P=f(R_P))
CM32, GNA32, GK32, GL32 = _C["C_M"], _C["G_NA"], _C["G_K"], _C["G_L"]
ENA32, EK32, EL32, EM32 = _C["E_NA"], _C["E_K"], _C["E_L"], _C["E_M"]
EEX32, EIN32, VT32, VS32, RP32 = _C["E_EX"], _C["E_IN"], _C["V_T"], _C["V_S"], _C["R_P"]
ONE, HALF, TWO, SIXTH = f(1.0), f(0.5), f(2.0), f(1.0 / 6.0)
K4, K5, K9, K18, K40 = f(4.0), f(5.0), f(9.0), f(18.0), f(40.0)
A13, A15, A17, A30, A10 = f(13.0), f(15.0), f(17.0), f(30.0), f(10.0)
C032, C028, C0128, C0032, C05 = f(0.32), f(0.28), f(0.128), f(0.032), f(0.5)
TINY = f(1e-4)


def _xe(x, k):
    if abs(x) < TINY:
        return k - HALF * x
    return x / (math.exp(x / k) - ONE)


def _make(dev):
    xe = dev(_xe)

    def _deriv(V, m, h, n, p, I, gM, gE0, gI0):
        vr = V - VT32
        am = C032 * xe(A13 - vr, K4)
        bm = C028 * xe(vr - K40, K5)
        ah = C0128 * math.exp(-(vr - VS32 - A17) / K18)
        bh = K4 / (ONE + math.exp(-(vr - VS32 - K40) / K5))
        an = C0032 * xe(A15 - vr, K5)
        bn = C05 * math.exp((A10 - vr) / K40)
        ap = RP32 * xe(-(V + A30), K9)
        bp = RP32 * xe(V + A30, K9)
        Iion = (GL32 * (V - EL32) + GNA32 * m * m * m * h * (V - ENA32)
                + GK32 * n * n * n * n * (V - EK32) + gM * p * (V - EM32)
                + gE0 * (V - EEX32) + gI0 * (V - EIN32))
        return ((I - Iion) / CM32, am * (ONE - m) - bm * m, ah * (ONE - h) - bh * h,
                an * (ONE - n) - bn * n, ap * (ONE - p) - bp * p)

    deriv = dev(_deriv)

    def _step(y, P, I):
        gM, gE0, gI0, hs = P[0], P[1], P[2], P[3]
        ns = int(P[4])
        V, m, h, n, p = y[0], y[1], y[2], y[3], y[4]
        v_prev = V
        for _ in range(ns):
            a1, b1, c1, d1, e1 = deriv(V, m, h, n, p, I, gM, gE0, gI0)
            a2, b2, c2, d2, e2 = deriv(V + HALF * hs * a1, m + HALF * hs * b1, h + HALF * hs * c1,
                                       n + HALF * hs * d1, p + HALF * hs * e1, I, gM, gE0, gI0)
            a3, b3, c3, d3, e3 = deriv(V + HALF * hs * a2, m + HALF * hs * b2, h + HALF * hs * c2,
                                       n + HALF * hs * d2, p + HALF * hs * e2, I, gM, gE0, gI0)
            a4, b4, c4, d4, e4 = deriv(V + hs * a3, m + hs * b3, h + hs * c3, n + hs * d3,
                                       p + hs * e3, I, gM, gE0, gI0)
            V = V + SIXTH * hs * (a1 + TWO * a2 + TWO * a3 + a4)
            m = m + SIXTH * hs * (b1 + TWO * b2 + TWO * b3 + b4)
            h = h + SIXTH * hs * (c1 + TWO * c2 + TWO * c3 + c4)
            n = n + SIXTH * hs * (d1 + TWO * d2 + TWO * d3 + d4)
            p = p + SIXTH * hs * (e1 + TWO * e2 + TWO * e3 + e4)
        # NEST hh_cond_exp_destexhe: emit when V is above V_thresh and falling (the peak);
        # the engine's 3 ms refractory count suppresses repeats
        spk = 0
        if V > P[5] and v_prev > V:
            spk = 1
        y[0], y[1], y[2], y[3], y[4] = V, m, h, n, p
        return spk

    return dev(_step)


step = _make(njit(inline="always"))
step_cuda = _make(cuda.jit(device=True, inline=True))


def params(N, dt, gM, gE0=G_E0, gI0=G_I0, h_max=0.025, v_thresh=V_THRESH):
    nsub = max(1, int(math.ceil(dt / h_max - 1e-9)))
    P = np.empty((N, NP), np.float32)
    P[:] = (0.0, gE0, gI0, dt / nsub, nsub, v_thresh)
    P[:, 0] = gM
    return P


def gates_at(V):
    """steady-state gates (m, h, n, p) at voltage V (NEST initialises the gates this way)"""
    vr = V - V_T
    xe = lambda x, k: k if abs(x) < 1e-9 else x / (math.exp(x / k) - 1.0)
    am, bm = 0.32 * xe(13 - vr, 4), 0.28 * xe(vr - 40, 5)
    ah, bh = 0.128 * math.exp(-(vr - V_S - 17) / 18), 4 / (1 + math.exp(-(vr - V_S - 40) / 5))
    an, bn = 0.032 * xe(15 - vr, 5), 0.5 * math.exp((10 - vr) / 40)
    ap, bp = R_P * xe(-(V + 30), 9), R_P * xe(V + 30, 9)
    return am / (am + bm), ah / (ah + bh), an / (an + bn), ap / (ap + bp)
