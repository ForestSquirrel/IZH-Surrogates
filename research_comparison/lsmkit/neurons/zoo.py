"""
zoo.py -- the pipeline's zoo targets as network neurons (float32, CPU + CUDA).

Equations transcribed from pipeline/models_offamily.py (hh_rhs, morris_lecar_c1_rhs) and
lsmkit/zoo_targets.py (adex_rhs); spike rules as in the pipeline: upward crossing
of v_th (reset-free targets, latch kept in the state) or the AdEx reset at V_peak. Every
model integrates with RK4 at the step it was characterised with in the pipeline (HH 0.005,
ML 0.01, AdEx 0.01 ms), as nsub substeps inside one engine step; the spike test runs after
every substep. Never Euler: these are stiff near the spike.

    hh       y = (V, m, h, n, above)   P = (hs, nsub, v_th)
    ml_c1    y = (V, N, above)         P = (hs, nsub, v_th)
    adex     y = (V, w)                P = (hs, nsub, V_r, b)       V_peak = 30
"""
import math
import numpy as np
from numba import njit, cuda

f = np.float32
ONE, HALF, TWO, SIXTH, ZERO = f(1), f(0.5), f(2), f(1 / 6), f(0)
TINY = f(1e-4)

# ---- HH (models_offamily.hh_rhs) ------------------------------------------------------
H_C40, H_C10, H_C65, H_C18, H_C20, H_C35, H_C55, H_C80 = map(f, (40, 10, 65, 18, 20, 35, 55, 80))
H_01, H_4, H_007, H_001, H_0125, H_GNA, H_GK, H_GL = map(f, (0.1, 4, 0.07, 0.01, 0.125, 120, 36, 0.3))
H_ENA, H_EK, H_EL = f(50), f(-77), f(-54.387)
# ---- ML class 1 (models_offamily.morris_lecar_c1_rhs) -----------------------------------
M_V1, M_V2, M_V3, M_V4, M_GCA, M_GK, M_GL = map(f, (-1.2, 18, 12, 17.4, 4, 8, 2))
M_ECA, M_EK, M_EL, M_C, M_PHI = f(120), f(-84), f(-60), f(20), f(0.0667)
# ---- AdEx (C 200, gL 10, EL -65, VT -50, DT 2, a 2, tauw 300) ---------------
A_C, A_GL, A_EL, A_VT, A_DT, A_A, A_TW, A_VPK, A_XMAX = map(f, (200, 10, -65, -50, 2, 2, 300, 30, 10))


def _hh_f(V, m, h, n, I):
    x = V + H_C40
    am = H_01 * x / (ONE - math.exp(-x / H_C10)) if abs(x) > TINY else ONE
    bm = H_4 * math.exp(-(V + H_C65) / H_C18)
    ah = H_007 * math.exp(-(V + H_C65) / H_C20)
    bh = ONE / (ONE + math.exp(-(V + H_C35) / H_C10))
    z = V + H_C55
    an = H_001 * z / (ONE - math.exp(-z / H_C10)) if abs(z) > TINY else H_01
    bn = H_0125 * math.exp(-(V + H_C65) / H_C80)
    dV = I - H_GNA * m * m * m * h * (V - H_ENA) - H_GK * n * n * n * n * (V - H_EK) - H_GL * (V - H_EL)
    return dV, am * (ONE - m) - bm * m, ah * (ONE - h) - bh * h, an * (ONE - n) - bn * n


def _ml_f(V, N, I):
    m_inf = HALF * (ONE + math.tanh((V - M_V1) / M_V2))
    w_inf = HALF * (ONE + math.tanh((V - M_V3) / M_V4))
    tau_w = ONE / math.cosh((V - M_V3) / (TWO * M_V4))
    dV = (I - M_GCA * m_inf * (V - M_ECA) - M_GK * N * (V - M_EK) - M_GL * (V - M_EL)) / M_C
    return dV, M_PHI * (w_inf - N) * tau_w


def _adex_f(V, w, I):
    x = (V - A_VT) / A_DT
    if x > A_XMAX:
        x = A_XMAX
    return (-A_GL * (V - A_EL) + A_GL * A_DT * math.exp(x) - w + I) / A_C, (A_A * (V - A_EL) - w) / A_TW


def _make(dev):
    hh_f, ml_f, adex_f = dev(_hh_f), dev(_ml_f), dev(_adex_f)

    def hh_step(y, P, I):
        hs, ns, vth = P[0], int(P[1]), P[2]
        V, m, h, n, up = y[0], y[1], y[2], y[3], y[4]
        spk = 0
        for _ in range(ns):
            a1, b1, c1, d1 = hh_f(V, m, h, n, I)
            a2, b2, c2, d2 = hh_f(V + HALF * hs * a1, m + HALF * hs * b1, h + HALF * hs * c1, n + HALF * hs * d1, I)
            a3, b3, c3, d3 = hh_f(V + HALF * hs * a2, m + HALF * hs * b2, h + HALF * hs * c2, n + HALF * hs * d2, I)
            a4, b4, c4, d4 = hh_f(V + hs * a3, m + hs * b3, h + hs * c3, n + hs * d3, I)
            V = V + SIXTH * hs * (a1 + TWO * a2 + TWO * a3 + a4)
            m = m + SIXTH * hs * (b1 + TWO * b2 + TWO * b3 + b4)
            h = h + SIXTH * hs * (c1 + TWO * c2 + TWO * c3 + c4)
            n = n + SIXTH * hs * (d1 + TWO * d2 + TWO * d3 + d4)
            if V >= vth:
                if up == ZERO:
                    spk = 1
                up = ONE
            else:
                up = ZERO
        y[0], y[1], y[2], y[3], y[4] = V, m, h, n, up
        return spk

    def ml_step(y, P, I):
        hs, ns, vth = P[0], int(P[1]), P[2]
        V, N, up = y[0], y[1], y[2]
        spk = 0
        for _ in range(ns):
            a1, b1 = ml_f(V, N, I)
            a2, b2 = ml_f(V + HALF * hs * a1, N + HALF * hs * b1, I)
            a3, b3 = ml_f(V + HALF * hs * a2, N + HALF * hs * b2, I)
            a4, b4 = ml_f(V + hs * a3, N + hs * b3, I)
            V = V + SIXTH * hs * (a1 + TWO * a2 + TWO * a3 + a4)
            N = N + SIXTH * hs * (b1 + TWO * b2 + TWO * b3 + b4)
            if V >= vth:
                if up == ZERO:
                    spk = 1
                up = ONE
            else:
                up = ZERO
        y[0], y[1], y[2] = V, N, up
        return spk

    def adex_step(y, P, I):
        hs, ns, vr, b = P[0], int(P[1]), P[2], P[3]
        V, w = y[0], y[1]
        spk = 0
        for _ in range(ns):
            a1, b1 = adex_f(V, w, I)
            a2, b2 = adex_f(V + HALF * hs * a1, w + HALF * hs * b1, I)
            a3, b3 = adex_f(V + HALF * hs * a2, w + HALF * hs * b2, I)
            a4, b4 = adex_f(V + hs * a3, w + hs * b3, I)
            V = V + SIXTH * hs * (a1 + TWO * a2 + TWO * a3 + a4)
            w = w + SIXTH * hs * (b1 + TWO * b2 + TWO * b3 + b4)
            if V >= A_VPK:
                V = vr
                w = w + b
                spk = 1
        y[0], y[1] = V, w
        return spk

    return dev(hh_step), dev(ml_step), dev(adex_step)


hh_step, ml_step, adex_step = _make(njit(inline="always"))
hh_step_cuda, ml_step_cuda, adex_step_cuda = _make(cuda.jit(device=True, inline=True))


class _M:
    def __init__(self, step, step_cuda, dim, np_):
        self.step, self.step_cuda, self.DIM, self.NP = step, step_cuda, dim, np_


HH = _M(hh_step, hh_step_cuda, 5, 3)
ML = _M(ml_step, ml_step_cuda, 3, 3)
ADEX = _M(adex_step, adex_step_cuda, 2, 4)

# key -> (module-like, model step h, spike param, resting y0, extra params)
SPECS = {
    "hh": dict(model="zoo_hh", h=0.005, v_th=0.0, y0=(-65.0, 0.053, 0.596, 0.318, 0.0)),
    "ml_c1": dict(model="zoo_ml", h=0.01, v_th=10.0, y0=(-60.0, 0.0, 0.0)),
    "adex_adapting": dict(model="zoo_adex", h=0.01, Vr=-65.0, b=60.0, y0=(-65.0, 0.0)),
}


def params(key, N, dt):
    s = SPECS[key]
    nsub = max(1, int(math.ceil(dt / s["h"] - 1e-9)))
    hs = dt / nsub
    if s["model"] == "zoo_adex":
        P = np.empty((N, 4), np.float32); P[:] = (hs, nsub, s["Vr"], s["b"])
    else:
        P = np.empty((N, 3), np.float32); P[:] = (hs, nsub, s["v_th"])
    return P
