"""
izh.py -- Izhikevich neuron for the network engines, float32, RK4 with substeps.

    dv/ds = 0.04 v^2 + 5 v + 140 - u + I,   du/ds = a (b v - u),   v >= 30: v -> c, u -> u + d
s is the model's own time. The engine step is dt of paper time; with the gauge
tau = T_cyc,target / T_cyc,surrogate one engine step is dt / tau of model time, split into
nsub RK4 substeps of hs = dt / (tau * nsub) <= h_max (0.01, the pipeline's step, so the
surrogate is integrated as it was fitted). The spike test runs after every substep
(pipeline order: RK4, then the v >= 30 test and reset).
P row: [a, b, c, d, hs, nsub].
"""
import math
import numpy as np
from numba import njit, cuda

NP = 6
DIM = 2
C004 = np.float32(0.04)
C5 = np.float32(5.0)
C140 = np.float32(140.0)
VPEAK = np.float32(30.0)
HALF = np.float32(0.5)
SIXTH = np.float32(1.0 / 6.0)
TWO = np.float32(2.0)


def _step(y, p, I):
    a, b, c, d, h = p[0], p[1], p[2], p[3], p[4]
    ns = int(p[5])
    v = y[0]
    u = y[1]
    spk = 0
    for _ in range(ns):
        # RK4, RHS written out so the same source compiles for CPU and CUDA
        k1v = C004 * v * v + C5 * v + C140 - u + I
        k1u = a * (b * v - u)
        vv = v + HALF * h * k1v
        uu = u + HALF * h * k1u
        k2v = C004 * vv * vv + C5 * vv + C140 - uu + I
        k2u = a * (b * vv - uu)
        vv = v + HALF * h * k2v
        uu = u + HALF * h * k2u
        k3v = C004 * vv * vv + C5 * vv + C140 - uu + I
        k3u = a * (b * vv - uu)
        vv = v + h * k3v
        uu = u + h * k3u
        k4v = C004 * vv * vv + C5 * vv + C140 - uu + I
        k4u = a * (b * vv - uu)
        v = v + SIXTH * h * (k1v + TWO * k2v + TWO * k3v + k4v)
        u = u + SIXTH * h * (k1u + TWO * k2u + TWO * k3u + k4u)
        if v >= VPEAK:
            v = c
            u = u + d
            spk = 1
    y[0] = v
    y[1] = u
    return spk


step = njit(inline="always")(_step)
step_cuda = cuda.jit(device=True, inline=True)(_step)


def substeps(dt, tau, h_max=0.01):
    """-> (hs, nsub): model-time substep and count for one engine step of dt."""
    ds = dt / tau
    nsub = max(1, int(math.ceil(ds / h_max - 1e-9)))
    return ds / nsub, nsub


def params(N, theta, dt, tau, h_max=0.01):
    a, b, c, d = theta
    hs, nsub = substeps(dt, tau, h_max)
    P = np.empty((N, NP), np.float32)
    P[:] = (a, b, c, d, hs, nsub)
    return P


def rest(theta, I):
    """Stable resting state (v, u) at constant model current I, or None if the cell has
    no rest there (above the saddle-node, it fires tonically). With u = b v at rest,
    0.04 v^2 + (5 - b) v + 140 + I = 0; the lower root is the node for the usual a, b."""
    a, b, c, d = theta
    disc = (5.0 - b) ** 2 - 4 * 0.04 * (140.0 + I)
    if disc < 0:
        return None
    v = (-(5.0 - b) - math.sqrt(disc)) / (2 * 0.04)
    # Jacobian [[0.08 v + 5, -1], [a b, -a]] must be stable
    tr = 0.08 * v + 5 - a
    det = -(0.08 * v + 5) * a + a * b
    if tr >= 0 or det <= 0:
        return None
    return v, b * v
