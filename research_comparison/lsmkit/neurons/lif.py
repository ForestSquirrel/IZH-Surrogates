"""
lif.py -- current-based leaky integrate-and-fire, exact exponential update.

    tau_m dV/dt = -(V - V_rest) + R I        spike at V >= V_th, V -> V_reset
P row: [decay = exp(-dt/tau_m), R, V_th, V_reset, V_rest]. State y = [V] (dim 1; the engine
allocates dim >= 1). Refractoriness (clamp at V_reset) is applied by the engine.
"""
import numpy as np
from numba import njit, cuda

NP = 5
DIM = 1


def _step(y, p, I):
    vinf = p[4] + p[1] * I
    y[0] = vinf + (y[0] - vinf) * p[0]
    if y[0] >= p[2]:
        y[0] = p[3]
        return 1
    return 0


step = njit(inline="always")(_step)              # CPU engine
step_cuda = cuda.jit(device=True, inline=True)(_step)   # GPU engine


def params(N, dt, tau_m, R, v_th, v_reset, v_rest=0.0):
    P = np.empty((N, NP), np.float32)
    P[:] = (np.exp(-dt / tau_m), R, v_th, v_reset, v_rest)
    return P
