"""
lif_euler.py -- Ivanov & Michmizos 2021 LIF in its own discrete form (paper eq. 1; the authors' code).

    v <- v + dt * (-v / tau_v + I)          spike when v >= theta (outside refractoriness), v -> 0
I is the summed synaptic current in the paper's unitless per-ms units. Refractoriness (2 ms)
only blocks spike registration; v keeps integrating and is not reset while refractory
(register_spikes in the authors' code), so the counter lives in the model state and the engine's
own refractory mechanism is not used (tref = 0).
State y = (v, refractory steps left); P row = (dt / tau_v, dt, theta, tref_steps).
"""
import numpy as np
from numba import njit, cuda

NP = 4
DIM = 2
ONE = np.float32(1.0)
ZERO = np.float32(0.0)


def _step(y, p, I):
    v = y[0] - y[0] * p[0] + p[1] * I
    y[0] = v
    if y[1] > ZERO:
        y[1] = y[1] - ONE
        return 0
    if v >= p[2]:
        y[0] = ZERO
        y[1] = p[3]
        return 1
    return 0


step = njit(inline="always")(_step)
step_cuda = cuda.jit(device=True, inline=True)(_step)


def params(N, dt=1.0, tau_v=64.0, theta=20.0, t_ref=2.0):
    P = np.empty((N, NP), np.float32)
    P[:] = (dt / tau_v, dt, theta, round(t_ref / dt))
    return P
