"""
v3_models.py -- the canonical Izhikevich right-hand sides.

These previously lived in `atlas_v3/v3_smoke.py`, a smoke-test file, and were
imported from there by eight modules including the evaluation runners and every
plot script. `izh_rhs_scaled_v3` was additionally defined a second time, with an
identical body, in `atlas_v3/v3_sensitivity.py`. Collecting both here is the one
structural change made while localizing the pipeline: the function the entire
schema is defined against should not be reachable only through a test.

Signature is the pipeline-wide RHS convention:

    rhs(t, y, params, I, dydt) -> None        writes into dydt

`params` is (a, b, c, d) for `izh_rhs_v3`. The reset (v -> c, u -> u + d) is
applied by the integrator, not here.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def izh_rhs_v3(t, y, params, I, dydt):
    v = y[0]; u = y[1]
    dydt[0] = 0.04 * v * v + 5.0 * v + 140.0 - u + I
    dydt[1] = params[0] * (params[1] * v - u)


@njit(cache=True)
def izh_rhs_scaled_v3(t, y, params, I, dydt):
    """The gauge RHS: params[4] = tau (time dilation), params[5] = kappa (current gain).

    params[0..3] stay (a, b, c, d) so the resets and the IC u0 = b*v0 are
    untouched -- tau is a pure dilation of the vector field, not a
    reparametrization. Used to measure the (tau, k_I) domain of validity.
    """
    v = y[0]; u = y[1]
    inv = 1.0 / params[4]
    dydt[0] = inv * (0.04 * v * v + 5.0 * v + 140.0 - u + params[5] * I)
    dydt[1] = inv * (params[0] * (params[1] * v - u))
