"""
my_model.py -- a runnable "bring your own model" example.

Two targets, one of each kind, plus a deliberate demonstration of the I_RH_MIN
units trap. Run them with:

    cd pipeline_final
    PYTHONPATH=examples python scripts/infer_continuous.py --target my_model:fhn
    PYTHONPATH=examples python scripts/infer_continuous.py --target my_model:tiny_current_bad
    PYTHONPATH=examples python scripts/infer_continuous.py --target my_model:tiny_current_good

The second and third are the SAME neuron. They differ only in the unit its
injected current is quoted in, and they score differently -- see section 7 of
README.md.
"""
import numpy as np
from numba import njit

from v3_an_features import V3Spec


# ---------------------------------------------------------------- example 1 --
# FitzHugh-Nagumo. Reset-free: V is a continuous ODE solution and a "spike" is
# whatever we declare it to be by choosing v_th.
@njit(cache=True)
def fhn_rhs(t, y, params, I, dydt):
    v, w = y[0], y[1]
    a, b, eps = params[0], params[1], params[2]
    dydt[0] = v - v * v * v / 3.0 - w + I
    dydt[1] = eps * (v + a - b * w)


fhn = V3Spec(
    rhs=fhn_rhs,
    y0=np.array([-1.2, -0.62]),      # y[0] is the observable AND the spike variable
    params=np.array([0.7, 0.8, 0.08, 0.0]),
    v_th=1.0,                        # a modelling choice, not a property of the model
    rheo_lo=0.0, rheo_hi=2.0,        # must BRACKET the true rheobase, in MODEL units
    h=0.005,                         # must resolve the fastest timescale
    boot_t_trans=100.0, boot_t_win=800.0, boot_t_disc=100.0,
    name="my FHN")


# ------------------------------------------------- on the I_RH_MIN units trap --
# A toy model was tried here and removed. It did show the four I_RH_MIN-gated
# columns returning when the current unit was rescaled (6/18 -> 9/18 defined),
# but it also failed to define nine OTHER columns because the toy never settles
# into a clean cycle -- so it demonstrated "badly specified model" at least as
# loudly as it demonstrated the units trap, which would mislead.
#
# For the units trap and its workaround, see section 7 of README.md.
