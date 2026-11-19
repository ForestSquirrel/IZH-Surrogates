"""
neuron_spec.py -- the NeuronSpec dataclass that `models_offamily.resetfree_specs()`
constructs.

Upstream this type lives in `src_any_neuron/features_any_neuron.py`, whose module
-level imports pull in `an_config`, `an_features`, `an_protocols`, `an_integrator`
and four modules of `src/` -- nine files, none of which this pipeline uses. The
type itself is a plain record, so it is restated here with the same field names
and the same literal defaults, which drops all nine.

Safe because of how it is actually consumed: `resetfree_specs()` never sets
`params` (all five specs take the zeros(4) default) and none of the reset-free
RHS functions read `params` at all; `v3_an_zoo.targets()` reads only
rhs / y0 / params / v_th / rheo_lo / rheo_hi / h / t_discard / t_win and
immediately repacks them into a `V3Spec`. `rheo_neg_lo` and `noinh` are
inhibition-era fields, retained for construction compatibility and read by
nothing in this pipeline.
"""
from dataclasses import dataclass, field

import numpy as np

# Literal values of the upstream an_config defaults, as of the v3 freeze.
DEFAULT_V_TH        = 30.0
DEFAULT_RHEO_LO     = 0.0
DEFAULT_RHEO_HI     = 200.0
DEFAULT_RHEO_NEG_LO = -200.0
DT                  = 0.01
DEFAULT_T_WIN       = 500.0
DEFAULT_T_DISCARD   = 50.0


@dataclass
class NeuronSpec:
    rhs: object                                    # njit rhs(t, y, params, I, dydt)
    y0: np.ndarray                                 # full initial state
    params: np.ndarray = field(default_factory=lambda: np.zeros(4))
    v_th: float = DEFAULT_V_TH                     # spike threshold on y[0]
    rheo_lo: float = DEFAULT_RHEO_LO               # rheobase bracket, model current units
    rheo_hi: float = DEFAULT_RHEO_HI
    rheo_neg_lo: float = DEFAULT_RHEO_NEG_LO       # unused here (inhibition is out of scope)
    h: float = DT                                  # RK4 step, model time units
    t_win: float = DEFAULT_T_WIN                   # bootstrap observation window
    t_discard: float = DEFAULT_T_DISCARD           # bootstrap transient discard
    noinh: bool = False                            # unused here
    name: str = "model"
