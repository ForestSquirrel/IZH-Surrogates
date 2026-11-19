"""
zoo_targets.py -- the out-of-family target models as specs, for the network studies.

spec(key) returns the pipeline's own V3Spec (pipeline/v3_an_zoo.py), so a target simulated
in a network is exactly the cell the pipeline was scored on. AdEx is not in the pipeline's reset-free zoo (it has a spike reset); its four regimes are
defined here with `reset_kind = "adex"`: a spike is the reset itself (v >= V_peak -> v = V_r,
w += b), not an upward crossing of v_th.

keys: fhn, adex_{tonic,adapting,initial_burst,bursting_L2}, ml_c1, ml_c2, hh,
      wilson, wilson_m025, wilson_m05, hr_008, hr_015, hr_020
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.dirname(os.path.dirname(HERE))                    # pipeline_final
for _p in (os.path.join(PF, "pipeline"), os.path.join(PF, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ZOO_NAME = {
    "fhn": "FitzHugh-Nagumo",
    "ml_c1": "Morris-Lecar C1", "ml_c2": "Morris-Lecar C2", "hh": "Hodgkin-Huxley",
    "wilson": "Wilson", "wilson_m025": "Wilson vth-0.25", "wilson_m05": "Wilson vth-0.5",
    "hr_008": "Hindmarsh-Rose r=0.008", "hr_015": "Hindmarsh-Rose r=0.015",
    "hr_020": "Hindmarsh-Rose r=0.02",
}

# ---------------------------------------------------------------- AdEx
DT_ADEX = 0.01
ADEX = dict(C=200.0, gL=10.0, EL=-65.0, VT=-50.0, DT=2.0, a=2.0, tauw=300.0, Vpeak=30.0,
            xclip=10.0)
ADEX_REGIMES = {
    "tonic":         dict(name="AdEx tonic",           Vr=-65.0,  b=0.0),
    "adapting":      dict(name="AdEx adapting",        Vr=-65.0,  b=60.0),
    "initial_burst": dict(name="AdEx initial burst",   Vr=-51.0,  b=100.0),
    "bursting":      dict(name="AdEx bursting",        Vr=-46.0,  b=120.0),
    # the original bursting target sits on its own tonic -> doublet transition at
    # gamma = 1.98 (= the anchor); this one keeps doublets on gamma in [1, 8]
    "bursting_L2":   dict(name="AdEx bursting (deep)", Vr=-45.08, b=150.0),
}


@njit(cache=False)
def adex_rhs(t, y, params, I, dydt):
    """AdEx with the ADEX constants. params = (V_r, b, -, -); the reset uses the first two."""
    V = y[0]
    w = y[1]
    x = (V - (-50.0)) / 2.0
    if x > 10.0:
        x = 10.0
    dydt[0] = (-10.0 * (V - (-65.0)) + 10.0 * 2.0 * np.exp(x) - w + I) / 200.0
    dydt[1] = (2.0 * (V - (-65.0)) - w) / 300.0


@dataclass
class AdExSpec:
    name: str
    rhs: object
    y0: tuple
    params: tuple
    v_th: float
    h: float
    reset_kind: str = "adex"
    v_peak: float = 30.0


def adex_spec(key):
    r = ADEX_REGIMES[key.replace("adex_", "")]
    return AdExSpec(name=r["name"], rhs=adex_rhs, y0=(-65.0, 0.0),
                    params=(r["Vr"], r["b"], 0.0, 0.0), v_th=ADEX["Vpeak"], h=DT_ADEX)


# ---------------------------------------------------------------- registry
def spec(key):
    """-> the target's spec, identical to the pipeline's (AdEx: AdExSpec above)."""
    if key.startswith("adex_"):
        return adex_spec(key)
    from v3_an_zoo import targets
    hits = [s for n, s in targets() if n == ZOO_NAME[key]]
    if len(hits) != 1:
        raise KeyError(key)
    return hits[0]
