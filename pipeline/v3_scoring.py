"""
v3_scoring.py -- the two scorers, and the settled-ISI extraction they need.

These were previously stranded inside the evaluation runners
(`v3_eval_infamily.py`, `v3_eval_offfamily.py`) and imported back out of them by
the plot scripts, which made a runner a library. They are collected here
unchanged.

THE TWO METRICS ARE NOT INTERCHANGEABLE, AND BOTH ARE REPORTED

  d2   feature-space distance, in atlas sigma. ATLAS-RELATIVE: sigma is fitted
       per atlas, so d2 is not comparable across atlases or schemas. Largely
       measures family distance.
  w1   Wasserstein-1 between settled-ISI distributions. Atlas-free, comparable
       across checkpoints and model families, well defined on atoms (a tonic
       settled train IS a point mass), and a proper metric, which d2 is not.

Measured to be close to orthogonal, from both directions: across 24 cells
spanning six families Spearman(regime instability, d2) = +0.418 and is carried
entirely by two extremes, while the same statistic predicts w1_shape at +0.723.
A verdict quoting only one
of them is incomplete.
"""
import numpy as np

import v3_config as C
from v3_models import izh_rhs_v3

MISS_PENALTY = 9.0        # z^2 charged per target dim the draw fails to define


def d2(a, b, sigma, eps=1e-12):
    """SUM of squared z over jointly-defined dims, PLUS a penalty per dim the
    target defines and the draw does not. Returns (score, n_joint).

    B14, and the reason it is required rather than cosmetic. A MEAN over joint
    dims is not comparable across draws with different |M|: a draw that defines
    a single feature and matches it scores ~0 and wins outright. That is not
    hypothetical -- it is what the first version of this scorer did on 4 of the
    first 5 presets, reporting `d2 = 0.000 (1d)` and selecting a degenerate cell
    while looking like a triumph.

    The penalty is charged against the TARGET's defined set, so failing to
    define a feature the target has costs MISS_PENALTY = 9 (a 3-sigma miss)
    rather than being free. Extra features the draw defines and the target does
    not are not charged: the target is the thing being matched.
    """
    dt = a > C.SENT_CUT
    m = dt & (b > C.SENT_CUT)
    if not m.any():
        return np.inf, 0
    z = (a[m] - b[m]) / np.maximum(sigma[m], eps)
    miss = int(dt.sum() - m.sum())
    return float(np.sum(z * z) + MISS_PENALTY * miss), int(m.sum())


def w1(x, y):
    """Wasserstein-1 between two ISI samples, by quantile coupling.

    No binning: several schema components are atoms and the tie policy alone
    moves derived quantities materially, so a histogram distance is the wrong
    tool here. Returns nan when either side has fewer than 2 intervals.
    """
    if x is None or y is None or len(x) < 2 or len(y) < 2:
        return np.nan
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def isis_of(theta, bufs, gamma=2.0, rhs=None, rheo_hi=None):
    """Settled ISI sequence under P1's reference step, for a RESET model.

    Returns (isis, landmarks) with isis in absolute time units, or (None, lm).

    `rhs` defaults to Izhikevich but MUST be passed for a foreign reset model.
    It was hardcoded here originally, and the out-of-family scorer then fed
    AdEx's parameter vector to the Izhikevich RHS -- simulating an Izhikevich
    cell with a=0, b=0, c=-65, d=0 rather than AdEx. Three of four AdEx regimes
    returned nan (that bogus cell has no anchor) and the fourth returned a w1
    computed from the wrong model, which is the more dangerous outcome.
    """
    from v3_landmarks import landmarks as v3_landmarks
    from v3_core import integrate_v3
    from v3_config import STIM_PARAMS_LEN, STATE_DIM, V_REST, DT
    if rhs is None:
        rhs = izh_rhs_v3
    lm = (v3_landmarks(rhs, theta, bufs) if rheo_hi is None
          else v3_landmarks(rhs, theta, bufs, rheo_hi=float(rheo_hi)))
    if lm["T_cycle"] < C.SENT_CUT or lm["i_rh"] < C.SENT_CUT:
        return None, lm
    tc = lm["T_cycle"]
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = C.K_TRANS * tc
    sp[1] = gamma * lm["i_rh"]
    y = np.empty(STATE_DIM); y[0] = V_REST; y[1] = theta[1] * V_REST
    st = bufs["spike_times"]
    t_end = sp[0] + C.K_WIN * tc
    ns, nt, inv = integrate_v3(rhs, 0, sp, 1.0, y, 0.0, t_end, DT,
                               st, theta[2], theta[3], theta)
    if inv or ns < 3 or ns != nt:
        return None, lm
    t = st[:ns]
    t = t[t >= sp[0] + C.K_DISCARD * tc]
    if len(t) < 3:
        return None, lm
    return np.diff(t), lm


def isis_an(spec, i_rh, tc, buf=1 << 16):
    """Settled ISI sequence for a RESET-FREE target, already in T_cycle units.

    Dimensionless on return, which is what makes w1_shape tau-invariant by
    construction: a foreign model's time units never enter the comparison.
    """
    from v3_an_core import integrate_an, make_sp
    st = np.empty(buf)
    trans = C.K_TRANS * tc
    sp = make_sp(trans, C.GAMMA_STEP * i_rh)
    ns, nt, inv = integrate_an(spec.rhs, C.STIM_STEP, sp, 1.0, spec.y0, 0.0,
                               trans + C.K_WIN * tc, spec.h, spec.v_th, st,
                               spec.params)
    if inv or ns != nt or ns < 3:
        return None
    t = st[:ns]
    t = t[t >= trans + C.K_DISCARD * tc]
    return np.diff(t) / tc if len(t) >= 3 else None
