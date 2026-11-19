"""
objective.py -- the search space and the objective.

Only the four Izhikevich shape parameters are searched. The time scale tau and the current
gain k_I are NOT decision variables: every protocol is stated in units of the cell's own
cycle and rheobase, so the features cannot depend on them, and after optimization they
follow by division. That is what turns a 6-D unit-dependent fit into a 4-D unit-free one.

    sp  = Space()                      # u in [0,1]^4  <->  theta
    tgt = Target(name, f_t, sigma)     # one target's frozen feature vector
    J   = tgt.J(F, lm, ok)             # from evalbatch.evaluate's output
"""
import numpy as np

import common as K
import v3_config as C
import evalbatch as EB


def box():
    """The parameter box, read from the atlas metadata rather than restated.

    It must be the atlas's box: the cINN was trained on cells drawn from it, so a
    comparison on a different box would not be a comparison of methods.
    """
    import os
    meta = K.jread(os.path.join(K.ATLAS, "izh_meta.json"), {}) or {}
    lo, hi = meta.get("param_lo"), meta.get("param_hi")
    if not lo or not hi:
        raise SystemExit(f"no param_lo/param_hi in {K.ATLAS}/izh_meta.json")
    return np.asarray(lo, float), np.asarray(hi, float)


class Space:
    """u in [0,1]^4 <-> theta = (a, b, c, d), with `a` optionally on a log scale.

    Log-scaling `a` is the one transform applied: it spans 0.001 to 0.3, two
    and a half decades, and enters as a rate, so a linear coordinate spends most of its
    range on cells whose recovery variable is effectively frozen.
    """

    names = ("a", "b", "c", "d")

    def __init__(self, log_a=None):
        self.lo, self.hi = box()
        self.log_a = K.LOG_A if log_a is None else log_a
        self.la, self.ha = np.log10(self.lo[0]), np.log10(self.hi[0])

    def to_theta(self, u):
        u = np.atleast_2d(np.asarray(u, float))
        th = self.lo[None, :] + np.clip(u, 0.0, 1.0) * (self.hi - self.lo)[None, :]
        if self.log_a:
            th[:, 0] = 10.0 ** (self.la + np.clip(u[:, 0], 0.0, 1.0) * (self.ha - self.la))
        return th

    def to_u(self, theta):
        th = np.atleast_2d(np.asarray(theta, float))
        u = (th - self.lo[None, :]) / (self.hi - self.lo)[None, :]
        if self.log_a:
            u[:, 0] = (np.log10(th[:, 0]) - self.la) / (self.ha - self.la)
        return u

    def at_bound(self, u, tol=0.01):
        """Spec G6: report a solution sitting on a bound, never widen the box in silence."""
        u = np.asarray(u, float).ravel()
        return {n: ("lo" if v < tol else "hi") for n, v in zip(self.names, u)
                if v < tol or v > 1.0 - tol}


class Target:
    """One target's frozen feature vector and the objective built on it."""

    def __init__(self, name, f_t, sigma, lam_period=None, meta=None):
        self.name = name
        self.f_t = np.asarray(f_t, float)[:C.N_ACTIVE]
        self.sigma = np.asarray(sigma, float)[:C.N_ACTIVE]
        self.mask = self.f_t > C.SENT_CUT
        self.n_D = int(self.mask.sum())
        self.lam_period = K.LAM_PERIOD if lam_period is None else lam_period
        ip = C.FEATURE_NAMES.index("p")
        self.p_t = float(self.f_t[ip]) if self.mask[ip] else None
        self.meta = meta or {}
        if self.n_D < K.N_DEF_MIN:
            raise ValueError(f"{name}: target defines only {self.n_D} features")

    def J(self, F, lm, ok):
        return EB.score(self.f_t, F, lm, ok, self.sigma,
                        lam_period=self.lam_period, p_t=self.p_t)

    def of_theta(self, theta):
        """Convenience for one-off checks; the runs go through the pooled evaluator."""
        F, lm, ok = EB.evaluate(theta)
        J, nj = self.J(F, lm, ok)
        return J, nj, F, lm, ok

    def gauge(self, lm_row):
        """tau and k_I by division, never searched.

        tau = T_cycle(target) / T_cycle(surrogate), k_I = i_rh(target) / i_rh(surrogate).
        """
        irh_s, tc_s = float(lm_row[0]), float(lm_row[1])
        if tc_s <= C.SENT_CUT or irh_s <= C.SENT_CUT:
            return None, None
        tc_t = self.meta.get("T_cycle")
        irh_t = self.meta.get("i_rh")
        if tc_t is None or irh_t is None:
            return None, None
        return float(tc_t) / tc_s, float(irh_t) / irh_s
