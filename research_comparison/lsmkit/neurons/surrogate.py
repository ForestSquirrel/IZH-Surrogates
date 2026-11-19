"""
surrogate.py -- a target's Izhikevich surrogate as a network neuron (gauge + voltage map).

posterior.json (written by lsmkit.infer) holds the pipeline's best draw theta = (a,b,c,d)
and its gauge:
    tau = T_cyc,target / T_cyc,surrogate     (paper time per model time)
    k_I = i_rh,target / i_rh,surrogate       (paper current per model current)
In a network the surrogate receives I_paper / k_I and advances dt / tau of its own time per
engine step. The gauge has no voltage term; voltage-dependent synapse terms and recordings
use an affine map V_paper = va * v + vb stored with the target.
"""
import json
from dataclasses import dataclass
import numpy as np

from . import izh
from ..spec import Binding


@dataclass
class Gauge:
    name: str
    theta: tuple
    tau: float
    k_I: float
    va: float
    vb: float
    view: str
    raw: dict

    @classmethod
    def load(cls, path, view="best"):
        js = json.load(open(path))
        b = js[view]
        vm = js["vmap"].get("views", {}).get(view, js["vmap"])     # per-view map if stored
        return cls(name=js["name"], theta=tuple(b["theta"]), tau=float(b["tau"]),
                   k_I=float(b["k_I"]), va=float(vm["va"]), vb=float(vm["vb"]),
                   view=view, raw=js)

    def binding(self, N, dt, bias, h_max=0.01):
        """Izhikevich binding for N neurons with the paper's bias current (paper units)."""
        f = np.float32
        return Binding(
            model="izh", P=izh.params(N, self.theta, dt, self.tau, h_max),
            tref=np.zeros(N, np.int32), clamp=np.zeros(N, np.int32),
            bias=np.broadcast_to(np.asarray(bias, f), (N,)).copy(),
            iscale=np.full(N, 1.0 / self.k_I, f), va=np.full(N, self.va, f),
            vb=np.full(N, self.vb, f), dim=2,
            meta=dict(gauge=dict(theta=list(self.theta), tau=self.tau, k_I=self.k_I,
                                 va=self.va, vb=self.vb, view=self.view)))

    def to_model_v(self, V):
        return (np.asarray(V, float) - self.vb) / self.va

    def rest(self, I_paper):
        return izh.rest(self.theta, I_paper / self.k_I)


def warm_state(g, I_paper, dt, T_ms=1000.0, h_max=0.01):
    """Isolated surrogate held at the paper bias I_paper for T_ms from (v, u) = (-65, -65 b)
    (the pipeline's initial condition). -> ((v, u), spikes in the second half). A non-zero
    count means the surrogate fires at the bias the target sits silently at (latching)."""
    from .izh import step
    p = izh.params(1, g.theta, dt, g.tau, h_max)[0]
    y = np.array([-65.0, -65.0 * g.theta[1]], np.float32)
    I = np.float32(I_paper / g.k_I)
    n = int(round(T_ms / dt))
    late = 0
    for k in range(n):
        if step(y, p, I) and k >= n // 2:
            late += 1
    return (float(y[0]), float(y[1])), late
