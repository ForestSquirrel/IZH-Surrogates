"""
circuit.py -- build the Schulte data-based circuit and bind HH targets / surrogates.

Source ids: neurons 0..559 (population blocks in spec.POPS order), then stream 1 channels
0..39 and stream 2 channels 40..79. Receptors: 0 exc synapse, 1 inh synapse,
2 / 3 intrinsic OU conductance noise (exc / inh; sigma 0 for the noise-free variant).
"""
import os
import numpy as np

import spec as S
from lsmkit.spec import Binding, Receptors
from lsmkit.topology import populations as TP
from lsmkit.neurons import hh_dp as H
from lsmkit.neurons.surrogate import Gauge

HERE = os.path.dirname(os.path.abspath(__file__))
IDX, N = TP.layout(S.POPS)
assert N == S.N
TYPES = np.concatenate([np.full(n, 1 if S.PTYPE[k] == 0 else -1, np.int8) for k, n in S.POPS.items()])


def receptors(noise=True):
    sig = S.NOISE["sig"] if noise else (0.0, 0.0)
    return Receptors(tau=np.array([*S.TAU_SYN, *S.NOISE["tau"]]),
                     cond=np.ones(4, np.int32),
                     E=np.array([S.E_EX, S.E_IN, S.E_EX, S.E_IN], np.float32),
                     sig=np.array([0.0, 0.0, *sig]))


def weight(A, src_type):
    E = S.E_EX if src_type == 0 else S.E_IN
    return A * S.G_L / abs(E - S.V_MEAN)


def build(rng, use_stp=True):
    P = {s: {d: v[1] for d, v in row.items()} for s, row in S.CONN.items()}
    W = {s: {d: S.S_RW * weight(v[0], S.PTYPE[s]) for d, v in row.items()} for s, row in S.CONN.items()}
    rec = TP.connect(IDX, S.PTYPE, P, W, S.W_CV, S.STP, S.DELAY, S.D_CV, S.DT, rng, use_stp, S.UDF_CV)
    w_in = weight(S.A_IN, 0)
    in1 = TP.inputs(IDX, N, 0, S.N_CH, S.IN1, S.S1 * w_in, S.W_CV, S.DELAY_IN, S.DT, rng)
    in2 = TP.inputs(IDX, N, S.N_CH, S.N_CH, S.IN2, S.S2 * w_in, S.W_CV, S.DELAY_IN, S.DT, rng)
    return TP.circuit(N, 2 * S.N_CH, TYPES, [rec, in1, in2])


def bind_target(dt=S.DT):
    f = np.float32
    gM = np.where(TYPES > 0, H.G_M_E, H.G_M_I)
    P = H.params(N, dt, 0.0)
    P[:, 0] = gM
    return Binding(model="hh_dp", P=P, tref=np.full(N, int(round(H.T_REF / dt)), np.int32),
                   clamp=np.zeros(N, np.int32), bias=np.zeros(N, f), iscale=np.ones(N, f),
                   va=np.ones(N, f), vb=np.zeros(N, f), dim=5, meta=dict(model="HH-DP"))


def gauges(view="best"):
    return {t: Gauge.load(os.path.join(HERE, "targets", f"hh_{t}.json"), view) for t in "EI"}


def bind_surrogate(g, dt=S.DT):
    """per-neuron Izhikevich parameters: E populations take the HH-E surrogate, I the HH-I"""
    bE, bI = g["E"].binding(N, dt, 0.0), g["I"].binding(N, dt, 0.0)
    e = TYPES > 0
    pick = lambda a, b: np.where(e[:, None] if a.ndim == 2 else e, a, b)
    return Binding(model="izh", P=pick(bE.P, bI.P).astype(np.float32), tref=np.zeros(N, np.int32),
                   clamp=np.zeros(N, np.int32), bias=np.zeros(N, np.float32),
                   iscale=pick(bE.iscale, bI.iscale), va=pick(bE.va, bI.va), vb=pick(bE.vb, bI.vb),
                   dim=2, meta=dict(model="izh", E=bE.meta["gauge"], I=bI.meta["gauge"]))


def y0_target(V):
    Y = np.zeros(V.shape + (5,), np.float32)
    Y[..., 0] = V
    for k in np.ndindex(V.shape):
        Y[k + (slice(1, 5),)] = H.gates_at(float(V[k]))
    return Y


def y0_surrogate(V, g):
    """the same random V0 through each neuron's voltage map; u = b v"""
    Y = np.zeros(V.shape + (2,), np.float32)
    e = TYPES > 0
    for t, m in (("E", e), ("I", ~e)):
        v = g[t].to_model_v(V[..., m])
        Y[..., m, 0] = v
        Y[..., m, 1] = g[t].theta[1] * v
    return Y
