"""
circuit.py -- Ivanov baseline LSM wiring with the input neurons as ordinary network neurons.

Neuron ids: 0..NIN-1 input neurons, NIN..NIN+999 liquid (E first). Sources NIN+1000.. are the
external stimulus channels, one per input neuron: a stimulus event at time t opens a square
current pulse of I_PULSE for T_PULSE on its input neuron (a +A edge with delay 1 step and a
-A edge with delay 1 + T_PULSE/dt steps on the never-decaying receptor 1).
Every IL / LL weight is the single liquid weight w (sign: liquid E +, I -, inputs random);
synaptic current kernel exp(-t/tau_u)/tau_u (receptor 0: jump w/tau_u, decay exp(-dt/tau_u)).
"""
import numpy as np

import spec as S
from lsmkit.spec import Circuit, Receptors
from lsmkit.topology import lattice


def receptors():
    return Receptors(tau=np.array([S.TAU_U, 1e30]), cond=np.zeros(2, np.int32),
                     E=np.zeros(2, np.float32))


def build(nin, w, dt, rng, pulse=S.I_PULSE, input_neurons=True):
    """-> Circuit. input_neurons=True: N = nin + 1000 (input neurons are network neurons,
    driven by square pulses from nin stimulus channels). False (LIF-input control): N = 1000,
    the nin channels are the input neurons' precomputed spike trains and project onto the
    liquid directly (same IL wiring and weights). The random draws are identical in both
    cases, so a seed gives the same liquid and IL graph."""
    pos = lattice.grid(S.SHAPE)
    NL = len(pos)
    t = lattice.types_exact(NL, S.F_INH, rng)                     # 0 E / 1 I, random sites
    o = np.argsort(t, kind="stable")                              # E first
    pos, t = pos[o], t[o]
    d2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
    p = np.asarray(S.C)[t[:, None], t[None, :]] * np.exp(-d2 / S.LAM ** 2)
    np.fill_diagonal(p, 0.0)
    pre, post = np.nonzero(rng.random((NL, NL)) < p)
    ll_w = np.where(t[pre] == 0, w, -w)
    m = rng.random((nin, NL)) < S.IN_DENSITY                      # input -> liquid, 15 %
    ipre, ipost = np.nonzero(m)
    il_w = np.where(rng.random(len(ipre)) < 0.5, w, -w)           # random sign
    liq_types = np.where(t == 0, 1, -1).astype(np.int8)
    meta = dict(n_ll=int(len(pre)), n_il=int(len(ipre)), w=w, input_neurons=input_neurons)
    if not input_neurons:
        N = NL
        src = np.concatenate([pre, N + ipre])
        dst = np.concatenate([post, ipost])
        ww = np.concatenate([ll_w, il_w]) / S.TAU_U
        n = len(src)
        return Circuit(N=N, nin=nin, types=liq_types, src=src.astype(np.int32), post=dst.astype(np.int32),
                       w=ww.astype(float), dly=np.ones(n, np.int32), rec=np.zeros(n, np.int32),
                       U=np.ones(n), D=np.ones(n), F=np.ones(n), stp=np.zeros(n, np.int32), meta=meta)
    N = nin + NL
    ns = int(round(S.T_PULSE / dt))
    ch = np.arange(nin)
    src = np.concatenate([pre + nin, ipre, N + ch, N + ch])
    dst = np.concatenate([post + nin, ipost + nin, ch, ch])
    ww = np.concatenate([ll_w, il_w, np.full(nin, pulse), np.full(nin, -pulse)])
    syn = np.concatenate([np.zeros(len(pre) + len(ipre), np.int32), np.ones(2 * nin, np.int32)])
    ww = np.where(syn == 0, ww / S.TAU_U, ww)
    dly = np.concatenate([np.ones(len(pre) + len(ipre) + nin, np.int32), np.full(nin, 1 + ns, np.int32)])
    n = len(src)
    types = np.concatenate([np.ones(nin, np.int8), liq_types])
    return Circuit(N=N, nin=nin, types=types, src=src.astype(np.int32), post=dst.astype(np.int32),
                   w=ww.astype(float), dly=dly, rec=syn, U=np.ones(n), D=np.ones(n), F=np.ones(n),
                   stp=np.zeros(n, np.int32), meta=meta)


def build_inputs_only(nin, dt, pulse):
    """only the input neurons and their stimulus pulses (no liquid): the '@fastin' input
    layer, whose spikes do not depend on the liquid"""
    ns = int(round(S.T_PULSE / dt))
    ch = np.arange(nin)
    n = 2 * nin
    return Circuit(N=nin, nin=nin, types=np.ones(nin, np.int8),
                   src=np.concatenate([nin + ch, nin + ch]).astype(np.int32),
                   post=np.concatenate([ch, ch]).astype(np.int32),
                   w=np.concatenate([np.full(nin, pulse), np.full(nin, -pulse)]).astype(float),
                   dly=np.concatenate([np.ones(nin, np.int32), np.full(nin, 1 + ns, np.int32)]),
                   rec=np.ones(n, np.int32), U=np.ones(n), D=np.ones(n), F=np.ones(n),
                   stp=np.zeros(n, np.int32), meta=dict(inputs_only=True))
