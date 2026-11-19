"""
spec.py -- host-side network description shared by the CPU and GPU engines.

A Circuit is one wired network instance (topology + synapse parameters). Sources are the
N reservoir neurons (ids 0..N-1) followed by NIN input channels (ids N..N+NIN-1); every
edge is an out-edge of a source:

    src, post        source id, reservoir id of the target
    w                signed amplitude in the paper's units (nA for current receptors,
                     nS for conductance receptors)
    dly              delay in steps (>= 1)
    rec              receptor index on the post neuron
    U, D, F          Tsodyks-Markram use / depression / facilitation (D, F in ms)
    stp              1 = dynamic synapse, 0 = static (amplitude w)

A Binding attaches a neuron model to the reservoir neurons of a circuit: per-neuron model
parameters P, refractory steps and clamp flag, bias current, and the gauge terms
(iscale: current multiplier into model units; va, vb: model voltage -> paper voltage,
V_paper = va * y[0] + vb, used by conductance receptors and recordings).

Receptors (shared by all neurons): exponential decay tau (ms), conductance flag, E_rev,
optional zero-mean OU fluctuation (sig) for intrinsic conductance noise.
Everything the engines touch is float32 / int32 / int64.
"""
from dataclasses import dataclass, field
import numpy as np

F32 = np.float32


@dataclass
class Receptors:
    tau: np.ndarray                  # (R,) ms
    cond: np.ndarray                 # (R,) 0 current, 1 conductance
    E: np.ndarray                    # (R,) reversal potential (paper voltage units)
    sig: np.ndarray = None           # (R,) stationary SD of a zero-mean OU fluctuation added
                                     # to the receptor every step (0 = none); Schulte's
                                     # intrinsic conductance noise (eq. 18-23)

    def decay(self, dt):
        return np.exp(-dt / np.asarray(self.tau, float)).astype(F32)

    def noise_amp(self, dt):
        """exact OU increment SD per step: sig * sqrt(1 - exp(-2 dt / tau))"""
        sig = np.zeros(len(self.tau)) if self.sig is None else np.asarray(self.sig, float)
        return (sig * np.sqrt(1 - np.exp(-2 * dt / np.asarray(self.tau, float)))).astype(F32)


@dataclass
class Circuit:
    N: int
    nin: int
    types: np.ndarray                # (N,) +1 E, -1 I
    src: np.ndarray
    post: np.ndarray
    w: np.ndarray
    dly: np.ndarray
    rec: np.ndarray
    U: np.ndarray
    D: np.ndarray
    F: np.ndarray
    stp: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def nsrc(self):
        return self.N + self.nin

    def csr(self):
        """-> (ptr (nsrc+1,), order) with edges grouped by source."""
        order = np.argsort(self.src, kind="stable")
        ptr = np.zeros(self.nsrc + 1, np.int64)
        np.add.at(ptr, self.src + 1, 1)
        return np.cumsum(ptr), order


@dataclass
class Binding:
    model: str                       # key into lsmkit.neurons.MODELS
    P: np.ndarray                    # (N, NP) float32
    tref: np.ndarray                 # (N,) int32 refractory steps
    clamp: np.ndarray                # (N,) int32 1 = state frozen while refractory
    bias: np.ndarray                 # (N,) float32 paper current units
    iscale: np.ndarray               # (N,) float32 paper current -> model current
    va: np.ndarray                   # (N,) float32
    vb: np.ndarray                   # (N,) float32
    dim: int
    meta: dict = field(default_factory=dict)


@dataclass
class Network:
    """Several circuits stacked for one engine call (all with the same N, nin, model)."""
    N: int
    nin: int
    model: str
    dim: int
    out_ptr: np.ndarray              # (C, nsrc+1) int64 absolute edge indices
    e_post: np.ndarray
    e_w: np.ndarray
    e_dly: np.ndarray
    e_rec: np.ndarray
    e_U: np.ndarray
    e_D: np.ndarray
    e_F: np.ndarray
    e_stp: np.ndarray
    P: np.ndarray                    # (C, N, NP)
    tref: np.ndarray                 # (C, N)
    clamp: np.ndarray
    bias: np.ndarray
    iscale: np.ndarray
    va: np.ndarray
    vb: np.ndarray
    rec: Receptors
    dmax: int


def stack(circuits, bindings, receptors):
    c0, b0 = circuits[0], bindings[0]
    assert all(c.N == c0.N and c.nin == c0.nin for c in circuits)
    assert all(b.model == b0.model for b in bindings)
    ptrs, cols = [], {k: [] for k in ("post", "w", "dly", "rec", "U", "D", "F", "stp")}
    off = 0
    for c in circuits:
        ptr, order = c.csr()
        ptrs.append(ptr + off)
        off += len(order)
        for k in cols:
            cols[k].append(getattr(c, k)[order])
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    dmax = int(cat["dly"].max()) + 1 if len(cat["dly"]) else 1
    sb = lambda k, t: np.ascontiguousarray(np.stack([getattr(b, k) for b in bindings]), t)
    return Network(
        N=c0.N, nin=c0.nin, model=b0.model, dim=b0.dim,
        out_ptr=np.ascontiguousarray(np.stack(ptrs), np.int64),
        e_post=cat["post"].astype(np.int32), e_w=cat["w"].astype(F32),
        e_dly=cat["dly"].astype(np.int32), e_rec=cat["rec"].astype(np.int32),
        e_U=cat["U"].astype(F32), e_D=cat["D"].astype(F32), e_F=cat["F"].astype(F32),
        e_stp=cat["stp"].astype(np.int32),
        P=sb("P", F32), tref=sb("tref", np.int32), clamp=sb("clamp", np.int32),
        bias=sb("bias", F32), iscale=sb("iscale", F32), va=sb("va", F32), vb=sb("vb", F32),
        rec=receptors, dmax=dmax)


def events(spike_lists, dt, nin):
    """spike_lists[k] = (times_ms, channels) for instance k -> CSR over instances of input
    events sorted by step (ev_ptr, ev_step, ev_src as channel index 0..nin-1)."""
    ptr = [0]
    steps, chans = [], []
    for t, ch in spike_lists:
        t = np.asarray(t, float)
        ch = np.asarray(ch, np.int32)
        s = np.floor(t / dt).astype(np.int64)
        keep = s >= 0
        s, ch = s[keep], ch[keep]
        o = np.argsort(s, kind="stable")
        steps.append(s[o])
        chans.append(ch[o])
        ptr.append(ptr[-1] + len(o))
    assert all((c < nin).all() for c in chans)
    cat = lambda a, t: np.concatenate(a).astype(t) if a else np.zeros(0, t)
    return np.asarray(ptr, np.int64), cat(steps, np.int64), cat(chans, np.int32)
