"""
showcase.py -- one test digit through every condition, for the raster panels (figs.py): input-layer
and liquid spike times of network 0, static LSM at the campaign optimum. Cached in
<out>/showcase/static_<cond>.npz.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import spec as S
import circuit as CI
import encode as EN
import run as R
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec

SAMPLE = 60000          # first MNIST test image
RMAX = 256


def _safe(m):
    return m.replace(":", "_").replace("*", "x").replace("@", "_at_")


def _spikes(sp, ids):
    _, spt, spn = sp
    i = np.concatenate([np.full(spn[0, j], j) for j in ids]) if len(ids) else np.zeros(0, int)
    t = np.concatenate([spt[0, j, :spn[0, j]] for j in ids]) if len(ids) else np.zeros(0, int)
    return t, i


def _inputs(model, base, mod, nin, data, dt):
    """(steps, channels) of the input spike channels for SAMPLE (ext modes)"""
    if mod == "fastin":
        p, st, ch = R.input_layer(model, base, nin, data, np.array([SAMPLE]), "mnist", dt)
        return st, ch
    e = R.stimulus(data, SAMPLE, "mnist")
    if mod == "lifin":
        e = EN.lif_input_spikes(e, nin)
    _, s, c = LS.events([e], dt, nin)
    return s, c


def static(cdir, cond, out, data, nin):
    f = os.path.join(out, "showcase", f"static_{_safe(cond)}.npz")
    if os.path.exists(f):
        return dict(np.load(f))
    import campaign as CA
    w = CA.cond_w(cdir, cond)
    base, _, mod = cond.partition("@")
    ext = mod in ("lifin", "fastin")
    dt = R.dt_for(cond)
    b, pulse = R.binding(base, S.N_LIQ if ext else nin + S.N_LIQ, dt)
    circ = CI.build(nin, w, dt, np.random.default_rng(100), pulse=pulse, input_neurons=not ext)
    net = LS.stack([circ], [b], CI.receptors())
    if ext:
        s, c = _inputs(cond, base, mod, nin, data, dt)
        ev = (np.array([0, len(s)], np.int64), s.astype(np.int64), c.astype(np.int32))
    else:
        ev = LS.events([R.stimulus(data, SAMPLE, "mnist")], dt, nin)
    o = engine_gpu.simulate(net, np.zeros(1, np.int32), np.array([int(round(S.T_SAMPLE / dt))]),
                            R.y0_for(base, 1, circ.N), ev, dt, Rec(spikes=(0,), rmax=RMAX))
    res = _pack(o["spikes"], circ, ext, nin, dt, ev)
    res.update(w=w)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, **res)
    return res


def _pack(sp, circ, ext, nin, dt, ev):
    if ext:
        lt, li = _spikes(sp, range(circ.N))
        it, ii = ev[1].astype(int), ev[2].astype(int)
        types = circ.types
    else:
        it, ii = _spikes(sp, range(nin))
        lt, li = _spikes(sp, range(nin, circ.N))
        li = li - nin
        types = circ.types[nin:]
    return dict(in_t=it * dt, in_i=ii, liq_t=lt * dt, liq_i=li, types=types, dt=dt, label=-1)
