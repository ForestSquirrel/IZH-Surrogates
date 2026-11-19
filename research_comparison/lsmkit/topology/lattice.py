"""
lattice.py -- Maass-style spatial column (Maass et al. 2002 App. B; Maass 2003 §18.3).

Neurons sit on the integer points of an sx x sy x sz grid; a random fraction is
inhibitory. A directed edge a -> b exists with probability C[ta][tb] * exp(-(D(a,b)/lam)^2),
each direction drawn independently, no autapses, no distance cut-off. Per edge:
amplitude |A| ~ gamma with the given mean and CV (Maass: SD 100 % of the mean -> shape 1),
sign from the presynaptic type; Tsodyks U, D, F ~ Gaussian(mean, cv*mean) with values
outside the range replaced by a uniform draw on (0, 2*mean) (and U <= 1); delay per type
pair; receptor = presynaptic type (0 E, 1 I).

Generalises the BDCC builder: that one normalises by in-degree and
has a distance cut-off; the literature columns do neither.
"""
import numpy as np

from ..spec import Circuit

E, I = 0, 1


def grid(shape):
    sx, sy, sz = shape
    g = np.stack(np.meshgrid(np.arange(sx), np.arange(sy), np.arange(sz), indexing="ij"), -1)
    return g.reshape(-1, 3).astype(float)


def types_exact(N, f_inh, rng):
    """exactly round(f_inh * N) inhibitory neurons, positions random -> (N,) 0 E / 1 I"""
    t = np.zeros(N, np.int8)
    t[rng.choice(N, int(round(f_inh * N)), replace=False)] = I
    return t


def stp_draw(mean, cv, rng, n, upper=None):
    m = float(mean)
    x = rng.normal(m, cv * m, n)
    bad = (x <= 0) | ((x > upper) if upper is not None else False)
    while bad.any():
        x[bad] = rng.uniform(0, 2 * m, bad.sum())
        bad = (x <= 0) | ((x > upper) if upper is not None else False)
    return x


def gamma_amp(mean, cv, rng, n):
    m = abs(float(mean))
    if cv <= 0:
        return np.full(n, m)
    k = 1.0 / cv ** 2
    return rng.gamma(k, m / k, n)


def column(shape, f_inh, C, lam, A, A_cv, UDF, udf_cv, delay_ms, dt, rng, stp=True):
    """Recurrent part. C, A, delay_ms: 2x2 [pre][post]; UDF: 2x2 of (U, D_ms, F_ms).
    -> dict of edge arrays + types (0/1) + positions."""
    pos = grid(shape)
    N = len(pos)
    t = types_exact(N, f_inh, rng)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    Cm = np.asarray(C, float)[t[:, None], t[None, :]]
    p = Cm * np.exp(-d2 / lam ** 2)
    np.fill_diagonal(p, 0.0)
    pre, post = np.nonzero(rng.random((N, N)) < p)
    m = len(pre)
    tp, tq = t[pre], t[post]
    w = np.empty(m)
    U = np.empty(m); D = np.empty(m); F = np.empty(m)
    dly = np.empty(m, np.int32)
    for a in (E, I):
        for b in (E, I):
            s = (tp == a) & (tq == b)
            n = int(s.sum())
            if not n:
                continue
            w[s] = np.sign(A[a][b]) * gamma_amp(A[a][b], A_cv, rng, n)
            u, dd, ff = UDF[a][b]
            U[s] = stp_draw(u, udf_cv, rng, n, upper=1.0)
            D[s] = stp_draw(dd, udf_cv, rng, n)
            F[s] = stp_draw(ff, udf_cv, rng, n)
            dly[s] = max(1, int(round(delay_ms[a][b] / dt)))
    return dict(N=N, types=t, pos=pos, src=pre.astype(np.int32), post=post.astype(np.int32),
                w=w, dly=dly, rec=tp.astype(np.int32), U=U, D=D, F=F,
                stp=np.full(m, int(stp), np.int32))


def inputs(N, types, nin, frac, A_in, A_cv, UDF_in, udf_cv, delay_ms, dt, rng, stp=True):
    """Each input channel projects to round(frac*N) distinct random reservoir neurons.
    A_in, UDF_in: per post type (E, I); receptor 0 (excitatory)."""
    k = int(round(frac * N))
    src, post = [], []
    for ch in range(nin):
        src.append(np.full(k, N + ch, np.int32))
        post.append(rng.choice(N, k, replace=False).astype(np.int32))
    src, post = np.concatenate(src), np.concatenate(post)
    tq = types[post]
    m = len(src)
    w = np.empty(m); U = np.empty(m); D = np.empty(m); F = np.empty(m)
    for b in (E, I):
        s = tq == b
        n = int(s.sum())
        if not n:
            continue
        w[s] = np.sign(A_in[b]) * gamma_amp(A_in[b], A_cv, rng, n)
        u, dd, ff = UDF_in[b]
        U[s] = stp_draw(u, udf_cv, rng, n, upper=1.0)
        D[s] = stp_draw(dd, udf_cv, rng, n)
        F[s] = stp_draw(ff, udf_cv, rng, n)
    dly = np.full(m, max(1, int(round(delay_ms / dt))), np.int32)
    return dict(src=src, post=post, w=w, dly=dly, rec=np.zeros(m, np.int32), U=U, D=D, F=F,
                stp=np.full(m, int(stp), np.int32))


def circuit(rec, inp, nin, meta=None):
    cat = lambda k: np.concatenate([rec[k], inp[k]])
    return Circuit(N=rec["N"], nin=nin, types=np.where(rec["types"] == E, 1, -1).astype(np.int8),
                   src=cat("src"), post=cat("post"), w=cat("w"), dly=cat("dly"), rec=cat("rec"),
                   U=cat("U"), D=cat("D"), F=cat("F"), stp=cat("stp"), meta=meta or {})
