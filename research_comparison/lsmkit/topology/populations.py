"""
populations.py -- population-block circuits (Haeusler & Maass 2007 / Schulte to Brinke 2022).

Neurons are grouped in named populations; every ordered pair (src, dst) has a Bernoulli
connection probability p and a mean weight; synapse parameters (Tsodyks U, D, F and the
delay) depend on the E/I types of the pair. Each parameter is drawn per synapse from a
Gaussian with SD = cv * mean; values outside (0, upper] are replaced by a uniform draw on
(0, min(2 * mean, upper)) (Schulte 2022 §2.1.3). No autapses.
"""
import numpy as np

from ..spec import Circuit


def gauss_pos(mean, cv, n, rng, upper=np.inf):
    m = float(mean)
    x = rng.normal(m, cv * abs(m), n)
    bad = (x <= 0) | (x > upper)
    x[bad] = rng.uniform(0, min(2 * m, upper), bad.sum())
    return x


def layout(sizes):
    """sizes: {pop: n} in order -> ({pop: index array}, N)"""
    idx, o = {}, 0
    for k, n in sizes.items():
        idx[k] = np.arange(o, o + n)
        o += n
    return idx, o


def connect(idx, ptype, P, W, w_cv, stp, delay, d_cv, dt, rng, use_stp=True, udf_cv=0.5):
    """P, W: {src: {dst: value}}; ptype: {pop: 0 E / 1 I}; stp: [ts][td] -> (U, D, F);
    delay: [ts][td] ms. -> dict of edge arrays (receptor = source type)."""
    cols = {k: [] for k in ("src", "post", "w", "dly", "rec", "U", "D", "F")}
    for s, row in P.items():
        for d, p in row.items():
            if p <= 0 or W[s][d] == 0:
                continue
            a, b = idx[s], idx[d]
            m = rng.random((len(a), len(b))) < p
            if s == d:
                np.fill_diagonal(m, False)
            pre, post = np.nonzero(m)
            n = len(pre)
            ts, td = ptype[s], ptype[d]
            U, D, F = stp[ts][td]
            cols["src"].append(a[pre]); cols["post"].append(b[post])
            cols["w"].append(gauss_pos(abs(W[s][d]), w_cv, n, rng))
            dl = gauss_pos(delay[ts][td], d_cv, n, rng)
            cols["dly"].append(np.maximum(1, np.round(dl / dt)).astype(np.int32))
            cols["rec"].append(np.full(n, ts, np.int32))
            cols["U"].append(gauss_pos(U, udf_cv, n, rng, upper=1.0))
            cols["D"].append(gauss_pos(D, udf_cv, n, rng))
            cols["F"].append(gauss_pos(F, udf_cv, n, rng))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["stp"] = np.full(len(out["src"]), int(use_stp), np.int32)
    return out


def inputs(idx, N, ch0, nch, P_in, w_in, w_cv, delay_ms, dt, rng, rec=0):
    """static input synapses: every channel ch0..ch0+nch-1 connects to each neuron of pop
    with probability P_in[pop]; source ids are N + channel."""
    cols = {k: [] for k in ("src", "post", "w")}
    for pop, p in P_in.items():
        if p <= 0:
            continue
        m = rng.random((nch, len(idx[pop]))) < p
        ch, post = np.nonzero(m)
        cols["src"].append(N + ch0 + ch); cols["post"].append(idx[pop][post])
        cols["w"].append(gauss_pos(w_in, w_cv, len(ch), rng))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(out["src"])
    out.update(dly=np.full(n, max(1, int(round(delay_ms / dt))), np.int32),
               rec=np.full(n, rec, np.int32), U=np.ones(n), D=np.ones(n), F=np.ones(n),
               stp=np.zeros(n, np.int32))
    return out


def circuit(N, nin, types, parts, meta=None):
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return Circuit(N=N, nin=nin, types=types, src=cat("src").astype(np.int32),
                   post=cat("post").astype(np.int32), w=cat("w"), dly=cat("dly"),
                   rec=cat("rec"), U=cat("U"), D=cat("D"), F=cat("F"), stp=cat("stp"),
                   meta=meta or {})
