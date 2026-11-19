"""
evalbatch.py -- the batched candidate evaluator shared by every optimizer run.

One evaluation is a full 18-feature protocol suite on an Izhikevich cell, which is the
same work the atlas build does per cell. It is therefore done exactly the way the atlas
does it -- `v3_gpu.landmarks_gpu` then `v3_gpu.features_gpu`, with the atlas build's own
gate -- so an optimizer's candidate and an atlas cell are the same measurement.

Two facts drive the design:

  * Per-cell cost collapses with batch size. Measured: 88.56 ms/cell at
    n = 1024 against 11.25 ms/cell marginal between large batches: at a single CMA-ES
    generation (32-64 candidates) the GPU runs at a few percent occupancy. So the callers
    pool every concurrently live run into ONE call here (see runners.py), and this module
    only ever sees large batches.
  * PASS 0 is about half the cost and already decides most infeasibility (no rheobase, no
    settled cycle, resolution gate). PASS 1 therefore runs on survivors only.

    F, lm, ok = evaluate(theta)        # theta (n, 4)
    J, nj     = score(f_t, F, lm, ok, sigma)
"""
import numpy as np

import common as K                                            # noqa: E402  (sys.path)
import v3_config as C                                         # noqa: E402
import v3_gpu as G                                            # noqa: E402
import prescreen as PS                                        # noqa: E402
from v3_scoring import MISS_PENALTY                            # noqa: E402

# ---------------------------------------------------------------------- cache
# Key: the f32 bytes of theta. The kernels cast their input to float32, so two parameter
# vectors with the same f32 image are not merely close -- they produce bit-identical
# results, which makes this dedupe exact rather than a tolerance. Avoids a rounded-key
# hash, whose rounding would impose plateaus of its own on the landscape.
_CACHE = {}
_STAT = dict(calls=0, asked=0, hits=0, pass0=0, pass1=0, gated0=0, slow=0,
             cache_drops=0)

# A full study is ~10^6 unique cells and each entry holds 23 doubles plus dict overhead,
# so the cache is bounded and emptied wholesale when it fills. Hits come overwhelmingly
# from within a generation and from runs sitting in the same basin, both of which are
# recent, so eviction policy buys nothing over starting a fresh dict.
MAX_CACHE = 1_500_000


def cache_stats():
    return dict(_STAT, size=len(_CACHE))


def cache_clear():
    _CACHE.clear()
    for k in _STAT:
        _STAT[k] = 0


def _keys(theta):
    f32 = np.ascontiguousarray(theta, dtype=np.float32)
    return [f32[i].tobytes() for i in range(f32.shape[0])], f32


def _gate(lm):
    """The atlas build's gate, verbatim from scripts/v3_build.py.

    `capped` means the anchor was measured on a spike record that hit the buffer bound,
    so the cell is not trustworthy; the second term is the resolution gate B5, dropping
    cells whose cycle is shorter than ten integration steps.
    """
    tc = lm[:, 1].astype(np.float64)
    capped = lm[:, 4] != 0.0
    return capped | ((tc > C.SENT_CUT) & (tc < C.GATE_K_DT * C.DT))


def _grade(lm_row):
    """How close an infeasible candidate came to being usable, in [0, 1].

    A constant penalty leaves a dead floor the search cannot descend; this ranks the
    infeasible region so "almost fires" beats "silent" and the search is pulled toward
    the firing region. Computed from PASS 0's landmarks alone, so it means the same thing
    whether or not PASS 1 ran -- "the kernel got further" is not the same as "the cell
    came closer to being a usable surrogate".

    A candidate dropped by the cost ceiling is ranked with the resolution-gated ones at
    2/3 rather than at the top: it does fire and does have a settled cycle, but giving it
    the best infeasible score would build a gradient pointing at the ceiling, and the
    ceiling is a budget decision, not a property of the target.
    """
    i_rh, tc = lm_row[0], lm_row[1]
    if len(lm_row) > 5 and lm_row[5] == PS.DROP_SLOW:
        return 2.0 / 3.0
    if tc > C.SENT_CUT:
        gated = bool(_gate(np.atleast_2d(lm_row))[0])
        return 2.0 / 3.0 if gated else 1.0   # has a settled cycle; gated = too fast/capped
    if i_rh > C.SENT_CUT:
        return 1.0 / 3.0                # fires at some current, but no settled cycle
    return 0.0                          # no rheobase at all


def evaluate(theta, tpb=64):
    """Measure a batch of Izhikevich cells.

    theta : (n, 4) array of (a, b, c, d)
    returns
      F  : (n, 18) features, C.NAN_SENTINEL where undefined or not measured
      lm : (n, 6)  i_rh, T_cycle, L_inf, p_hat, capped, drop code (prescreen.DROP_*)
      ok : (n,)    bool, the cell passed the atlas gate and the cost ceiling, has a
                   settled cycle, and its features were measured
    """
    th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    n = th.shape[0]
    keys, _ = _keys(th)
    _STAT["calls"] += 1
    _STAT["asked"] += n

    F = np.full((n, C.N_ACTIVE), C.NAN_SENTINEL)
    lm = np.full((n, 6), C.NAN_SENTINEL)
    ok = np.zeros(n, dtype=bool)

    # unique misses only: an optimizer re-proposing a point costs nothing, and several
    # runs exploring the same basin share the work
    todo, seen = [], {}
    for i, k in enumerate(keys):
        hit = _CACHE.get(k)
        if hit is not None:
            F[i], lm[i], ok[i] = hit
            _STAT["hits"] += 1
        elif k in seen:
            todo[seen[k]][1].append(i)
        else:
            seen[k] = len(todo)
            todo.append((i, [i]))

    if todo:
        idx = np.array([t[0] for t in todo])
        sub = np.ascontiguousarray(th[idx])
        if K.TC_MAX:
            lm_s, afford = PS.landmarks_capped(sub, K.TC_MAX, tpb=tpb)
            _STAT["slow"] += int((lm_s[:, 5] == PS.DROP_SLOW).sum())
        else:
            lm_s = np.concatenate([G.landmarks_gpu(sub, tpb=tpb),
                                   np.zeros((len(idx), 1))], axis=1)
            afford = np.ones(len(idx), dtype=bool)
        _STAT["pass0"] += len(idx)
        gated = _gate(lm_s) | ~afford
        live = np.flatnonzero(~gated)
        F_s = np.full((len(idx), C.N_ACTIVE), C.NAN_SENTINEL)
        ok_s = np.zeros(len(idx), dtype=bool)
        _STAT["gated0"] += int(gated.sum())
        if live.size:
            th_l = np.ascontiguousarray(sub[live])
            out, _ = G.features_gpu(th_l, lm=np.ascontiguousarray(lm_s[live, :5]),
                                    tpb=tpb)
            _STAT["pass1"] += int(live.size)
            F_s[live] = np.asarray(out[:, :C.N_ACTIVE], dtype=np.float64)
            # an anchor is required: without T_cycle nothing is normalized, and such a
            # cell must not outrank one that merely fires too fast
            ok_s[live] = lm_s[live, 1] > C.SENT_CUT
        for (_, dests), fr, lr, okr in zip(todo, F_s, lm_s, ok_s):
            ent = (fr.copy(), np.asarray(lr, dtype=np.float64).copy(), bool(okr))
            _CACHE[keys[dests[0]]] = ent
            for j in dests:
                F[j], lm[j], ok[j] = ent

    if len(_CACHE) > MAX_CACHE:
        _CACHE.clear()
        _STAT["cache_drops"] += 1
    return F, lm, ok


def score(f_t, F, lm, ok, sigma, lam_period=0.0, p_t=None):
    """d2 of every row against one target, with the infeasible branch.

    f_t : (18,) target features, sentinel where undefined
    returns (J (n,), n_joint (n,) int)

    Vectorizes v3_scoring.d2 exactly -- sum of squared z over jointly defined dims plus
    MISS_PENALTY per dim the TARGET defines and the candidate does not. A mean would let a
    candidate defining one feature and matching it win outright, which is the failure
    v3_scoring documents.
    """
    f_t = np.asarray(f_t, dtype=np.float64)[:C.N_ACTIVE]
    dt = f_t > C.SENT_CUT                       # D: what the target defines
    n_D = int(dt.sum())
    md = F > C.SENT_CUT                         # per-candidate definedness
    nd = md.sum(axis=1)

    feas = ok & (nd >= K.N_DEF_MIN)
    joint = md & dt[None, :]
    nj = joint.sum(axis=1)

    z = np.zeros_like(F)
    sg = np.maximum(np.asarray(sigma, dtype=np.float64)[:C.N_ACTIVE], 1e-12)
    np.divide(F - f_t[None, :], sg[None, :], out=z, where=joint)
    J = np.einsum("ij,ij->i", z, z) + MISS_PENALTY * (n_D - nj)
    J[nj == 0] = np.inf                         # nothing to compare: v3_scoring returns inf

    if lam_period and p_t is not None and p_t > C.SENT_CUT:
        ip = C.FEATURE_NAMES.index("p")
        J = J + lam_period * ((F[:, ip] > C.SENT_CUT) & (F[:, ip] != p_t))

    if (~feas).any():
        bad = np.flatnonzero(~feas)
        g = np.array([_grade(lm[i]) for i in bad])
        J[bad] = (K.J_INF_BASE + MISS_PENALTY * (n_D - nd[bad]) - K.KAPPA * g)
        nj[bad] = 0
    return J, nj.astype(int)
