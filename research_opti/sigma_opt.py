"""
sigma_opt.py -- the per-feature dispersion scales sigma_j, frozen before any run.

sigma_j IS the metric: d2 is a sum of squared (dF_j / sigma_j), so changing sigma changes
the optimum (spec G9). Two are produced, and both are frozen and reported:

  sigma_atlas   the pipeline's own, via v3_inference.load_sigma -- 1.4826*MAD over the
                10^7-cell atlas, with an atlas-subsample std fallback for the seven atomic
                columns whose MAD is zero. Every published cINN d2 is in these units, so
                the head-to-head has to use it or the two methods are not on one scale.
  sigma_lhs     the same recipe on a Latin-hypercube/Sobol sample drawn here. The compact
                paper's identification method may not lean on the atlas, so this is the
                value it would declare; it also shows how much of d2's scale is a property
                of the atlas rather than of the feature set.

    $PY sigma_opt.py --n 100000        -> data/sigma_lhs.json, data/sigma_atlas.json
    $PY sigma_opt.py --compare         -> print the two side by side
"""
import os
import time
import argparse
import numpy as np

import common as K
K.ensure_dirs()
import v3_config as C                                         # noqa: E402

ATLAS_JSON = os.path.join(K.DATA, "sigma_atlas.json")
LHS_JSON = os.path.join(K.DATA, "sigma_lhs.json")
N_DEFAULT = 100_000
SEED = 20261002
MAD_K = 1.4826                 # MAD -> sigma for a normal, as the pipeline uses
MIN_DEFINED = 100              # below this a column cannot support a scale


def atlas_sigma(force=False):
    d = None if force else K.jread(ATLAS_JSON)
    if d is None:
        from v3_inference import load_sigma
        S = load_sigma(K.ATLAS)
        d = dict(kind="atlas", atlas=os.path.basename(K.ATLAS),
                 source="v3_inference.load_sigma (1.4826*MAD, atlas-subsample std "
                        "fallback for zero-MAD columns)",
                 sigma=dict(zip(C.FEATURE_NAMES, S.tolist())))
        K.jwrite(ATLAS_JSON, d)
    return np.array([d["sigma"][n] for n in C.FEATURE_NAMES]), d


def _sobol(n, lo, hi, seed):
    """Sobol points in the box, with `a` sampled log-uniformly if that is the coordinate.

    The sample defines the metric, so it is drawn over the same box the atlas used, and
    uniformly in the SAME coordinates the optimizer searches -- otherwise the scales would
    describe a different population from the one the search walks through.
    """
    from scipy.stats import qmc
    u = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
    import objective as OB
    return OB.Space().to_theta(u), u


def lhs_sigma(n=N_DEFAULT, seed=SEED, force=False, chunk=25_000):
    d = None if force else K.jread(LHS_JSON)
    if d is not None and d.get("n") == n and d.get("seed") == seed:
        return np.array([d["sigma"][nm] for nm in C.FEATURE_NAMES]), d

    import evalbatch as EB
    import objective as OB
    sp = OB.Space()
    lo, hi = sp.lo, sp.hi
    th, _ = _sobol(n, lo, hi, seed)
    cols = [[] for _ in range(C.N_ACTIVE)]
    n_ok = 0
    t0 = time.time()
    for i in range(0, n, chunk):
        F, lm, ok = EB.evaluate(th[i:i + chunk])
        n_ok += int(ok.sum())
        for j in range(C.N_ACTIVE):
            v = F[ok, j]
            cols[j].append(v[v > C.SENT_CUT])
        el = time.time() - t0
        done = min(i + chunk, n)
        print(f"  {done:7d}/{n}  {1e3*el/done:6.2f} ms/cell  eta "
              f"{el/done*(n-done)/60:6.2f} min", flush=True)
        EB.cache_clear()                     # the sample is unique; keep memory flat

    S, detail = np.zeros(C.N_ACTIVE), {}
    for j, nm in enumerate(C.FEATURE_NAMES):
        v = np.concatenate(cols[j]) if cols[j] else np.empty(0)
        mad = float(np.median(np.abs(v - np.median(v)))) if v.size else 0.0
        s_mad = MAD_K * mad
        if s_mad > 0:
            S[j], how = s_mad, "mad"
        elif v.size >= MIN_DEFINED:
            S[j], how = float(np.std(v)), "std"            # atomic column: MAD is 0
        else:
            S[j], how = 1.0, "fallback_1"
        detail[nm] = dict(n_defined=int(v.size), frac_defined=float(v.size / max(n_ok, 1)),
                          mad=mad, sigma=float(S[j]), how=how,
                          n_unique=int(np.unique(v).size) if v.size else 0)

    d = dict(kind="lhs", n=int(n), seed=int(seed), n_feasible=int(n_ok),
             frac_feasible=float(n_ok / n), sampler="scipy.stats.qmc.Sobol(scramble=True)",
             log_a=sp.log_a, box_lo=lo.tolist(), box_hi=hi.tolist(),
             mad_k=MAD_K, min_defined=MIN_DEFINED,
             seconds=round(time.time() - t0, 1),
             source="1.4826*MAD over this sample, std fallback for zero-MAD columns",
             sigma=dict(zip(C.FEATURE_NAMES, S.tolist())), detail=detail,
             env=K.gpu_info())
    K.jwrite(LHS_JSON, d)
    return S, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    Sa, da = atlas_sigma(force=a.force)
    if a.compare:
        Sl, dl = lhs_sigma(a.n, a.seed)
    else:
        print(f"LHS sigma on {a.n} Sobol cells ...", flush=True)
        Sl, dl = lhs_sigma(a.n, a.seed, force=a.force)
        print(f"feasible {dl['n_feasible']}/{dl['n']} = {100*dl['frac_feasible']:.1f}%  "
              f"in {dl['seconds']/60:.2f} min")
    print(f"\n{'feature':12s} {'atlas':>11s} {'lhs':>11s} {'ratio':>7s} {'how':>5s} "
          f"{'defined':>8s} {'uniq':>6s}")
    for j, nm in enumerate(C.FEATURE_NAMES):
        det = dl["detail"][nm]
        print(f"{nm:12s} {Sa[j]:11.5g} {Sl[j]:11.5g} {Sl[j]/Sa[j]:7.3f} "
              f"{det['how']:>5s} {100*det['frac_defined']:7.1f}% {det['n_unique']:6d}")


if __name__ == "__main__":
    main()
