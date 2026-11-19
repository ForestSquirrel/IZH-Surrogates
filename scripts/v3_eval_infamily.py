"""
v3_eval_infamily.py -- in-family (izh -> izh) evaluation of the v3 surrogate.

The control condition: the target IS an Izhikevich cell, so a good pipeline must
recover theta_true up to posterior degeneracy, with tau ~ 1 and k_I ~ 1. Anything
it gets wrong here it cannot get right out of family.

Everything runs through v3's OWN extraction, which is the point -- §7.2 makes the
extraction pipeline part of the checkpoint's contract, so a v3 model must be
conditioned on v3 features and scored against v3 landmarks.

Scale recovery uses v3's anchor (B7):

    k_I = i_rh_target    / i_rh(theta)
    tau = T_cycle_target / T_cycle(theta)

Reported per target:
  * posterior mean +- sd, and the recovery error against theta_true
  * d^2 of the best draw, in v3's own atlas sigma
  * W1 between target and surrogate ISI distributions (log and shape)
  * tau and k_I, which should both be 1
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # pipeline_final/scripts
ROOT = os.path.dirname(HERE)                       # pipeline_final
PKG     = os.path.join(ROOT, "pipeline")
DATA    = os.path.join(ROOT, "data")
MODELS  = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
PLOTS   = os.path.join(ROOT, "plots")
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)   # sibling runners (v3_plot_pair -> v3_plot_family)

import v3_config as C
from v3_core import alloc_buffers
from v3_features import compute_features_v3
from v3_landmarks import landmarks as v3_landmarks
from v3_models import izh_rhs_v3
from v3_regime import regime_report_izh, format_report


MISS_PENALTY = 9.0        # z^2 charged per target dim the draw fails to define


def d2(a, b, sigma, eps=1e-12):
    """SUM of squared z over jointly-defined dims, PLUS a penalty per dim the
    target defines and the draw does not. Returns (score, n_joint).

    B14, and the reason it is required rather than cosmetic. A MEAN over joint
    dims is not comparable across draws with different |M| (§3.2): a draw that
    defines a single feature and matches it scores ~0 and wins outright. That is
    not hypothetical -- it is what the first run of this script did on 4 of the
    first 5 presets, reporting `d2 = 0.000 (1d)` and selecting a degenerate cell.

    The penalty is charged against the TARGET's defined set, so failing to
    define a feature the target has costs MISS_PENALTY = 9 (a 3-sigma miss)
    rather than being free. Extra features the draw defines and the target does
    not are not charged: the target is the thing being matched.
    """
    dt = a > C.SENT_CUT
    m = dt & (b > C.SENT_CUT)
    if not m.any():
        return np.inf, 0
    z = (a[m] - b[m]) / np.maximum(sigma[m], eps)
    miss = int(dt.sum() - m.sum())
    return float(np.sum(z * z) + MISS_PENALTY * miss), int(m.sum())


def isis_of(theta, bufs, gamma=2.0, rhs=None, rheo_hi=None):
    """Settled ISI sequence under P1's reference step, at the cell's own scale.

    `rhs` defaults to Izhikevich but MUST be passed for a foreign reset model.
    It was hardcoded here originally, and the out-of-family scorer then fed AdEx's
    parameter vector to the Izhikevich RHS -- simulating an Izhikevich cell with
    a=0, b=0, c=-65, d=0 rather than AdEx. Three of four AdEx regimes returned nan
    (that bogus cell has no anchor) and the fourth returned a w1 computed from the
    wrong model, which is the more dangerous outcome of the two.
    """
    if rhs is None:
        rhs = izh_rhs_v3
    theta = np.ascontiguousarray(theta)
    lm = (v3_landmarks(rhs, theta, bufs) if rheo_hi is None
          else v3_landmarks(rhs, theta, bufs, rheo_hi=float(rheo_hi)))
    if lm["T_cycle"] < C.SENT_CUT or lm["i_rh"] < C.SENT_CUT:
        return None, lm
    from v3_core import integrate_v3
    from v3_config import STIM_PARAMS_LEN, STATE_DIM, V_REST, DT
    tc = lm["T_cycle"]
    sp = np.zeros(STIM_PARAMS_LEN)
    sp[0] = C.K_TRANS * tc
    sp[1] = gamma * lm["i_rh"]
    y = np.empty(STATE_DIM); y[0] = V_REST; y[1] = theta[1] * V_REST
    st = bufs["spike_times"]
    t_end = sp[0] + C.K_WIN * tc
    ns, nt, inv = integrate_v3(rhs, 0, sp, 1.0, y, 0.0, t_end, DT,
                               st, theta[2], theta[3], theta)
    if inv or ns < 3 or ns != nt:
        return None, lm
    t = st[:ns]
    t = t[t >= sp[0] + C.K_DISCARD * tc]
    if len(t) < 3:
        return None, lm
    return np.diff(t), lm


def w1(x, y):
    """1-D Wasserstein between two samples, via sorted quantiles."""
    if x is None or y is None or len(x) < 2 or len(y) < 2:
        return np.nan
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(MODELS, "cinn_v3_p015"))
    ap.add_argument("--atlas", default=os.path.join(DATA, "atlas_v3_1e7"))
    ap.add_argument("--sigma", default=os.path.join(DATA, "atlas_v3_1e7", "sigma_v3.json"))
    ap.add_argument("--n-targets", type=int, default=0,
                    help="atlas draws in addition to the canonical presets")
    ap.add_argument("--n-posterior", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "v3_infamily.json"))
    args = ap.parse_args()

    import run_surrogate as rs
    from data import AtlasView

    sig = json.load(open(args.sigma))
    view = AtlasView(args.atlas)
    S = np.zeros(C.N_ACTIVE)
    rng0 = np.random.default_rng(1)
    rows0 = np.sort(rng0.choice(view.n, 200_000, replace=False))
    A0 = np.asarray(view._feats[rows0], float)[:, view.keep]
    for j, nm in enumerate(C.FEATURE_NAMES):
        s = sig.get(nm, 0.0)
        if s > 0:
            S[j] = s
        else:
            c = A0[:, j]; d = c > C.SENT_CUT
            S[j] = float(np.std(c[d])) if d.sum() > 100 else 1.0

    targets = []
    from v3_presets import PRESETS
    for nm, kw in PRESETS.items():
        targets.append((nm, np.array([kw["a"], kw["b"], kw["c"], kw["d"]])))
    if args.n_targets:
        rng = np.random.default_rng(args.seed)
        for i in rng.integers(0, view.n, size=args.n_targets):
            targets.append((f"atlas:{int(i)}",
                            np.asarray(view._params[int(i)], float)))

    bufs = alloc_buffers()
    out = np.full(C.N_COLS, C.NAN_SENTINEL)
    res = []
    t0 = time.time()
    for name, th_true in targets:
        th_true = np.ascontiguousarray(th_true, dtype=np.float64)
        f_t, info_t = compute_features_v3(izh_rhs_v3, th_true, bufs, out=out)
        f_t = np.asarray(f_t[:C.N_ACTIVE], float).copy()
        isi_t, lm_t = isis_of(th_true, bufs)
        if info_t["gated"] or (f_t > C.SENT_CUT).sum() < 5:
            res.append(dict(name=name, status="target undefined")); continue

        # §53.5: reported, never gating. The posterior is drawn regardless.
        reg = regime_report_izh(izh_rhs_v3, th_true, lm_t["i_rh"],
                                lm_t["T_cycle"], bufs) \
            if lm_t["T_cycle"] > C.SENT_CUT else None

        theta, _ = rs.cinn_infer(args.model_dir, f_t, args.n_posterior,
                                 temperature=args.temperature)
        theta = np.asarray(theta, float)

        best = None
        for k in range(len(theta)):
            fk, ik = compute_features_v3(izh_rhs_v3,
                                         np.ascontiguousarray(theta[k]), bufs, out=out)
            if ik["gated"]:
                continue
            fk = np.asarray(fk[:C.N_ACTIVE], float).copy()
            v, nn = d2(f_t, fk, S)
            if np.isfinite(v) and (best is None or v < best[0]):
                best = (v, nn, theta[k].copy(), fk)
        if best is None:
            res.append(dict(name=name, status="no scorable draw")); continue

        d2v, nd, th_b, f_b = best
        isi_b, lm_b = isis_of(th_b, bufs)
        tau = (lm_t["T_cycle"] / lm_b["T_cycle"]
               if lm_b["T_cycle"] > C.SENT_CUT and lm_t["T_cycle"] > C.SENT_CUT else np.nan)
        kI = (lm_t["i_rh"] / lm_b["i_rh"]
              if lm_b["i_rh"] > C.SENT_CUT and lm_b["i_rh"] > 0 else np.nan)
        lo = np.asarray(view.meta["param_lo"]); hi = np.asarray(view.meta["param_hi"])
        err = np.abs(th_b - th_true) / (hi - lo)
        res.append(dict(name=name, status="ok",
                        theta_true=th_true.tolist(), theta_best=th_b.tolist(),
                        theta_mean=theta.mean(0).tolist(),
                        theta_sd=theta.std(0).tolist(),
                        err_width=err.tolist(), d2=d2v, n_dims=nd,
                        tau=float(tau), k_I=float(kI),
                        w1_log=w1(np.log(isi_t) if isi_t is not None else None,
                                  np.log(isi_b) if isi_b is not None else None),
                        w1_raw=w1(isi_t, isi_b),
                        defined_target=int((f_t > C.SENT_CUT).sum()),
                        defined_best=int((f_b > C.SENT_CUT).sum()),
                        regime=reg))
        r = res[-1]
        print(f"  {name:>26s}  d2={d2v:9.3f} ({nd:2d}d)  tau={tau:6.3f}  k_I={kI:6.3f}  "
              f"w1_log={r['w1_log']:.4f}  err={np.mean(err):.4f}"
              + (f"  [{reg['verdict']}]" if reg else ""), flush=True)

    ok = [r for r in res if r.get("status") == "ok"]
    print(f"\n{len(ok)}/{len(res)} scorable, {time.time()-t0:.0f}s")
    if ok:
        for k in ("d2", "tau", "k_I", "w1_log"):
            v = np.array([r[k] for r in ok], float); v = v[np.isfinite(v)]
            print(f"   {k:>7s}  median {np.median(v):9.4f}   p90 {np.percentile(v,90):9.4f}")
        e = np.array([r["err_width"] for r in ok])
        print(f"   recovery error (prior widths): "
              + "  ".join(f"{n}={np.median(e[:,j]):.4f}" for j, n in enumerate("abcd")))
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
