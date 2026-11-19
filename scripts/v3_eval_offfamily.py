"""
v3_eval_offfamily.py -- out-of-family scoring for the v3 surrogate.

Target features come from the RESET-FREE path (v3_an_features), surrogate draws
from the Izhikevich path (v3_features). Both emit the same 18-column v3 schema on
the same cell-relative geometry, which is the whole point of the B4 port.

Three surrogate viewpoints on ONE posterior (§19 / cinn.md), because they say
different things and disagreeing is informative:

  best    global best feature match, any period
  mean    posterior mean -- meaningful only if the posterior is unimodal
  pmatch  best draw that keeps the TARGET's period (burst vs tonic)

Metrics, all reported with N-of-M scorable beside them (§3.2):

  d2       SUM of z^2 over joint dims + 9.0 per dim the target defines and the
           draw does not. NOT a mean -- §47.3 is the record of what a mean does.
  w1_shape W1 between ISI/T_cycle distributions. Dimensionless on BOTH sides, so
           it is tau-invariant by construction and a foreign model's time units
           never enter.
  w1_log   the same on log(ISI/T_cycle)
  tau, k_I recovered scales: T_cycle^model / T_cycle^izh, i_rh^model / i_rh^izh
  support  fraction of posterior draws that are scorable at all
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
from v3_landmarks import landmarks as izh_landmarks
from v3_models import izh_rhs_v3
from v3_an_features import compute_features_an
from v3_an_zoo import targets as zoo_targets, reset_targets
from v3_scoring import d2, MISS_PENALTY, isis_of
from v3_regime import regime_report_izh, regime_report_an, format_report


def isis_an(spec, i_rh, tc, buf=1 << 16):
    """Settled ISI sequence for a foreign target, in T_cycle units."""
    from v3_an_core import integrate_an, make_sp
    st = np.empty(buf)
    trans = C.K_TRANS * tc
    sp = make_sp(trans, C.GAMMA_STEP * i_rh)
    ns, nt, inv = integrate_an(spec.rhs, C.STIM_STEP, sp, 1.0, spec.y0, 0.0,
                               trans + C.K_WIN * tc, spec.h, spec.v_th, st,
                               spec.params)
    if inv or ns != nt or ns < 3:
        return None
    t = st[:ns]
    t = t[t >= trans + C.K_DISCARD * tc]
    return np.diff(t) / tc if len(t) >= 3 else None


def w1(x, y):
    if x is None or y is None or len(x) < 2 or len(y) < 2:
        return np.nan
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(MODELS, "cinn_v3_p015"))
    ap.add_argument("--atlas", default=os.path.join(DATA, "atlas_v3_1e7"))
    ap.add_argument("--sigma", default=os.path.join(DATA, "atlas_v3_1e7", "sigma_v3.json"))
    ap.add_argument("--n-posterior", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--only", default=None,
                    help="substring filter on target names")
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing --out instead of replacing it")
    ap.add_argument("--out", default=os.path.join(RESULTS, "v3_offfamily.json"))
    args = ap.parse_args()

    import run_surrogate as rs
    from data import AtlasView
    view = AtlasView(args.atlas)
    sig = json.load(open(args.sigma))
    rng = np.random.default_rng(1)
    A0 = np.asarray(view._feats[np.sort(rng.choice(view.n, 200_000, replace=False))],
                    float)[:, view.keep]
    S = np.zeros(C.N_ACTIVE)
    for j, nm in enumerate(C.FEATURE_NAMES):
        s = sig.get(nm, 0.0)
        if s > 0:
            S[j] = s
        else:
            c = A0[:, j]; d = c > C.SENT_CUT
            S[j] = float(np.std(c[d])) if d.sum() > 100 else 1.0

    bufs = alloc_buffers()
    out = np.full(C.N_COLS, C.NAN_SENTINEL)
    # Two kinds of target, and the difference is structural (§47): a RESET model
    # (AdEx) goes through v3's Izhikevich path, a reset-free one through the B4
    # port. Both emit the same 18-column schema, which is what makes them
    # comparable at all.
    work = [(n, "resetfree", sp, None) for n, sp in zoo_targets()]
    work += [(n, "reset", (rhs, pr), rh) for n, rhs, pr, rh in reset_targets()]

    if args.only:
        work = [w for w in work if args.only.lower() in w[0].lower()]
    res = []
    for name, kind, spec, rheo_hi in work:
        t0 = time.time()
        if kind == "resetfree":
            f_t, info_t = compute_features_an(spec)
        else:
            rhs_m, pr_m = spec
            f_t, info_t = compute_features_v3(rhs_m, np.ascontiguousarray(pr_m),
                                              bufs, out=out, rheo_hi=rheo_hi)
        f_t = np.asarray(f_t[:C.N_ACTIVE], float).copy()
        if info_t["gated"] or (f_t > C.SENT_CUT).sum() < 5:
            res.append(dict(name=name, status=f"target undefined: {info_t['reason']}"))
            print(f"{name:>22s}  SKIP ({info_t['reason']})"); continue
        tc_m, irh_m = info_t["T_cycle"], info_t["i_rh"]

        # §53.5 regime diagnostic. REPORTED, NEVER GATING -- the posterior below
        # is drawn identically whatever this says. An unstable verdict means the
        # conditioning vector mixes regimes, not that the answer is wrong:
        # AdEx tonic is unstable here and matches at w1_shape 0.0017.
        if kind == "resetfree":
            reg = regime_report_an(spec, irh_m, tc_m)
        else:
            reg = regime_report_izh(spec[0], spec[1], irh_m, tc_m, bufs,
                                    rheo_hi=rheo_hi)
        print(f"{name:>24s}  {format_report(reg).splitlines()[0]}", flush=True)
        if kind == "resetfree":
            isi_m = isis_an(spec, irh_m, tc_m)
        else:
            im_, _ = isis_of(np.ascontiguousarray(spec[1]), bufs,
                             rhs=spec[0], rheo_hi=rheo_hi)
            isi_m = im_ / tc_m if im_ is not None else None
        p_t = f_t[C.FEATURE_NAMES.index("p")]

        theta, _ = rs.cinn_infer(args.model_dir, f_t, args.n_posterior,
                                 temperature=args.temperature)
        theta = np.asarray(theta, float)

        rows = []
        for k in range(len(theta)):
            th = np.ascontiguousarray(theta[k])
            fk, ik = compute_features_v3(izh_rhs_v3, th, bufs, out=out)
            if ik["gated"]:
                continue
            fk = np.asarray(fk[:C.N_ACTIVE], float).copy()
            if (fk > C.SENT_CUT).sum() < 5:
                continue
            v, nn = d2(f_t, fk, S)
            rows.append((v, nn, th.copy(), fk))
        support = len(rows) / len(theta)
        if not rows:
            res.append(dict(name=name, status="no scorable draw", support=0.0))
            print(f"{name:>22s}  no scorable draw"); continue

        rows.sort(key=lambda r: r[0])
        views = {"best": rows[0]}
        pm = [r for r in rows
              if abs(r[3][C.FEATURE_NAMES.index("p")] - p_t) < 1e-9]
        views["pmatch"] = pm[0] if pm else None
        thm = np.ascontiguousarray(theta.mean(0))
        fm, im = compute_features_v3(izh_rhs_v3, thm, bufs, out=out)
        fm = np.asarray(fm[:C.N_ACTIVE], float).copy()
        views["mean"] = None if im["gated"] else (d2(f_t, fm, S)[0], d2(f_t, fm, S)[1], thm, fm)

        entry = dict(name=name, kind=kind, status="ok", support=support,
                     regime=reg,
                     n_scorable=len(rows), n_posterior=len(theta),
                     target_defined=int((f_t > C.SENT_CUT).sum()),
                     T_cycle_model=tc_m, i_rh_model=irh_m, p_target=float(p_t),
                     theta_mean=theta.mean(0).tolist(), theta_sd=theta.std(0).tolist())
        for vname, r in views.items():
            if r is None:
                entry[vname] = None; continue
            d2v, nd, th, fk = r
            lm = izh_landmarks(izh_rhs_v3, th, bufs)
            tau = tc_m / lm["T_cycle"] if lm["T_cycle"] > C.SENT_CUT else np.nan
            kI = irh_m / lm["i_rh"] if lm["i_rh"] > C.SENT_CUT and lm["i_rh"] > 0 else np.nan
            isi_s, _ = isis_of(th, bufs)
            isi_s = isi_s / lm["T_cycle"] if (isi_s is not None and
                                              lm["T_cycle"] > C.SENT_CUT) else None
            entry[vname] = dict(
                d2=d2v, n_dims=nd, theta=th.tolist(), tau=float(tau), k_I=float(kI),
                w1_shape=w1(isi_m, isi_s),
                w1_log=w1(np.log(isi_m) if isi_m is not None else None,
                          np.log(isi_s) if isi_s is not None else None),
                p=float(fk[C.FEATURE_NAMES.index("p")]))
        res.append(entry)
        b = entry["best"]
        print(f"{name:>22s}  d2={b['d2']:9.3f} ({b['n_dims']:2d}d)  tau={b['tau']:7.3f}  "
              f"k_I={b['k_I']:8.3f}  w1_shape={b['w1_shape']:.4f}  "
              f"support={support:.3f}  p {p_t:.0f}->{b['p']:.0f}  "
              f"[{reg['verdict']}]  {time.time()-t0:.0f}s", flush=True)

    if args.merge and os.path.exists(args.out):
        old = json.load(open(args.out))
        keep = [r for r in old if r["name"] not in {x["name"] for x in res}]
        res = keep + res
    json.dump(res, open(args.out, "w"), indent=2)
    ok = [r for r in res if r.get("status") == "ok"]
    print(f"\n{len(ok)}/{len(res)} scored")
    if ok:
        for k in ("d2", "tau", "k_I", "w1_shape", "w1_log"):
            v = np.array([r["best"][k] for r in ok], float); v = v[np.isfinite(v)]
            if len(v):
                print(f"   best.{k:<9s} median {np.median(v):9.4f}")
        print(f"   support     median {np.median([r['support'] for r in ok]):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
