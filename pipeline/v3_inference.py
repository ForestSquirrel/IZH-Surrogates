"""
v3_inference.py -- the inference core shared by both endpoints.

ONE posterior, three views, and a support figure. The flow is the fit; `d2` is
only a SELECTOR among draws (see v3_scoring). The two endpoints differ solely in
which extractor measures the target:

    reset model       compute_features_v3   -- V_TH = 30 is folded at compile
                                             time into the reset integrator
    reset-free model  compute_features_an   -- v_th is a runtime field of V3Spec

That asymmetry is not cosmetic. A reset model whose threshold is not 30 cannot
go through the reset path at all; it must either be rescaled or (better) be
expressed as a reset-free system (`examples/my_model.py` is a template).
"""
import json
import numpy as np

import v3_config as C
from v3_core import alloc_buffers
from v3_features import compute_features_v3
from v3_landmarks import landmarks as izh_landmarks
from v3_models import izh_rhs_v3
from v3_scoring import d2, w1, isis_of, isis_an
from v3_regime import regime_report_izh, regime_report_an


def load_sigma(atlas_dir, sigma_path=None, n_sample=200_000, seed=1):
    """Per-feature standardization scale: robust sigma, std fallback.

    The stored sigma_v3.json is 1.4826*MAD; seven columns have zero MAD (they
    are atoms) and fall back to the empirical std over an atlas subsample. d2 is
    therefore ATLAS-RELATIVE -- a score computed against one atlas is not
    comparable to one computed against another.
    """
    from data import AtlasView
    view = AtlasView(atlas_dir)
    sig = json.load(open(sigma_path or f"{atlas_dir}/sigma_v3.json"))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(view.n, min(n_sample, view.n), replace=False))
    A0 = np.asarray(view._feats[idx], float)[:, view.keep]
    S = np.zeros(C.N_ACTIVE)
    for j, nm in enumerate(C.FEATURE_NAMES):
        s = sig.get(nm, 0.0)
        if s > 0:
            S[j] = s
        else:
            col = A0[:, j]; dfn = col > C.SENT_CUT
            S[j] = float(np.std(col[dfn])) if dfn.sum() > 100 else 1.0
    return S


def _views(rows, theta, f_t, S, bufs, out, isi_m, tc_m, irh_m):
    """best / mean / pmatch over one posterior, plus the gauge and w1 for each."""
    p_t = f_t[C.FEATURE_NAMES.index("p")]
    rows.sort(key=lambda r: r[0])
    sel = {"best": rows[0]}
    pm = [r for r in rows
          if abs(r[3][C.FEATURE_NAMES.index("p")] - p_t) < 1e-9]
    sel["pmatch"] = pm[0] if pm else None
    thm = np.ascontiguousarray(theta.mean(0))
    fm, im = compute_features_v3(izh_rhs_v3, thm, bufs, out=out)
    fm = np.asarray(fm[:C.N_ACTIVE], float).copy()
    if im["gated"]:
        sel["mean"] = None
    else:
        v, nn = d2(f_t, fm, S)
        sel["mean"] = (v, nn, thm, fm)

    res = {}
    for name, r in sel.items():
        if r is None:
            res[name] = None; continue
        d2v, nd, th, fk = r
        lm = izh_landmarks(izh_rhs_v3, np.ascontiguousarray(th), bufs)
        tau = tc_m / lm["T_cycle"] if lm["T_cycle"] > C.SENT_CUT else np.nan
        kI = (irh_m / lm["i_rh"] if lm["i_rh"] > C.SENT_CUT and lm["i_rh"] > 0
              else np.nan)
        isi_s, _ = isis_of(np.ascontiguousarray(th), bufs)
        isi_s = (isi_s / lm["T_cycle"]
                 if (isi_s is not None and lm["T_cycle"] > C.SENT_CUT) else None)
        res[name] = dict(
            d2=float(d2v), n_dims=int(nd), theta=np.asarray(th).tolist(),
            tau=float(tau), k_I=float(kI),
            w1_shape=w1(isi_m, isi_s),
            w1_log=w1(np.log(isi_m) if isi_m is not None else None,
                      np.log(isi_s) if isi_s is not None else None),
            p=float(fk[C.FEATURE_NAMES.index("p")]))
    return res


def infer(kind, target, model_dir, S, n_posterior=256, temperature=1.0,
          rheo_hi=None, bufs=None, name="target"):
    """Measure -> draw -> score. Returns a result dict (JSON-serializable).

    kind : "reset"      target = (rhs, params) and optionally rheo_hi
           "continuous" target = a V3Spec
    """
    import run_surrogate as rs
    from v3_an_features import compute_features_an

    bufs = bufs or alloc_buffers()
    out = np.full(C.N_COLS, C.NAN_SENTINEL)

    if kind == "continuous":
        f_t, it = compute_features_an(target)
    else:
        rhs_m, pr_m = target
        f_t, it = compute_features_v3(rhs_m, np.ascontiguousarray(pr_m), bufs,
                                      out=out, rheo_hi=rheo_hi)
    f_t = np.asarray(f_t[:C.N_ACTIVE], float).copy()
    n_def = int((f_t > C.SENT_CUT).sum())
    base = dict(name=name, kind=kind, n_defined=n_def,
                i_rh=float(it["i_rh"]), T_cycle=float(it["T_cycle"]))
    if it["gated"] or n_def < 5:
        return dict(base, status=f"target undefined: {it.get('reason','')}")

    tc_m, irh_m = it["T_cycle"], it["i_rh"]

    # Advisory only -- NEVER gates the draw. An unstable verdict says the
    # conditioning vector mixes regimes, not that the answer is wrong.
    reg = (regime_report_an(target, irh_m, tc_m) if kind == "continuous"
           else regime_report_izh(target[0], target[1], irh_m, tc_m, bufs,
                                  rheo_hi=rheo_hi))
    isi_m = (isis_an(target, irh_m, tc_m) if kind == "continuous"
             else (lambda im: im / tc_m if im is not None else None)(
                 isis_of(np.ascontiguousarray(target[1]), bufs,
                         rhs=target[0], rheo_hi=rheo_hi)[0]))

    theta, _ = rs.cinn_infer(model_dir, f_t, n_posterior, temperature=temperature)
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
        return dict(base, status="no scorable draw", support=0.0, regime=reg)

    res = dict(base, status="ok", support=support, regime=reg,
               n_scorable=len(rows), n_posterior=len(theta),
               theta_mean=theta.mean(0).tolist(),
               theta_sd=theta.std(0).tolist(),
               features=f_t.tolist())
    res.update(_views(rows, theta, f_t, S, bufs, out, isi_m, tc_m, irh_m))
    return res


def format_result(r):
    if r.get("status") != "ok":
        return f"{r['name']:>26s}  {r.get('status')}"
    b = r["best"]
    w = b["w1_shape"]
    return (f"{r['name']:>26s}  d2={b['d2']:9.3f} ({b['n_dims']:2d}d)  "
            f"tau={b['tau']:9.4g}  k_I={b['k_I']:9.4g}  "
            f"w1_shape={w:.5g}  support={r['support']:.3f}  "
            f"defined {r['n_defined']}/18  [{r['regime']['verdict']}]")
