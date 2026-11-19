"""
cinn_baseline.py -- the amortized arm of study 1.

The cINN is given exactly the same target feature vector and the same sigma as the
optimizers, and its draws are scored through the same GPU evaluator, so the two arms
differ only in HOW a theta is proposed. Reported per target:

  d2 of the best of N draws        what an optimizer's final J is compared against
  d2 of the median draw, support   the posterior is a distribution, not a point
  wall clock of the forward pass   the per-target cost, against the optimizer's budget
  the offline cost it rests on     atlas build + flow training, amortized over all targets

It also measures one thing the two arms cannot avoid disagreeing about. Candidate features
here come from the float32 GPU kernels (as the atlas itself does), while the pipeline's
published d2 values scored candidates on the float64 CPU path. On random cells those
differ by 8e-9 sigma at the median but 0.15 sigma at the 99th percentile, with spike-count
ratios (`fbar`, `Pi`) worst, so the same theta can carry two different d2 values. Each
draw is therefore scored BOTH ways here and the difference is reported on the cells that
actually matter, rather than left as a caveat.

    $PY cinn_baseline.py --n-posterior 256
Writes data/cinn_baseline.json.
"""
import os
import time
import argparse
import numpy as np

import common as K
K.ensure_dirs()
import v3_config as C                                         # noqa: E402
import evalbatch as EB                                        # noqa: E402
import objective as OB                                        # noqa: E402
import targets as T                                           # noqa: E402
import sigma_opt as SG                                        # noqa: E402


def cpu_features(theta):
    from v3_core import alloc_buffers
    from v3_features import compute_features_v3
    from v3_models import izh_rhs_v3
    bufs = alloc_buffers()
    out = np.full((len(theta), C.N_ACTIVE), C.NAN_SENTINEL)
    ok = np.zeros(len(theta), dtype=bool)
    for i, th in enumerate(theta):
        buf = np.full(C.N_COLS, C.NAN_SENTINEL)
        f, it = compute_features_v3(izh_rhs_v3, np.ascontiguousarray(th), bufs, out=buf)
        out[i] = np.asarray(f[:C.N_ACTIVE], float)
        ok[i] = not it["gated"]
    return out, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-posterior", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model-dir", default=K.MODEL_DIR)
    ap.add_argument("--cpu-check", action="store_true", default=True)
    ap.add_argument("--no-cpu-check", dest="cpu_check", action="store_false")
    ap.add_argument("--which", default="study1", choices=("study1", "study2", "both"))
    a = ap.parse_args()

    S, _ = SG.atlas_sigma()
    import run_surrogate as rs

    work = []
    if a.which in ("study1", "both"):
        work += list(T.study1())
    if a.which in ("study2", "both"):
        for fam, pname, items in T.study2():
            work += [(s, l, k, p, r) for s, l, x, k, p, r in items]

    # Draw every target's posterior FIRST, then measure all the draws in one call. The
    # forward pass is milliseconds; the candidate measurement is the cost, and at 256
    # cells per call it runs at ~1200 ms/cell against ~30 ms/cell when pooled.
    drawn = []
    for slug, label, kind, payload, rh in work:
        d = T.cached(slug, label, kind, payload, rh)
        if d["gated"]:
            print(f"{slug}: gated target, skipped ({d['reason']})")
            continue
        tgt = T.target_of(d, S)
        t0 = time.time()
        theta, _ = rs.cinn_infer(a.model_dir, tgt.f_t, a.n_posterior,
                                 temperature=a.temperature)
        drawn.append((slug, label, tgt, np.asarray(theta, float),
                      time.time() - t0))
    print(f"{len(drawn)} posteriors drawn; measuring "
          f"{sum(len(x[3]) for x in drawn)} candidates in one batch ...", flush=True)
    TH = np.concatenate([x[3] for x in drawn], axis=0)
    F_all, lm_all, ok_all = EB.evaluate(TH)

    rows, off = [], 0
    for slug, label, tgt, theta, t_fwd in drawn:
        sl = slice(off, off + len(theta))
        off += len(theta)
        F, lm, ok = F_all[sl], lm_all[sl], ok_all[sl]
        J, nj = tgt.J(F, lm, ok)
        fin = np.isfinite(J) & (J < K.J_INF_BASE / 2)
        i_best = int(np.argmin(J))
        sp = OB.Space()
        row = dict(target=slug, label=label, n_posterior=int(len(theta)),
                   temperature=a.temperature, model_dir=a.model_dir,
                   n_D=tgt.n_D, forward_s=round(t_fwd, 4),
                   support=float(fin.mean()),
                   J_best=float(J[i_best]), n_joint_best=int(nj[i_best]),
                   J_median=float(np.median(J[fin])) if fin.any() else None,
                   J_p10=float(np.quantile(J[fin], 0.1)) if fin.any() else None,
                   theta_best=theta[i_best].tolist(),
                   u_best=sp.to_u(theta[i_best][None, :])[0].tolist(),
                   at_bound=sp.at_bound(sp.to_u(theta[i_best][None, :])[0]),
                   i_rh_s=float(lm[i_best, 0]), T_cycle_s=float(lm[i_best, 1]),
                   theta_spread=np.std(theta[fin], axis=0).tolist() if fin.any() else None)
        row["tau"], row["k_I"] = tgt.gauge(lm[i_best])

        if a.cpu_check:
            Fc, okc = cpu_features(theta)
            Jc, njc = EB.score(tgt.f_t, Fc, lm, okc, S)
            fc = np.isfinite(Jc) & (Jc < K.J_INF_BASE / 2)
            m = fin & fc
            row["cpu_check"] = dict(
                J_best_cpu=float(Jc.min()),
                same_best_draw=bool(int(np.argmin(Jc)) == i_best),
                n_both=int(m.sum()),
                dJ_med=float(np.median(np.abs(J[m] - Jc[m]))) if m.any() else None,
                dJ_max=float(np.max(np.abs(J[m] - Jc[m]))) if m.any() else None,
                dJ_rel_med=float(np.median(np.abs(J[m] - Jc[m]) / J[m])) if m.any() else None,
                support_cpu=float(fc.mean()))
        rows.append(row)
        cc = row.get("cpu_check") or {}
        print(f"{slug:14s} |D| {tgt.n_D:2d}  J_best {row['J_best']:10.4f} "
              f"(cpu {cc.get('J_best_cpu', float('nan')):10.4f})  "
              f"J_med {(row['J_median'] or np.nan):10.4f}  support {row['support']:.3f}  "
              f"fwd {t_fwd*1e3:7.1f} ms  tau {(row['tau'] or np.nan):.4g}", flush=True)

    K.jwrite(os.path.join(K.DATA, "cinn_baseline.json"),
             dict(rows=rows, sigma="atlas", env=K.gpu_info(),
                  note="offline cost this arm rests on: ~31 h atlas build (10^7 cells) "
                       "plus ~1.4 h flow training, amortized over all targets"))
    print(f"\nwrote {os.path.relpath(os.path.join(K.DATA, 'cinn_baseline.json'), K.HERE)}")


if __name__ == "__main__":
    main()
