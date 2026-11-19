"""
check_controls.py -- the end-to-end controls (plan verification 4, 5, 6).

A. Optimizer sanity. Each backend on a 4-D quadratic in u, driven through the same
   ask/tell interface the studies use. Confirms the adapters and the restart logic, with
   no simulation involved.
B. In-family self-recovery -- the decisive control. The target is a KNOWN Izhikevich
   cell's own feature vector, so a correct chain must drive J to ~0. If it does not, the
   objective, the gauge or an adapter is wrong, and no out-of-family number from this
   folder would mean anything. Parameter recovery is reported but NOT required: the
   parameter-to-behaviour map is many-to-one (spec G4; `c` is weakly identified), so a
   different theta at J ~ 0 is the degeneracy, not a failure.
C. Null. The same target vector with its defined values permuted between columns. It is
   still a "plausible" vector dimension by dimension, but not one any cell produces, so
   the attainable J must stay far above the self-recovery value. This is what rules out
   "the optimizer always finds something that looks good".

    $PY check_controls.py --budget 2000
Writes data/check_controls.json.
"""
import os
import argparse
import numpy as np

import common as K
K.ensure_dirs()
import v3_config as C                                         # noqa: E402
import evalbatch as EB                                        # noqa: E402
import objective as OB                                        # noqa: E402
import runners as R                                           # noqa: E402
import sigma_opt as SG                                        # noqa: E402

# A plain tonic Izhikevich cell, well inside the box AND inside the cost ceiling.
#
# Izhikevich's own tonic-spiking preset (0.02, 0.2, -65, 6) was the first choice and is
# NOT usable here: its cycle time at gamma = 2 is 51.19 ms, just past the 50 ms ceiling,
# so the prescreen rejects it and the control would have no target at all. Worth recording
# rather than quietly swapping, because it says something about the ceiling: the textbook
# presets are slow in absolute milliseconds and are not typical of the atlas prior, whose
# median cycle time is 0.97 ms. This cell runs at 10.84 ms, a factor 4.6 inside the
# ceiling, and defines 16 of 18 features.
THETA_TRUE = np.array([0.1, 0.2, -65.0, 2.0])


def control_a(budget=2000):
    """Each backend on sum((u - 0.3)^2); the optimum is 0 at u = 0.3."""
    out = {}
    for m in K.METHODS:
        rng = np.random.default_rng(1)
        be = R._backend(m, 1, K.POP, rng)
        n, best, xb = 0, np.inf, None
        while n < budget:
            while be.exhausted():
                be.restart()
            u = np.clip(be.ask(), 0, 1)
            f = ((u - 0.3) ** 2).sum(axis=1)
            be.tell(f)
            n += len(f)
            if f.min() < best:
                best, xb = float(f.min()), u[int(np.argmin(f))]
        out[m] = dict(evals=n, best=best, x=np.round(xb, 4).tolist(),
                      max_coord_err=float(np.max(np.abs(xb - 0.3))))
    return out


def control_targets(S, seed=1):
    """(in-family target, permuted-null target) -- the two controls as ordinary Targets.

    Exposed so `run_all.py` can pool them into the studies' own schedule. They then ride
    along in the same rounds and cost nothing extra: wall clock is set by the number of
    rounds and each round's slowest candidate, not by how many runs are in flight.
    """
    import evalbatch as _EB
    F, lm, ok = _EB.evaluate(THETA_TRUE[None, :])
    if not ok[0]:
        raise SystemExit("the in-family reference cell is gated -- pick another theta")
    f_t = F[0].copy()
    tgt = OB.Target("infamily", f_t, S,
                    meta=dict(i_rh=float(lm[0, 0]), T_cycle=float(lm[0, 1]),
                              label="Izhikevich (0.02, 0.2, -65, 6)"))
    rng = np.random.default_rng(seed)
    f_null = f_t.copy()
    idx = np.flatnonzero(f_t > C.SENT_CUT)
    f_null[idx] = f_t[rng.permutation(idx)]
    null = OB.Target("null", f_null, S,
                     meta=dict(i_rh=float(lm[0, 0]), T_cycle=float(lm[0, 1]),
                               label="permuted features"))
    return tgt, null


def _run_all(tgt, budget, seed, study, methods=K.METHODS):
    """All three methods on one target in ONE schedule, so the batches are pooled.

    Run separately they are 32-candidate launches and cost ~8x more per cell; the studies
    pool far more than this, so the controls should not be the one place that does not.
    """
    runs = [R.Run(m, tgt, seed, budget, study, traj=False) for m in methods]
    rows = []
    R.schedule(runs, verbose=True, on_finish=rows.append)
    return {r["method"]: r for r in rows}


def from_runs(budget_note=None):
    """Assemble data/check_controls.json from the pooled run's control rows.

    The controls are run inside `run_all.py` (three 64-candidate launches of their own
    would be the worst batch size there is), so their optimizer rows already exist in
    data/results_control.jsonl. Only control A -- the quadratic, which touches no GPU --
    is run here.
    """
    import json
    rows = [json.loads(l) for l in
            open(os.path.join(K.DATA, "results_control.jsonl")) if l.strip()]
    if not rows:
        raise SystemExit("no control rows; run run_all.py (controls are pooled into it)")
    budget = rows[0]["budget"]
    res = dict(budget=budget, seed=rows[0]["seed"], theta_true=THETA_TRUE.tolist(),
               source="runs pooled into run_all.py; control A run here",
               env=K.gpu_info())
    print(f"A. optimizers on a 4-D quadratic (budget {budget})")
    res["quadratic"] = control_a(budget)
    for m, d in res["quadratic"].items():
        print(f"   {m:4s} best {d['best']:.3e}  max coord err {d['max_coord_err']:.4f}")
    sp = OB.Space()
    for which in ("infamily", "null"):
        res[which] = {}
        for r in rows:
            if r["target"] != which:
                continue
            if which == "infamily" and r.get("theta"):
                th = np.asarray(r["theta"], float)
                r = dict(r, theta_err=(th - THETA_TRUE).tolist(),
                         u_err=(sp.to_u(th)[0]
                                - sp.to_u(THETA_TRUE[None, :])[0]).tolist())
            res[which][r["method"]] = r
        print(f"\n{'B' if which == 'infamily' else 'C'}. {which}: " +
              ", ".join(f"{m} J={v['J']:.4g}" for m, v in sorted(res[which].items())))
    jb = min(v["J"] for v in res["infamily"].values())
    jn = min(v["J"] for v in res["null"].values())
    res["verdict"] = dict(best_infamily_J=jb, best_null_J=jn,
                          ratio=float(jn / max(jb, 1e-12)))
    F, lm, ok = __import__("evalbatch").evaluate(THETA_TRUE[None, :])
    res["theta_true_n_defined"] = int((F[0] > C.SENT_CUT).sum())
    res["theta_true_landmarks"] = dict(i_rh=float(lm[0, 0]), T_cycle=float(lm[0, 1]))
    K.jwrite(os.path.join(K.DATA, "check_controls.json"), res)
    print(f"\nbest in-family J {jb:.4g}   best null J {jn:.4g}   "
          f"ratio {res['verdict']['ratio']:.3g}x")
    print("wrote data/check_controls.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-runs", action="store_true",
                    help="assemble from the pooled run's control rows (no new GPU work)")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.from_runs:
        from_runs()
        return

    res = dict(budget=a.budget, seed=a.seed, theta_true=THETA_TRUE.tolist(),
               env=K.gpu_info())

    print("A. optimizers on a 4-D quadratic")
    res["quadratic"] = control_a(a.budget)
    for m, d in res["quadratic"].items():
        print(f"   {m:4s} best {d['best']:.3e}  max coord err {d['max_coord_err']:.4f}")

    S, _ = SG.atlas_sigma()
    sp = OB.Space()

    # -- the in-family target, measured by the SAME evaluator the candidates use, so the
    #    only thing between J and zero is the search
    F, lm, ok = EB.evaluate(THETA_TRUE[None, :])
    if not ok[0]:
        raise SystemExit("the in-family reference cell is gated -- pick another theta")
    f_t = F[0].copy()
    res["theta_true_n_defined"] = int((f_t > C.SENT_CUT).sum())
    res["theta_true_landmarks"] = dict(i_rh=float(lm[0, 0]), T_cycle=float(lm[0, 1]))
    tgt = OB.Target("infamily", f_t, S,
                    meta=dict(i_rh=float(lm[0, 0]), T_cycle=float(lm[0, 1]),
                              label="Izhikevich (0.02, 0.2, -65, 6)"))
    print(f"\nB. in-family self-recovery  (|D| = {tgt.n_D}/18)")
    res["infamily"] = {}
    got = _run_all(tgt, a.budget, a.seed, "control")
    for m in K.METHODS:
        row = got[m]
        th = np.asarray(row["theta"], float)
        row["theta_err"] = (th - THETA_TRUE).tolist()
        row["u_err"] = (sp.to_u(th)[0] - sp.to_u(THETA_TRUE[None, :])[0]).tolist()
        res["infamily"][m] = row
        print(f"   {m:4s} J {row['J']:12.5g}  nj {row['n_joint']:2d}/{tgt.n_D}  "
              f"theta {np.round(th, 4).tolist()}  tau {row['tau']:.4f}  "
              f"k_I {row['k_I']:.4f}")

    # -- the null: the same values, in the wrong columns
    rng = np.random.default_rng(a.seed)
    f_null = f_t.copy()
    idx = np.flatnonzero(f_t > C.SENT_CUT)
    f_null[idx] = f_t[rng.permutation(idx)]
    tgt_n = OB.Target("null", f_null, S, meta=dict(i_rh=1.0, T_cycle=1.0,
                                                   label="permuted features"))
    print("\nC. null (defined values permuted between columns)")
    res["null"] = {}
    got_n = _run_all(tgt_n, a.budget, a.seed, "control")
    for m in K.METHODS:
        row = got_n[m]
        res["null"][m] = row
        print(f"   {m:4s} J {row['J']:12.5g}  nj {row['n_joint']:2d}/{tgt_n.n_D}")

    jb = min(v["J"] for v in res["infamily"].values())
    jn = min(v["J"] for v in res["null"].values())
    res["verdict"] = dict(best_infamily_J=jb, best_null_J=jn,
                          ratio=float(jn / max(jb, 1e-12)))
    res["cache"] = EB.cache_stats()
    K.jwrite(os.path.join(K.DATA, "check_controls.json"), res)
    print(f"\nbest in-family J {jb:.4g}   best null J {jn:.4g}   "
          f"ratio {res['verdict']['ratio']:.1f}x")
    print(f"wrote {os.path.relpath(os.path.join(K.DATA, 'check_controls.json'), K.HERE)}")


if __name__ == "__main__":
    main()
