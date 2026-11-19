"""
driver.py -- the shared driver behind run_study1.py and run_study2.py.

Both studies are the same loop over (target, method, seed): build every run, hand them all
to the lockstep scheduler at once so each GPU launch carries thousands of candidates, and
append each finished run to results_<study>.jsonl the moment it ends. A run already in
that file is skipped, so an interrupted sweep resumes.
"""
import os
import time
import numpy as np

import common as K
import evalbatch as EB
import runners as R
import targets as T
import sigma_opt as SG


def build(study, work, methods, seeds, budget, traj, S):
    """[(Run, ...)] for everything not already on disk."""
    have = R.completed(study)
    runs, skipped, gated = [], 0, []
    for slug, label, kind, payload, rh in work:
        d = T.cached(slug, label, kind, payload, rh)
        if d["gated"] or d["n_defined"] < K.N_DEF_MIN:
            gated.append((slug, d["reason"], d["n_defined"]))
            continue
        tg = T.target_of(d, S)
        for m in methods:
            for sd in seeds:
                rid = f"{study}|{tg.name}|{m}|s{int(sd)}|b{int(budget)}"
                if rid in have:
                    skipped += 1
                    continue
                runs.append(R.Run(m, tg, sd, budget, study, traj=traj))
    return runs, skipped, gated


def run(study, work, methods=K.METHODS, seeds=K.SEEDS, budget=10_000, traj=True,
        sigma="atlas", tpb=64, max_live=None):
    """Drive a study to completion. Returns the rows written in this call."""
    K.ensure_dirs()
    S = (SG.atlas_sigma() if sigma == "atlas" else SG.lhs_sigma())[0]
    runs, skipped, gated = build(study, work, methods, seeds, budget, traj, S)
    for slug, why, nd in gated:
        print(f"  {slug}: target unusable (|D|={nd}, {why}) -- no runs")
    if skipped:
        print(f"  {skipped} run(s) already finished, skipped (resume)")
    if not runs:
        print("  nothing to do")
        return []

    tot_eval = len(runs) * budget
    print(f"  {len(runs)} runs x {budget} evals = {tot_eval} evaluations")
    rows = []

    def done(row):
        rows.append(row)
        R.append_result(study, row)

    t0 = time.time()
    # max_live caps how many runs are in flight, in case the pooled batch outgrows memory;
    # None keeps them all live, which is what makes the launches large
    groups = ([runs] if not max_live else
              [runs[i:i + max_live] for i in range(0, len(runs), max_live)])
    for gi, grp in enumerate(groups):
        if len(groups) > 1:
            print(f"  group {gi+1}/{len(groups)}: {len(grp)} runs")
        R.schedule(grp, tpb=tpb, on_finish=done)
    el = time.time() - t0
    n = sum(r["n_eval"] for r in rows)
    st = EB.cache_stats()
    print(f"\n  {n} evaluations in {el/3600:.3f} h = {1e3*el/max(n,1):.2f} ms/eval; "
          f"cache hits {100*st['hits']/max(st['asked'],1):.1f}%, "
          f"PASS 0 rejected {100*st['gated0']/max(st['pass0'],1):.1f}%")
    K.jwrite(os.path.join(K.DATA, f"cost_{study}.json"),
             dict(study=study, n_runs=len(runs), budget=budget, evals=n,
                  wall_s=round(el, 1), ms_per_eval=round(1e3 * el / max(n, 1), 3),
                  cache=st, methods=list(methods), seeds=[int(s) for s in seeds],
                  sigma=sigma, env=K.gpu_info()))
    return rows


def run_many(jobs, sigma="atlas", tpb=64, extra=None):
    """Drive several studies in ONE lockstep schedule.

    jobs = [(study, work, methods, seeds, budget, traj), ...]

    Worth doing rather than running the studies back to back. Wall clock here is
    (budget / population) rounds times the cost of each round's slowest candidate, and that
    cost is a property of the candidate, not of the batch: a round of 11000 candidates
    takes about as long as a round of 2900. Pooling independent runs is therefore close to
    free, and the measured per-cell cost falls with batch size for the same reason
    (1635 ms/cell at a batch of 64, 357 at 1024).
    """
    K.ensure_dirs()
    S = (SG.atlas_sigma() if sigma == "atlas" else SG.lhs_sigma())[0]
    runs, meta = [], []
    for study, work, methods, seeds, budget, traj in jobs:
        rs, skipped, gated = build(study, work, methods, seeds, budget, traj, S)
        for slug, why, nd in gated:
            print(f"  {study}: {slug} unusable (|D|={nd}, {why})")
        print(f"  {study}: {len(rs)} runs to do, {skipped} already finished")
        runs += rs
        meta.append((study, len(rs), budget))
    if extra is not None:
        study, methods, seeds, budget, traj = extra
        import check_controls as CC
        have = R.completed(study)
        for tg in CC.control_targets(S):
            for m in methods:
                for sd in seeds:
                    rid = f"{study}|{tg.name}|{m}|s{int(sd)}|b{int(budget)}"
                    if rid in have:
                        continue
                    runs.append(R.Run(m, tg, sd, budget, study, traj=traj))
        n_ctrl = sum(1 for r in runs if r.study == study)
        print(f"  {study}: {n_ctrl} control runs (in-family self-recovery and the "
              f"permuted null), pooled so they cost no extra rounds")
        meta.append((study, n_ctrl, budget))
    if not runs:
        print("  nothing to do")
        return []
    print(f"  pooled: {len(runs)} runs, {sum(r.pop0 for r in runs)} candidates per round, "
          f"{sum(r.budget for r in runs)} evaluations total")
    rows, timing = [], []
    t0 = time.time()
    R.schedule(runs, tpb=tpb, timing=timing,
               on_finish=lambda row: (rows.append(row),
                                      R.append_result(row["study"], row)))
    el = time.time() - t0
    n = sum(r["n_eval"] for r in rows)
    st = EB.cache_stats()
    print(f"\n  {n} evaluations in {el/3600:.3f} h = {1e3*el/max(n,1):.2f} ms/eval; "
          f"cache hits {100*st['hits']/max(st['asked'],1):.1f}%, "
          f"PASS 0 rejected {100*st['gated0']/max(st['pass0'],1):.1f}% "
          f"(of which too slow: {st['slow']})")
    K.jwrite(os.path.join(K.DATA, "cost_pooled.json"),
             dict(jobs=[dict(study=s, n_runs=k, budget=b) for s, k, b in meta],
                  evals=n, wall_s=round(el, 1),
                  ms_per_eval=round(1e3 * el / max(n, 1), 3), cache=st,
                  rounds_timing=timing, sigma=sigma, env=K.gpu_info()))
    return rows


def table(rows):
    """Median final J per (target, method), the quantity study 1 compares."""
    by = {}
    for r in rows:
        by.setdefault((r["target"], r["method"]), []).append(r["J"])
    print(f"\n  {'target':14s} {'method':6s} {'n':>3s} {'J median':>12s} "
          f"{'J min':>12s} {'J max':>12s}")
    for (tg, m), v in sorted(by.items()):
        a = np.array(v)
        print(f"  {tg:14s} {m:6s} {len(v):3d} {np.median(a):12.4f} "
              f"{a.min():12.4f} {a.max():12.4f}")
