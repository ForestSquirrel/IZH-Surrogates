"""
run_all.py -- both studies in ONE pooled schedule.

Preferred over running `run_study1.py` and `run_study2.py` back to back. Every pooled run
advances in lockstep, so the wall clock is `(budget / population)` rounds times the cost of
each round's SLOWEST candidate -- and that cost belongs to the candidate, not to the batch.
A round of 11000 candidates therefore costs about what a round of 2900 costs, and pooling
the two studies is close to free. The same effect is why per-cell cost falls so steeply
with batch size: 1635 ms/cell at a batch of 64, 357 at 1024 (measured).

Results still go to `results_study1.jsonl` and `results_study2.jsonl` separately, and both
resume independently.

    $PY run_all.py --budget 10000 --seeds1 10 --seeds2 3
    $PY run_all.py --budget 10000 --dry-run         # just the shape and the count
"""
import argparse

import common as K
import driver as D
import targets as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=10_000)
    ap.add_argument("--seeds1", type=int, default=len(K.SEEDS),
                    help="seeds per (target, method) in study 1")
    ap.add_argument("--seeds2", type=int, default=3,
                    help="seeds per configuration in study 2")
    ap.add_argument("--methods", default=",".join(K.METHODS))
    ap.add_argument("--method2", default="cma",
                    help="the one method study 2 uses (chosen by study 1)")
    ap.add_argument("--only", default="both", choices=("both", "study1", "study2"))
    ap.add_argument("--traj2", action="store_true",
                    help="store per-evaluation records for study 2 as well")
    ap.add_argument("--sigma", default="atlas", choices=("atlas", "lhs"))
    ap.add_argument("--tpb", type=int, default=64)
    ap.add_argument("--no-controls", dest="controls", action="store_false",
                    help="skip the in-family and permuted-null controls")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs = []
    if a.only in ("both", "study1"):
        jobs.append(("study1", list(T.study1()), a.methods.split(","),
                     K.SEEDS[:a.seeds1], a.budget, True))
    if a.only in ("both", "study2"):
        work = []
        for fam, pname, items in T.study2():
            work += [(s, l, k, p, r) for s, l, x, k, p, r in items]
        jobs.append(("study2", work, a.method2.split(","),
                     K.SEEDS[:a.seeds2], a.budget, a.traj2))

    for study, work, methods, seeds, budget, traj in jobs:
        n = len(work) * len(methods) * len(seeds)
        print(f"{study}: {len(work)} targets x {len(methods)} methods x {len(seeds)} "
              f"seeds = {n} runs, {n*budget} evaluations, trajectories {'on' if traj else 'off'}")
    if a.dry_run:
        tot = sum(len(w) * len(m) * len(s) for _, w, m, s, _, _ in jobs)
        print(f"\npooled: {tot} runs, {tot*K.POP} candidates per round, "
              f"{a.budget//K.POP} rounds at population {K.POP}")
        return

    rows = D.run_many(jobs, sigma=a.sigma, tpb=a.tpb,
                      extra=_controls(a) if a.controls else None)
    if rows:
        D.table(rows)


def _controls(a):
    """The two controls, as a third job in the same schedule.

    They belong in the pooled run rather than in a script of their own: on their own they
    are three 64-candidate launches, which is the worst batch size there is (1635 ms/cell
    against 28.7 at 16384), and pooled they are free.
    """
    return ("control", a.methods.split(","), K.SEEDS[:1], a.budget, True)


if __name__ == "__main__":
    main()
