"""
run_study1.py -- the head-to-head: three optimizer families against the cINN, on three
targets at the configuration the pipeline's zoo defines (HH, Morris-Lecar C1, AdEx tonic).

Every run minimizes the same d2, in the same sigma, measured by the same GPU evaluator the
cINN arm uses (cinn_baseline.py), so the arms differ only in how a theta is proposed.
Equal evaluation budget per run; the cINN's cost is reported separately, because one
forward pass against an offline atlas is a different kind of cost, not a smaller number of
the same kind.

    $PY run_study1.py --budget 10000 --seeds 10
    $PY run_study1.py --budget 10000 --seeds 10 --methods cma      # resume/extend
"""
import argparse

import common as K
import driver as D
import targets as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, default=len(K.SEEDS))
    ap.add_argument("--methods", default=",".join(K.METHODS))
    ap.add_argument("--targets", default="hh,ml_c1,adex_tonic")
    ap.add_argument("--sigma", default="atlas", choices=("atlas", "lhs"))
    ap.add_argument("--no-traj", dest="traj", action="store_false")
    ap.add_argument("--tpb", type=int, default=64)
    a = ap.parse_args()

    want = set(a.targets.split(","))
    work = [r for r in T.study1() if r[0] in want]
    print(f"study 1: {[r[0] for r in work]}")
    rows = D.run("study1", work, methods=a.methods.split(","),
                 seeds=K.SEEDS[:a.seeds], budget=a.budget, traj=a.traj,
                 sigma=a.sigma, tpb=a.tpb)
    if rows:
        D.table(rows)


if __name__ == "__main__":
    main()
