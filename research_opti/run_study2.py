"""
run_study2.py -- the generalization showcase: one optimizer, swept along each target
model's own parameter, to show the identification works over a spread of configurations
rather than at one hand-picked cell.

Axes (targets.py): Morris-Lecar V3 2 -> 14 (Class 2 to Class 1), Hodgkin-Huxley gK
24 -> 40, AdEx spike-triggered w jump 0 -> 120 at V_reset = -65. The cINN is run at every
one of these configurations too (cinn_baseline.py --which study2): at 256 draws it is
nearly free, so both methods get the same map.

Trajectories are off by default here -- the per-configuration result is what the figure
needs, and 30 configurations x seeds of per-evaluation records would be the bulk of the
data for no gain.

    $PY run_study2.py --method cma --budget 10000 --seeds 3
"""
import argparse

import common as K
import driver as D
import targets as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cma", help="the one method chosen by study 1")
    ap.add_argument("--budget", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--family", default=None, help="substring: Morris, Hodgkin, AdEx")
    ap.add_argument("--sigma", default="atlas", choices=("atlas", "lhs"))
    ap.add_argument("--traj", action="store_true", help="store per-evaluation records too")
    ap.add_argument("--tpb", type=int, default=64)
    a = ap.parse_args()

    work = []
    for fam, pname, items in T.study2():
        if a.family and a.family.lower() not in fam.lower():
            continue
        print(f"study 2: {fam} along {pname}, {len(items)} configurations")
        work += [(s, l, k, p, r) for s, l, x, k, p, r in items]
    rows = D.run("study2", work, methods=a.method.split(","),
                 seeds=K.SEEDS[:a.seeds], budget=a.budget, traj=a.traj,
                 sigma=a.sigma, tpb=a.tpb)
    if rows:
        D.table(rows)


if __name__ == "__main__":
    main()
