"""
analyse.py -- the report: what each method reached, how it compares to the cINN, and what
the spread across seeds says.

Reads only the append-only artifacts, so it can be rerun at any time without touching the
GPU. Per the specification's protocol for the optimizer-agnostic claim it reports the
median and IQR of the final J, the spike-train validation of each chosen theta, the spread
of theta across seeds (a degeneracy indicator, not an error), and whether a solution sits
on a box bound.

    $PY analyse.py --study study1          -> data/summary_study1.json, printed tables
"""
import os
import json
import argparse
import numpy as np

import common as K
import v3_config as C
import runners as R


def load(study):
    rows, val = [], {}
    p = os.path.join(K.DATA, f"results_{study}.jsonl")
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p) if l.strip()]
    q = os.path.join(K.DATA, f"validated_{study}.jsonl")
    if os.path.exists(q):
        val = {v["run_id"]: v for v in (json.loads(l) for l in open(q) if l.strip())}
    return rows, val


def cinn():
    d = K.jread(os.path.join(K.DATA, "cinn_baseline.json"))
    return {r["target"]: r for r in (d or {}).get("rows", [])}


_TRAJ_INDEX = None


def traj_index():
    """run_id -> (path, n_cols), built once from the trajectory sidecars."""
    global _TRAJ_INDEX
    if _TRAJ_INDEX is None:
        _TRAJ_INDEX = {}
        for cand in sorted(os.listdir(K.TRAJ)) if os.path.isdir(K.TRAJ) else []:
            if not cand.endswith(".json"):
                continue
            meta = K.jread(os.path.join(K.TRAJ, cand))
            if meta and meta.get("run_id"):
                _TRAJ_INDEX[meta["run_id"]] = (
                    os.path.join(K.TRAJ, cand[:-5] + ".bin"), len(meta["cols"]))
    return _TRAJ_INDEX


def traj(run_id, study=None):
    """(n, 8) float32 per-evaluation records, or None if they were not stored."""
    hit = traj_index().get(run_id)
    if hit is None or not os.path.exists(hit[0]):
        return None
    return np.fromfile(hit[0], dtype=np.float32).reshape(-1, hit[1])


def best_so_far(study, run_id, n_grid=200):
    """(evals, best J so far) on a log grid, for the convergence figure."""
    t = traj(run_id, study)
    if t is None:
        return None, None
    e, J = t[:, 0], t[:, 5].astype(np.float64)
    J = np.minimum.accumulate(J)
    g = np.unique(np.round(np.geomspace(1, max(e[-1], 2), n_grid)).astype(int))
    idx = np.searchsorted(e, g, side="right") - 1
    keep = idx >= 0
    return g[keep], J[idx[keep]]


def q(v, lo=25, hi=75):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)),
                iqr=[float(np.percentile(v, lo)), float(np.percentile(v, hi))],
                min=float(v.min()), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="study1")
    a = ap.parse_args()
    rows, val = load(a.study)
    if not rows:
        raise SystemExit(f"no rows for {a.study}")
    CB = cinn()

    by = {}
    for r in rows:
        by.setdefault((r["target"], r["method"]), []).append(r)

    out = {"study": a.study, "n_rows": len(rows), "by": {}}
    print(f"\n=== {a.study}: final objective, {len(rows)} runs ===")
    print(f"{'target':14s} {'method':6s} {'n':>3s} {'J median':>10s} {'J IQR':>21s} "
          f"{'J best':>10s} {'cINN best':>10s} {'evals to best':>13s} {'rst':>4s}")
    for (tg, m), rs in sorted(by.items()):
        J = [r["J"] for r in rs]
        st = q(J)
        cb = CB.get(tg, {}).get("J_best")
        ba = q([r["best_at_eval"] for r in rs])
        rec = dict(J=st, best_at_eval=ba,
                   restarts=q([r["n_restart"] for r in rs]),
                   wall_s=q([r["wall_s"] for r in rs]),
                   cinn_J_best=cb, budget=rs[0]["budget"],
                   at_bound=[r["at_bound"] for r in rs if r["at_bound"]])
        # theta spread across seeds, in u units: the degeneracy, not an error
        U = np.array([r["u"] for r in rs if r["u"]])
        if len(U) > 1:
            rec["u_iqr"] = (np.percentile(U, 75, axis=0)
                            - np.percentile(U, 25, axis=0)).tolist()
            rec["u_median"] = np.median(U, axis=0).tolist()
        vs = [val[r["run_id"]] for r in rs if r["run_id"] in val]
        if vs:
            rec["w1_shape"] = q([v["w1_shape"] for v in vs])
            rec["w1_log"] = q([v["w1_log"] for v in vs])
            pm = [v["period_match"] for v in vs if v["period_match"] is not None]
            rec["period_match_frac"] = float(np.mean(pm)) if pm else None
        out["by"][f"{tg}|{m}"] = rec
        print(f"{tg:14s} {m:6s} {st['n']:3d} {st['median']:10.4f} "
              f"[{st['iqr'][0]:9.4f},{st['iqr'][1]:9.4f}] {st['min']:10.4f} "
              f"{(cb if cb is not None else np.nan):10.4f} "
              f"{ba.get('median', np.nan):13.0f} {rec['restarts'].get('median', 0):4.0f}")

    if val:
        print(f"\n=== spike-train validation of the chosen theta ===")
        print(f"{'target':14s} {'method':6s} {'w1_shape med':>13s} {'w1_log med':>11s} "
              f"{'period match':>12s} {'cINN w1_shape':>14s}")
        for (tg, m), rs in sorted(by.items()):
            rec = out["by"][f"{tg}|{m}"]
            if "w1_shape" not in rec:
                continue
            pmf = rec.get("period_match_frac")
            pm_txt = "--" if pmf is None else f"{100 * pmf:.0f}%"
            cw = CB.get(tg, {}).get("w1_shape")
            print(f"{tg:14s} {m:6s} {rec['w1_shape'].get('median', np.nan):13.5f} "
                  f"{rec['w1_log'].get('median', np.nan):11.5f} "
                  f"{pm_txt:>12s} {(cw if cw is not None else np.nan):14.5f}")

    print(f"\n=== theta spread across seeds (IQR in u units; 1.0 = the whole box) ===")
    print(f"{'target':14s} {'method':6s} " + " ".join(f"{n:>8s}" for n in "abcd"))
    for key, rec in sorted(out["by"].items()):
        if "u_iqr" not in rec:
            continue
        tg, m = key.split("|")
        print(f"{tg:14s} {m:6s} " + " ".join(f"{v:8.4f}" for v in rec["u_iqr"]))
    bnd = {k: v["at_bound"] for k, v in out["by"].items() if v["at_bound"]}
    if bnd:
        print("\nsolutions at a box bound (reported, not corrected):")
        for k, v in bnd.items():
            print(f"  {k}: {v}")

    K.jwrite(os.path.join(K.DATA, f"summary_{a.study}.json"), out)
    print(f"\nwrote data/summary_{a.study}.json")


if __name__ == "__main__":
    main()
