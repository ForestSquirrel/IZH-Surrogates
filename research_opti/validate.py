"""
validate.py -- spike-train validation of each run's chosen theta, as a separate pass.

d2 is a feature-space distance in atlas sigma and is not a spike-train distance: across
model families the two correlate poorly, and a low d2 can be bought by matching the
features both models define while firing differently. So the optimization is scored by
d2, and the result is then judged by quantities d2 never saw:

  w1_shape   Wasserstein-1 between the settled ISI distributions under the reference step
             (gamma = 2), both sides divided by their own cycle time -- dimensionless, so
             it does not reward the gauge, and well defined on atoms (a tonic train is a
             point mass), which a histogram distance is not
  w1_log     the same on log ISIs, which weighs the short intervals of a burst
  period     whether the ISI-map period matches, which is the firing-regime check

This runs after the fact, from results_<study>.jsonl, and appends to
validated_<study>.jsonl. Nothing is recomputed: a row already validated is skipped, and
each target's own ISI sequence is measured once and cached.

    $PY validate.py --study study1
    $PY validate.py --study study2 --regime        # also the 9-point regime ladder
"""
import os
import json
import argparse
import numpy as np

import common as K
K.ensure_dirs()
import v3_config as C                                         # noqa: E402
import targets as T                                           # noqa: E402

GAMMA = C.GAMMA_STEP            # the reference step the ISIs are measured at


def _payload(slug):
    """(label, kind, payload, rheo_hi) for a slug, from either study's definition."""
    for s, label, kind, payload, rh in T.study1():
        if s == slug:
            return label, kind, payload, rh
    for fam, pname, items in T.study2():
        for s, label, x, kind, payload, rh in items:
            if s == slug:
                return label, kind, payload, rh
    raise KeyError(slug)


def target_isis(slug):
    """The target's settled ISIs at gamma = 2, in units of its own cycle time. Cached."""
    p = os.path.join(K.TARGETS, f"{slug}_isi.npy")
    if os.path.exists(p):
        return np.load(p)
    from v3_scoring import isis_an, isis_of
    from v3_core import alloc_buffers
    label, kind, payload, rh = _payload(slug)
    d = T.cached(slug, label, kind, payload, rh)
    if kind == "continuous":
        isi = isis_an(payload, d["i_rh"], d["T_cycle"])        # already in T_cycle units
    else:
        rhs, pr = payload
        raw, lm = isis_of(np.ascontiguousarray(pr), alloc_buffers(), gamma=GAMMA,
                          rhs=rhs, rheo_hi=rh)                 # rhs MUST be passed here
        isi = None if raw is None else np.asarray(raw) / lm["T_cycle"]
    isi = np.empty(0) if isi is None else np.asarray(isi, float)
    np.save(p, isi)
    return isi


def izh_isis(theta, bufs):
    """The surrogate's settled ISIs at gamma = 2, in units of its own cycle time."""
    from v3_scoring import isis_of
    raw, lm = isis_of(np.ascontiguousarray(np.asarray(theta, float)), bufs, gamma=GAMMA)
    if raw is None or lm["T_cycle"] < C.SENT_CUT:
        return None
    return np.asarray(raw) / lm["T_cycle"]


def period_of(isi):
    from v3_landmarks import isi_period
    if isi is None or len(isi) < 2:
        return None
    x = np.ascontiguousarray(np.asarray(isi, float))
    return int(isi_period(x, len(x), C.P_MAX, C.PERIOD_TOL))


def augment_cinn(path=None):
    """Add the spike-train metrics to data/cinn_baseline.json's best draw per target.

    The cINN arm has to be judged by the same quantities as the optimizer arm, and d2 is
    not one of them: a feature-space distance can be low while the spike train is wrong,
    which is the whole reason w1 is reported separately.
    """
    import numpy as np
    from v3_core import alloc_buffers
    from v3_scoring import w1
    path = path or os.path.join(K.DATA, "cinn_baseline.json")
    d = K.jread(path)
    if not d:
        raise SystemExit("no cinn_baseline.json -- run cinn_baseline.py first")
    bufs = alloc_buffers()
    for r in d["rows"]:
        isi_t = target_isis(r["target"])
        isi_s = izh_isis(r["theta_best"], bufs)
        r["p_target"] = period_of(isi_t)
        r["p_sur"] = period_of(isi_s)
        r["period_match"] = (None if r["p_target"] is None or r["p_sur"] is None
                             else bool(r["p_target"] == r["p_sur"]))
        if isi_s is not None and len(isi_t) >= 2 and len(isi_s) >= 2:
            r["w1_shape"] = w1(isi_t, isi_s)
            r["w1_log"] = w1(np.log(isi_t), np.log(isi_s))
        else:
            r["w1_shape"] = r["w1_log"] = None
        print(f"{r['target']:14s} J {r['J_best']:9.4f}  w1_shape "
              f"{(r['w1_shape'] if r['w1_shape'] is not None else float('nan')):9.5f}  "
              f"period {r['p_target']}->{r['p_sur']}", flush=True)
    K.jwrite(path, d)
    print(f"updated {os.path.relpath(path, K.HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="study1")
    ap.add_argument("--cinn", action="store_true",
                    help="instead: add w1 and the period to cinn_baseline.json")
    ap.add_argument("--regime", action="store_true",
                    help="also run the 9-point regime ladder on the surrogate")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cinn:
        augment_cinn()
        return

    src = os.path.join(K.DATA, f"results_{a.study}.jsonl")
    dst = os.path.join(K.DATA, f"validated_{a.study}.jsonl")
    if not os.path.exists(src):
        raise SystemExit(f"no {os.path.relpath(src, K.HERE)}")
    rows = [json.loads(l) for l in open(src) if l.strip()]
    have = set()
    if os.path.exists(dst):
        have = {json.loads(l)["run_id"] for l in open(dst) if l.strip()}
    todo = [r for r in rows if r["run_id"] not in have and r.get("theta")]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(rows)} rows, {len(have)} already validated, {len(todo)} to do")

    from v3_core import alloc_buffers
    from v3_scoring import w1
    bufs = alloc_buffers()
    cache_t = {}
    out = open(dst, "a")
    for i, r in enumerate(todo):
        slug = r["target"]
        if slug not in cache_t:
            it = target_isis(slug)
            cache_t[slug] = (it, period_of(it))
        isi_t, p_t = cache_t[slug]
        isi_s = izh_isis(r["theta"], bufs)
        p_s = period_of(isi_s)
        v = dict(run_id=r["run_id"], study=r["study"], target=slug, method=r["method"],
                 seed=r["seed"], J=r["J"], theta=r["theta"], tau=r["tau"], k_I=r["k_I"],
                 n_isi_target=int(len(isi_t)), n_isi_sur=0 if isi_s is None else len(isi_s),
                 w1_shape=None, w1_log=None, p_target=p_t, p_sur=p_s,
                 period_match=None if (p_t is None or p_s is None) else bool(p_t == p_s))
        if isi_s is not None and len(isi_t) >= 2 and len(isi_s) >= 2:
            v["w1_shape"] = w1(isi_t, isi_s)
            v["w1_log"] = w1(np.log(isi_t), np.log(isi_s))
            v["isi_mean_target"] = float(np.mean(isi_t))
            v["isi_mean_sur"] = float(np.mean(isi_s))
        if a.regime:
            from v3_regime import regime_report_izh
            from v3_models import izh_rhs_v3
            rep = regime_report_izh(izh_rhs_v3, np.asarray(r["theta"], float),
                                    r["i_rh_s"], r["T_cycle_s"], bufs)
            v["regime"] = {k: rep[k] for k in ("verdict", "n_jump", "max_jump")
                           if k in rep}
        out.write(json.dumps(v, default=float) + "\n")
        out.flush()
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}", flush=True)
    out.close()

    val = [json.loads(l) for l in open(dst) if l.strip()]
    by = {}
    for v in val:
        by.setdefault((v["target"], v["method"]), []).append(v)
    print(f"\n{'target':14s} {'method':6s} {'n':>3s} {'J med':>10s} {'w1_shape med':>13s} "
          f"{'w1_log med':>11s} {'period match':>12s}")
    for (tg, m), vs in sorted(by.items()):
        ws = [v["w1_shape"] for v in vs if v["w1_shape"] is not None]
        wl = [v["w1_log"] for v in vs if v["w1_log"] is not None]
        pm = [v["period_match"] for v in vs if v["period_match"] is not None]
        print(f"{tg:14s} {m:6s} {len(vs):3d} "
              f"{np.median([v['J'] for v in vs]):10.4f} "
              f"{(np.median(ws) if ws else np.nan):13.5f} "
              f"{(np.median(wl) if wl else np.nan):11.5f} "
              f"{(f'{100*np.mean(pm):.0f}%' if pm else '--'):>12s}")
    print(f"\nwrote {os.path.relpath(dst, K.HERE)}")


if __name__ == "__main__":
    main()
