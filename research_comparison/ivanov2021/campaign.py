"""
campaign.py -- the Ivanov MNIST campaign: sweep ranges -> weight sweep -> full runs (resumable).

Conditions (16):
  lif                              the paper's LSM (reproduction)
  hh, ml_c1, adex_adapting         zoo neurons in the liquid AND the input layer
  <zoo>@fastin                     input neurons time-scaled to follow the LIF input code
                                   (spec.FASTIN_SPEED, input_speed.py)
  <zoo>@lifin                      LIF-input control: zoo liquid driven by the LIF input layer
  sur:<zoo>                        drop-in surrogate (target's currents through the gauge)
  sur*:<zoo>                       self-calibrated surrogate (own 1 ms-pulse unit)
Stages (each skips work already on disk):
  ranges  liquid rate vs w on 200 samples over 0.05-51.2 (log); sweep grid = 7 log-spaced w
          between the lowest w with mean liquid rate >= 5 Hz and the highest <= 150 Hz
          (LIF: the paper's 0.4-1.2). No w in band -> condition marked dead (no sweep).
  sweep   one network (seed 100), 10k / 2k / 2k subset, every grid w
  full    the best-validation w of the sweep, networks 0..2, 50k / 10k / 10k; test-set spike
          counts saved (identical stimuli across conditions: CKA / agreement analyses)

    PY=python
    $PY campaign.py --eta                       # ETA from a short bench, nothing written
    $PY campaign.py --stage all                 # ranges + sweep + full (MNIST, ~8 h)
    $PY campaign.py --stage all --smoke --out <scratch>   # tiny sizes, end-to-end check
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R
import readout as RO
import spec as S
import circuit as CI, encode as EN
from lsmkit import spec as LS, engine_gpu
from lsmkit.engine_cpu import Rec

ZOO = ("hh", "ml_c1", "adex_adapting")
CONDITIONS = (["lif"] + list(ZOO) + [f"{z}@fastin" for z in ZOO] + [f"{z}@lifin" for z in ZOO]
              + [f"sur:{z}" for z in ZOO] + [f"sur*:{z}" for z in ZOO])
# the zoo comparison (user decision 2026-11-06): identical topology and input code, only the
# neuron model swapped -- the paper's LIF, each zoo neuron with its time-scaled input layer, and
# its self-calibrated surrogate with the surrogate of that input layer (fastin_sur.py)
FINAL = ["lif"] + [f"{z}@fastin" for z in ZOO] + [f"sur*:{z}@fastin" for z in ZOO]
# gain-matched surrogates (gain.py): physical current units, own input layer, w_s = g w*_target
MATCHED = [f"sur:{z}@fastin" for z in ZOO]
STATIC_FINAL = ["lif"] + [f"{z}@fastin" for z in ZOO] + MATCHED
SCAN_W = tuple(0.05 * 2 ** k for k in range(11))          # 0.05 .. 51.2
BAND = (5.0, 150.0)
LIF_GRID = tuple(np.round(np.linspace(0.4, 1.2, 7), 4))
SIZES = dict(scan=200, sweep=(10000, 2000, 2000), full=(50000, 10000, 10000), nets=3, grid=7)
SMOKE = dict(scan=40, sweep=(400, 100, 100), full=(600, 150, 150), nets=1, grid=3)


def safe(m):
    return m.replace(":", "_").replace("*", "x").replace("@", "_at_")


def rate(model, w, nin, data, n, dataset):
    dt = R.dt_for(model)
    cnt, comp, _ = R.counts_for(model, w, nin, data, np.arange(n), dataset, dt, 100, chunk=n)
    return float(cnt.mean() / (S.T_SAMPLE / 1000)), comp


def stage_ranges(out, conds, nin, data, Z, dataset):
    path = os.path.join(out, "ranges.json")
    rng = json.load(open(path)) if os.path.exists(path) else {}
    for m in conds:
        if m in rng:
            continue
        if m == "lif":
            rng[m] = dict(grid=list(LIF_GRID), scan=None, dead=False)
        else:
            sc = {}
            for w in SCAN_W:
                sc[w] = rate(m, w, nin, data, Z["scan"], dataset)[0]
                if sc[w] > 10 * BAND[1]:
                    break
            ok = [w for w, r in sc.items() if BAND[0] <= r <= BAND[1]]
            if ok:
                lo, hi = min(ok), max(ok)
                hi = hi if hi > lo else lo * 2
                grid = list(np.round(np.geomspace(lo, hi, Z["grid"]), 4))
            else:
                grid = []
            rng[m] = dict(grid=grid, scan={str(k): v for k, v in sc.items()}, dead=not ok)
        print(f"[ranges] {m:22s} grid {rng[m]['grid']}" + ("  DEAD" if rng[m]["dead"] else ""), flush=True)
        json.dump(rng, open(path, "w"), indent=1)
    return rng


def one(model, w, net, sizes, nin, data, y, dataset, save_test=None):
    tr, va, te = R.split_idx(sizes)
    idx = np.concatenate([tr, va, te])
    dt = R.dt_for(model)
    t0 = time.time()
    cnt, comp, circ = R.counts_for(model, w, nin, data, idx, dataset, dt, 100 + net,
                                   chunk=2000 if "hh" not in model else 1000)
    n1, n2 = len(tr), len(tr) + len(va)
    r = RO.train(cnt[:n1], y[tr], cnt[n1:n2], y[va], cnt[n2:], y[te], seed=net)
    r.update(model=model, w=float(w), net=net, sizes=list(sizes), dt=dt,
             rate_hz=float(cnt.mean() / (S.T_SAMPLE / 1000)), silent=float((cnt.sum(0) == 0).mean()),
             compute=comp, wall_total_s=time.time() - t0)
    if save_test:
        np.savez_compressed(save_test, counts=cnt[n2:], labels=y[te])
    return r


def stage_sweep(out, rng, nin, data, y, Z, dataset):
    best = {}
    for m, rg in rng.items():
        if rg["dead"]:
            continue
        d = os.path.join(out, "sweep", safe(m))
        os.makedirs(d, exist_ok=True)
        rows = []
        for w in rg["grid"]:
            f = os.path.join(d, f"w{w:g}.json")
            if os.path.exists(f):
                rows.append(json.load(open(f))); continue
            r = one(m, w, 0, Z["sweep"], nin, data, y, dataset)
            json.dump(r, open(f, "w"), indent=1)
            rows.append(r)
            print(f"[sweep] {m:22s} w {w:<8g} val {r['val'] * 100:6.2f}  test {r['test'] * 100:6.2f}  "
                  f"rate {r['rate_hz']:6.1f} Hz  {r['wall_total_s']:.0f}s", flush=True)
        b = max(rows, key=lambda r: r["val"])
        best[m] = b["w"]
    f = os.path.join(out, "best_w.json")
    allb = json.load(open(f)) if os.path.exists(f) else {}
    allb.update(best)                    # merge: a subset run keeps the other conditions' optima
    json.dump(allb, open(f, "w"), indent=1)
    return best


def full_dir(out, m, w):
    """full/<cond>; a later best w (after 'extend') gets full/<cond>_w<w> so finished runs at the
    old w are never overwritten"""
    d = os.path.join(out, "full", safe(m))
    f = os.path.join(d, "net0.json")
    if os.path.exists(f) and abs(json.load(open(f))["w"] - float(w)) > 1e-9:
        d = os.path.join(out, "full", f"{safe(m)}_w{float(w):g}")
    return d


def stage_extend(out, conds, nin, data, y, Z, dataset, n_add=2):
    """sweep optimum on a grid edge -> add n_add grid points beyond that edge (same spacing:
    linear for the LIF paper grid, log otherwise), sweep them; repeat until interior (max 3x).
    Stops in a direction once a point leaves [BAND[0] / 2, 2 BAND[1]] Hz (the grids are defined
    by the 5-150 Hz band; this allows a margin, not a runaway liquid)"""
    rng = json.load(open(os.path.join(out, "ranges.json")))
    for m in conds:
        for _ in range(3):
            d = os.path.join(out, "sweep", safe(m))
            rows = {float(json.load(open(os.path.join(d, f)))["w"]): json.load(open(os.path.join(d, f)))
                    for f in os.listdir(d) if f.endswith(".json")}
            ws = sorted(rows)
            b = max(rows.values(), key=lambda r: r["val"])["w"]
            if ws[0] < b < ws[-1]:
                break
            lin = m == "lif"
            step = (ws[1] - ws[0]) if lin else ws[1] / ws[0]
            if b == ws[0]:
                new = [ws[0] - step * (k + 1) if lin else ws[0] / step ** (k + 1) for k in range(n_add)]
            else:
                new = [ws[-1] + step * (k + 1) if lin else ws[-1] * step ** (k + 1) for k in range(n_add)]
            new = [float(np.round(w, 4)) for w in new if w > 0]
            if not new:
                break
            out_band = False
            for w in new:
                r = one(m, w, 0, Z["sweep"], nin, data, y, dataset)
                json.dump(r, open(os.path.join(d, f"w{w:g}.json"), "w"), indent=1)
                print(f"[extend] {m:22s} w {w:<8g} val {r['val'] * 100:6.2f}  test {r['test'] * 100:6.2f}  "
                      f"rate {r['rate_hz']:6.1f} Hz", flush=True)
                if not BAND[0] / 2 <= r["rate_hz"] <= 2 * BAND[1]:
                    out_band = True
                    break
            rng[m]["grid"] = sorted(set(rng[m]["grid"]) | set(new))
            json.dump(rng, open(os.path.join(out, "ranges.json"), "w"), indent=1)
            if out_band:
                break
    best = {}
    for m in conds:
        d = os.path.join(out, "sweep", safe(m))
        rows = [json.load(open(os.path.join(d, f))) for f in os.listdir(d) if f.endswith(".json")]
        best[m] = max(rows, key=lambda r: r["val"])["w"]
    allb = json.load(open(os.path.join(out, "best_w.json")))
    allb.update(best)
    json.dump(allb, open(os.path.join(out, "best_w.json"), "w"), indent=1)
    return best


def stage_pred(out, best, nin, data, y, Z, dataset):
    """confusion matrices for full runs that predate them: recompute the counts (same seeds and
    stimuli) and the readout -> net<k>_pred.json next to the original (never modified)"""
    for m, w in best.items():
        d = full_dir(out, m, w)
        for net in range(Z["nets"]):
            f = os.path.join(d, f"net{net}.json")
            fp = os.path.join(d, f"net{net}_pred.json")
            if not os.path.exists(f) or "confusion" in json.load(open(f)) or os.path.exists(fp):
                continue
            r = one(m, w, net, Z["full"], nin, data, y, dataset)
            r["original_test"] = json.load(open(f))["test"]
            json.dump(r, open(fp, "w"), indent=1)
            print(f"[pred] {m:22s} net {net}  test {r['test'] * 100:6.2f} (original {r['original_test'] * 100:6.2f})", flush=True)


def stage_full(out, best, nin, data, y, Z, dataset):
    for m, w in best.items():
        d = full_dir(out, m, w)
        os.makedirs(d, exist_ok=True)
        for net in range(Z["nets"]):
            f = os.path.join(d, f"net{net}.json")
            if os.path.exists(f):
                continue
            r = one(m, w, net, Z["full"], nin, data, y, dataset,
                    save_test=os.path.join(d, f"net{net}_test_counts.npz"))
            json.dump(r, open(f, "w"), indent=1)
            print(f"[full] {m:22s} w {w:<8g} net {net}  test {r['test'] * 100:6.2f}  val {r['val'] * 100:6.2f}  "
                  f"rate {r['rate_hz']:6.1f} Hz  {r['wall_total_s'] / 60:.1f} min", flush=True)
    summ = json.load(open(os.path.join(out, "summary.json"))) if os.path.exists(os.path.join(out, "summary.json")) else {}
    for m in best:
        d = full_dir(out, m, best[m])
        rows = [json.load(open(os.path.join(d, f"net{k}.json"))) for k in range(Z["nets"])
                if os.path.exists(os.path.join(d, f"net{k}.json"))]
        if rows:
            acc = np.array([r["test"] for r in rows]) * 100
            summ[m] = dict(w=best[m], test_mean=float(acc.mean()), test_sd=float(acc.std()),
                           test_top=float(acc.max()), n=len(rows),
                           rate_hz=float(np.mean([r["rate_hz"] for r in rows])))
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    for m, s in summ.items():
        print(f"{m:22s} {s['test_mean']:6.2f} ± {s['test_sd']:.2f} %  (top {s['test_top']:.2f}, w {s['w']:g})")


def cond_w(out, m):
    """liquid weight of a condition: the sweep optimum (best_w.json) or, for a gain-matched
    surrogate, w_s = g w*_target (matched_w.json)"""
    for f in ("best_w.json", "matched_w.json"):
        p = os.path.join(out, f)
        if os.path.exists(p):
            js = json.load(open(p))
            if m in js:
                return js[m]["w_s"] if isinstance(js[m], dict) else js[m]
    return None


def stage_matched(out, nin, data):
    """predicted gain per zoo model at the target optimum (gain.py, no network simulation)
    -> matched_w.json {sur:<z>@fastin: {w_t, g, w_s, nu, ...}}; returns {cond: w_s}"""
    import gain as G
    f = os.path.join(out, "matched_w.json")
    js = json.load(open(f)) if os.path.exists(f) else {}
    bw = json.load(open(os.path.join(out, "best_w.json")))
    for z, m in zip(ZOO, MATCHED):
        if m not in js:
            p = G.predict(z, bw[f"{z}@fastin"], data=data, nin=nin)
            assert p.get("w_s"), f"no gain-matched weight for {z}: {p}"
            js[m] = p
            json.dump(js, open(f, "w"), indent=1)
        print(f"[matched] {m:22s} w_t {js[m]['w_t']:g}  nu* {js[m]['nu']:.1f} Hz  g {js[m]['g']:.3f}  w_s {js[m]['w_s']:.4g}", flush=True)
    return {m: js[m]["w_s"] for m in MATCHED}


def eta(conds, nin, data, dataset, Z):
    per = {}
    for m in conds:
        R.counts_for(m, 1.0, nin, data, np.arange(16), dataset, R.dt_for(m), 0, chunk=16)  # compile
        n = 500
        _, comp = rate(m, 1.0, nin, data, n, dataset)
        per[m] = comp["wall_s"] / n
    enc_read = 3e-3                          # s per sample: encoding + readout share (LIF probe)
    sw = sum((per[m] + enc_read) * sum(Z["sweep"]) * Z["grid"] for m in conds)
    fu = sum((per[m] + enc_read) * sum(Z["full"]) * Z["nets"] for m in conds)
    for m in conds:
        print(f"  {m:22s} {per[m] * 1e3:6.2f} ms/sample (sim)")
    print(f"ETA: ranges ~{len(conds) * 0.5:.0f} min, sweep ~{sw / 3600:.1f} h, full ~{fu / 3600:.1f} h "
          f"(upper bound: assumes {Z['grid']} grid points for every condition)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--stage", default="all", choices=("ranges", "sweep", "full", "all", "final", "matched"))
    ap.add_argument("--conditions", default=None, help="default: all CONDITIONS (stage final: FINAL)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eta", action="store_true")
    a = ap.parse_args()
    Z = SMOKE if a.smoke else SIZES
    conds = a.conditions.split(",") if a.conditions else (FINAL if a.stage == "final" else CONDITIONS)
    data, y, nin = R.load(a.dataset)
    if a.eta:
        eta(conds, nin, data, a.dataset, Z)
        return
    out = a.out or os.path.join(HERE, "data", f"campaign_{a.dataset}")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if a.stage == "matched":
        best = stage_matched(out, nin, data)
        stage_full(out, best, nin, data, y, Z, a.dataset)
        print(f"done in {(time.time() - t0) / 3600:.2f} h")
        return
    rng = stage_ranges(out, conds, nin, data, Z, a.dataset)
    if a.stage == "ranges":
        return
    if a.stage == "final":
        # zoo comparison: sweep the new conditions, extend edge optima, full runs, confusion
        stage_sweep(out, {m: rng[m] for m in conds}, nin, data, y, Z, a.dataset)
        best = stage_extend(out, conds, nin, data, y, Z, a.dataset)
        stage_full(out, best, nin, data, y, Z, a.dataset)
        stage_pred(out, best, nin, data, y, Z, a.dataset)
        print(f"done in {(time.time() - t0) / 3600:.2f} h")
        return
    if a.stage in ("sweep", "all"):
        best = stage_sweep(out, {m: rng[m] for m in conds}, nin, data, y, Z, a.dataset)
    else:
        best = json.load(open(os.path.join(out, "best_w.json")))
    if a.stage in ("full", "all"):
        stage_full(out, best, nin, data, y, Z, a.dataset)
    print(f"done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
