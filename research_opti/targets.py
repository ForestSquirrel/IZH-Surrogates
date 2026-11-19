"""
targets.py -- the targets both studies identify a surrogate for, and their frozen feature
vectors.

Study 1 uses three models at the configuration the pipeline's zoo already defines, so the
optimizer results sit next to published cINN numbers for the same cells:

    hh            Hodgkin-Huxley            reset-free
    ml_c1         Morris-Lecar, Class 1     reset-free
    adex_tonic    AdEx, tonic regime        reset (needs rheo_hi = 2000 pA)

Study 2 sweeps each model along one of its OWN parameters. ML and HH reuse the sweeps
`pipeline/sweeps.py` already defines (ML V3 2 -> 14, which carries Class 2 to Class 1; HH
gK 24 -> 40, capped at 40 because the model enters depolarization block above it). AdEx has
no such sweep -- the pipeline deliberately treats it as four named regimes, because those
move V_reset and the spike-triggered w jump together -- so a continuous axis is defined
here: the w jump at fixed V_reset = -65, with the named regimes marked on it.

A target's features are measured once, with the TARGET's own integration step, initial
condition and spike threshold, and cached. Nothing about a target depends on the
optimizer, so no run ever re-measures one.

    $PY targets.py                 # extract and cache every target, print a table
    $PY targets.py --only hh
"""
import os
import argparse
import numpy as np

import common as K
import v3_config as C


# --------------------------------------------------------------------- study 1
def study1():
    """[(slug, label, kind, payload, rheo_hi)] -- the zoo configs, unmodified."""
    from v3_an_zoo import targets as zoo, reset_targets
    want = {"Hodgkin-Huxley": "hh", "Morris-Lecar C1": "ml_c1"}
    out = []
    for name, spec in zoo():
        if name in want:
            out.append((want[name], name, "continuous", spec, None))
    for lbl, rhs, pr, rh in reset_targets():
        if lbl == "AdEx tonic":
            out.append(("adex_tonic", lbl, "reset", (rhs, pr), rh))
    order = {"hh": 0, "ml_c1": 1, "adex_tonic": 2}
    out.sort(key=lambda r: order[r[0]])
    if len(out) != 3:
        raise SystemExit(f"expected 3 study-1 targets, got {[r[0] for r in out]}")
    return out


# --------------------------------------------------------------------- study 2
# AdEx's continuous axis. params = (unused, 0, V_reset, b): index 1 must stay 0 so the
# pipeline's initial condition w(0) = params[1]*V_REST is AdEx's correct w(0) = 0.
ADEX_VRESET = -65.0
ADEX_B = np.linspace(0.0, 120.0, 10)
ADEX_NAMED = {0.0: "tonic", 60.0: "adapting"}       # the named regimes on this axis


def study2():
    """[(family, pname, [(slug, label, x, kind, payload, rheo_hi)])]."""
    from sweeps import resetfree_sweeps
    from v3_an_features import V3Spec
    from models_offamily import adex_rhs
    out = []
    for sw in resetfree_sweeps():
        if sw["family"] not in ("Morris-Lecar", "Hodgkin-Huxley"):
            continue
        cfgs = sw["configs"]
        Pm = np.array([np.asarray(c[1], float) for c in cfgs])
        vi = int(np.argmax(Pm.std(axis=0)))                    # the swept coordinate
        pname = cfgs[0][0].split("=")[0].split()[-1]
        fam = "ml" if sw["family"] == "Morris-Lecar" else "hh"
        items = []
        for lbl, pr in cfgs:
            sp = V3Spec(rhs=sw["rhs"], y0=np.asarray(sw["y0"], float),
                        params=np.asarray(pr, float), v_th=sw["v_th"],
                        rheo_lo=0.0, rheo_hi=sw["rheo_hi"], h=sw["h"],
                        boot_t_trans=sw["t_discard"], boot_t_win=sw["t_win"],
                        boot_t_disc=sw["t_discard"], name=lbl)
            items.append((f"{fam}_sw_{float(pr[vi]):g}".replace(".", "p"),
                          lbl, float(pr[vi]), "continuous", sp, None))
        out.append((sw["family"], pname, items))
    items = []
    for b in ADEX_B:
        pr = np.array([0.0, 0.0, ADEX_VRESET, float(b)])
        nm = ADEX_NAMED.get(float(b))
        lbl = f"AdEx b={b:.0f}" + (f" ({nm})" if nm else "")
        items.append((f"adex_sw_{b:g}".replace(".", "p"), lbl, float(b),
                      "reset", (adex_rhs, pr), 2000.0))
    out.append(("AdEx", "b", items))
    return out


# ------------------------------------------------------------------ extraction
def extract(kind, payload, rheo_hi=None):
    """(features(18,), info) measured on the target itself."""
    if kind == "continuous":
        from v3_an_features import compute_features_an
        f, it = compute_features_an(payload)
    else:
        from v3_core import alloc_buffers
        from v3_features import compute_features_v3
        rhs, pr = payload
        buf = np.full(C.N_COLS, C.NAN_SENTINEL)
        f, it = compute_features_v3(rhs, np.ascontiguousarray(pr), alloc_buffers(),
                                    out=buf, rheo_hi=rheo_hi)
    return np.asarray(f[:C.N_ACTIVE], float).copy(), it


def cached(slug, label, kind, payload, rheo_hi=None, force=False):
    """The target's frozen feature vector, measured once and reused forever."""
    path = os.path.join(K.TARGETS, f"{slug}.json")
    if not force:
        d = K.jread(path)
        if d is not None:
            return d
    import time
    t0 = time.time()
    f, it = extract(kind, payload, rheo_hi)
    d = dict(slug=slug, label=label, kind=kind,
             rheo_hi=None if rheo_hi is None else float(rheo_hi),
             f_t=f.tolist(),
             mask=[bool(x) for x in (f > C.SENT_CUT)],
             n_defined=int((f > C.SENT_CUT).sum()),
             i_rh=float(it["i_rh"]), T_cycle=float(it["T_cycle"]),
             L_inf=float(it.get("L_inf", C.NAN_SENTINEL)),
             p_hat=float(it.get("p_hat", C.NAN_SENTINEL)),
             gated=int(it.get("gated", 0)), reason=str(it.get("reason", "")),
             boot_t_win=float(it.get("boot_t_win", np.nan)),
             boot_cycles=float(it.get("boot_cycles", np.nan)),
             boot_retries=int(it.get("boot_retries", 0)),
             seconds=round(time.time() - t0, 2))
    if kind == "continuous":
        sp = payload
        d["spec"] = dict(v_th=float(sp.v_th), h=float(sp.h),
                         rheo_hi=float(sp.rheo_hi), y0=np.asarray(sp.y0).tolist(),
                         params=np.asarray(sp.params).tolist(),
                         boot_t_win=float(sp.boot_t_win),
                         boot_t_trans=float(sp.boot_t_trans))
    else:
        d["spec"] = dict(v_th=float(C.V_TH), h=float(C.DT),
                         params=np.asarray(payload[1]).tolist())
    K.jwrite(path, d)
    return d


def target_of(d, sigma):
    """A cached dict -> objective.Target."""
    import objective as OB
    return OB.Target(d["slug"], np.asarray(d["f_t"], float), sigma,
                     meta=dict(i_rh=d["i_rh"], T_cycle=d["T_cycle"],
                               label=d["label"], kind=d["kind"],
                               rheo_hi=d["rheo_hi"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="slug substring")
    ap.add_argument("--which", default="both", choices=("study1", "study2", "both"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    K.ensure_dirs()

    work = []
    if a.which in ("study1", "both"):
        work += [("study1", *r) for r in study1()]
    if a.which in ("study2", "both"):
        for fam, pname, items in study2():
            for slug, lbl, x, kind, payload, rh in items:
                work.append((f"study2:{fam}", slug, lbl, kind, payload, rh))
    if a.only:
        work = [w for w in work if a.only in w[1]]

    print(f"{'group':14s} {'slug':16s} {'label':24s} {'n_def':>5s} {'i_rh':>10s} "
          f"{'T_cycle':>10s} {'boot_cyc':>8s} {'s':>6s}")
    for grp, slug, lbl, kind, payload, rh in work:
        d = cached(slug, lbl, kind, payload, rh, force=a.force)
        bad = " GATED" if d["gated"] else ""
        print(f"{grp:14s} {slug:16s} {lbl[:24]:24s} {d['n_defined']:5d} "
              f"{d['i_rh']:10.4g} {d['T_cycle']:10.4g} {d['boot_cycles']:8.2f} "
              f"{d['seconds']:6.1f}{bad}", flush=True)


if __name__ == "__main__":
    main()
