"""
smoke.py -- self-test for the localized pipeline.

    python scripts/smoke.py --isolation     import graph only, no GPU, seconds
    python scripts/smoke.py --numeric       feature equivalence, ~2 min
    python scripts/smoke.py                 both

WHAT --isolation PROVES, AND WHY IT IS THE IMPORTANT ONE

The package is meant to have no dependency on the research tree it was extracted
from. Running a script and seeing it work does NOT establish that: executed from
the original repo, a missed import can silently resolve against the old tree and
everything passes. So this check imports every module with ONLY the package
directory on sys.path, then asserts that every module Python actually loaded
resolves to a file inside pipeline_final/. A bare "it ran" is not evidence.
"""
import os, sys, argparse, importlib, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG  = os.path.join(ROOT, "pipeline")

STDLIB_OK = ("numpy", "numba", "torch", "matplotlib", "scipy", "llvmlite")


def isolation():
    # only the package dir -- deliberately NOT the repo root or any sibling tree
    sys.path.insert(0, PKG)
    mods = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(PKG, "*.py")))
    print(f"[isolation] importing {len(mods)} modules with sys.path = [pipeline/]")
    failed, foreign = [], []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:                     # noqa: BLE001
            failed.append((m, f"{type(e).__name__}: {e}"))
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or name.startswith(STDLIB_OK) or name.split(".")[0] in sys.builtin_module_names:
            continue
        f = os.path.abspath(f)
        if f.startswith(ROOT + os.sep):
            continue
        # stdlib and site-packages are fine; anything in the parent repo is not
        if "site-packages" in f or f.startswith(sys.prefix) or f.startswith(sys.base_prefix):
            continue
        foreign.append((name, f))
    for m, e in failed:
        print(f"  IMPORT FAILED  {m}: {e}")
    for n, f in foreign:
        print(f"  FOREIGN MODULE {n} -> {f}")
    ok = not failed and not foreign
    print(f"[isolation] {'PASS' if ok else 'FAIL'} "
          f"({len(mods) - len(failed)}/{len(mods)} imported, {len(foreign)} foreign)")
    return ok


def numeric():
    """Feature vectors from this copy must be BIT-IDENTICAL to the research tree.

    Not 'close': the code is a copy, the step size is the same and the integrator
    is deterministic, so any difference at all means a copy or edit error.
    """
    sys.path.insert(0, PKG)
    import numpy as np
    import v3_config as C
    from v3_core import alloc_buffers
    from v3_features import compute_features_v3
    from v3_models import izh_rhs_v3
    from v3_presets import PRESETS

    bufs = alloc_buffers()
    out = np.full(C.N_COLS, C.NAN_SENTINEL)
    names = ["A tonic spiking", "C tonic bursting", "G Class 1",
             "H Class 2", "K resonator"]
    res = {}
    for nm in names:
        kw = PRESETS[nm]
        th = np.ascontiguousarray([kw["a"], kw["b"], kw["c"], kw["d"]], dtype=np.float64)
        f, info = compute_features_v3(izh_rhs_v3, th, bufs, out=out)
        v = np.asarray(f[:C.N_ACTIVE], float).copy()
        res[nm] = v
        nd = int((v > C.SENT_CUT).sum())
        print(f"  {nm:<20s} defined {nd:2d}/18  i_rh={info['i_rh']:.6f}  "
              f"T_cycle={info['T_cycle']:.6f}")
    ref = os.environ.get("SMOKE_REF")
    if ref:
        old = np.load(ref)
        bad = [nm for nm in names if not np.array_equal(res[nm], old[nm])]
        print(f"[numeric] vs reference {ref}: "
              f"{'PASS -- bit-identical' if not bad else 'FAIL ' + str(bad)}")
        return not bad
    dump = os.environ.get("SMOKE_DUMP")
    if dump:
        np.savez(dump, **res); print(f"[numeric] wrote reference {dump}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--isolation", action="store_true")
    ap.add_argument("--numeric", action="store_true")
    a = ap.parse_args()
    run_all = not (a.isolation or a.numeric)
    ok = True
    if a.isolation or run_all:
        ok &= isolation()
    if a.numeric or run_all:
        ok &= numeric()
    sys.exit(0 if ok else 1)
