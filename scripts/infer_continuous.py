"""
infer_continuous.py -- draw a posterior for a RESET-FREE (continuous) target.

Use this endpoint when the target neuron has no spike-reset: its voltage is a
continuous solution of an ODE and a spike is a threshold crossing you declare.
FitzHugh-Nagumo, Morris-Lecar, Hodgkin-Huxley, Hindmarsh-Rose and Wilson all
belong here.

    # a built-in zoo target
    python scripts/infer_continuous.py --target "Hodgkin-Huxley"
    python scripts/infer_continuous.py --all

    # your own model: a module exposing a V3Spec (or a zero-arg factory)
    python scripts/infer_continuous.py --target mymodels:my_spec

WRITING YOUR OWN TARGET

    from v3_an_features import V3Spec
    my_spec = V3Spec(rhs=my_rhs, y0=np.array([...]), params=np.array([...]),
                     v_th=0.0, rheo_lo=0.0, rheo_hi=200.0, h=0.01,
                     boot_t_trans=50.0, boot_t_win=500.0, boot_t_disc=50.0,
                     name="my model")

`rhs` must be numba-njit with signature rhs(t, y, params, I, dydt). y[0] is BOTH
the variable the spike detector thresholds and the variable the plots draw, so
order the state to put your observable first. `rheo_hi` must bracket the
rheobase -- if it is too low the search reports "no rheobase" for a cell that
fires perfectly well.

If your target's rheobase comes out below I_RH_MIN = 1e-3, you will silently
lose eta, N_reb, g_reb and gamma_ramp. That is a UNITS problem, not a dynamics
problem -- see the I_RH_MIN section of README.md.
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # pipeline_final/scripts
ROOT = os.path.dirname(HERE)                       # pipeline_final
PKG     = os.path.join(ROOT, "pipeline")
DATA    = os.path.join(ROOT, "data")
MODELS  = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)


def resolve(target):
    """'Name' -> built-in zoo spec;  'module:attr' -> user spec or factory."""
    if ":" in target:
        mod, _, attr = target.partition(":")
        import importlib
        obj = getattr(importlib.import_module(mod), attr)
        return [(getattr(obj, "name", attr), obj() if callable(obj) else obj)]
    from v3_an_zoo import targets as zoo
    hits = [(n, s) for n, s in zoo() if target.lower() in n.lower()]
    if not hits:
        raise SystemExit(f"no zoo target matching {target!r}; have: "
                         + ", ".join(n for n, _ in zoo()))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="zoo name substring, or module:attribute")
    ap.add_argument("--all", action="store_true", help="every built-in zoo target")
    ap.add_argument("--n-posterior", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model-dir", default=os.path.join(MODELS, "cinn_v3_p015"))
    ap.add_argument("--atlas", default=os.path.join(DATA, "atlas_v3_1e7"))
    ap.add_argument("--sigma", default=None)
    ap.add_argument("--out", default=os.path.join(RESULTS, "infer_continuous.json"))
    args = ap.parse_args()
    if not (args.target or args.all):
        ap.error("give --target or --all")

    from v3_core import alloc_buffers
    from v3_inference import load_sigma, infer, format_result

    if args.all:
        from v3_an_zoo import targets as zoo
        work = list(zoo())
    else:
        work = resolve(args.target)

    S = load_sigma(args.atlas, args.sigma)
    bufs = alloc_buffers()
    res = []
    for name, spec in work:
        t0 = time.time()
        r = infer("continuous", spec, args.model_dir, S,
                  n_posterior=args.n_posterior, temperature=args.temperature,
                  bufs=bufs, name=name)
        print(format_result(r) + f"  [{time.time()-t0:.0f}s]", flush=True)
        res.append(r)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
