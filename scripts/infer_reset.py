"""
infer_reset.py -- draw a posterior for a RESET target.

Use this endpoint when the target declares a spike at a fixed threshold and then
JUMPS its state -- Izhikevich itself, and AdEx.

    # an Izhikevich preset (the native control: tau = k_I = 1 by construction)
    python scripts/infer_reset.py --preset "A tonic spiking"
    python scripts/infer_reset.py --all-presets

    # a built-in reset target from the zoo
    python scripts/infer_reset.py --target "AdEx bursting"

    # your own: a module exposing (rhs, params) or (rhs, params, rheo_hi)
    python scripts/infer_reset.py --target mymodels:my_reset

THE THRESHOLD CONSTRAINT -- READ THIS BEFORE PORTING A MODEL

The reset integrator folds V_TH = 30 as a COMPILE-TIME constant. AdEx works here
only because its threshold really is 30 mV. A reset model with any other
threshold CANNOT use this endpoint: either rescale the model's voltage so its
threshold is 30, or express it as a reset-free system and use
infer_continuous.py.

The reset itself is applied by the integrator as v -> params[2], u -> u + params[3].
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", help="Izhikevich canonical preset, name substring")
    ap.add_argument("--all-presets", action="store_true")
    ap.add_argument("--target", help="zoo reset target, or module:attribute")
    ap.add_argument("--rheo-hi", type=float, default=None,
                    help="rheobase bracket ceiling; AdEx needs 2000 (pA units), "
                         "the Izhikevich-tuned 200 clips its bursting regimes")
    ap.add_argument("--n-posterior", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model-dir", default=os.path.join(MODELS, "cinn_v3_p015"))
    ap.add_argument("--atlas", default=os.path.join(DATA, "atlas_v3_1e7"))
    ap.add_argument("--sigma", default=None)
    ap.add_argument("--out", default=os.path.join(RESULTS, "infer_reset.json"))
    args = ap.parse_args()
    if not (args.preset or args.all_presets or args.target):
        ap.error("give --preset, --all-presets or --target")

    from v3_core import alloc_buffers
    from v3_models import izh_rhs_v3
    from v3_inference import load_sigma, infer, format_result

    work = []   # (name, rhs, params, rheo_hi)
    if args.preset or args.all_presets:
        from v3_presets import PRESETS
        for nm, kw in PRESETS.items():
            if args.all_presets or args.preset.lower() in nm.lower():
                work.append((nm, izh_rhs_v3,
                             np.ascontiguousarray([kw["a"], kw["b"], kw["c"], kw["d"]],
                                                  dtype=np.float64), args.rheo_hi))
        if not work:
            raise SystemExit(f"no preset matching {args.preset!r}")
    if args.target:
        if ":" in args.target:
            mod, _, attr = args.target.partition(":")
            import importlib
            obj = getattr(importlib.import_module(mod), attr)
            obj = obj() if callable(obj) else obj
            rhs, pr = obj[0], np.ascontiguousarray(obj[1], dtype=np.float64)
            rh = obj[2] if len(obj) > 2 else args.rheo_hi
            work.append((attr, rhs, pr, rh))
        else:
            from v3_an_zoo import reset_targets
            hits = [(n, r, np.ascontiguousarray(p, dtype=np.float64),
                     args.rheo_hi if args.rheo_hi is not None else rh)
                    for n, r, p, rh in reset_targets()
                    if args.target.lower() in n.lower()]
            if not hits:
                raise SystemExit(f"no reset target matching {args.target!r}")
            work += hits

    S = load_sigma(args.atlas, args.sigma)
    bufs = alloc_buffers()
    res = []
    for name, rhs, pr, rh in work:
        t0 = time.time()
        r = infer("reset", (rhs, pr), args.model_dir, S,
                  n_posterior=args.n_posterior, temperature=args.temperature,
                  rheo_hi=rh, bufs=bufs, name=name)
        print(format_result(r) + f"  [{time.time()-t0:.0f}s]", flush=True)
        res.append(r)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
