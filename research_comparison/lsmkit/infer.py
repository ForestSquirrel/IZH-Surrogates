"""
infer.py -- draw a target's Izhikevich surrogate with the pipeline and write posterior.json.

Wraps pipeline_final/pipeline/v3_inference.infer (the endpoint scripts/infer_*.py use).
The output adds what a network needs and the pipeline does not record: the voltage map
V_paper = va * v_model + vb, and the target's own description.

    posterior.json = {name, kind, i_rh_target, T_cycle_target, vmap{va, vb, note},
                      best{theta, tau, k_I, d2, n_dims, ...}, mean{...}, pmatch{...},
                      support, n_defined, regime, target{...}}
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # pipeline_final
PKG = os.path.join(ROOT, "pipeline")
MODEL_DIR = os.path.join(ROOT, "models", "cinn_v3_p015")
ATLAS = os.path.join(ROOT, "data", "atlas_v3_1e7")


def _pipeline():
    if PKG not in sys.path:
        sys.path.insert(0, PKG)


def run(kind, target, out_json, vmap, name, rheo_hi=None, n_posterior=256,
        temperature=1.0, target_info=None, force=False):
    """kind 'reset': target = (rhs, params); 'continuous': target = V3Spec.
    vmap: dict(va, vb, note) or a callable(result) -> dict, for maps that need the drawn
    theta (rest <-> rest, threshold <-> threshold). Refuses to overwrite out_json unless force."""
    if os.path.exists(out_json) and not force:
        raise SystemExit(f"{out_json} exists; pass force=True to redraw")
    _pipeline()
    from v3_core import alloc_buffers
    from v3_inference import load_sigma, infer, format_result
    S = load_sigma(ATLAS)
    t0 = time.time()
    if kind == "reset":
        rhs, pr = target
        r = infer("reset", (rhs, np.ascontiguousarray(pr, dtype=np.float64)), MODEL_DIR, S,
                  n_posterior=n_posterior, temperature=temperature, rheo_hi=rheo_hi,
                  bufs=alloc_buffers(), name=name)
    else:
        r = infer("continuous", target, MODEL_DIR, S, n_posterior=n_posterior,
                  temperature=temperature, name=name)
    print(format_result(r) + f"  [{time.time() - t0:.0f}s]", flush=True)
    if r.get("status") != "ok":
        raise SystemExit(f"inference failed: {r.get('status')}")
    r["i_rh_target"] = r["i_rh"]
    r["T_cycle_target"] = r["T_cycle"]
    r["vmap"] = vmap(r) if callable(vmap) else vmap
    r["target"] = target_info or {}
    r["model_dir"] = os.path.relpath(MODEL_DIR, ROOT)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(r, open(out_json, "w"), indent=1, default=float)
    print("wrote", out_json)
    return r
