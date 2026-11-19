"""
run_surrogate.py -- the flow-facing API: draw a posterior, and record a trace.

Two functions lifted from the repo-root `run_surrogate.py` of the research tree.
That file is ~700 lines whose `main()` pulls in the whole v1/v2 feature stack
(`src/`, `src_any_neuron/`, optionally `src_cellwin/`) plus matplotlib. None of
it is reachable from these two: every first-party import in the original is lazy,
`cinn_infer` needs only `checkpoint` and `data`, and `simulate_trace` imports
nothing at all. Lifting them is what lets this package drop that stack entirely.

The MODULE NAME IS DELIBERATE. Callers say `import run_surrogate as rs` and
`from run_surrogate import simulate_trace`; keeping the name means those lines
are byte-identical to the research tree, so the copied runners needed no edit
here and cannot silently diverge.
"""
import numpy as np


def cinn_infer(model_dir, features_active, n_samples, temperature=1.0):
    """features_active (F_active,) -> (theta samples (n_samples, 4), stats).

    `features_active` must already be reduced to the model's active feature set
    and carry NAN_SENTINEL in undefined slots; `build_condition` zeroes those and
    flags them in the mask channel.

    `temperature` widens the latent draw. Leave at 1.0 for the true posterior;
    raise it only when the target is far out of the atlas and the draws are being
    used as a search proposal rather than as a density.

    THE DRAW IS SEEDED. `torch.manual_seed(0)` runs immediately before sampling,
    so repeating a call reproduces it bit-for-bit -- a re-run is NOT a replicate.
    Genuine replicates require shifting that seed explicitly.
    """
    import torch
    from checkpoint import load_checkpoint, has_checkpoint
    from data import build_condition, unstandardize_params
    if not has_checkpoint(model_dir):
        raise SystemExit(
            f"no cINN checkpoint in {model_dir}\n"
            f"train one first:  python scripts/train_flow.py --model-dir {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)   # reproducible posterior draw
    model, stats, _, _ = load_checkpoint(model_dir, device=device)
    cond = build_condition(features_active[None, :], stats["feature_stats"])
    cond_t = torch.from_numpy(cond).float().to(device)
    theta_std = model.sample(cond_t, n_samples, temperature=temperature)
    theta = unstandardize_params(theta_std.cpu().numpy(), stats["param_stats"])
    return theta, stats


def simulate_trace(rhs, params, y0, stim, v_th, h, t_end,
                   reset_v=None, reset_du=None):
    """RK4 trace recorder. `stim` is a callable stim(t) -> injected current I(t).

    Returns (t, v, I, spike_times). Applies a reset only if `reset_v` is given
    (Izhikevich and other reset models); reset-free otherwise.

    NOTE `vs[k] = y[0]`: the recorded trace is the FIRST state component, which
    is also the component the spike detector thresholds. For a model whose
    observable is not its first state variable, order the state so that it is.
    """
    params = np.asarray(params, dtype=np.float64)
    n_steps = int(t_end / h) + 1
    ts = np.empty(n_steps); vs = np.empty(n_steps); is_ = np.empty(n_steps)
    y = np.asarray(y0, dtype=np.float64).copy()
    nd = y.shape[0]
    d1 = np.empty(nd); d2 = np.empty(nd); d3 = np.empty(nd); d4 = np.empty(nd)
    spikes = []
    t = 0.0
    for k in range(n_steps):
        ts[k] = t; vs[k] = y[0]
        I = stim(t)   # current held across the RK4 substeps (fine for small h)
        is_[k] = I
        rhs(t, y, params, I, d1)
        rhs(t + 0.5 * h, y + 0.5 * h * d1, params, I, d2)
        rhs(t + 0.5 * h, y + 0.5 * h * d2, params, I, d3)
        rhs(t + h, y + h * d3, params, I, d4)
        v_prev = y[0]
        y = y + (h / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
        if v_prev < v_th and y[0] >= v_th:
            spikes.append(t)
            if reset_v is not None:
                y[0] = reset_v
                y[1] += reset_du
        t += h
    return ts, vs, is_, np.array(spikes)
