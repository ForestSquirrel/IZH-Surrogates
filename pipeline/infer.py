"""
infer.py — inverse pass: features -> posterior over theta = (a, b, c, d).

Loads model_params.bin + stats.json + config.json from model_dir, reads a
single-row features binary (same on-disk format as izh_features.bin, n=1),
standardizes it with the FROZEN stats, then draws n_posterior_samples latents and
inverts once per draw. Multimodal / collision-prone inputs come back as visibly
spread samples rather than a single averaged point (cinn.md §6).
"""

import torch

from data import read_bin, build_condition, unstandardize_params, select_features
from checkpoint import load_checkpoint


def _resolve_device(want):
    if want.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return want


def infer(cfg):
    device = _resolve_device(cfg.device)
    model, stats, arch_cfg, _ = load_checkpoint(cfg.model_dir, device=device)

    fnames = stats["feature_names"]                                 # active set
    all_fnames = stats.get("all_feature_names", fnames)             # on-disk schema
    if not cfg.infer_input:
        raise ValueError("cfg.infer_input must point to a single-row features .bin")

    feats = read_bin(cfg.infer_input, b"IZHF")                      # (1, F_in) float32
    # Accept either a reduced (active) row or a full-schema row: a noinh model
    # takes a full 22-col vector and drops the inhibition columns by name here.
    if feats.shape[1] == len(all_fnames) and len(all_fnames) != len(fnames):
        feats = select_features(feats, all_fnames, fnames)
    elif feats.shape[1] != len(fnames):
        raise ValueError(f"{cfg.infer_input}: {feats.shape[1]} feature cols match "
                         f"neither the active ({len(fnames)}) nor the full "
                         f"({len(all_fnames)}) schema in stats.json")
    if feats.shape[0] != 1:
        print(f"[infer] note: {feats.shape[0]} rows present; using row 0 only")

    cond = build_condition(feats[:1], stats["feature_stats"])       # (1, 2F)
    cond_t = torch.from_numpy(cond).float().to(device)

    theta_std = model.sample(cond_t, cfg.n_posterior_samples)       # (n, 4)
    theta = unstandardize_params(theta_std.cpu().numpy(), stats["param_stats"])

    pnames = stats["param_names"]
    print(f"[infer] {cfg.n_posterior_samples} posterior samples for {cfg.infer_input}")
    means = theta.mean(0)
    stds = theta.std(0)
    print("[infer] mean  " + " ".join(f"{pnames[j]}={means[j]:.4f}" for j in range(len(pnames))))
    print("[infer] std   " + " ".join(f"{pnames[j]}={stds[j]:.4f}" for j in range(len(pnames))))
    for row in theta:
        print(" ".join(f"{v:.5f}" for v in row))

    return theta
