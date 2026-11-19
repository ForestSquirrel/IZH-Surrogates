"""
checkpoint.py — self-contained checkpoint I/O for a trained cINN run.

A checkpoint is the directory `model_dir` holding four files (cinn.md §5):
    model_params.bin  the LAST training state: {model, optim, epoch, val_nll, ...}
    best_params.bin   the BEST-validating weights seen, written only on improvement
    stats.json        frozen robust-zscore constants + feature/param names
    config.json       the exact PipelineConfig used (records the architecture)

Loading rebuilds the architecture from config.json, so inference never needs the
original data_dir.

Why two weight files
--------------------
There used to be one. `save_checkpoint` overwrote it unconditionally at every
epoch end and every `ckpt_every_steps`, so a run shipped whichever step it
happened to stop on rather than its best. In the 400k-step 1e8 run that cost the
epoch-31 model (val_nll -14.13) in favour of epoch 34 (-13.90) -- and with the
+/-3 nat epoch-to-epoch swing that run had, which one you got was luck.

`best_params.bin` is written only when validation improves, and may hold EMA
rather than raw weights (see the "weights" key). It is for INFERENCE only and is
never resumed from: it carries no coherent optimizer state, and resuming from it
would silently rewind training. `load_checkpoint` therefore defaults to "best"
(what infer.py and run_surrogate.py want) while train.py asks for "last".
"""

import os
import json
import torch

from dataclasses import fields

from cinn_config import PipelineConfig
from data import load_stats
from model import CINN

_WEIGHTS = "model_params.bin"
_BEST = "best_params.bin"
_CONFIG = "config.json"


def save_checkpoint(model_dir, model, optimizer, epoch, val_nll, cfg,
                    global_step=0, filename=_WEIGHTS, state_dict=None,
                    weights="raw", best_val=None, since_best=0, ema=None,
                    sched=None):
    """Write one checkpoint file plus config.json.

    `state_dict` overrides `model.state_dict()` (used to save EMA weights without
    disturbing the live model); `weights` records which of the two it is.

    The write goes to a temp file and is then renamed, so a crash mid-save leaves
    the previous checkpoint intact rather than a truncated one.
    """
    os.makedirs(model_dir, exist_ok=True)
    payload = {
        "model": model.state_dict() if state_dict is None else state_dict,
        "optim": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "global_step": global_step,   # read back with .get(): older
                                      # checkpoints predate this field
        "val_nll": val_nll,
        "cond_dim": model.cond_dim,
        "param_dim": model.param_dim,
        # --- resume state; all read back with .get() for old checkpoints ---
        "weights": weights,           # "raw" | "ema"
        "best_val": best_val,
        "since_best": since_best,
        "ema": ema,                   # EMA shadow state dict, or None
        "sched": sched,               # {"warmup":…, "total":…, "base_lr":…}
    }
    path = os.path.join(model_dir, filename)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)             # atomic on POSIX

    # config.json (written every save so it always matches the run)
    with open(os.path.join(model_dir, _CONFIG), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def has_checkpoint(model_dir):
    return os.path.exists(os.path.join(model_dir, _WEIGHTS))


def _weights_path(model_dir, which):
    """Resolve "best" | "last" to a file, falling back when best is absent.

    The fallback is what keeps model dirs written before best/last existed
    loadable, and covers a run killed before its first evaluation.
    """
    if which == "last":
        return os.path.join(model_dir, _WEIGHTS)
    if which != "best":
        raise ValueError(f"which must be 'best' or 'last', got {which!r}")
    best = os.path.join(model_dir, _BEST)
    return best if os.path.exists(best) else os.path.join(model_dir, _WEIGHTS)


def load_config(model_dir):
    """Reconstruct the PipelineConfig recorded at training time."""
    with open(os.path.join(model_dir, _CONFIG)) as f:
        raw = json.load(f)
    valid = {f.name for f in fields(PipelineConfig)}
    return PipelineConfig(**{k: v for k, v in raw.items() if k in valid})


def load_checkpoint(model_dir, device="cpu", which="best"):
    """
    Returns (model, stats, arch_cfg, ckpt). `arch_cfg` is the architecture
    recorded in config.json; the caller keeps its own launch-time config for
    runtime knobs (device, n_posterior_samples, ...).

    `which="best"` (the default, and what inference wants) prefers
    best_params.bin, falling back to model_params.bin when it is absent.
    Training must pass `which="last"`: best_params.bin holds no coherent
    optimizer state and may hold EMA weights.
    """
    arch_cfg = load_config(model_dir)
    stats = load_stats(model_dir)
    ckpt = torch.load(_weights_path(model_dir, which), map_location=device,
                      weights_only=False)

    model = CINN(arch_cfg, cond_dim=ckpt["cond_dim"], param_dim=ckpt["param_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, stats, arch_cfg, ckpt
