"""
cinn_config.py — Single source of all constants for the Izhikevich cINN
surrogate (training + inference).

Mirrors the philosophy of src/config.py: every tunable lives here as a plain
Python value, there is NO YAML/JSON config loading. Scripts import ``CFG`` (a
frozen-ish ``PipelineConfig`` instance) and read fields off it. A checkpoint
still serializes the *effective* config into ``model_dir/config.json`` so a
trained run records its own architecture — but that file is written BY the
pipeline, never read as the primary config source at launch time.

``main.py`` exposes a thin argparse layer that overrides only the runtime knobs
(mode / resume / infer_input / device / epochs / batch and the streaming and
step-budget flags); the architecture and data-shaping defaults are edited here.
See main.py's docstring for the flag list and per-atlas-size launch recipes.
"""

import os
from dataclasses import dataclass, asdict, field

# Repo root = parent of this src_cinn/ directory. The dataset binaries
# (izh_features.bin / izh_params.bin / izh_meta.json) live at the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- preprocessing sentinels (must match the feature generator) ----
# NAN_SENTINEL = -9999.0 is written by the feature pipeline for gated/undefined
# features. SENTINEL_CUT is the decision threshold: anything below it is treated
# as "undefined" (mask = 0). A generous cut (-9000) tolerates any float32 rounding
# of the sentinel while staying far below every real standardized feature value.
NAN_SENTINEL = -9999.0
SENTINEL_CUT = -9000.0

# ---- inhibition-regime features (removed when the atlas was built --noinh) ----
# Must match src/config.INHIBITION_FEATURES. When izh_meta.json marks "noinh":
# true these columns are all-sentinel; data.AtlasView drops them by NAME so the
# cINN trains on the reduced feature set (cond_dim shrinks accordingly).
INHIBITION_FEATURES = ("nu_inh", "p_inh", "rho_inh")


@dataclass
class PipelineConfig:
    # --- data ---
    data_dir: str = _REPO_ROOT                       # holds izh_features.bin / izh_params.bin / izh_meta.json
    model_dir: str = os.path.join(_REPO_ROOT, "cinn_model")  # weights + stats.json + config.json

    # --- run mode ---
    mode: str = "train"      # "train" | "infer"
    resume: bool = False     # if True and model_dir has a checkpoint, continue training

    # --- training ---
    epochs: int = 200
    batch_size: int = 65536
    lr: float = 1e-3
    weight_decay: float = 1e-5
    # Global grad-norm clip (0 disables); flows spike early without it. Was 10.0,
    # which is loose for a flow whose log-det is heavy-tailed: in the 400k-step
    # 1e8 run, 290 of the last 5000 logged steps sat above -12 against a p50 of
    # -13.5, and those outliers were taking full-size optimizer steps.
    grad_clip: float = 1.0
    # 0.05 of a 1e8 atlas is 5M validation rows -- ~100x oversampled for a 4-D
    # density, and enough to make per-step evaluation unaffordable. 0.0025 keeps
    # 250k rows, which is still far more than the estimator needs.
    val_frac: float = 0.0025
    seed: int = 0
    log_every: int = 50      # batches between stdout progress lines
    device: str = "cuda"     # falls back to "cpu" if unavailable
    # Gaussian noise (in atlas sigma) added to the standardized VALUE channels of
    # the conditioning vector during training only; masked slots stay at 0.
    # 0 reproduces the pre-jitter behaviour exactly. See SlabLoader's docstring:
    # without it the flow has a 0.05-0.2 sigma
    # robustness margin around its own training manifold, which is the mechanism
    # behind the posterior collapse on out-of-family targets.
    cond_jitter: float = 0.0
    # Retain a step-tagged copy (snap_<step>.bin) at every ckpt_every_steps
    # write, instead of only the overwritten model_params.bin. Turns one run into
    # a training-budget trajectory, which §18 showed matters: the budget effect
    # on out-of-family collapse is family-dependent IN SIGN (15k->60k helps FHN
    # and HH, hurts HR).
    keep_snapshots: bool = False

    # --- LR schedule (linear warmup -> cosine decay), applied per optimizer step ---
    # The 1e8 run held lr=1e-3 flat for all 400k steps and never annealed; its
    # val_nll plateaued around epoch 21 and then swung +/-3 nats between epochs
    # on a FIXED validation set, i.e. the model itself was thrashing.
    warmup_steps: int = 2000
    lr_final_frac: float = 0.01   # cosine floor, as a fraction of lr

    # --- weight EMA ---
    # Direct antidote to the same thrashing: the shipped weights become an
    # average over ~1/(1-decay) steps rather than whichever step training
    # happened to stop on. Evaluated alongside the raw weights every eval.
    use_ema: bool = True
    ema_decay: float = 0.9999     # ~10k-step time constant

    # --- evaluation cadence ---
    # 0 = evaluate at epoch end only (the old behavior). An epoch of the 1e8
    # atlas is ~2900 steps at batch 32768, so epoch-end evaluation gives a
    # ~35-point curve over a full run -- too coarse to drive best-checkpointing
    # or early stopping.
    eval_every_steps: int = 2000

    # --- compilation ---
    # torch.compile on the nll step. Measured 141 -> 57 ms/step at batch 32768
    # on the RTX 5070 Ti, together with the branch-free spline rewrite.
    compile: bool = True

    # --- streaming / large-atlas knobs ---
    # The dataset is never resident: data.open_atlas memmaps it and loader.SlabLoader
    # streams contiguous slabs, so host RAM and VRAM both stay flat in N.
    #
    # The step is launch-bound (10 coupling blocks x ActNorm/SoftPermute/coupling
    # = ~30 modules of tiny kernels), so throughput rises with batch size until
    # activations (~82 KB/row) stop paying for themselves.
    #
    # Measured end-to-end, one epoch of the 1e8 atlas on an RTX 5070 Ti:
    #     B= 65536   237 s/epoch   0.40 M rows/s   6.5 GB VRAM   <- default
    #     B=131072   203 s/epoch   0.47 M rows/s  ~11 GB VRAM
    # 131072 is ~15% faster but more than doubles the VRAM; 65536 is the safer
    # default and leaves room for a wider architecture. On a SMALL atlas prefer a
    # small batch instead (--batch-size 8192): 200 epochs of 2e6 rows at 65536 is
    # only ~5.8k optimizer steps, well short of what the model needs.
    # A synthetic fixed-tensor benchmark ranks these the other way round; trust
    # the end-to-end numbers, since the real loop also pays per-batch Python and
    # gather overhead that favours fewer, larger batches.
    slab_rows: int = 1_000_000     # rows per streamed slab (upload + shuffle unit)
    stat_samples: int = 5_000_000  # rows subsampled to fit the median/MAD constants
    max_steps: int = 0             # stop after N optimizer steps (0 = use epochs)
    ckpt_every_steps: int = 0      # mid-epoch checkpoint cadence (0 = epoch end only)
    early_stop_patience: int = 0   # epochs without val_nll improvement before stopping (0 = off)

    # --- architecture ---
    #
    # Sized from the 1e8 run, which showed a NEGATIVE generalization gap: median
    # train loss -13.54 against val_nll -13.90 after 34 passes over 95M rows with
    # 1.76M parameters. The model could not overfit 95M rows, so it was capacity-
    # and optimization-limited, not data-limited -- nothing should be spent on
    # regularization, everything on capacity.
    #
    # Measured ladder (torch.compile + branch-free spline, batch 32768, 5070 Ti):
    #     10blk h256 se2 b8      68 ms   14.7 st/s   1.76M   <- old
    #     + bins16               70 ms   14.4 st/s   1.83M
    #     + hidden512            98 ms   10.2 st/s   5.29M
    #     + h512 all-spline b16  57 ms   17.6 st/s   3.98M   <- this
    #     + 14 blocks           147 ms    6.8 st/s   5.50M
    # All-spline is what makes the branch-free spline pay off, which is why the
    # widest useful config is also the fastest here.
    embed_hidden: int = 128       # SwiGLU embedder width / conditioning dim H
    embed_layers: int = 4         # number of residual SwiGLU blocks
    n_coupling_blocks: int = 10
    coupling_hidden: int = 512    # width of each coupling block's internal MLP
    spline_every: int = 1         # every k-th block uses spline coupling instead of affine
                                  #   (0 or >= n_coupling_blocks => all-affine baseline; 1 => all spline)
    spline_bins: int = 16         # bins for rational-quadratic spline coupling
    permute_soft: bool = True     # keep True: PermuteRandom adds an int64 buffer,
                                  # which the EMA shadow would have to special-case

    # --- inference ---
    infer_input: str = ""         # path to a single-row features .bin for the inverse pass
    n_posterior_samples: int = 256

    def to_dict(self):
        return asdict(self)


# The one config instance the whole pipeline reads.
CFG = PipelineConfig()
