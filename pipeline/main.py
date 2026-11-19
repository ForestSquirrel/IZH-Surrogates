"""
main.py — entry point. Dispatches train / infer off the one config object.

Config lives in cinn_config.py (no YAML/JSON config loading, per project style).
This thin argparse layer only overrides the handful of RUNTIME knobs; edit
cinn_config.py to change architecture or data-shaping defaults.

    python main.py                              # train with cinn_config defaults
    python main.py --mode infer --infer-input row.bin
    python main.py --mode train --resume --epochs 400

Flags
-----
Run selection
    --mode train|infer      what to do (default: train)
    --data-dir DIR          atlas dir: izh_features.bin / izh_params.bin / izh_meta.json
    --model-dir DIR         checkpoint dir: model_params.bin / stats.json / config.json
    --device cuda|cpu       falls back to cpu automatically if CUDA is missing
    --resume                continue from the checkpoint in --model-dir

Training
    --epochs N              passes over the atlas. Only bounds the run when
                            --max-steps is 0; otherwise the step budget wins.
    --batch-size N          rows per optimizer step
    --lr F, --seed N
    --max-steps N           stop after N optimizer steps, mid-epoch if need be
                            (0 = run --epochs). The knob to reach for on a large
                            atlas — see "Sizing a run" below.
    --grad-clip F           global grad-norm clip (0 disables)
    --cond-jitter F         train-time Gaussian noise on the conditioning value
                            channels, in atlas sigma (0 disables). Widens the
                            flow's robustness margin around the atlas manifold
                            (see SlabLoader's docstring in loader.py)
    --early-stop-patience N stop after N *evaluations* with no val_nll
                            improvement (0 = off). Rarely needed now that the LR
                            schedule defines the end of the run.
    --ckpt-every-steps N    mid-epoch checkpoint cadence (0 = epoch end only).
                            Worth setting on multi-hour runs.

LR schedule (linear warmup, then cosine decay; applied per optimizer step)
    --warmup-steps N        warmup length, capped at 10% of the step budget
    --lr-final-frac F       cosine floor, as a fraction of --lr

Validation / checkpoint selection
    --val-frac F            held-out tail fraction. 250k rows is plenty for a
                            4-D density; the old 0.05 was ~100x that at 1e8.
    --eval-every-steps N    evaluate + best-checkpoint every N steps
                            (0 = epoch end only)
    --ema / --no-ema        track an EMA of the weights and evaluate it beside
                            the raw ones each time; whichever is better is what
                            best_params.bin gets
    --ema-decay F
    --compile / --no-compile
                            torch.compile the nll step

Checkpoints
    best_params.bin  written only when validation improves — this is what
                     inference loads by default. May hold EMA weights.
    model_params.bin the last training state, with optimizer + EMA + schedule;
                     this is what --resume reads.

Streaming (rarely need changing)
    --slab-rows N           rows per streamed slab; the upload + shuffle unit.
                            Clamped up to --batch-size if smaller.
    --stat-samples N        rows subsampled to fit the median/MAD constants. The
                            mean/std fallback is always fit exactly, so this only
                            affects the quantile-based features.

Inference
    --infer-input PATH      single-row features .bin for the inverse pass
    --n-posterior-samples N draws from the posterior (default 256)

Sizing a run
------------
The dataset is streamed (data.open_atlas + loader.SlabLoader), so **memory does
not scale with the atlas** — ~2.4 GB host and ~6.5 GB VRAM whether it has 1e6
rows or 1e9. Only wall time moves.

The useful budget is **optimizer steps, not epochs**. The original 1e6-row run
was 200 epochs at batch 8192 ≈ 23k steps. A bigger atlas does not need more
steps — it makes each step see fresher data. So `--epochs 200` on a 1e8 atlas
would be ~100x the training it was ever tuned for, which is why --max-steps
exists.

One caveat on "25k steps is enough", which the numbers below used to assume: the
400k-step 1e8 run ended with median train loss (-13.54) WORSE than val_nll
(-13.90), i.e. a negative generalization gap after 34 passes over 95M rows. The
model was capacity- and optimization-limited, not data-limited, so a long
annealed run does keep paying. Set --max-steps from the wall time available, and
let the cosine schedule land on it.

Consequence: wall time is roughly FLAT at ~1-2 h across atlas sizes.

    ~1e6-2e6 rows   the tuned baseline; small batch to get enough steps
        python main.py --batch-size 8192 --epochs 200
        # ~19 s/epoch, ~46k steps, ~1 h

    ~1e7 rows
        python main.py --batch-size 65536 --max-steps 25000 \
                       --early-stop-patience 5 --ckpt-every-steps 2000
        # ~24 s/epoch, ~170 epochs, ~70 min

    ~1e8 rows   (measured on 1e8_noinh_data, RTX 5070 Ti)
        python main.py --data-dir ../1e8_noinh_data --model-dir ../cinn_model_1e8_v2 \
                       --batch-size 32768 --max-steps 600000 \
                       --warmup-steps 2000 --eval-every-steps 2000 \
                       --ckpt-every-steps 20000 --val-frac 0.0025 --grad-clip 1.0
        # 57 ms/step, 17.6 steps/s -> ~9.5 h for the full budget
        # one epoch alone already reaches val_nll ~ -6.1
        #
        # Batch 32768 rather than 65536: with the branch-free spline and
        # torch.compile the step is no longer launch-bound the way the note
        # below describes, and 32768 gives MORE optimizer steps per second than
        # 8192 did before (17.6 vs 11.1) while also seeing 4x the rows per step.

    ~1e9 rows   (extrapolated, not yet run)
        python main.py --data-dir ../1e9_data \
                       --batch-size 65536 --max-steps 25000 \
                       --slab-rows 4000000 --ckpt-every-steps 1000
        # ~40 min/epoch, under 2 epochs; still compute-bound, since the ~137 s of
        # per-epoch disk reads hide behind the loader's prefetch thread

Throughput notes: the step is launch-bound, so a larger --batch-size buys
throughput until activations (~82 KB/row) stop paying for themselves — measured
0.40 M rows/s at 65536 (6.5 GB VRAM) vs 0.47 at 131072 (~11 GB). Set
--early-stop-patience on any long run; val_nll on a large atlas tends to flatten
well before the step budget is spent.
"""

import argparse
import dataclasses

from cinn_config import CFG
from train import train
from infer import infer


def _parse_overrides(cfg):
    p = argparse.ArgumentParser(description="Izhikevich cINN surrogate")
    p.add_argument("--mode", choices=["train", "infer"], default=cfg.mode)
    p.add_argument("--data-dir", default=cfg.data_dir)
    p.add_argument("--model-dir", default=cfg.model_dir)
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--resume", action="store_true", default=cfg.resume)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--max-steps", type=int, default=cfg.max_steps,
                   help="stop after N optimizer steps (0 = run --epochs). Prefer "
                        "this to --epochs on a large atlas: 200 epochs at 1e8 rows "
                        "is ~100x the step budget the 1e6 run was tuned at")
    p.add_argument("--slab-rows", type=int, default=cfg.slab_rows,
                   help="rows per streamed slab (upload + shuffle unit)")
    p.add_argument("--stat-samples", type=int, default=cfg.stat_samples,
                   help="rows subsampled to fit the median/MAD constants")
    p.add_argument("--ckpt-every-steps", type=int, default=cfg.ckpt_every_steps,
                   help="mid-epoch checkpoint cadence (0 = epoch end only)")
    p.add_argument("--early-stop-patience", type=int, default=cfg.early_stop_patience,
                   help="stop after N evaluations without val_nll improvement (0 = off)")
    p.add_argument("--eval-every-steps", type=int, default=cfg.eval_every_steps,
                   help="evaluate + best-checkpoint every N optimizer steps "
                        "(0 = epoch end only)")
    p.add_argument("--val-frac", type=float, default=cfg.val_frac,
                   help="fraction of the atlas held out as the validation tail")
    p.add_argument("--grad-clip", type=float, default=cfg.grad_clip,
                   help="global grad-norm clip (0 disables)")
    p.add_argument("--keep-snapshots", dest="keep_snapshots",
                   action="store_true", default=cfg.keep_snapshots,
                   help="keep a step-tagged snap_<step>.bin at every periodic "
                        "checkpoint instead of overwriting model_params.bin only")
    p.add_argument("--cond-jitter", type=float, default=cfg.cond_jitter,
                   help="stddev (in atlas sigma) of Gaussian noise added to the "
                        "conditioning value channels during training; 0 disables")
    p.add_argument("--warmup-steps", type=int, default=cfg.warmup_steps,
                   help="linear LR warmup length, in optimizer steps")
    p.add_argument("--lr-final-frac", type=float, default=cfg.lr_final_frac,
                   help="cosine LR floor as a fraction of --lr")
    p.add_argument("--ema", dest="use_ema", default=cfg.use_ema,
                   action=argparse.BooleanOptionalAction,
                   help="track an EMA of the weights and evaluate it alongside raw")
    p.add_argument("--ema-decay", type=float, default=cfg.ema_decay)
    p.add_argument("--compile", default=cfg.compile,
                   action=argparse.BooleanOptionalAction,
                   help="torch.compile the nll step")
    p.add_argument("--infer-input", default=cfg.infer_input)
    p.add_argument("--n-posterior-samples", type=int, default=cfg.n_posterior_samples)
    args = p.parse_args()

    # Every flag's argparse dest is spelled exactly like its PipelineConfig field,
    # so the override is a filtered splat rather than a hand-maintained kwarg
    # list. The old form duplicated each knob in both places, and a flag missing
    # from the second list parsed fine while silently doing nothing.
    valid = {f.name for f in dataclasses.fields(cfg)}
    unknown = set(vars(args)) - valid
    if unknown:
        raise AssertionError(f"argparse dests with no PipelineConfig field: "
                             f"{sorted(unknown)}")
    return dataclasses.replace(cfg, **vars(args))


def main():
    cfg = _parse_overrides(CFG)
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "infer":
        infer(cfg)
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")


if __name__ == "__main__":
    main()
