"""
train_flow.py -- train the conditional flow on an atlas.

Thin wrapper over pipeline/main.py that (a) applies the shipped defaults, and
(b) refuses to overwrite an existing checkpoint.

    python scripts/train_flow.py --data-dir data/atlas_v3_1e7 \
                                 --model-dir /scratch/my_flow

DESTRUCTIVE-WRITE WARNING
    Training writes model_params.bin, best_params.bin, config.json and
    stats.json into --model-dir, and invalidates any stale best_params.bin it
    finds there first. Pointing it at a checkpoint you want to keep destroys
    that checkpoint. The guard below requires --force (overwrite) or --resume
    (continue) to touch a directory that already holds weights.

Shipped defaults reproduce models/cinn_v3_p015: cond-jitter 0.15, 60k steps,
batch 32768, EMA 0.9999, cosine schedule after 2000 warmup steps. The jitter is
not optional garnish -- see README.md, it is what makes out-of-family
conditioning vectors land somewhere sane.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))   # pipeline_final/scripts
ROOT = os.path.dirname(HERE)                       # pipeline_final
PKG  = os.path.join(ROOT, "pipeline")
sys.path.insert(0, PKG)

DEFAULTS = ["--mode", "train", "--cond-jitter", "0.15"]


def _arg(argv, name, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def main():
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        # let the underlying parser print its own help; the guard below would
        # otherwise reject --help for lack of --model-dir
        sys.argv = [sys.argv[0]] + DEFAULTS + argv
        import main as flow_main
        return flow_main.main()
    force = "--force" in argv
    if force:
        argv.remove("--force")

    model_dir = _arg(argv, "--model-dir")
    if model_dir is None:
        raise SystemExit(
            "--model-dir is required.\n"
            "Do NOT point it at models/cinn_v3_p015 unless you intend to "
            "replace the shipped checkpoint; use a fresh directory.")

    existing = [n for n in ("model_params.bin", "best_params.bin")
                if os.path.exists(os.path.join(model_dir, n))]
    if existing and not force and "--resume" not in argv:
        raise SystemExit(
            f"refusing to train into {model_dir!r}: it already contains "
            f"{', '.join(existing)}.\n"
            f"Training would invalidate the stale best_params.bin and then "
            f"overwrite the weights.\n"
            f"Use a NEW --model-dir, or pass --resume to continue that run, "
            f"or --force to overwrite it.")

    data_dir = _arg(argv, "--data-dir")
    if data_dir is None:
        argv += ["--data-dir", os.path.join(ROOT, "data", "atlas_v3_1e7")]

    sys.argv = [sys.argv[0]] + DEFAULTS + argv
    import main as flow_main
    flow_main.main()


if __name__ == "__main__":
    main()
