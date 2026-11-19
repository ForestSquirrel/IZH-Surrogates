# Artifacts

The code in this repository is useless without two binary artifacts, and neither
is in git:

| artifact | path | size | what it is |
|---|---|---|---|
| atlas | `data/atlas_v3_1e7/` | 1.6 GB | 10,250,000 simulated Izhikevich cells: features, parameters, landmarks |
| trained flow | `models/cinn_v3_p015/` | 77 MB | the cINN checkpoint inference conditions on |

## Why they are not in git

`izh_features.bin` is 1.3 GB. GitHub rejects any file over 100 MB outright, and
Git LFS on a free plan provides 1 GB of storage — less than the atlas alone.
Committing them is not an option without a paid LFS data pack or external
hosting.

## Getting them

**Download them** from HuggingFace throw hf CLI:
```
# macOS & Linux (Standalone Installer)
curl -LsSf https://hf.co/cli/install.sh | bash

# or

# Windows powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Run these from the root. They put the files exactly where the pipeline looks for them:

```
# trained flow (77 MB)  -> models/cinn_v3_p015/
hf download ForestSquirrel/Izh_cINN --local-dir models/cinn_v3_p015

# training atlas (1.6 GB) -> data/atlas_v3_1e7/
hf download ForestSquirrel/IzhCells --repo-type dataset --local-dir data/atlas_v3_1e7

```

Verify after copying — a truncated atlas fails in confusing ways rather than
loudly:

```bash
sha256sum -c ARTIFACTS.sha256
```

**Or regenerate them.** Both are reproducible from this code:

```bash
python scripts/v3_build.py --out /scratch/atlas_new      # ~29 h on one GPU
python scripts/train_flow.py --data-dir /scratch/atlas_new \
                             --model-dir /scratch/flow_new   # ~1 h
```

The atlas build is seeded (`--seed 20260826`) and the training seed is fixed, but
neither is bit-reproducible across different GPUs or numba/torch versions.

## Do not overwrite them

`scripts/v3_build.py` and `scripts/train_flow.py` overwrite their output
directory, and a shipped atlas directory contains a `progress.json` that makes it
*live to a resume*. Both scripts refuse a non-empty target without `--force`.
See section 2 of README.md.
