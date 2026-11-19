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

**Copy them** from wherever the working set lives. The layout must be exactly:

```
pipeline_final/
  data/atlas_v3_1e7/{izh_features.bin,izh_params.bin,izh_landmarks.bin,
                     izh_meta.json,sigma_v3.json,progress.json}
  models/cinn_v3_p015/{model_params.bin,best_params.bin,config.json,stats.json}
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
