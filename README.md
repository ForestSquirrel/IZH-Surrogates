# pipeline_final

Spike-train-equivalent Izhikevich surrogates for arbitrary neuron models.

Give it a neuron model of any family; it measures an 18-component dimensionless
feature vector on a fixed protocol suite, conditions a trained normalizing flow
(cINN) on that vector, and returns a posterior over Izhikevich `(a, b, c, d)`.
The two physical scales are recovered separately, as a **gauge**, not inferred:

```
φ  =  (τ, k_I)  ∘  (a, b, c, d)
      └ gauge ┘    └ inferred ┘
```

The package (`pipeline/`, `scripts/`, `examples/`) is **self-contained**: it
imports nothing outside this folder (`scripts/smoke.py --isolation` asserts
this).

## Installation

Python 3.12 (tested 3.12.3). From the repository root:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

| package | needed by | tested version |
|---|---|---|
| `numpy` | everything | 2.4.6 |
| `numba` | everything (all kernels are `@njit`; the atlas build uses `numba.cuda`) | 0.66.0 |
| `matplotlib` | plot scripts | 3.11.1 |
| `torch` | the flow: inference and training | 2.11.0+cu128 |
| `FrEIA==0.2` | the flow (`pipeline/model.py`) | 0.2 |
| `scipy` | `research_comparison/schulte2022`, `research_opti` | 1.18.0 |
| `pymoo`, `cma` | `research_opti` only | 0.6.2, 4.5.0 |

`FrEIA` is pinned: 0.2 has no rational-quadratic spline coupling, so
`pipeline/spline.py` provides it and the soft permutation, against the 0.2 API.

**GPU.** Inference, evaluation and plotting run on CPU; torch uses a GPU if one is
present. A CUDA GPU is **required** only to rebuild the atlas
(`scripts/v3_build.py`, `numba.cuda`) and is strongly advised for retraining the
flow.

- *torch on a Blackwell GPU* (RTX 50xx) needs the CUDA 12.8 wheel:
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- *numba.cuda* needs the NVIDIA driver and a CUDA toolkit with libNVVM. The tested
  environment used the pip packages `cuda-toolkit` 12.8.1 and `cuda-bindings` 12.9.4.

**Fonts (optional).** The figure scripts use Times New Roman when matplotlib can
find it; set `FONT_DIR` to a folder holding `times.ttf`, `timesbd.ttf`, `timesi.ttf`
and `timesbi.ttf` to load it explicitly. Otherwise they fall back to Nimbus Roman,
TeX Gyre Termes or STIX.

## Artifacts (not in git)

The code is useless without two binary artifacts, which are **not** in the
repository: the atlas `data/atlas_v3_1e7/` (1.6 GB) and the trained flow
`models/cinn_v3_p015/` (77 MB). **`ARTIFACTS.md`** says how to obtain, verify
(`sha256sum -c ARTIFACTS.sha256`) or regenerate them.

More generally the repository tracks **source only**: the Python code that
reproduces the experiments, the READMEs, and this file. Data, results JSON, plots,
PDFs, LaTeX, logs, datasets and caches are excluded by `.gitignore` (a whitelist),
so the studies in section 9 need their data inputs in place before they run.

## Quickstart

```bash
python scripts/smoke.py                       # self-test: imports + numerics

# infer for a built-in target
python scripts/infer_continuous.py --target "Hodgkin-Huxley"
python scripts/infer_reset.py --preset "A tonic spiking"

# figures
python scripts/v3_plots.py --which offfamily --only "Hodgkin"
```

Reset target (Izhikevich, AdEx: spike at threshold, then a state jump) →
`scripts/infer_reset.py`. Continuous target (HH, FHN, Morris–Lecar, HR, Wilson:
a spike is a threshold crossing) → `scripts/infer_continuous.py`. See
section 3.

> ⚠ `scripts/v3_build.py` and `scripts/train_flow.py` **overwrite their output
> directory**, and the shipped atlas is ~29 h of GPU time. Read section 2 before
> running either.

## Contents

- [1. Layout and entry points](#1-layout-and-entry-points)
- [2. ⚠ Destructive writes](#2--destructive-writes)
- [3. Choosing an endpoint: reset vs continuous](#3-choosing-an-endpoint-reset-vs-continuous)
- [4. Function reference](#4-function-reference)
- [5. Runner reference](#5-runner-reference)
- [6. Worked examples](#6-worked-examples)
- [7. The I_RH_MIN units workaround](#7-the-i_rh_min-units-workaround)
- [8. Gotchas](#8-gotchas)
- [9. Network and optimization studies](#9-network-and-optimization-studies)

---

## 1. Layout and entry points

```
pipeline/            the package: every module, flat, one sys.path entry
scripts/             runnable entry points (table below)
examples/            my_model.py: bring-your-own-model template
data/                atlas, 10,250,000 cells, 1.6 GB          (not in git, see ARTIFACTS.md)
models/              trained flow, cINN, 3.98 M params         (not in git)
results/, plots/     JSON and figures written by the runners   (not in git)
research_comparison/ literature networks: authors' neuron vs surrogate (section 9)
research_opti/       surrogate identification by optimization (section 9)
```

Every runner puts exactly one directory on `sys.path` — `pipeline/` — plus its
own `scripts/` directory so the plot scripts can import each other. Nothing
outside `pipeline_final/` is ever imported; `scripts/smoke.py --isolation`
asserts this.

| task | script |
|---|---|
| self-test | `scripts/smoke.py` |
| posterior for a reset target | `scripts/infer_reset.py` |
| posterior for a continuous target | `scripts/infer_continuous.py` |
| score all 20 Izhikevich presets | `scripts/v3_eval_infamily.py` |
| score the whole out-of-family zoo | `scripts/v3_eval_offfamily.py` |
| per-protocol target-vs-surrogate figures | `scripts/v3_plots.py` |
| per-family regime landscape (9 panels) | `scripts/v3_plot_family.py` |
| one wide 2-panel trace pair | `scripts/v3_plot_pair.py` |
| (τ, k_I) domain of validity | `scripts/v3_plot_validity.py`, `..._wide.py` |
| rebuild the atlas (GPU) | `scripts/v3_build.py` |
| retrain the flow | `scripts/train_flow.py` |

### Package modules

| module | contents |
|---|---|
| `v3_config` | every constant: `DT`, window multiples `K_*`, gammas, `I_RH_MIN`, sentinels |
| `v3_core` | reset integrator `integrate_v3`, `stim_v3`, `alloc_buffers` |
| `v3_landmarks` | `landmarks`, `rheobase_v3`, `t_cycle_win`, `l_inf_v3`, `isi_period` |
| `v3_protocols` | P1–P10 protocol kernels |
| `v3_features` | `compute_features_v3`, `Geometry`, `IDX`, `mask_of` |
| `v3_models` | `izh_rhs_v3`, `izh_rhs_scaled_v3` |
| `v3_gpu` | CUDA atlas kernels: `landmarks_gpu`, `features_gpu` |
| `v3_an_core` | reset-free integrator `integrate_an`, `make_sp`, `count_an` |
| `v3_an_features` | `V3Spec`, `compute_features_an`, `landmarks_an` |
| `v3_scoring` | `d2`, `w1`, `isis_of`, `isis_an`, `MISS_PENALTY` |
| `v3_regime` | `regime_report_izh/_an`, `summarize`, `format_report`, `GAMMA_LADDER` |
| `v3_inference` | `load_sigma`, `infer`, `format_result` — the endpoint core |
| `v3_an_zoo` | built-in out-of-family targets |
| `v3_presets` | Izhikevich's 20 canonical regimes |
| `models_offamily`, `sweeps` | foreign model RHS functions and sweeps |
| `neuron_spec` | `NeuronSpec` record used by `resetfree_specs()` |
| `run_surrogate` | `cinn_infer`, `simulate_trace` |
| `cinn_config`, `data`, `loader`, `model`, `spline`, `checkpoint`, `train`, `infer`, `main` | the flow |

---

## 2. ⚠ Destructive writes

**Two scripts overwrite their output directory. The shipped atlas is ~29 hours
of GPU time. Read this before running either.**

| script | what it does to a directory that already holds artifacts |
|---|---|
| `scripts/v3_build.py` | `makedirs(exist_ok=True)` accepts it. If `progress.json` is present it takes the **resume** branch and **truncates** the `.part` files to the recorded row count; otherwise it zero-truncates them. On completion it rewrites `izh_features.bin`, `izh_params.bin`, `izh_landmarks.bin`, `izh_meta.json`. |
| `scripts/train_flow.py` | Invalidates the stale `best_params.bin`, then writes `model_params.bin`, `best_params.bin`, `config.json`, `stats.json`. |

**A shipped atlas directory contains a `progress.json`, so it is live to a
resume.** A mistyped `--out` therefore does not error — it silently begins
consuming the atlas.

Both scripts now refuse to write into a non-empty artifact directory:

```
$ python scripts/v3_build.py --out data/atlas_v3_1e7 --n-train 10
refusing to build into 'data/atlas_v3_1e7': it already contains
izh_features.bin, izh_params.bin, izh_landmarks.bin, progress.json.
That directory holds an atlas or a resumable build, and this script would
truncate and overwrite it.
Point --out at a NEW directory, or pass --force if you really mean to
resume/overwrite this one.
```

`--force` (build) and `--force` / `--resume` (train) override the guard. Treat
them as you would `rm -rf`.

**Recommended practice**

- Always point `--out` / `--model-dir` at a **new** directory.
- `chmod -R a-w` any atlas or checkpoint you intend to keep. The copies shipped
  here are left writable so you can manage them, but locking them costs nothing.
- Keep a `sha256sum` manifest of artifacts you cannot regenerate cheaply.

---

## 3. Choosing an endpoint: reset vs continuous

| | reset | continuous (reset-free) |
|---|---|---|
| script | `infer_reset.py` | `infer_continuous.py` |
| extractor | `compute_features_v3` | `compute_features_an` |
| integrator | `integrate_v3` | `integrate_an` |
| spike | `v[0]` crosses **`V_TH = 30`**, then state jumps | `y[0]` crosses `spec.v_th` |
| threshold | **compile-time constant** | runtime field |
| state dim | 2 | any |
| examples | Izhikevich, AdEx | HH, FHN, Morris-Lecar, HR, Wilson |

**The threshold constraint is the thing that catches people.** The reset
integrator folds `V_TH = 30`. AdEx works only because its threshold genuinely is
30 mV. A reset model with any other threshold cannot use the reset path; you
must either rescale its voltage so the threshold is 30, or express it as a
reset-free system.

The second route is usually the better one: rescaling the voltage gives up the
model's physical units, and the pipeline is scale-free in time anyway (exact over
10 decades of τ). `examples/my_model.py` is a template for a reset-free target.

---

## 4. Function reference

### Measuring a target

```python
compute_features_v3(rhs, params, bufs, out=None, lm=None,
                    rheo_hi=None, auto_boot=False) -> (features, info)
```
One reset cell. `rhs(t, y, params, I, dydt)` must be numba-njit; `params` is
`(a, b, c, d)` and the reset `v→params[2]`, `u→u+params[3]` is applied by the
integrator. `bufs` comes from `alloc_buffers()`. `features` is a 32-slot array
whose first 18 entries are the active schema; `info` carries `i_rh`, `T_cycle`,
`L_inf`, `gated` and `reason`, so a dropped cell tells you *why*.
`rheo_hi` overrides the rheobase bracket ceiling (AdEx needs 2000).
`auto_boot=True` widens the bootstrap window when the anchor does not converge.

```python
compute_features_an(spec, buf_n=1<<16) -> (features, info)
```
Same contract for a reset-free `V3Spec`. Same 18-column schema on both sides —
that is what makes the two comparable at all.

```python
@dataclass V3Spec:
    rhs, y0, params=zeros(4), v_th=0.0, rheo_lo=0.0, rheo_hi=200.0,
    h=DT, boot_t_trans=.., boot_t_win=.., boot_t_disc=.., name="model"
```
`rheo_lo/hi` are in the **model's own current units** — Izhikevich's `[0, 200]`
is meaningless for Hodgkin-Huxley. `h` is the model's own integration step.

```python
landmarks(rhs, params, bufs, gamma_tc=2.0, rheo_hi=..., auto_boot=False) -> dict
landmarks_an(spec, st, isi, auto_boot=True) -> dict
```
The anchors alone (`i_rh`, `T_cycle`, `L_inf`, `p_hat`) without the protocol
suite. Cheap; use when you only need the gauge.

```python
alloc_buffers(n_spikes=None) -> dict
```
Reusable work buffers. Allocate once and pass to every call in a loop.

### Scoring

```python
d2(a, b, sigma, eps=1e-12) -> (score, n_joint)
```
**SUM** of squared z over jointly-defined dimensions, **plus `MISS_PENALTY = 9.0`
per dimension the target defines and the draw does not.** Not a mean — a mean
lets a draw that defines one feature and matches it score ~0 and win outright.
Atlas-relative: `sigma` is fitted per atlas, so scores are not comparable across
atlases or schemas.

```python
w1(x, y) -> float
```
Wasserstein-1 between two ISI samples by quantile coupling. No binning (several
schema columns are atoms). Atlas-free, comparable across checkpoints and
families, a proper metric. `nan` if either side has <2 intervals.

```python
isis_of(theta, bufs, gamma=2.0, rhs=None, rheo_hi=None) -> (isis, landmarks)
isis_an(spec, i_rh, tc, buf=1<<16) -> isis            # already in T_cycle units
```
Settled ISI sequences under the reference step. **Pass `rhs` for a foreign reset
model** — it defaults to Izhikevich, and feeding AdEx's parameters to the
Izhikevich RHS silently simulates a different cell.

### Regime diagnostic

```python
regime_report_izh(rhs, params, i_rh, t_cycle, bufs, gammas=GAMMA_LADDER, rheo_hi=None)
regime_report_an(spec, i_rh, t_cycle, gammas=GAMMA_LADDER, buf_n=1<<16)
format_report(rep, name="") -> str
```
Sweeps γ over the amplitude ladder and records the distinct settled-ISI count at
each. Verdict: `stable` / `marginal` / `unstable`, plus `block_above` for
depolarization block, which is reported **separately** (a cell that stops firing
is in a different state, not at an extreme of the same one).

**This never gates inference.** `gates_posterior` is always `False`. It predicts
`w1_shape`, not `d2`, so an unstable verdict is a statement about the
conditioning vector, not a prediction of failure — AdEx tonic scores 17 here and
still matches at `w1_shape = 0.0017`.

*Known limitation:* it cannot distinguish genuine regime stability from an
**inert drive axis**. A target with `i_rh ≈ 0` reports `stable` because nothing
varies along a ladder that is doing nothing. The tell is an identical spike
count at every γ.

### Flow

```python
cinn_infer(model_dir, features_active, n_samples, temperature=1.0) -> (theta, stats)
```
`features_active` is the 18-vector with `NAN_SENTINEL` in undefined slots.
Temperature 1.0 is the true posterior. **The draw is seeded** — repeating a call
reproduces it bit-for-bit, so a re-run is not a replicate.

```python
simulate_trace(rhs, params, y0, stim, v_th, h, t_end,
               reset_v=None, reset_du=None) -> (t, v, I, spikes)
```
RK4 trace recorder; `stim` is a callable `stim(t) -> I`. Applies a reset only if
`reset_v` is given. **Records `y[0]`**, which is also the component the spike
detector thresholds — order your state to put the observable first.

```python
load_sigma(atlas_dir, sigma_path=None) -> np.ndarray      # v3_inference
infer(kind, target, model_dir, S, n_posterior=256, ...) -> dict
```
`kind` is `"reset"` (target = `(rhs, params)`) or `"continuous"` (target =
`V3Spec`). Returns `best` / `mean` / `pmatch` views, `support`, `regime`, the
recovered `tau` / `k_I` and both `w1` figures.

### GPU (atlas build only)

```python
landmarks_gpu(theta, tpb=64)
features_gpu(theta, lm=None, tpb=64, sort=True)
```
Batched CUDA kernels. `GPU_MAX_SPIKES = 4096` is folded into `cuda.local.array`
at compile time — changing it **requires a distinct `NUMBA_CACHE_DIR`** or you
will silently get the previous value.

---

## 5. Runner reference

Common arguments: `--model-dir` (default `models/cinn_v3_p015`), `--atlas`
(default `data/atlas_v3_1e7`), `--n-posterior` (default 256), `--temperature`
(default 1.0), `--out`.

| script | key arguments | writes | cost |
|---|---|---|---|
| `smoke.py` | `--isolation`, `--numeric` | — | seconds / ~2 min |
| `infer_reset.py` | `--preset`, `--all-presets`, `--target`, `--rheo-hi` | `results/infer_reset.json` | ~1–2 min/target |
| `infer_continuous.py` | `--target`, `--all` | `results/infer_continuous.json` | ~1–5 min/target |
| `v3_eval_infamily.py` | `--only` | `results/v3_infamily.json` | ~20 min (20 presets) |
| `v3_eval_offfamily.py` | `--only`, `--merge` | `results/v3_offfamily.json` | ~30 min (14 targets) |
| `v3_plots.py` | `--which {infamily,offfamily,both}`, `--only` | `plots/{in,off}family/` | ~1 min/target |
| `v3_plot_family.py` | `--family`, `--all` | `plots/family/`, `results/v3_family_*.json` | ~3 min/family |
| `v3_plot_pair.py` | `--family`, `--label`, `--protocol`, `--cycles` | `plots/family/` | ~1 min |
| `v3_build.py` | `--out` (**required**), `--n-train`, `--n-val`, `--force` | a new atlas | **~29 h** at 1e7, GPU |
| `train_flow.py` | `--model-dir` (**required**), `--data-dir`, `--max-steps`, `--force`, `--resume` | a new checkpoint | ~1 h, GPU |

---

## 6. Worked examples

### (a) Score a built-in out-of-family target

```bash
python scripts/infer_continuous.py --target "Hodgkin-Huxley"
```
```
            Hodgkin-Huxley  d2=   14.459 (16d)  tau=  18.40  k_I=  0.2354
                            w1_shape=0.01539  support=1.000  defined 17/18  [stable]
```
Read it as: the surrogate is ~18× faster than HH in its own time units, needs
0.24× the current, and reproduces the spike train to `w1_shape = 0.015`. The d²
of 14.5 is mid-table and mostly reflects family distance, not spike-train error
— quote both.

### (b) Bring your own reset-free model

```python
# mymodels.py
import numpy as np
from numba import njit
from v3_an_features import V3Spec

@njit(cache=True)
def my_rhs(t, y, params, I, dydt):
    v, w = y[0], y[1]
    dydt[0] = v - v*v*v/3.0 - w + I
    dydt[1] = params[0] * (v + params[1] - params[2] * w)

my_spec = V3Spec(rhs=my_rhs, y0=np.array([-1.2, -0.62]),
                 params=np.array([0.08, 0.7, 0.8, 0.0]),
                 v_th=1.0, rheo_lo=0.0, rheo_hi=2.0, h=0.005,
                 boot_t_trans=100.0, boot_t_win=800.0, boot_t_disc=100.0,
                 name="my FHN")
```
```bash
PYTHONPATH=. python scripts/infer_continuous.py --target mymodels:my_spec
```
Checklist: `y[0]` is your observable **and** the spike-detection variable;
`rheo_hi` must bracket the true rheobase (too low ⇒ "no rheobase" for a cell
that fires fine); `h` must resolve your fastest timescale; `boot_t_win` must
hold several cycles or the anchor is measured on a starved window.

### (c) Bring your own reset model

Only if its threshold is 30. Otherwise use (b).
```python
# mymodels.py
my_reset = (adex_like_rhs, np.array([0.0, 0.0, -65.0, 60.0]), 2000.0)  # rhs, params, rheo_hi
```
```bash
PYTHONPATH=. python scripts/infer_reset.py --target mymodels:my_reset
```

### (d) Figures for a target

```bash
python scripts/v3_eval_offfamily.py --only "Wilson"      # scores first
python scripts/v3_plots.py --which offfamily --only "Wilson"
python scripts/v3_plot_family.py --family Wilson         # regime landscape
python scripts/v3_plot_pair.py --family Wilson --label "gT=2.00"
```
The plot scripts read the JSON the eval scripts write, so score before you plot.
The surrogate drawn in every figure is the **lowest-d² scorable draw**; `mean`
and `pmatch` are computed and stored but not plotted.

### (e) Rebuild the atlas

```bash
python scripts/v3_build.py --out /scratch/atlas_new --n-train 10000000 --n-val 250000
```
~29 h on one GPU. **`--out` must be a new directory** (§2). Verify with a short
run first: `--n-train 2000 --n-val 200`.

### (f) Retrain the flow

```bash
python scripts/train_flow.py --data-dir data/atlas_v3_1e7 --model-dir /scratch/flow_new
```
Defaults reproduce the shipped checkpoint: `--cond-jitter 0.15`, 60k steps,
batch 32768, EMA 0.9999. The jitter is load-bearing, not garnish: the atlas is a
4-D sheet in feature space and a displacement of 0.19σ moves 96.6% of posterior
mass outside the prior box. Real targets are off-manifold by construction.

---

## 7. The `I_RH_MIN` units workaround

**`I_RH_MIN = 1e-3` is an absolute constant.** The pipeline is scale-free in
*time* — exact over 10 decades of τ — but there is **no equivalent in current**.

If your target's rheobase, *in whatever units you quoted it*, falls below
`I_RH_MIN`, four columns are gated out:

> `eta`, `N_reb`, `g_reb`, `gamma_ramp`

That is not a cosmetic loss. Those four are where `a` and `c` information
concentrates: three of the five strongest `a`-correlates and
both strongest `c`-correlates. `a` and `c` are already the two hardest
parameters to identify.

### Diagnostic

**If your target loses exactly those four columns, check your current unit
before you look at anything else.**

### The workaround

Rescale the current unit *inside the RHS*: a pure change of unit, not of
dynamics. Add a unit factor to the parameters and multiply the stimulus by it:

```python
# params = (..., i_unit);  i_unit = physical current per unit of I
dV = (... + I * params[-1]) / C
```

If the model's rheobase is, say, `6.6e-6` A, then `i_unit = 1e-6` states the
stimulus in microamperes and moves `i_rh` to `6.6`, clearing the floor. The
trajectories are unchanged (check that `T_cycle` agrees between the two framings);
only the unit the stimulus is quoted in differs. The rescaled framing also makes
`k_I` meaningful.

Expect the restored columns to make d² *larger*: they are hard to match, and the
score over fewer dimensions was flattering. Quote the fuller one.

**This is a workaround for a known defect, not a design feature.** The proper
fix is to make `I_RH_MIN` relative to the target's own current scale; that is
untested.

---

## 8. Gotchas

**Numba caching.** Kernels are `@njit(cache=True)` and fold constants (`DT`,
`GPU_MAX_SPIKES`, window geometry) at compile time. If you change any of those,
**use a distinct `NUMBA_CACHE_DIR`** or you will silently get the previous
build's values. This has caused two false results in this project's history.

**Inference is seeded.** `torch.manual_seed(0)` runs immediately before
sampling. Re-running a measurement reproduces it bit-for-bit — that is *not* a
replicate. Genuine replicates need the seed shifted explicitly.

**`best` is a best-of-n statistic.** Measured seed-to-seed variation is 1.78×
median and 4.34× p90 — larger than most effects of interest — against 1.04× for
the posterior mean. Do not draw a conclusion from a `best` difference between
two runs; compare within a run on identical draws, or compare `w1`.

**`d2` is atlas-relative, `w1` is not.** `sigma` is fitted per atlas, so a `d2`
threshold must be quoted with the atlas that defines it. Across schemas the
numbers are meaningless. `w1` references no atlas.

**The two metrics are near-orthogonal.** Across 24 cells from six families,
Spearman(regime instability, d²) = +0.418 and is carried entirely by two
extremes; within Hindmarsh-Rose it is −0.200. A verdict quoting only one metric
is incomplete.

**Validation NLL is not comparable across schemas.** The shipped checkpoint's
−7.8927 cannot be compared to a v2 number: different feature definitions,
different conditioning dimension.

**Two different `adex_rhs` exist.** `models_offamily.adex_rhs` (C = 200) and
`sweeps.adex_rhs` (C = 50, conductances scaled 4× so the rheobase lands inside
the `[0, 200]` bracket) are different functions with the same name. They are not
interchangeable.

**Out of scope by design.** Sub-threshold/waveform-defined regimes (no component
observes them), inhibition (removed end-to-end in v3), and targets that cross a
stimulus-amplitude bifurcation boundary or occupy a chaotic band — their feature
vector is a collage of several dynamical systems. Hindmarsh-Rose is the
documented case; no window or step-size change repairs it.

---

## 9. Network and optimization studies

These folders sit beside the package and import it. Neither is needed to run
`pipeline/` or `scripts/`. Each has its own README, which is the reference for its
commands.

| folder | what it is | README |
|---|---|---|
| `research_comparison/` | published networks re-run with the authors' neuron and with the Izhikevich surrogate: Ivanov 2021 (LSM on MNIST, manuscript §4.1) and Schulte 2022 (HH microcircuit, §4.2). `lsmkit/` is the shared network engine (CPU and CUDA), with its unit tests in `lsmkit/tests/` (`python -m pytest -q lsmkit/tests` from `research_comparison/`) | `README.md`, then the README in each study folder |
| `research_opti/` | identifying the surrogate by optimizing `d2` directly: study 1 pits CMA-ES, DE and PSO against the flow on the same targets; study 2 sweeps the best optimizer along each target's own parameter | `README.md` |

**Data inputs they expect** (not in git):

- **MNIST** for Ivanov 2021: the four IDX files in `research_comparison/datasets/mnist/`
  (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`,
  `t10k-labels.idx1-ubyte`), read by `lsmkit/datasets.py`.
- **Zoo surrogate posteriors** for Ivanov 2021: `research_comparison/ivanov2021/data/zoo_posteriors/{hh,ml_c1,adex_adapting}.json`,
  read by `ivanov2021/calib.py` and `lsmkit/tests/test_zoo.py`. Each holds the pipeline's
  posterior summary for one target (`best{theta, tau, k_I, d2}`, `i_rh_target`,
  `T_cycle_target`) and is distributed with the data. The target models themselves are
  defined in `lsmkit/zoo_targets.py`.
- Each study's own run outputs (`data/` under the study folder), which its figure
  scripts read. The study READMEs say which runner writes each one.
