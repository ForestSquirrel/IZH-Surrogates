# research_opti -- surrogate identification by optimization

Two studies, both self-contained in this folder.

1. **Study 1, the head-to-head.** Identify an Izhikevich surrogate by minimizing the
   pipeline's feature distance `d2` with three conventional derivative-free optimizers --
   CMA-ES, DE and PSO -- and compare with the amortized cINN on the same targets, the same
   objective, the same metric and the same evaluator. Targets: Hodgkin-Huxley,
   Morris-Lecar C1, AdEx tonic, at the configurations the pipeline's zoo defines.
2. **Study 2, the generalization showcase.** The best optimizer, swept along each target
   model's own parameter, so the claim is "it works across a spread of configurations",
   not "it works on three cells". Axes: Morris-Lecar `V3` 2->14 (Class 2 to Class 1),
   Hodgkin-Huxley `gK` 24->40, AdEx spike-triggered `w` jump 0->120.

## The short paper's figures

`paper_figs.py` produces four figures for the short paper, whose claim is about the
**feature schema and its protocols**, not about any one way of fitting them. The cINN
appears in none of them, and surrogate-against-model fidelity is out of that paper's
scope.

| figure | |
|---|---|
| `figs/fig5_features.pdf` | **the schema figure**: eight protocols, each on the Izhikevich regime that displays the feature, with the measured value. Supersedes fig1 |
| `figs/fig1_schema.pdf` | superseded by fig5 -- it placed features as percentiles within the atlas, which belongs to the amortized-inference work | (a) the nine protocols as drives in the cell's own rheobase and cycle time, each labelled with the features it yields; (b) the resulting 18-number fingerprint of three targets, as percentiles within the atlas, undefined entries hatched |
| `figs/fig2_optimizers.pdf` | (a, b) final `d2` and `W1` per method per target, 10 seeds, with the largest spread between methods quoted; (c, d) the same along each model's own parameter, 30 configurations |
| `figs/fig3_performance.pdf` | convergence and evaluations-to-converge. Separate so it can be dropped |
| `figs/fig4_waveforms.pdf` | target and its three identified surrogates under the reference step, one column per model |

Nothing here modifies `pipeline/`; it is imported. Nothing is written outside this folder.

## Running it

```bash
PY=python     # needs torch and numba.cuda
cd pipeline_final/research_opti

$PY targets.py                   # measure and cache every target's feature vector
$PY sigma_opt.py --n 100000      # freeze both metrics
$PY check_controls.py            # quadratic / in-family self-recovery / permuted null

$PY cinn_baseline.py                                  # the amortized arm
$PY run_all.py --budget 5000 --seeds1 10 --seeds2 3   # BOTH studies + controls, pooled
# or separately, which is slower for the reason in the last section:
#   $PY run_study1.py --budget 5000 --seeds 10
#   $PY run_study2.py --method cma --budget 5000 --seeds 3

$PY validate.py --study study1   # w1_shape, w1_log, ISI-map period of each chosen theta
$PY analyse.py  --study study1
$PY plot_opt.py --study study1
$PY paper_figs.py                # the paper figures (fig1..fig4)
```

Every runner is resumable: a run already in `data/results_<study>.jsonl` is skipped, so an
interrupted sweep continues instead of restarting, and no file is ever rewritten in place.

## Layout

| | |
|---|---|
| `common.py` | paths, the frozen constants, environment capture |
| `evalbatch.py` | the batched GPU evaluator, the dedupe cache, the vectorized `d2` |
| `objective.py` | the search space (`u` <-> theta, log `a`) and the objective per target |
| `sigma_opt.py` | the two frozen metrics: atlas and atlas-free |
| `targets.py` | both studies' targets and their cached feature vectors |
| `runners.py` | the three backends behind one ask/tell interface, the lockstep scheduler, logging |
| `driver.py` | the shared study driver (build, schedule, resume) |
| `prescreen.py` | PASS 0 with a cost ceiling, so one slow candidate cannot stall a batch |
| `run_all.py` | both studies and the controls in one pooled schedule -- the way to run it |
| `run_study1.py`, `run_study2.py` | the individual runs |
| `cinn_baseline.py` | the cINN arm, scored through the same evaluator |
| `validate.py` | spike-train validation of each chosen theta, as a separate pass |
| `analyse.py`, `plot_opt.py` | tables and internal figures, from the logs only |
| `traces.py` | record a voltage trace under the reference step (figures only) |
| `paper_figs.py` | the paper figures and `FIGURES.md`, cINN-free |
| `fig_schema.py` | the schema figure: what each feature measures, drawn on the dynamics |
| `check_controls.py` | the end-to-end controls (control targets for `run_all.py`) |

## The one design decision worth knowing

One evaluation is a full protocol suite on one Izhikevich cell, and per-cell cost collapses
with batch size: 88.56 ms/cell at a batch of 1024 against 11.25 ms/cell marginal between
large batches. A single optimizer generation is 32 candidates, which
would run the GPU at a few percent occupancy.

So every concurrently live run -- each method, each seed, each target -- contributes to one
kernel launch per round. Candidate features do not depend on the target, so one launch
serves every target at once and only the `d2` reduction is per-target. This is why all
three optimizers are driven through `ask()`/`tell()` rather than by handing a callable to a
library's own loop, and why `run_all.py` pools both studies.

The second half of the same problem: a round waits for its SLOWEST candidate, and v3's
PASS 0 re-measures the rheobase over 100 of the candidate's own cycle times, so that cost
is unbounded. `prescreen.py` caps it. Without the cap a random batch of 64 cells took over
two minutes; with it, the full 930 000-evaluation run took 7.2 h.
