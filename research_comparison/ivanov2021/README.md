# ivanov2021 — Ivanov & Michmizos 2021 baseline LSM (MNIST / N-MNIST): LIF vs continuous zoo neurons vs surrogates

Paper: NeurIPS 2021, arXiv 2111.01760 (baseline LSM: static weights).
The authors' public code was consulted only for values the paper omits.

```
PY=python
$PY calib.py                                   # per-model units -> data/calib.json
$PY rate_scan.py                               # liquid / input rates vs w -> data/rate_scan_mnist.json
$PY run.py --dataset mnist --model lif --bench
$PY run.py --dataset mnist --model lif --w 0.8 --nets 1 --tag probe
$PY campaign.py --eta && $PY campaign.py --stage all      # the MNIST campaign (~6.6 h)
```
Models: `lif` (paper), `hh`, `ml_c1`, `adex_adapting` (zoo, RK4 at their pipeline step), `sur:<key>` (drop-in surrogate:
the target's current through the gauge), `sur*:<key>` (self-calibrated surrogate: its own 1 ms-pulse unit).

## Design (2026-11-05)
- **Liquid:** 1000 neurons on a 10×10×10 grid (grid shape CHOSEN), 80/20 E/I. Connection probability C·exp(−(D/λ)²) with C = EE .2, EI .1, IE .05, II .3 and λ 3. Input→liquid density 15% with random sign. One liquid weight w for all connections. Synaptic kernel e^{−t/1 ms}/1 ms.
- **LIF** = the paper's discrete model: Euler at 1 ms, θ 20, τ_v 64, 2 ms refractoriness that blocks spike registration but lets v keep integrating.
- **Input neurons are the same model as the liquid** (user decision). A stimulus event (MNIST: Bernoulli(pixel) each ms; N-MNIST: an event in its 1 ms bin) delivers an exact 1 ms square pulse: a +A edge, then a −A edge one millisecond later.
- **Units for continuous models:** 1 ms-pulse threshold from rest (LIF: 20), so I = u·thr/20. DC rheobase is the wrong unit for brief inputs: HH's thr is 1.1 i_rh, ML's 22.7, AdEx's 24.9. The input pulse is 2·thr (one spike per event; the paper's 5× makes AdEx burst 3 spikes). The LIF keeps 100.
- **Integration:** zoo models use RK4 at their pipeline step inside dt = 0.1 ms (HH 20 substeps, ML/AdEx 10). Convergence is checked in `../lsmkit/tests/test_zoo.py`: same spike counts, mean ISI within 1e-4 of float64 RK4 at h/4. Euler is used only for the paper's LIF.
- **Readout:** spike counts over 250 ms → linear layer, softmax cross-entropy plus L2 5e-10, Adam lr 0.1, batch 250, zero init. Training stops at the best validation epoch, with patience 100 (paper: until validation peaks, at most 5000 epochs).

## Input-neuron time scaling (smoke test, `input_speed.py`, data/input_speed.json)
Input neurons can run s-fold faster in network time (like the gauge τ; parameters
unchanged). For the paper's code (Bernoulli 1 ms pulses, p = pixel/255) the input layer must
satisfy two conditions:
- one pulse fires the cell: A ≥ the 1 ms-pulse threshold at speed s;
- fused pulses (p → 1 means DC of amplitude A) keep the cell firing repetitively: i_rh ≤ A < block.

The rule A = max(2·thr, 2·i_rh) at **HH s = 4, ML C1 s = 32, AdEx s = 16** matches the LIF
rate-vs-p curve to a mean |log error| of 0.18–0.21. Unscaled (s = 1) the errors are 1.06 / 1.97 / 0.40;
there, HH emits one spike per run of fused pulses and ML enters depolarization block.
**Added as `<zoo>@fastin`** (`spec.FASTIN_SPEED`, `run.fastin`). On MNIST the input layer's mean rate is
HH 45 / ML 63 / AdEx 60 Hz (LIF 52; unscaled HH 13).

The input layer is simulated once per sample (it gets no liquid input) and cached in
`data/input_cache/mnist_<zoo>_fastin/b<block>.npz` (1000 samples per block, ~60 MB). It is replayed into
every weight and network. The cached run's liquid spike counts are identical to the coupled network's
for every neuron.

## Status
- **LIF probe:** MNIST test **94.75%** at w = 0.8 (paper's best 95.44%, the best of its sweep and 10 networks), 51 Hz, 3 min end to end.
- **Calibration:** brief-pulse threshold, surrogate / target: HH **35×** (the drop-in HH surrogate's liquid is silent), ML 1.37×, AdEx 1.59×.
- **Rate scan:** input-neuron rates LIF 52, HH 13, ML 4.5, AdEx 83 Hz. Swapping the input layer filters the image code per model. Working w ranges differ by ~10× between models.
- **LIF-input control** (`<zoo>@lifin`): the exact LIF input-layer spike trains (`encode.lif_input_spikes`, identical to the network's input neurons spike for spike) drive the zoo liquid directly.
- **Campaign ready** (`campaign.py`): ranges → sweep → full, **16 conditions**, MNIST, ETA ≈ 7 h, plus ~1 h of one-off
  `@fastin` input caching (~14 GB). Smoke-tested end to end.
- **One stimulus realisation:** seeded per (seed 1000, sample index), so it is identical across networks, conditions,
  sweep and full runs. Networks differ only in wiring.
- **Engine:** in static circuits, input events are delivered by all of the block's threads (the old serial thread-0
  path was the bottleneck for dense input layers). Static path CPU vs GPU: identical counts.
- **Drop-in HH surrogate:** fires only at w ≳ 51 (not dead, just ~35× less sensitive to brief input).

## Surrogate input layer for @fastin (`fastin_sur.py`)
- **The posterior is exactly gauge-equivariant.** Fast HH (x4) gets τ 4.6 = 18.4/4, with the same θ, k_I 0.2354 and d² 14.46 as the HH surrogate (`data/fastin_posteriors/hh_s4.json`).
- **Neither input pulse matches the target's rate curve.** The fast-HH target saturates near 300 Hz (101 / 243 / 293 Hz for p < 0.3 / 0.3–0.7 / > 0.7). Its 1 ms threshold is 2.53 and its pulse 12.34; the surrogate's 1 ms threshold is 63.9 (25×, the brief-pulse charge mismatch again).
  - Self-calibrated pulse (2 × 63.9): 225 / 908 / 1605 Hz. Fused pulses act as ~20× rheobase DC with no saturation, and the 192-spike cache buffer overflowed.
  - Drop-in pulse (12.34 through the gauge): 0 spikes per single pulse → 0 / 86 / 268 Hz. Dim pixels are lost; bright pixels match.
- **Rate-matched pulse (chosen):** 16.43 target units (target pulse 12.34). Fitted on 50 train images, 10 pixel bins; RMSE 76 Hz. `data/fastin_posteriors/hh_s4_pulse.json`.
  - The curve shapes differ. Target and surrogate rates: p ≤ 0.3: 46–166 vs 0–12 Hz (single pulses don't fire). p > 0.9: 295 vs 348 Hz.
  - Total input spikes per sample: 8,344 surrogate vs 8,908 target (−6%).

## Zoo comparison (MNIST; run with `overnight.py`, results in the manuscript §4.1)
- **Conditions** (identical topology and input code; only the neuron model is swapped):
  - `lif`;
  - `<zoo>@fastin` (each zoo neuron with its own time-scaled input layer);
  - `sur:<zoo>@fastin` (its gain-matched surrogate, w_s = g·w*_target from `gain.py`, with the surrogate of that input layer);
  - zoo = hh, ml_c1, adex_adapting.
  - No @lifin or other mixed conditions.
- **Network:** static LSM with the paper's baseline wiring, `campaign.py --stage final`.
- **Input-layer surrogates** (`fastin_sur.py`, `data/fastin_posteriors/`):

  | model | posterior (time-scaled target) | rate-matched pulse (target pulse) | RMSE |
  |---|---|---|---|
  | HH x4 | τ 4.6 = 18.4/4, θ / k_I / d² identical (exact equivariance) | 16.43 (12.34) | 76 Hz |
  | ML C1 x32 | τ 0.2709 = 8.668/32, exact | 108.9 (111.1) | 18 Hz |
  | AdEx x16 | τ 0.609 vs 0.572 predicted, k_I 50.1 vs 46.6, d² 0.593 vs 0.621 | 728 (658) | 30 Hz |

  - AdEx deviates because the reset path integrates at a fixed DT 0.01, an effective 0.16 model-ms step.
  - Input caches carry the pulse in the directory name (`mnist_sur*:<zoo>_fastin_p<pulse>`).
- **Additions:**
  - `campaign.py`: FINAL set; stages extend (edge optima, rate guard 2.5–300 Hz), pred (confusion for older full runs, written next to them), `full_dir` (a new w gets its own dir); `best_w.json` is merged, never truncated.
  - `readout.py` returns the test confusion matrix and the validation curve.
  - `showcase.py`: one test digit through every condition (rasters).
  - `figs.py`: `ivanov_static`.
- **Rehearsal** (`overnight.py --smoke --root <scratch>`): all steps, all 7 conditions and the figures, end to end in 8 min. Layout checked.
- **ETA (bench):** ~3 h for the static campaign, plus ~10 min for the figures.
