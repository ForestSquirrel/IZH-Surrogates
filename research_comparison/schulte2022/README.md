# schulte2022 — Häusler & Maass 2007 data-based circuit (Schulte to Brinke et al. 2022), HH vs Izhikevich surrogate

Main non-LIF anchor. 560 Destexhe-Paré HH neurons in L2/3, L4, L5 (E/I), conductance synapses
with Tsodyks STP, intrinsic OU conductance noise; spike-pattern tasks (tcl, delayed tcl, XOR),
kappa from NNLS readouts at L2/3-E and L5-E. Constants with sources in `spec.py`.

```
PY=python
$PY targets.py                    # HH-E / HH-I surrogates -> targets/hh_{E,I}.json (~1 min each)
$PY run.py --bench                # ETA (20 runs run in parallel: wall = one run's time)
$PY run.py --tag noise            # Table 6 condition, 20 runs x 1800 trials (~46 min, 3 models)
$PY run.py --no-noise --tag nonoise
$PY sd_check.py                   # single-cell probes -> data/sd_check_best.json (10 s)
$PY figs.py                       # tex/figs/schulte_{noise,nonoise,cell}.{pdf,png} (45 s)
cd tex && latexmk -pdf main_schulte.tex   # standalone build of tex/sec_schulte.tex
```

Manuscript entry: `tex/sec_schulte.tex` (subsection, `\input`-able; macros from
`pipeline_final/tex/main.tex`; extra bib entries in `tex/refs_schulte.bib` to merge into
`refs.bib`). `tex/figs/schulte2022_fig1.png` is the paper's Fig 1 cropped from the PDF
(CC BY 4.0; credited in the caption). Figure fonts: Tinos, a TrueType font with Times New
Roman metrics (the TeX Gyre Termes OTF embeds as a mismatched font type).

| file | role |
|---|---|
| `spec.py` | Fig 1 / Tables 1–5 constants; PAPER / CODE / CHOSEN marked |
| `targets.py` | HH-DP (E: gM 100, I: gM 0; tonic noise means included) as continuous pipeline targets; voltage map |
| `circuit.py` | population-block circuit (`lsmkit/topology/populations.py`), receptors, target / surrogate bindings |
| `tasks.py` | continuous pattern input (15 × 2 templates per stream, 20 Hz, 1 ms jitter), targets |
| `run.py` | GPU runs, NNLS readouts, kappa, rates, agreement (CKA, rate correlation over neurons, Γ), compute; `best@gain`, `best@x<fE>:<fI>` gain variants |
| `sd_check.py` | single-cell probes: strength–duration, PSP, EPSC-pair coincidence (target vs surrogate) |
| `figs.py` | manuscript figures |
| `tex/` | `sec_schulte.tex`, `main_schulte.tex` (standalone), `refs_schulte.bib`, `figs/`, `tables/` |

## Surrogates (targets/)

| target | i_rh | T_cyc | view | θ (a, b, c, d) | τ | k_I | d² (dims) | w1_shape |
|---|---|---|---|---|---|---|---|---|
| HH-E (gM 100) | 669 pA | 11.38 ms | best = pmatch | 0.228, −0.010, −80.3, 0.421 | 4.41 | 39.5 | 10.95 (16) | 0.0009 |
| | | | mean | 0.136, −0.014, −79.0, 0.988 | 4.17 | 39.0 | 12.05 (16) | 0.0005 |
| HH-I (gM 0) | 551 pA | 11.29 ms | best = pmatch | 0.078, 0.187, −79.6, −0.935 | 2.19 | 122.7 | 0.06 (16) | 0.0004 |
| | | | mean | 0.169, 0.159, −77.7, −0.864 | 2.48 | 85.5 | 0.27 (16) | 0.0004 |

Voltage map (per view): HH rest at I = 0 (−65.6 / −65.3 mV) ↔ Izhikevich node at I = 0;
HH threshold (dV/dt = 10 mV/ms, first spike at 1.2 i_rh; −49.0 / −48.9 mV) ↔ Izhikevich
saddle at I = 0. Used for the conductance synapses' driving force, the OU noise
conductances, the initial V and the L2 voltage record.

## Reproduction choices

- **Neuron:** code values (Table 1: gNa 17872 nS, ENa 50, VT −63, EM −80, r = 0.001).
  The noise conductance means (12 / 57 nS) belong to the neuron and are inside the inference
  target. The zero-mean OU fluctuations (σ 3 / 6.6 nS) are engine receptors, so the
  surrogate gets them through the voltage map. Spike = NEST rule (peak above −30 mV, 3 ms
  emission refractoriness).
- **dt** 0.1 ms (paper: NEST 0.2 ms). HH runs as 4 × 0.025 ms RK4 substeps; the surrogate
  runs at h ≤ 0.01 model ms (5 substeps).
- **Parameter scatter:** per paper, Gaussian with SD = b × mean; values out of range are
  redrawn uniform on (0, 2·mean).
- **Things the paper doesn't give (taken from the code):** 20 Hz pattern rate, one discarded
  30 ms segment, static input synapses with 1 ms delay. Readouts are L2/3-E and L5-E (the text
  says L4, but the code and Table 6 say L5).
- **Readout filter:** paper exp(−t/15 ms) on spikes. The code uses an iaf_psc_exp filter
  neuron (τ_syn 2 ms, τ_m 15 ms, 1 ms delay).

**Code vs paper (we follow the paper):**
- **Scatter SDs:** the code uses absolute SDs (`np.random.normal(mean, 0.7)`), so its weights,
  U and τ barely vary.
- **Stream 2 templates:** the code builds stream 2 from stream 1's templates.
- **Inhibitory redraws:** in the code, out-of-range inhibitory weights are redrawn on
  (2·mean, 0).

## Results (2026-11-05): 20 runs × 1800 trials, `data/runs/{noise,nonoise}/`

Kappa, mean ± SD over 20 runs. The surrogate column shows the paired difference from the
target, mean ± SE, on the same circuit and input; * marks Wilcoxon p < 0.01. `best` and
`mean` views agree within ~0.03.

| task | paper | target (noise) | surrogate − target (noise) | target (no noise) | surrogate − target (no noise) |
|---|---|---|---|---|---|
| tcl1 L23 | 0.870 | 0.930 ± 0.074 | −0.049 ± 0.012 * | 0.981 | −0.018 ± 0.010 |
| tcl2 L23 | 0.947 | 0.911 ± 0.056 | −0.066 ± 0.019 * | 0.945 | −0.055 ± 0.014 * |
| tcl1(t−1) L23 | 0.389 | 0.331 ± 0.104 | **+0.122 ± 0.023 *** | 0.642 | **+0.096 ± 0.031 *** |
| tcl2(t−1) L23 | 0.352 | 0.235 ± 0.131 | **+0.093 ± 0.027 *** | 0.340 | +0.070 ± 0.035 |
| XOR L23 | 0.304 | 0.313 ± 0.113 | **−0.185 ± 0.026 *** | 0.491 | **−0.236 ± 0.034 *** |
| tcl1 L5 | 0.693 | 0.798 ± 0.131 | −0.151 ± 0.030 * | 0.867 | −0.142 ± 0.038 * |
| tcl2 L5 | 0.968 | 0.948 ± 0.059 | −0.033 ± 0.017 | 0.969 | −0.027 ± 0.015 |
| tcl1(t−1) L5 | 0.418 | 0.300 ± 0.101 | **+0.119 ± 0.025 *** | 0.493 | **+0.113 ± 0.038 *** |
| tcl2(t−1) L5 | 0.490 | 0.350 ± 0.115 | +0.056 ± 0.020 | 0.464 | +0.061 ± 0.036 |
| XOR L5 | 0.383 | 0.397 ± 0.109 | **−0.165 ± 0.031 *** | 0.587 | **−0.243 ± 0.035 *** |
| mean of the 10 | 0.581 | 0.551 | 0.525 | 0.678 | 0.640 |

- **Target reproduces Table 6.** The mean of the 10 tasks is 0.551 against the paper's
  0.581. XOR is 0.313 / 0.397 against 0.304 / 0.383. The delayed tasks come out ~0.06–0.14
  low. Removing the noise raises performance, as the paper reports.
- **Surrogate: similar overall, different profile.**
  - Instantaneous classification: −0.03 to −0.15.
  - Delayed (memory) tasks: +0.06 to +0.12. The surrogate remembers more.
  - XOR (nonlinear): about half the target's. Same sign and size with and without noise.
- **Internal agreement** (noise / no noise):
  - spike-count correlation across neurons: 0.974 / 0.973
  - readout-state CKA: 0.67 / 0.76
  - population rate ratio: E23 0.82, E4 0.83, E5 0.60–0.66, I5 0.4–0.5
  - coincidence factor Γ ≈ 0: no spike-time agreement during ongoing activity
- **Hypothesis, not tested here:** the surrogate integrates over a longer window relative to
  its cycle. The equivalence-band study found this for classic HH (strength–duration, 14–57×). A longer window
  would give more memory and less sharp coincidence or nonlinear mixing. The next check is to
  measure the HH-DP surrogate's strength–duration.
- **Hypothesis test (`sd_check.py`, `data/sd_check_best.json`): the integration window is
  NOT much longer.**

  | | chronaxie | single-EPSC threshold | PSP half-width / decay |
  |---|---|---|---|
  | E: HH / surrogate | 4.98 / 8.33 ms | 43.1 / 63.0 nS | 8.5 / 7.4, 7.5 / 6.3 ms |
  | I: HH / surrogate | 5.59 / 6.96 ms | 40.7 / 38.9 nS | 8.7 / 7.8, 7.7 / 6.5 ms |

  Chronaxie is 1.25–1.7× longer (classic HH in the equivalence-band study: 14–57×), and the surrogate's
  PSPs are slightly SHORTER. The pair-coincidence window is about 10 ms for both. The clear
  difference is excitability: the E surrogate needs 1.46× the conductance to fire from a
  single EPSC, which fits its lower rates.
- **Gain control (20 runs × 300 trials, scratchpad):** E-synaptic gain applied to the
  surrogate.

  | E gain | rate ratio E23 / E4 / E5 | XOR L23 / L5 (target 0.20 / 0.28) |
  |---|---|---|
  | 1 (untuned) | 0.83 / 0.83 / 0.61 | 0.146 / 0.176 |
  | 1.13 (rates matched) | 0.98 / 1.05 / 0.85 | 0.064 / 0.145 |
  | 1.46 (threshold-matched) | 1.34 / 1.54 / 1.49 | 0.063 / 0.091 |

  More drive lowers XOR, so the XOR deficit is not a rate or excitability artifact. Main
  results keep the untuned pipeline surrogate. `run.py` accepts `best@gain` and
  `best@x<fE>:<fI>`.
- **Rates:** the target fires above the paper (E23 58 / E4 50 / E5 14 Hz against ~40 / ~40 /
  3–9 Hz).
- **Compute** (RTX 5070 Ti, 20 runs in parallel, 8.1 M steps per run): target 17.5 / 16.3
  min, surrogate 10.5 / 10.1 min, a wall-time ratio of 0.60. The run is latency-bound
  (one block per run). Per neuron step, HH takes 4 RK4 substeps of a 5-state model with 8
  exponentials; Izhikevich takes 5 substeps of a 2-state polynomial.

## Earlier smoke notes (2026-11-05)

- **Smoke run** (2 runs × 150 trials, 112 for training): target kappas are already close to
  Table 6. tcl1 / tcl2 are 0.80 / 1.00 on L23 (paper 0.87 / 0.947) and 0.72 / 0.84 on L5
  (paper 0.693 / 0.968).
- **Rates:** the target runs at E23 58, E4 51 and E5 16 Hz. The paper reports about 40 for
  L2/3 and L4, and 3–9 Hz for L5. The surrogate's rates are 0.84 / 0.83 / 0.64 of the
  target's for E23 / E4 / E5.
- **Agreement (3 runs × 30 trials):**
  - CKA of the readout states: 0.86
  - spike-count correlation across neurons: 0.97
  - Γ = 0.00: spike times decorrelate in an ongoing noisy run
- **Compute (GPU):** the surrogate takes 0.57× the target's wall time.
- **Full 20-run reproduction:** not run yet (~46 min).
