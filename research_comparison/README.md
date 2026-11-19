# research_comparison — literature LSMs: authors' neuron vs Izhikevich surrogate

Each study folder has its own README with its commands, sources and results.
Environment: the top-level `requirements.txt` (numba, scipy, torch for inference).

| folder | content | status |
|---|---|---|
| `lsmkit/` | shared skeleton: `spec` (Circuit / Binding / Network, CSR, OU receptors), `neurons/` (lif, izh, hh_dp, surrogate gauge + V-map), `topology/` (lattice, populations), `inputs`, `engine_cpu`, `engine_gpu`, `readout`, `metrics/` (task, agreement, compute), `infer` (pipeline wrapper → posterior.json), `tests/` | CPU + GPU engines, tested (analytic checks, zoo-neuron convergence) |
| `ivanov2021/` | Ivanov & Michmizos 2021 LSM on MNIST: LIF vs continuous targets (HH, Morris–Lecar C1, AdEx) vs their gain-matched surrogates | **In the manuscript (§4.1, Fig. 10, Tables 7–8).** Static LSM, 3 networks per condition: LIF 94.59 ± 0.16 %, HH 94.14 ± 0.18, ML C1 92.92 ± 0.11, AdEx 95.31 ± 0.16; surrogates HH 93.68 ± 0.22, ML C1 95.09 ± 0.03, AdEx 94.97 ± 0.24 (paper's best LSM 95.44). Neuron-integration cost 18× (HH), 5.8× (ML C1) and 2.7× (AdEx) lower with the surrogate. |
| `schulte2022/` | Häusler & Maass microcircuit in the replication of Schulte to Brinke et al. 2022 (560 HH neurons), spike-pattern classification tasks | **In the manuscript (§4.2).** Target reproduces their Table 6 (mean κ 0.551 vs 0.581). Surrogate network, 20 paired runs: mean κ preserved (Δκ −0.003 with noise, −0.027 without); delayed classification higher, XOR 0.12–0.23 lower, instantaneous classification slightly lower |

Tests (from this folder): `$PY lsmkit/tests/test_engine.py`, `$PY lsmkit/tests/test_zoo.py`.
