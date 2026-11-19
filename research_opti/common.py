"""
common.py -- paths, environment and the frozen settings every script here shares.

The two studies in this folder (optimizer-vs-cINN, and the parameter-sweep showcase) share
these settings. This module exists so that no number is written twice: anything a reader of
the paper would need to reproduce a run is defined here or read from the pipeline.

Nothing here writes outside research_opti/. The pipeline package is imported, never
modified.

    PY=python     # torch + numba.cuda
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                         # pipeline_final
PKG = os.path.join(ROOT, "pipeline")
ATLAS = os.path.join(ROOT, "data", "atlas_v3_1e7")
MODEL_DIR = os.path.join(ROOT, "models", "cinn_v3_p015")

DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
TRAJ = os.path.join(DATA, "traj")
TARGETS = os.path.join(DATA, "targets")

if PKG not in sys.path:
    sys.path.insert(0, PKG)


def ensure_dirs():
    for d in (DATA, FIGS, TRAJ, TARGETS):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------- the objective
# MISS_PENALTY lives in the pipeline (v3_scoring.MISS_PENALTY = 9.0) and is
# read from there rather than restated; these are the constants added on top.
#
# J_INF_BASE is the infeasibility barrier. It must exceed any feasible d2 that can occur,
# and d2 is a SUM of unbounded squared z-scores, so no finite
# calibrated constant can guarantee the barrier: one pathological feasible cell with a
# z of 10^3 would outrank the infeasible region and invert the ordering the penalty is
# there to create. All three optimizers used here are comparison-based, so only the
# ORDER matters, and a barrier far above any representable d2 gives the intended order
# unconditionally. Infeasible candidates still rank among themselves through the two
# terms that follow it.
J_INF_BASE = 1.0e9
KAPPA = 1.0          # weight of the graded "how close to firing" term
LAM_PERIOD = 0.0     # optional ISI-map period constraint; 0 = OFF.

# Optimize in u in [0,1]^4, with `a` on a log scale because it spans 2.5
# decades and the slow-variable rate acts multiplicatively.
LOG_A = True

# Resolution/definedness gates. GATE_K_DT * DT and the `capped` flag are the
# atlas build's own gate (scripts/v3_build.py) and are read from v3_config.
N_DEF_MIN = 5        # a candidate defining fewer features than this is infeasible
N_REQ_SPIKES = 3     # "almost fires" reference count, = v3_config.N_MIN_SPIKES

# ------------------------------------------------------------------- cost ceiling
# A cost gate, not a scientific one, and the one deliberate restriction of the search
# relative to the space the cINN was trained on. v3's PASS 0 measures the rheobase a
# second time over K_WIN = 100 of the candidate's OWN cycle times, so cost grows linearly
# with that cycle time and is unbounded: a candidate at 500 ms runs a 40-step bisection
# over a 50 s window in one GPU lane while every other lane in the launch waits. Measured:
# a random batch of 64 cells took over two minutes, and the atlas's own cycle times reach
# 1586 ms (p90 13 ms, p99 73 ms).
#
# Candidates above the ceiling are charged the infeasibility barrier, so the optimizers
# search a strict subset of the box -- which can only handicap them, never flatter them.
# 50 ms excludes 1.76% of the atlas, and every surrogate any method has returned for these
# targets is far below it, because the gauge makes a surrogate's cycle time the target's
# divided by tau. prescreen.py applies it and reports the excluded fraction.
TC_MAX = 50.0        # ms; None disables the ceiling

# ------------------------------------------------------------------ optimizers
# One evolution strategy, one swarm method, one evolutionary algorithm -- three distinct
# families, so "the result does not depend on the optimizer" is a measured claim rather
# than three variants of the same mechanism.
#
# Population sizes are deliberately far above each library's 4-D default: the landscape
# has large plateaus (spec G1, the atomic features) where a small population ties on rank
# and step-size control stalls.
# 64, the top of the specification's range (3.1: "population lambda = 32-64; default ~8
# is too small for the plateaus"). It is also the cheaper choice here, which is not
# obvious: every pooled run advances in lockstep, so the wall clock is
# (budget / population) rounds times the cost of each round's SLOWEST candidate, and that
# slowest-lane cost does not fall when the round is smaller. Doubling the population
# therefore halves the number of rounds at almost no extra cost per round.
POP = 64
CMA_SIGMA0 = 0.3     # in u units
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

METHODS = ("cma", "de", "pso")


def gpu_info():
    """Device, driver and library versions, recorded with each run."""
    out = {}
    try:
        from numba import cuda
        import numba
        d = cuda.get_current_device()
        out["gpu"] = d.name.decode() if isinstance(d.name, bytes) else str(d.name)
        out["gpu_cc"] = "%d.%d" % d.compute_capability
        out["gpu_sms"] = d.MULTIPROCESSOR_COUNT
        out["numba"] = numba.__version__
        out["cuda_driver"] = ".".join(str(x) for x in cuda.cudadrv.driver.driver.get_version())
    except Exception as e:                                   # noqa: BLE001
        out["gpu_error"] = repr(e)
    for mod in ("numpy", "cma", "pymoo", "scipy", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                    # noqa: BLE001
            out[mod] = None
    return out


def jwrite(path, obj):
    """Atomic JSON write: never leave a half-written artifact behind."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


def jread(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)
