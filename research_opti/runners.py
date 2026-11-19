"""
runners.py -- the three optimizers behind one ask/tell interface, the lockstep scheduler
that pools them into single GPU launches, and the append-only logging.

WHY LOCKSTEP. One optimizer generation is 32 candidates. At that batch size the feature
kernel runs at a few percent occupancy and costs about 8x more per cell than it does on a
large batch (measured: 88.56 ms/cell at n=1024 against 11.25 ms/cell marginal
between large batches). Candidate features do not depend on the target, so every
concurrently live run -- every method, seed and target -- contributes to ONE launch per
round, and only the d2 reduction is per-target.

WHY RESTARTS. The landscape has large plateaus (spec G1: p, p_tr, delta_star, g_reb and
N_reb are atomic, so ranks tie) and jumps (G2: the mask term moves J by 9 when a feature
appears or vanishes). An inner optimizer that converges early therefore has not found the
optimum, it has run out of gradient signal; the budget is spent on a fresh start instead,
IPOP-style with a doubled population. Tolerance-based stopping is switched off (G5): the
stopping rule is the evaluation budget, as the spec requires.

    runs = [make_run("cma", tgt, seed=1, budget=10_000), ...]
    schedule(runs)
"""
import os
import json
import time
import numpy as np

import common as K
import evalbatch as EB
import objective as OB
import v3_config as C

TRAJ_COLS = ("eval", "a", "b", "c", "d", "J", "n_joint", "feasible")


# ----------------------------------------------------------------- the backends
class _CMA:
    """pycma. Rank-based, learns parameter correlations -- the primary method."""

    name = "cma"

    def __init__(self, seed, pop, rng):
        import cma
        self.cma = cma
        self.seed = seed
        self.pop = pop
        self.rng = rng
        self.version = cma.__version__
        self._new(np.full(4, 0.5), pop)

    def _new(self, x0, pop):
        # every tolerance zeroed: the stopping rule is the budget, not convergence (G5)
        self.es = self.cma.CMAEvolutionStrategy(
            list(x0), K.CMA_SIGMA0,
            {"bounds": [0, 1], "popsize": int(pop), "seed": int(self.seed),
             "verbose": -9, "verb_log": 0, "verb_disp": 0,
             "tolfun": 0, "tolfunhist": 0, "tolx": 0, "tolstagnation": 0,
             "tolfunrel": 0})
        self._xs = None

    def ask(self):
        self._xs = self.es.ask()
        return np.asarray(self._xs, float)

    def tell(self, J):
        self.es.tell(self._xs, [float(v) for v in J])

    def exhausted(self):
        return bool(self.es.stop())

    def restart(self):
        self.pop *= 2
        self._new(self.rng.random(4), self.pop)

    def settings(self):
        return dict(library="cma", version=self.version, popsize=int(self.pop),
                    sigma0=K.CMA_SIGMA0, bounds="[0,1] (pycma repair)",
                    tolerances="all zeroed; stop on budget")


class _Pymoo:
    """pymoo DE (DE/rand/1/bin) and PSO, driven through ask/tell.

    pymoo evaluates through its own Evaluator; here the objective is external, so F is set
    on the population directly and the evaluation counter is advanced by hand.
    """

    def __init__(self, which, seed, pop, rng):
        import pymoo
        from pymoo.core.problem import Problem
        self.pymoo = pymoo
        self.which = which
        self.name = which
        self.seed = seed
        self.pop = pop
        self.rng = rng
        self.version = pymoo.__version__

        class _P(Problem):
            def __init__(s):
                super().__init__(n_var=4, n_obj=1, xl=np.zeros(4), xu=np.ones(4))

        self._P = _P
        self._kw = (dict(pop_size=pop, variant="DE/rand/1/bin", CR=0.9, F=0.8,
                         dither="vector", jitter=False) if which == "de"
                    else dict(pop_size=pop, w=0.7, c1=1.5, c2=1.5))
        self._new(pop)

    def _alg(self, pop):
        if self.which == "de":
            from pymoo.algorithms.soo.nonconvex.de import DE as A
            kw = dict(self._kw, pop_size=pop)
        else:
            from pymoo.algorithms.soo.nonconvex.pso import PSO as A
            kw = dict(self._kw, pop_size=pop)
        return A(**kw)

    def _new(self, pop):
        self.al = self._alg(int(pop))
        # a termination far beyond the budget: the scheduler owns the stopping rule
        self.al.setup(self._P(), termination=("n_gen", 10 ** 9),
                      seed=int(self.seed) + 1000 * int(np.log2(max(pop, 1))),
                      verbose=False)
        self._pop = None

    def ask(self):
        self._pop = self.al.ask()
        return np.atleast_2d(self._pop.get("X")).astype(float)

    def tell(self, J):
        J = np.asarray(J, float).reshape(-1, 1)
        self._pop.set("F", J)
        self.al.evaluator.n_eval += len(J)
        self.al.tell(infills=self._pop)

    def exhausted(self):
        return not self.al.has_next()

    def restart(self):
        self.pop *= 2
        self._new(self.pop)

    def settings(self):
        return dict(library="pymoo", version=self.version, **self._kw)


def _backend(method, seed, pop, rng):
    if method == "cma":
        return _CMA(seed, pop, rng)
    if method in ("de", "pso"):
        return _Pymoo(method, seed, pop, rng)
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------- one run
class Run:
    """One optimizer instance on one target, with its own budget and trajectory file."""

    def __init__(self, method, target, seed, budget, study, space=None,
                 pop=None, traj=True):
        self.method = method
        self.target = target
        self.seed = int(seed)
        self.budget = int(budget)
        self.study = study
        self.space = space or OB.Space()
        self.pop0 = int(pop or K.POP)
        self.rng = np.random.default_rng(self.seed)
        self.be = _backend(method, self.seed, self.pop0, self.rng)
        self.n_eval = 0
        self.n_restart = 0
        self.best = (np.inf, None, 0, None)        # J, theta, n_joint, lm
        self.best_at = 0
        self.t0 = time.time()
        self.wall = 0.0
        self._traj = None
        self._rows = []
        if traj:
            self._traj = open(self.path_traj + ".part", "wb")

    # -- identity and paths ----------------------------------------------------
    @property
    def run_id(self):
        return f"{self.study}|{self.target.name}|{self.method}|s{self.seed}|b{self.budget}"

    @property
    def slug(self):
        return f"{self.study}_{self.target.name}_{self.method}_s{self.seed}_b{self.budget}"

    @property
    def path_traj(self):
        return os.path.join(K.TRAJ, self.slug + ".bin")

    # -- the ask/tell cycle ----------------------------------------------------
    def done(self):
        return self.n_eval >= self.budget

    def ask(self):
        """(m, 4) thetas, clipped to the remaining budget."""
        while self.be.exhausted():
            self.be.restart()
            self.n_restart += 1
        u = np.clip(self.be.ask(), 0.0, 1.0)
        left = self.budget - self.n_eval
        self._u = u[:left] if len(u) > left else u
        self._full = u
        return self.space.to_theta(self._u)

    def tell(self, J, nj, F, lm, ok):
        J = np.asarray(J, float)
        th = self.space.to_theta(self._u)
        # the backend must be told about exactly what it asked for; candidates trimmed by
        # the budget are charged the barrier so the shapes line up and the optimizer is
        # not handed a fabricated low value
        if len(self._full) != len(self._u):
            pad = np.full(len(self._full) - len(self._u), K.J_INF_BASE)
            self.be.tell(np.concatenate([J, pad]))
        else:
            self.be.tell(J)

        i = int(np.argmin(J))
        if J[i] < self.best[0]:
            self.best = (float(J[i]), th[i].copy(), int(nj[i]),
                         np.asarray(lm[i], float).copy())
            self.best_at = self.n_eval + i + 1
        if self._traj is not None:
            blk = np.empty((len(J), len(TRAJ_COLS)), dtype=np.float32)
            blk[:, 0] = self.n_eval + np.arange(1, len(J) + 1)
            blk[:, 1:5] = th
            blk[:, 5] = J
            blk[:, 6] = nj
            blk[:, 7] = ok[:len(J)]
            self._traj.write(blk.tobytes())
        self.n_eval += len(J)

    # -- finishing -------------------------------------------------------------
    def finish(self):
        self.wall = time.time() - self.t0
        if self._traj is not None:
            self._traj.close()
            os.replace(self.path_traj + ".part", self.path_traj)
            K.jwrite(self.path_traj.replace(".bin", ".json"),
                     dict(run_id=self.run_id, cols=list(TRAJ_COLS), dtype="float32",
                          n=int(self.n_eval)))
        J, th, nj, lm = self.best
        tau, kI = (self.target.gauge(lm) if lm is not None else (None, None))
        u = self.space.to_u(th)[0] if th is not None else None
        return dict(
            run_id=self.run_id, study=self.study, target=self.target.name,
            label=self.target.meta.get("label"), method=self.method, seed=self.seed,
            budget=self.budget, pop0=self.pop0, pop_final=int(self.be.pop),
            n_eval=int(self.n_eval), n_restart=int(self.n_restart),
            J=float(J), n_joint=int(nj), n_D=int(self.target.n_D),
            best_at_eval=int(self.best_at),
            theta=None if th is None else th.tolist(),
            u=None if u is None else u.tolist(),
            at_bound=None if u is None else self.space.at_bound(u),
            i_rh_s=None if lm is None else float(lm[0]),
            T_cycle_s=None if lm is None else float(lm[1]),
            p_hat_s=None if lm is None else float(lm[3]),
            tau=tau, k_I=kI,
            wall_s=round(self.wall, 3),
            settings=self.be.settings(),
            log_a=self.space.log_a, lam_period=self.target.lam_period,
            finished=time.strftime("%Y-%m-%dT%H:%M:%S"))


# ----------------------------------------------------------------- the scheduler
def results_path(study):
    return os.path.join(K.DATA, f"results_{study}.jsonl")


def completed(study):
    """run_ids already on disk, so an interrupted sweep resumes instead of restarting."""
    p = results_path(study)
    if not os.path.exists(p):
        return set()
    out = set()
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.add(json.loads(line)["run_id"])
                except Exception:                             # noqa: BLE001
                    pass                                      # a torn last line is ignored
    return out


def append_result(study, row):
    with open(results_path(study), "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def schedule(runs, tpb=64, verbose=True, on_finish=None, timing=None):
    """Drive every run in lockstep, one pooled GPU launch per round.

    `timing`, if a list, collects (batch size, seconds, T_cycle quantiles) per round.
    Every window in the feature suite is a multiple of the candidate's OWN cycle time and
    the rheobase search sweeps a fixed absolute window at 24 amplitudes, so the cost of a
    round is set by its slowest candidates, not by its size. That is worth recording: the
    atlas's 11.25 ms/cell is an average over the prior, and a search does not sample the
    prior.
    """
    live = [r for r in runs if not r.done()]
    rounds = 0
    t0 = time.time()
    while live:
        asks = [(r, r.ask()) for r in live]
        sizes = [len(a[1]) for a in asks]
        TH = np.concatenate([a[1] for a in asks], axis=0)
        tr0 = time.time()
        F, lm, ok = EB.evaluate(TH, tpb=tpb)
        if timing is not None:
            tc = lm[:, 1]
            d = tc[tc > C.SENT_CUT]
            timing.append(dict(n=int(len(TH)), s=round(time.time() - tr0, 4),
                               n_anchored=int(d.size),
                               tc=[float(x) for x in np.quantile(d, [.5, .9, 1.0])]
                               if d.size else None))
        o = 0
        for (r, _), m in zip(asks, sizes):
            sl = slice(o, o + m)
            J, nj = r.target.J(F[sl], lm[sl], ok[sl])
            r.tell(J, nj, F[sl], lm[sl], ok[sl])
            o += m
        rounds += 1
        still = []
        for r in live:
            if r.done():
                row = r.finish()
                if on_finish is not None:
                    on_finish(row)
            else:
                still.append(r)
        if verbose and (rounds % 10 == 0 or not still):
            el = time.time() - t0
            tot = sum(r.n_eval for r in runs)
            st = EB.cache_stats()
            print(f"  round {rounds:5d}  live {len(still):3d}  evals {tot:8d}  "
                  f"batch {len(TH):5d}  {1e3*el/max(tot,1):6.2f} ms/eval  "
                  f"hits {st['hits']}  gated0 {st['gated0']}  {el/60:6.2f} min",
                  flush=True)
        live = still
    return rounds
