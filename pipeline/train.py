"""
train.py — cINN training (plain flow MLE, no auxiliary losses).

    L = 0.5 * ||z||^2 - log|det J|

The dataset is streamed, never resident: data.open_atlas memmaps the binaries and
loader.SlabLoader feeds batches a slab at a time, so host RAM and VRAM are both
flat in N. A 1e6-row atlas and a 1e9-row atlas cost the same memory; only wall
time differs.

Step budget, not epoch count, is the thing to set on a large atlas. The original
1e6-row run was 200 epochs = ~23k optimizer steps; at 1e8 rows that many steps is
reached in roughly 2 epochs, so ``--epochs 200`` there would be ~100x more
training than it was ever tuned for. Use ``--max-steps`` (and optionally
``early_stop_patience``) instead.

Three things here are less obvious than they look, all of them lessons from the
400k-step 1e8 run:

**The LR is scheduled per step, by a plain function, not a torch scheduler.**
That run held lr=1e-3 flat for all 400k steps; its val_nll plateaued around epoch
21 and then swung +/-3 nats between consecutive epochs *on a fixed validation
set*, which is the model thrashing, not eval noise. A ``LambdaLR`` cannot be
resumed cleanly here — checkpoints written before this change have no
``initial_lr`` in their param groups, so reconstructing one with
``last_epoch=global_step-1`` raises, and building it before
``optimizer.load_state_dict`` is fragile because that call replaces
``param_groups`` wholesale. ``lr_at()`` needs no state beyond ``global_step``,
which was already checkpointed.

**Evaluation drives everything and runs on a step cadence.** Best-checkpointing
and early stopping both hang off ``_maybe_eval``; there is deliberately no
second epoch-end copy of that logic. Keeping both would break early stopping
outright: ``best_val`` would already hold the minimum over that epoch's step
evals, so the epoch-end comparison could never fire, ``since_best`` would
increment every epoch unconditionally, and the run would die after `patience`
epochs no matter how well it was going.

**Ordering around ActNorm is load-bearing.** ``Fm.ActNorm`` initializes its
scale/bias in place from the first batch it ever sees. So the EMA shadow is
created lazily *after* the first optimizer step (a shadow taken earlier captures
scale=bias=0 and then chases a step-1 discontinuity), the first iteration runs
through eager rather than compiled code (so that in-place init happens outside
the compiled region), and nothing is ever evaluated before the first training
forward — an eval-first model would initialize ActNorm from validation data
under no_grad, and the EMA swap-back would then write zeros over it while the
"already initialized" flag stays set, silently disabling ActNorm with no error.
"""

import contextlib
import copy
import itertools
import math
import os
import time
import torch

from data import (
    open_atlas, sample_rows, fit_feature_stats_streaming, fit_param_stats,
    save_stats, load_stats,
)
from loader import SlabLoader
from model import CINN
from checkpoint import (
    save_checkpoint, has_checkpoint, load_checkpoint, _BEST, _WEIGHTS,
)


def _resolve_device(want):
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA unavailable — falling back to CPU")
        return "cpu"
    return want


def lr_at(step, base, warmup, total, final_frac):
    """Linear warmup to `base`, then cosine decay to `base * final_frac`.

    `step` is 0-based. The progress clamp matters: past `total` an unclamped
    cosine turns back upward and would return the LR to half of peak.
    """
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    p = min((step - warmup) / max(1, total - warmup), 1.0)
    return base * (final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * p)))


class _EMA:
    """Exponential moving average of the model parameters.

    Bias-corrected: the coupling subnets are zero-initialized, so an uncorrected
    average stays dominated by the near-identity init for thousands of steps and
    would lose every best-checkpoint race against the raw weights early on.

    Parameters only — verified that this model has zero buffers, so that is the
    whole state. (Which is also why `permute_soft` should stay True:
    `PermuteRandom` would add an int64 permutation buffer.)
    """

    def __init__(self, decay):
        self.decay = decay
        self.shadow = None
        self.n = 0

    def update(self, model):
        params = [p.detach() for p in model.parameters()]
        if self.shadow is None:                       # first call: after step 1
            self.shadow = [p.clone() for p in params]
            self.n = 1
            return
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        torch._foreach_lerp_(self.shadow, params, 1.0 - d)

    @contextlib.contextmanager
    def swap_into(self, model):
        """Context manager: EMA weights live in `model` for the duration.

        Only ever copies in place. Rebinding (`p.data = t`) would invalidate
        dynamo's guards on the compiled step. The restore is in a `finally` so a
        failed evaluation cannot leave EMA weights in the training model.
        """
        params = list(model.parameters())
        backup = [p.detach().clone() for p in params]
        with torch.no_grad():
            torch._foreach_copy_(params, self.shadow)
        try:
            yield
        finally:
            with torch.no_grad():
                torch._foreach_copy_(params, backup)

    def state(self, model):
        return {"n": self.n,
                "shadow": {n: t.cpu() for (n, _), t in
                           zip(model.named_parameters(), self.shadow)}}

    def load(self, model, state):
        if not state:
            return
        sh = state["shadow"]
        self.shadow = [sh[n].to(p.device, p.dtype) for n, p in model.named_parameters()]
        self.n = state.get("n", 0)


def _prepare_loaders(cfg, device, reuse_stats):
    """Memmap the atlas, settle the frozen stats, build the two loaders.

    The train/val split is a **contiguous tail**, not a random permutation: the
    rows are Latin-hypercube ordered and therefore exchangeable (see the
    loader.SlabLoader docstring), so the last val_frac of the file is already a
    uniform random subset. A permutation would cost an 0.8 GB index array plus
    full-array gathers at 1e8, for no statistical gain.

    Returns the stats rather than writing them. Writing here — before the model
    is built or a resume has loaded — is how a failed launch used to leave a
    model dir holding NEW stats against OLD weights, which infer.py and
    run_surrogate.py would then run without complaint and quietly get wrong.

    On resume the stats are read back, not refit. Refitting is not idempotent
    across a changed `val_frac`: the fit range is the train split, so moving the
    split moves the exact mean/std columns, and a resumed model would silently
    see differently-scaled inputs.
    """
    print(f"[train] opening atlas at {cfg.data_dir}")
    view = open_atlas(cfg.data_dir)
    n_val = max(1, int(round(cfg.val_frac * view.n)))
    n_tr = view.n - n_val
    if n_tr <= 0:
        raise ValueError(f"val_frac {cfg.val_frac} leaves no training rows")
    print(f"[train] {view.n} rows, {view.feat_dim} features, {view.param_dim} params"
          f"{'  [noinh]' if view.noinh else ''}")

    if reuse_stats:
        stats = load_stats(cfg.model_dir)
        fstats, pstats = stats["feature_stats"], stats["param_stats"]
        if stats["feature_names"] != list(view.fnames):
            raise ValueError(
                f"stats.json feature order {stats['feature_names']} does not match "
                f"the atlas at {cfg.data_dir} ({list(view.fnames)}) — the "
                f"checkpoint and this atlas are not compatible")
        print(f"[train] stats: reusing {cfg.model_dir}/stats.json (resume)")
        write_stats = None
    else:
        # Frozen constants, fit on the TRAIN range only. Quantile-based parts come
        # from a subsample; the mean/std fallback is computed exactly (see
        # fit_feature_stats_streaming — half this atlas's features land on it).
        fit_rows = min(n_tr, max(1, cfg.stat_samples))
        fstats, n_sub, exact_cols = fit_feature_stats_streaming(view, 0, n_tr, fit_rows)
        _, s_params = sample_rows(view, fit_rows, 0, n_tr)
        pstats = fit_param_stats(s_params)
        del s_params

        def write_stats():
            save_stats(cfg.model_dir, fstats, pstats, view.fnames, view.pnames,
                       all_fnames=view.all_fnames, noinh=view.noinh,
                       fit_rows=n_sub, n_rows=n_tr, exact_cols=exact_cols)
            print(f"[train] stats: median/MAD on {n_sub} subsampled rows, "
                  f"mean/std exact for {len(exact_cols)} feature(s) -> "
                  f"{cfg.model_dir}/stats.json")

    common = dict(fstats=fstats, pstats=pstats, batch_size=cfg.batch_size,
                  slab_rows=cfg.slab_rows, device=device, seed=cfg.seed)
    # jitter is a TRAIN-side augmentation only: val_nll must stay measured on the
    # clean conditioning distribution or the arms of a jitter sweep are each
    # scored against a different validation objective and cannot be compared.
    train_loader = SlabLoader(view, (0, n_tr), shuffle=True,
                              jitter=cfg.cond_jitter, **common)
    val_loader = SlabLoader(view, (n_tr, view.n), shuffle=False, **common)
    print(f"[train] split: {n_tr} train / {n_val} val  | cond_dim "
          f"{2 * view.feat_dim} param_dim {view.param_dim} | "
          f"{train_loader.n_batches} batches/epoch, slab {cfg.slab_rows}")
    return view, train_loader, val_loader, write_stats


@torch.no_grad()
def _evaluate(model, loader):
    """Mean val NLL. Deliberately runs the EAGER nll: evaluation is well under 1%
    of step time at these val sizes, and routing it through the compiled callable
    would add eval-shaped graphs to a recompile budget that silently falls back
    to eager once exceeded."""
    model.eval()
    total, count = 0.0, 0
    for cond, theta in loader:
        nll = model.nll(theta, cond)
        total += float(nll.sum())
        count += nll.shape[0]
    model.train()
    return total / max(count, 1)


def train(cfg):
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    resuming = bool(cfg.resume and has_checkpoint(cfg.model_dir))
    view, train_loader, val_loader, write_stats = _prepare_loaders(
        cfg, device, reuse_stats=resuming)
    cond_dim, param_dim = 2 * view.feat_dim, view.param_dim
    model = CINN(cfg, cond_dim=cond_dim, param_dim=param_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_spline = sum(1 for m in model.inn.module_list
                   if type(m).__name__ == "RQSplineCoupling")
    print(f"[train] model built: {n_params/1e6:.2f}M params  "
          f"({cfg.n_coupling_blocks} coupling blocks, {n_spline} spline, "
          f"spline_every={cfg.spline_every}, bins={cfg.spline_bins})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    n_batches = train_loader.n_batches
    total_steps = cfg.max_steps if cfg.max_steps else cfg.epochs * n_batches
    warmup = min(cfg.warmup_steps, max(1, total_steps // 10))
    sched = {"warmup": warmup, "total": total_steps, "base_lr": cfg.lr}

    ema = _EMA(cfg.ema_decay) if cfg.use_ema else None
    start_epoch, global_step = 0, 0
    best_val, since_best = float("inf"), 0

    if resuming:
        _, _, _, ckpt = load_checkpoint(cfg.model_dir, device=device, which="last")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optim"):
            optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)      # absent in older checkpoints
        # NaN guard: mid-epoch saves record last_val_nll, which is NaN for the
        # whole first epoch. Seeding best_val with it would make every later
        # `x < best_val - 1e-4` false forever, so best_params.bin would never be
        # written again for the rest of the run — silently.
        prev_best = ckpt.get("best_val")
        if prev_best is not None and math.isfinite(prev_best):
            best_val = prev_best
        since_best = ckpt.get("since_best", 0) or 0
        if ema is not None:
            ema.load(model, ckpt.get("ema"))
        prev_sched = ckpt.get("sched")
        if prev_sched and prev_sched != sched:
            print(f"[train] WARNING: LR horizon differs from the checkpoint's "
                  f"({prev_sched} -> {sched}); keeping the checkpoint's so the "
                  f"schedule stays continuous. Pass matching flags to override.")
            sched = prev_sched
        print(f"[train] resumed from epoch {ckpt['epoch']} step {global_step} "
              f"(val_nll {ckpt['val_nll']:.4f}, best {best_val:.4f})")
    else:
        # A fresh run into a dir that already holds a best from an earlier run
        # would leave that stale file to be picked up by load_checkpoint("best").
        stale = os.path.join(cfg.model_dir, _BEST)
        if os.path.exists(stale):
            os.remove(stale)
            print(f"[train] removed stale {_BEST} from a previous run")

    if write_stats is not None:      # only now that the model/resume has succeeded
        write_stats()

    train_loader.set_epoch(start_epoch)

    nll_eager = model.nll
    nll_step = torch.compile(model.nll) if cfg.compile else model.nll
    if cfg.compile:
        print("[train] torch.compile enabled (iteration 0 runs eager so ActNorm "
              "initializes outside the compiled region)")

    print(f"[train] schedule: warmup {sched['warmup']} -> cosine to "
          f"{cfg.lr_final_frac:g}*lr over {sched['total']} steps | "
          f"eval every {cfg.eval_every_steps or n_batches} steps | "
          f"ema {'on (decay %g)' % cfg.ema_decay if ema is not None else 'off'}")

    last_val_nll = float("nan")
    last_eval_step = -1
    stop = False

    def _maybe_eval(epoch):
        """Evaluate raw (and EMA) weights, update best, write best_params.bin.

        The single home for best-tracking and patience — see the module docstring
        on why there is no epoch-end duplicate.
        """
        nonlocal best_val, since_best, last_val_nll, last_eval_step, stop
        raw = _evaluate(model, val_loader)
        chosen, tag, sd = raw, "raw", None
        if ema is not None and ema.shadow is not None:
            with ema.swap_into(model):
                ema_nll = _evaluate(model, val_loader)
                if ema_nll < raw:
                    chosen, tag = ema_nll, "ema"
                    sd = copy.deepcopy(model.state_dict())
            print(f"[train] eval @ {global_step}: raw {raw:.4f} | ema "
                  f"{ema_nll:.4f} -> {tag}", flush=True)
        else:
            print(f"[train] eval @ {global_step}: raw {raw:.4f}", flush=True)

        last_val_nll = chosen
        last_eval_step = global_step
        if chosen < best_val - 1e-4:
            best_val, since_best = chosen, 0
            save_checkpoint(cfg.model_dir, model, None, epoch, chosen, cfg,
                            global_step=global_step, filename=_BEST,
                            state_dict=sd, weights=tag, best_val=best_val,
                            since_best=since_best, sched=sched)
            print(f"[train]   new best {best_val:.4f} ({tag}) -> {_BEST}")
        else:
            since_best += 1
            if cfg.early_stop_patience and since_best >= cfg.early_stop_patience:
                print(f"[train] early stop: no val_nll improvement in "
                      f"{since_best} evals (best {best_val:.4f})")
                stop = True

    def _save_last(epoch):
        save_checkpoint(cfg.model_dir, model, optimizer, epoch, last_val_nll, cfg,
                        global_step=global_step, filename=_WEIGHTS,
                        weights="raw", best_val=best_val, since_best=since_best,
                        ema=ema.state(model) if (ema and ema.shadow) else None,
                        sched=sched)

    eval_every = cfg.eval_every_steps
    epoch = start_epoch
    # --max-steps is the authoritative budget when set, so the epoch loop must
    # not also terminate the run: 600k steps at batch 32768 is ~197 epochs,
    # close enough to the default 200 that an unrelated batch-size change could
    # silently end training early. --epochs still bounds a run with max_steps=0.
    epoch_iter = (itertools.count(start_epoch) if cfg.max_steps
                  else range(start_epoch, cfg.epochs))
    for epoch in epoch_iter:
        model.train()
        t0 = time.time()
        for i, (cond_b, theta_b) in enumerate(train_loader):
            lr = lr_at(global_step, sched["base_lr"], sched["warmup"],
                       sched["total"], cfg.lr_final_frac)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            # Iteration 0 of a fresh run goes through eager: that is the forward
            # that performs ActNorm's in-place data-dependent init, and doing it
            # inside the compiled region costs a guard failure and a full
            # recompile at best.
            fn = nll_eager if (global_step == 0 and not resuming) else nll_step
            loss = fn(theta_b, cond_b).mean()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            global_step += 1

            if ema is not None:          # after the step, so the first shadow is
                ema.update(model)        # taken from an ActNorm-initialized model

            if i % cfg.log_every == 0:
                print(f"[train] epoch {epoch:04d} step {i:05d}/{n_batches} "
                      f"(global {global_step}) lr {lr:.3e} loss {loss.item():.4f} "
                      f"val_nll {last_val_nll:.4f}", flush=True)

            if eval_every and global_step % eval_every == 0:
                _maybe_eval(epoch)

            if cfg.ckpt_every_steps and global_step % cfg.ckpt_every_steps == 0:
                _save_last(epoch)
                if cfg.keep_snapshots:
                    # _save_last overwrites model_params.bin, so a run's
                    # intermediate states are normally unrecoverable -- which is
                    # why locating where the HR collapse emerges (§17.4) needed a
                    # fresh run. Keep a step-tagged copy: a plain file copy after
                    # the real save, so it cannot affect training.
                    import shutil
                    src = os.path.join(cfg.model_dir, "model_params.bin")
                    dst = os.path.join(cfg.model_dir, f"snap_{global_step}.bin")
                    try:
                        shutil.copyfile(src, dst)
                        print(f"[train] snapshot -> {os.path.basename(dst)}")
                    except OSError as exc:
                        print(f"[train] snapshot failed ({exc!r}); continuing")

            if cfg.max_steps and global_step >= cfg.max_steps:
                print(f"[train] reached max_steps {cfg.max_steps}")
                stop = True
            if stop:
                break

        if not eval_every:                       # epoch-cadence evaluation
            _maybe_eval(epoch)
        dt = time.time() - t0
        rows = train_loader.n
        print(f"[train] epoch {epoch:04d} done — val_nll {last_val_nll:.4f} "
              f"({dt:.1f}s, {rows/dt/1e6:.2f} M rows/s)")
        _save_last(epoch)

        if stop:
            break

    if last_eval_step != global_step:             # final state was never scored
        _maybe_eval(epoch)
        _save_last(epoch)

    print(f"[train] finished at step {global_step}. best val_nll {best_val:.4f} "
          f"-> {cfg.model_dir}/{_BEST}")
