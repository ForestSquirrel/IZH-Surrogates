"""
loader.py — slab-streaming batch source for atlases too large to hold in memory.

The old training path held the entire dataset as one GPU-resident tensor, which
works at 1e6 rows and dies at 1e8 (16.8 GB of conditioning matrix against 15.9 GB
of VRAM, after ~60 GB of host peak just to build it). This module replaces it
with a stream whose cost is flat in N:

    for each epoch:
        for each slab (1M rows) in a random slab order:
            read the slab from the memmap        (host, on a prefetch thread)
            upload it and build the condition    (device, ~4 ms)
            yield batches in a random within-slab order

Host RAM and VRAM are then bounded by one slab plus one prefetch buffer,
regardless of whether the atlas has 1e6 rows or 1e9.

Why slabs are a legitimate shuffle
----------------------------------
The atlas rows are **exchangeable by construction**: ``src/dataset.py::_sample_lhs``
draws parameters with ``scipy.stats.qmc.LatinHypercube``, which permutes each
dimension independently, so a row's position carries no information about its
contents. That licenses three things which would be wrong on ordered data:

* a contiguous slab is an unbiased sample of the atlas,
* a contiguous tail is a valid validation split (see train.py), and
* shuffling slab order + shuffling within a slab is equivalent to a global
  shuffle, because the only structure a global shuffle would break does not exist.

If a future atlas is ever generated in a correlated order (a grid sweep, or rows
appended from separate runs), this assumption breaks and the loader must be given
a real global shuffle instead. ``sampling`` in izh_meta.json records which
generator was used; SlabLoader warns when it is not "lhs".
"""

import queue
import threading

import numpy as np
import torch

from data import build_condition_torch, standardize_params_torch


class SlabLoader:
    """Iterable over (cond, theta) batches for one row range of an atlas.

    Parameters
    ----------
    view        : data.AtlasView
    row_range   : (start, stop) half-open row range this loader owns
    fstats/pstats : frozen standardization constants (dicts from data.fit_*_stats)
    batch_size  : optimizer batch
    slab_rows   : rows per slab; the upload and shuffle unit
    device      : torch device for the resident slab
    seed        : base seed; the shuffle is reseeded per epoch as seed + epoch
    shuffle     : False for validation (deterministic sequential pass)
    jitter      : stddev of Gaussian noise added to the STANDARDIZED value
                  channels of the conditioning vector, in atlas sigma. 0 disables.

    Why jitter exists
    -----------------
    The atlas manifold is 4-D embedded in 19-D feature space, so a conditional
    flow trained by MLE is only ever shown a measure-zero sheet and its behaviour
    off that sheet is unconstrained by the objective. Measured, the consequence
    is severe: displacing a conditioning vector by d^2 = 0.036 --
    0.19 sigma RMS per feature, and within 0.19 sigma of a real atlas cell --
    moves 96.6% of the posterior mass outside the prior box, and ~25% of random
    0.2-sigma directions do the same on ordinary atlas cells, on every checkpoint.
    Real targets are off-manifold by construction, so this is the mechanism behind
    the posterior collapse.

    Jitter forces the flow to be smooth in a neighbourhood of the manifold rather
    than interpolating it arbitrarily. It is isotropic because §16.1 measured the
    fragile directions to be low-rank at each cell (participation ratio 1.65 of 4)
    but essentially cell-specific (|cos| between cells' top singular vectors 0.227
    against a chance level of 0.19), with the linear Jacobian predicting the
    actual blow-up at only r = 0.20 -- so there is no global subspace to shape it
    along, and no basis for a curvature-aware scheme.

    Applied per BATCH rather than per slab: a slab is reused for ~30 optimizer
    steps, so jittering at slab-build time would show each row a single fixed
    perturbation instead of a fresh one per visit. The cost is one randn of the
    batch's value block.

    Masked slots are left at exactly 0. `build_condition` zeroes undefined values
    and flags them in the mask channel; adding noise there would inject signal
    into slots that carry none and teach the flow to read a channel that is
    definitionally empty.
    """

    def __init__(self, view, row_range, fstats, pstats, batch_size,
                 slab_rows=1_000_000, device="cuda", seed=0, shuffle=True,
                 jitter=0.0):
        self.view = view
        self.start, self.stop = int(row_range[0]), int(row_range[1])
        self.n = self.stop - self.start
        if self.n <= 0:
            raise ValueError(f"empty row range [{self.start}, {self.stop})")
        self.batch_size = int(batch_size)
        self.slab_rows = max(int(slab_rows), self.batch_size)
        self.device = device
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.jitter = float(jitter)
        self.epoch = 0

        if shuffle and view.meta.get("sampling", "lhs") != "lhs":
            print(f"[loader] WARNING: meta sampling is "
                  f"{view.meta.get('sampling')!r}, not 'lhs' — slab shuffling "
                  f"assumes exchangeable row order (see module docstring)")

        t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
        self._fcenter, self._fscale = t(fstats["center"]), t(fstats["scale"])
        self._pcenter, self._pscale = t(pstats["center"]), t(pstats["scale"])

        self._bounds = [(s, min(s + self.slab_rows, self.stop))
                        for s in range(self.start, self.stop, self.slab_rows)]
        # batches per epoch: partial slabs yield a partial final batch
        self.n_batches = sum((b - a + self.batch_size - 1) // self.batch_size
                             for a, b in self._bounds)

    def __len__(self):
        return self.n_batches

    # -- host-side prefetch -------------------------------------------------
    def _reader(self, order, out, stop):
        """Read slabs on a worker thread so host I/O overlaps GPU compute.

        At 1e8 this hides ~14 s of reads per epoch behind ~150 s of compute; at
        1e9 it hides ~137 s behind ~1,480 s, so the stream stays compute-bound.

        Every put is bounded and re-checks `stop`, because the queue holds one
        item: when the consumer leaves early (--max-steps mid-epoch) this thread
        is otherwise parked in put() forever and the generator's join stalls for
        the whole timeout.
        """
        def offer(item):
            while not stop.is_set():
                try:
                    out.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        try:
            for i in order:
                if stop.is_set():
                    return
                a, b = self._bounds[i]
                item = self.view.slab(a, b)
                self.view.release()
                if not offer(item):
                    return
        except Exception as exc:                     # surface, don't hang
            offer(exc)
        else:
            offer(None)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = (rng.permutation(len(self._bounds)) if self.shuffle
                 else np.arange(len(self._bounds)))

        q = queue.Queue(maxsize=1)
        stop = threading.Event()
        thread = threading.Thread(target=self._reader, args=(order, q, stop),
                                  daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                h_feats, h_params = item

                # slab -> device, then build the conditioning here rather than
                # ever materializing it host-side for the whole atlas
                d_feats = torch.from_numpy(h_feats).to(self.device, non_blocking=True)
                d_params = torch.from_numpy(h_params).to(self.device, non_blocking=True)
                cond = build_condition_torch(d_feats, self._fcenter, self._fscale)
                theta = standardize_params_torch(d_params, self._pcenter, self._pscale)
                del d_feats, d_params

                m = cond.shape[0]
                idx = (torch.randperm(m, device=self.device) if self.shuffle
                       else torch.arange(m, device=self.device))
                F = cond.shape[1] // 2
                for i in range(0, m, self.batch_size):
                    sel = idx[i:i + self.batch_size]
                    c = cond[sel]
                    if self.jitter > 0.0:
                        # noise on the value half only, gated by the mask half so
                        # undefined slots stay exactly 0
                        c = c.clone()
                        c[:, :F] += (self.jitter * torch.randn_like(c[:, :F])
                                     * c[:, F:])
                    yield c, theta[sel]
                del cond, theta
        finally:
            # advance the shuffle seed even on an early exit, and unblock the
            # reader before joining so the exit is prompt rather than timing out
            self.epoch += 1
            stop.set()
            thread.join(timeout=10)

    def set_epoch(self, epoch):
        """Pin the shuffle seed (used on resume so a run is reproducible)."""
        self.epoch = int(epoch)
