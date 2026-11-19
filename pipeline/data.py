"""
data.py — dataset ingestion, robust-zscore standardization, and masking for the
Izhikevich cINN.

Binary format is identical to the feature generator (src/dataset.py):

    [magic(4B)] [n(int64 LE)] [k(uint32 LE)] [n*k float32 row-major]

The conditioning vector fed to the embedder is the concatenation of, for each of
the 22 features, a (standardized value, mask bit) pair -> 44 dims. Undefined
(sentinel) slots are imputed to a neutral 0.0 and flagged by their mask bit; no
hand-derived branch/gate features (those go stale on a param-bounds sweep — the
embedder rediscovers mask correlations itself).

Standardization constants (per-feature and per-parameter median/MAD) are fit
ONCE on the training split and frozen into model_dir/stats.json. Inference reuses
these exact constants and never refits on a single row.

Nothing here is ever fully resident. ``open_atlas`` memmaps the binaries and
``loader.SlabLoader`` streams contiguous slabs, so an atlas of any size costs the
same host RAM. Two consequences worth knowing:

* Everything is float32 end to end. The old float64 ingestion needed 26 GB just
  to open a 1e8-row atlas (8.8 GB read + a 17.6 GB upcast, both live during the
  cast) and ~60 GB by the time the conditioning matrix existed.
* The (N, 2F) conditioning matrix is never materialized. It is a pure elementwise
  function of the raw features, so ``build_condition_torch`` rebuilds it per slab
  on the GPU instead — that array alone was 30 GB at 1e8.
"""

import os
import json
import struct
import numpy as np

from cinn_config import NAN_SENTINEL, SENTINEL_CUT, INHIBITION_FEATURES


# ============================================================
# Raw binary I/O  (copied from the feature pipeline's on-disk contract)
# ============================================================

def read_bin(path, magic):
    """Read a ``[magic][n int64][k uint32][n*k float32]`` binary into (n, k) float32."""
    with open(path, "rb") as f:
        got = f.read(4)
        if got != magic:
            raise ValueError(f"{path}: bad magic {got!r} (expected {magic!r})")
        n, k = struct.unpack("<qI", f.read(12))
        arr = np.fromfile(f, dtype=np.float32)
    if arr.size != n * k:
        raise ValueError(f"{path}: expected {n*k} floats, got {arr.size}")
    return arr.reshape(n, k)


def select_features(feats, all_names, keep_names):
    """
    Select the columns of `feats` (aligned to all_names) named in keep_names,
    returning them in keep_names order. The one place feature-set reduction
    happens — training (drop inhibition off the meta marker) and inference
    (match the trained model's active set) both route through here.
    """
    idx = [all_names.index(n) for n in keep_names]
    return np.asarray(feats)[:, idx]


def active_feature_names(all_names, noinh, drop=()):
    """The feature names kept for training/inference.

    Three exclusions, all BY NAME so that a schema change cannot silently re-map
    an ablation (§4.6):

      * `_reserved_*` — anonymous on-disk slack. v3 ships 32 columns with 18
        active so that a later schema change is not a migration; the surplus is
        permanently sentinel and must never reach the conditioner, or it puts
        14 dead channels into the mask topology §5 is trying to keep readable.
      * inhibition features, when noinh is set (the v2 path, unchanged)
      * anything named in `drop` — the runtime ablation hook. Passing a name
        here is how a column is removed from training WITHOUT rebuilding the
        atlas, which is the whole reason a near-binary column like δ* is a cheap
        bet rather than a commitment.
    """
    names = [n for n in all_names if not n.startswith("_reserved")]
    if noinh:
        names = [n for n in names if n not in INHIBITION_FEATURES]
    if drop:
        drop = set(drop)
        names = [n for n in names if n not in drop]
    return names


def _open_memmap(path, magic):
    """Validate a dataset binary's header and memmap its payload read-only.

    Same contract as read_bin, but returns an (n, k) float32 memmap instead of
    reading the file — opening a 8.8 GB atlas costs nothing.
    """
    with open(path, "rb") as f:
        got = f.read(4)
        if got != magic:
            raise ValueError(f"{path}: bad magic {got!r} (expected {magic!r})")
        n, k = struct.unpack("<qI", f.read(12))
    expect = 16 + n * k * 4
    actual = os.path.getsize(path)
    if actual != expect:
        raise ValueError(f"{path}: expected {expect} bytes for {n}x{k} float32, "
                         f"got {actual}")
    return np.memmap(path, dtype=np.float32, mode="r", offset=16, shape=(n, k))


class AtlasView:
    """A memmapped atlas directory. Rows are fetched a slab at a time.

    Attributes
    ----------
    n            : int        — row count
    feat_dim     : int        — active feature count F (after the --noinh drop)
    fnames       : list[str]  — active feature names, in conditioning order
    all_fnames   : list[str]  — the full on-disk schema (pre-drop)
    keep         : list[int]  — active column indices into the on-disk schema
    noinh        : bool

    The active-column reduction is carried as ``keep`` and applied per slab.
    Slicing the memmap itself (``feats[:, keep]``) would materialize the whole
    array — 7.6 GB at 1e8 — which is exactly what this class exists to avoid.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._feats = _open_memmap(os.path.join(data_dir, "izh_features.bin"), b"IZHF")
        self._params = _open_memmap(os.path.join(data_dir, "izh_params.bin"), b"IZHP")
        with open(os.path.join(data_dir, "izh_meta.json")) as f:
            self.meta = json.load(f)

        self.all_fnames = self.meta["feature_names"]
        self.pnames = self.meta.get("param_names", ["a", "b", "c", "d"])
        self.lo = np.asarray(self.meta.get("param_lo", []), dtype=np.float64)
        self.hi = np.asarray(self.meta.get("param_hi", []), dtype=np.float64)
        if self._feats.shape[1] != len(self.all_fnames):
            raise ValueError(f"feature_names ({len(self.all_fnames)}) != feature cols "
                             f"({self._feats.shape[1]})")
        if self._params.shape[0] != self._feats.shape[0]:
            raise ValueError("params / feats row mismatch")

        self.noinh = bool(self.meta.get("noinh", False))
        self.fnames = active_feature_names(self.all_fnames, self.noinh)
        self.keep = [self.all_fnames.index(nm) for nm in self.fnames]
        self.n = int(self._feats.shape[0])
        self.feat_dim = len(self.keep)
        self.param_dim = int(self._params.shape[1])
        if self.noinh:
            dropped = [n for n in self.all_fnames if n not in self.fnames]
            print(f"[data] noinh atlas: dropped {len(dropped)} inhibition feature(s) "
                  f"{dropped} -> {self.feat_dim} features")

    def slab(self, start, stop):
        """Rows [start, stop) as (feats (m, F) float32, params (m, P) float32).

        Both are owned, writable copies. This must not hand back a memmap view:
        release() below madvises the mapping away, and a plain slice would still
        be pointing into it (and would make torch.from_numpy warn about a
        non-writable buffer). Column selection already copies; the params slice
        would not, hence the explicit copy.
        """
        f = np.array(self._feats[start:stop, self.keep], dtype=np.float32, copy=True)
        p = np.array(self._params[start:stop], dtype=np.float32, copy=True)
        return f, p

    def release(self):
        """Drop consumed memmap pages.

        Mapped pages count against RSS, so a full sweep of a 8.8 GB atlas would
        otherwise leave ~10 GB resident even though nothing is retained. The
        pages are clean and read-only, so dropping them is free.
        """
        import mmap
        if not hasattr(mmap, "MADV_DONTNEED"):
            return
        for a in (self._feats, self._params):
            m = getattr(a, "_mmap", None)
            if m is not None:
                try:
                    m.madvise(mmap.MADV_DONTNEED)
                except (OSError, ValueError):
                    pass


def open_atlas(data_dir):
    """Memmap the atlas in data_dir. Nothing is read until .slab() is called."""
    return AtlasView(data_dir)


def sample_rows(view, n_rows, lo_row=0, hi_row=None, block=8192):
    """Subsample rows from [lo_row, hi_row) as evenly spaced contiguous blocks.

    Used to fit the standardization constants without a full pass. Contiguous
    blocks rather than a stride because a row is only ~88 bytes: ``feats[::step]``
    would touch every page and read the entire atlas just to sample it. Evenly
    spaced blocks are unbiased here because the rows are in Latin-hypercube order
    (src/dataset.py::_sample_lhs), i.e. exchangeable — see loader.SlabLoader.

    Returns (feats (m, F) float32, params (m, P) float32).
    """
    hi_row = view.n if hi_row is None else hi_row
    span = hi_row - lo_row
    if span <= 0:
        raise ValueError(f"empty row range [{lo_row}, {hi_row})")
    rows = min(span, max(1, n_rows))
    blk = min(span, block)
    nblk = max(1, rows // blk)
    if nblk == 1:
        starts = [lo_row]
    else:
        starts = np.linspace(lo_row, hi_row - blk, nblk).astype(np.int64)
    fs, ps = [], []
    for s in starts:
        f, p = view.slab(int(s), int(s) + blk)
        fs.append(f)
        ps.append(p)
    view.release()
    return np.concatenate(fs), np.concatenate(ps)


# ============================================================
# Robust z-score:  (x - median) / (1.4826 * MAD),  mean/std fallback
# ============================================================

_MAD_K = 1.4826   # MAD -> sigma scale for a normal distribution


def _robust_center_scale(x, mask=None):
    """
    Fit (center, scale) for one column. If mask is given, statistics use only
    mask==True entries. Falls back to mean/std when MAD collapses to ~0, and to
    scale=1 when even std collapses (a constant column).
    """
    v = x[mask] if mask is not None else x
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = _MAD_K * mad
    if scale < 1e-12:
        # MAD collapsed (mostly-constant / heavily-tied column) -> mean/std
        med = float(np.mean(v))
        scale = float(np.std(v))
        if scale < 1e-12:
            scale = 1.0
    return med, scale


def fit_feature_stats(feats):
    """
    Fit per-feature robust standardization over the masked (defined) entries.

    Returns dict with float lists:
        center : (F,)  per-feature center (median / mean fallback)
        scale  : (F,)  per-feature scale  (1.4826*MAD / std / 1.0 fallback)
    Sentinel entries never enter the statistics.
    """
    F = feats.shape[1]
    center = np.zeros(F)
    scale = np.ones(F)
    for j in range(F):
        mask = feats[:, j] > SENTINEL_CUT
        center[j], scale[j] = _robust_center_scale(feats[:, j], mask)
    return {"center": center.tolist(), "scale": scale.tolist()}


def fit_feature_stats_streaming(view, lo_row, hi_row, subsample_rows,
                                chunk_rows=4_000_000, verbose=True):
    """Fit feature standardization over rows [lo_row, hi_row) without loading them.

    Same estimator as fit_feature_stats, but split by which half of it is
    statistically fragile:

    * **median / MAD** come from a subsample. They are quantiles, so a few
      million rows pin them tightly (measured: <3% drift between disjoint 5M
      subsamples of the 1e8 atlas).
    * **the mean/std fallback** is computed EXACTLY, in one streaming pass. On
      this atlas 10 of 19 features have MAD identically 0 — they are dominated by
      point masses (delta_star is ~78% zeros, p takes 9 values) — so they all land
      on this branch, and std of a heavy-tailed feature is driven by rare extremes
      rather than by the bulk. Subsampling it gave 14% run-to-run wobble on lam
      and chi; the exact pass costs ~15 s at 1e8 and removes that entirely.

    Returns (stats_dict, n_subsampled, exact_cols).
    """
    s_feats, _ = sample_rows(view, subsample_rows, lo_row, hi_row)
    F = view.feat_dim
    center = np.zeros(F)
    scale = np.ones(F)
    need = []
    for j in range(F):
        m = s_feats[:, j] > SENTINEL_CUT
        v = s_feats[m, j].astype(np.float64)
        if v.size == 0:
            continue
        med = float(np.median(v))
        sc = _MAD_K * float(np.median(np.abs(v - med)))
        center[j] = med
        if sc < 1e-12:
            need.append(j)          # MAD collapsed -> exact mean/std below
        else:
            scale[j] = sc
    n_sub = s_feats.shape[0]
    del s_feats

    if need:
        if verbose:
            print(f"[data] MAD collapsed for {len(need)} feature(s) "
                  f"{[view.fnames[j] for j in need]}; fitting mean/std exactly "
                  f"over {hi_row - lo_row} rows")
        cnt = np.zeros(len(need), np.float64)
        ssum = np.zeros(len(need), np.float64)
        ssq = np.zeros(len(need), np.float64)
        for a in range(lo_row, hi_row, chunk_rows):
            b = min(a + chunk_rows, hi_row)
            f, _ = view.slab(a, b)
            view.release()
            for i, j in enumerate(need):
                col = f[:, j]
                x = col[col > SENTINEL_CUT].astype(np.float64)
                cnt[i] += x.size
                ssum[i] += x.sum()
                ssq[i] += (x * x).sum()
        for i, j in enumerate(need):
            if cnt[i] == 0:
                continue
            mean = ssum[i] / cnt[i]
            std = float(np.sqrt(max(ssq[i] / cnt[i] - mean * mean, 0.0)))
            center[j] = mean
            scale[j] = std if std >= 1e-12 else 1.0

    return ({"center": center.tolist(), "scale": scale.tolist()}, n_sub,
            [view.fnames[j] for j in need])


def fit_param_stats(params):
    """Fit per-parameter robust standardization (no sentinels in params)."""
    P = params.shape[1]
    center = np.zeros(P)
    scale = np.ones(P)
    for j in range(P):
        center[j], scale[j] = _robust_center_scale(params[:, j], None)
    return {"center": center.tolist(), "scale": scale.tolist()}


# ============================================================
# Apply / build conditioning vectors and standardized params
# ============================================================

def build_condition(feats, fstats):
    """
    Turn raw features (N, F) into the (N, 2F) conditioning matrix:
    columns [0:F] = standardized values (0.0 where undefined),
    columns [F:2F] = mask bits (1.0 defined, 0.0 undefined).

    Uses the frozen fstats center/scale — no refit here. float32 out: this is the
    single-row path (infer.py, run_surrogate.py); training uses the torch variant
    below, which never builds this array for the whole atlas.
    """
    feats = np.asarray(feats, dtype=np.float32)
    center = np.asarray(fstats["center"], dtype=np.float32)
    scale = np.asarray(fstats["scale"], dtype=np.float32)
    mask = (feats > SENTINEL_CUT)
    value = (feats - center) / scale
    value = np.where(mask, value, np.float32(0.0))
    return np.concatenate([value, mask.astype(np.float32)], axis=1)


def build_condition_torch(feats_t, center_t, scale_t):
    """On-device build_condition for a resident slab: (m, F) -> (m, 2F).

    Identical arithmetic to build_condition, kept on the GPU so the (N, 2F)
    conditioning matrix never exists on the host (30 GB at 1e8 rows). Called once
    per slab, not per batch, so its cost is amortized across ~15 optimizer steps.
    """
    import torch
    mask = feats_t > SENTINEL_CUT
    value = torch.where(mask, (feats_t - center_t) / scale_t,
                        torch.zeros((), dtype=feats_t.dtype, device=feats_t.device))
    return torch.cat([value, mask.to(feats_t.dtype)], dim=1)


def standardize_params(params, pstats):
    """(a,b,c,d) -> standardized, using frozen pstats."""
    center = np.asarray(pstats["center"])
    scale = np.asarray(pstats["scale"])
    return (np.asarray(params, dtype=np.float64) - center) / scale


def standardize_params_torch(params_t, center_t, scale_t):
    """On-device standardize_params for a resident slab."""
    return (params_t - center_t) / scale_t


def unstandardize_params(theta_std, pstats):
    """standardized theta -> (a,b,c,d) units."""
    center = np.asarray(pstats["center"])
    scale = np.asarray(pstats["scale"])
    return np.asarray(theta_std, dtype=np.float64) * scale + center


# ============================================================
# stats.json  (self-contained checkpoint sidecar)
# ============================================================

def save_stats(model_dir, fstats, pstats, fnames, pnames,
               all_fnames=None, noinh=False, fit_rows=None, n_rows=None,
               exact_cols=None):
    os.makedirs(model_dir, exist_ok=True)
    stats = {
        "feature_stats": fstats,
        "param_stats": pstats,
        "feature_names": list(fnames),                       # active set (cond order)
        "all_feature_names": list(all_fnames if all_fnames is not None else fnames),
        "noinh": bool(noinh),                                # inhibition features dropped?
        "param_names": list(pnames),
        "nan_sentinel": NAN_SENTINEL,
        "sentinel_cut": SENTINEL_CUT,
        # how the constants above were fit: median/MAD over `fit_rows` rows
        # subsampled from a train split of `n_rows`, except `exact_mean_std_cols`
        # whose MAD collapsed and whose mean/std were fit exactly over every row
        # (see data.fit_feature_stats_streaming)
        "fit_rows": int(fit_rows) if fit_rows is not None else None,
        "train_rows": int(n_rows) if n_rows is not None else None,
        "exact_mean_std_cols": list(exact_cols) if exact_cols is not None else [],
    }
    with open(os.path.join(model_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_stats(model_dir):
    with open(os.path.join(model_dir, "stats.json")) as f:
        return json.load(f)
