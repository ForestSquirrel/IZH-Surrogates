"""
datasets.py -- loaders for the downloaded datasets in ../datasets/ (not in git; see the top-level README, section 9).

mnist(kind) / fashion_mnist(kind) -> (images uint8 (n, 784), labels int64 (n,)), kind 'train'
(60 000) or 'test' (10 000), from the IDX files.

nmnist(kind, t_max_ms=250) -> (ptr, t_ms, ch, labels): events of every sample in CSR form,
truncated to the first t_max_ms. Channel = polarity * 34 * 34 + y * 34 + x (2312 channels,
Ivanov 2021's 68 x 34 layout). Files are Orchard et al.'s binary format: 5 bytes per event,
x (8 bit), y (8 bit), polarity (1 bit) + timestamp (23 bit, us). Sample order: digit folder,
then file name. Parsed once and cached in ../datasets/cache/nmnist_<kind>_<t_max>.npz.
"""
import os, glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
CACHE = os.path.join(ROOT, "cache")


def _idx(path):
    with open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = np.frombuffer(raw, ">u4", ndim, 4)
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def mnist(kind="train", folder="mnist"):
    pre = "train" if kind == "train" else "t10k"
    d = os.path.join(ROOT, folder)
    img = _idx(os.path.join(d, f"{pre}-images-idx3-ubyte" if folder == "fashion-mnist" else f"{pre}-images.idx3-ubyte"))
    lab = _idx(os.path.join(d, f"{pre}-labels-idx1-ubyte" if folder == "fashion-mnist" else f"{pre}-labels.idx1-ubyte"))
    return img.reshape(len(img), -1).copy(), lab.astype(np.int64)


def fashion_mnist(kind="train"):
    return mnist(kind, "fashion-mnist")


def _read_nmnist(path, t_max_us):
    b = np.fromfile(path, np.uint8).reshape(-1, 5).astype(np.uint32)
    x, y = b[:, 0], b[:, 1]
    pol = b[:, 2] >> 7
    ts = ((b[:, 2] & 0x7F) << 16) | (b[:, 3] << 8) | b[:, 4]
    keep = ts < t_max_us
    return ts[keep] / 1000.0, (pol[keep] * 34 * 34 + y[keep] * 34 + x[keep]).astype(np.int32)


def nmnist(kind="train", t_max_ms=250):
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, f"nmnist_{kind}_{t_max_ms}.npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        return z["ptr"], z["t"], z["ch"], z["y"]
    sub = "Train/Train" if kind == "train" else "Test/Test"
    files, labels = [], []
    for digit in range(10):
        fs = sorted(glob.glob(os.path.join(ROOT, "n-mnist", sub, str(digit), "*.bin")))
        files += fs
        labels += [digit] * len(fs)
    ptr, T, C = [0], [], []
    for f in files:
        t, ch = _read_nmnist(f, t_max_ms * 1000)
        T.append(t.astype(np.float32)); C.append(ch)
        ptr.append(ptr[-1] + len(t))
    ptr = np.asarray(ptr, np.int64)
    T, C, y = np.concatenate(T), np.concatenate(C), np.asarray(labels, np.int64)
    np.savez(cpath, ptr=ptr, t=T, ch=C, y=y)
    return ptr, T, C, y
