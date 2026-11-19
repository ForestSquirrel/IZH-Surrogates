"""
task.py -- end-task scores.
"""
import numpy as np


def wta_error(scores, y):
    """fraction misclassified by winner-take-all over readout outputs"""
    return float((np.argmax(scores, 1) != y).mean())


def maass_S(out, positive):
    """Hopfield & Brody / Maass recognition score S = Nfp/Ncp + Nfn/Ncn for one binary
    readout (fires iff out >= 0). positive: bool targets. Uses max(., 1) denominators."""
    fire = out >= 0
    pos = np.asarray(positive, bool)
    nfp = int((fire & ~pos).sum()); ncp = int((fire & pos).sum())
    nfn = int((~fire & pos).sum()); ncn = int((~fire & ~pos).sum())
    return nfp / max(ncp, 1) + nfn / max(ncn, 1)


def maass_S_mean(scores, y):
    k = scores.shape[1]
    return float(np.mean([maass_S(scores[:, j], y == j) for j in range(k)]))


def kappa(pred, y, k):
    """Cohen's kappa (Haeusler & Maass 2007 / Schulte 2022 classification score)"""
    po = float((pred == y).mean())
    pe = sum(float((pred == c).mean()) * float((y == c).mean()) for c in range(k))
    return (po - pe) / (1 - pe) if pe < 1 else 0.0
