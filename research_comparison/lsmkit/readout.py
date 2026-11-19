"""
readout.py -- linear readouts on liquid states (numpy only).
"""
import numpy as np


def fit(X, Y, lam=0.0):
    """Least squares (lam = 0, Maass' 'linear regression') or ridge with a bias column.
    X: (n, d), Y: (n, m) -> W (d+1, m)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    if lam <= 0:
        return np.linalg.lstsq(Xb, Y, rcond=None)[0]
    A = Xb.T @ Xb
    reg = lam * np.eye(A.shape[0])
    reg[-1, -1] = 0.0
    return np.linalg.solve(A + reg, Xb.T @ Y)


def predict(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def onehot_pm(y, k):
    """+1 for the class, -1 otherwise (Maass' perceptron targets)."""
    T = -np.ones((len(y), k))
    T[np.arange(len(y)), y] = 1.0
    return T


def ridge_cv(X, Y, lams, folds=5, rng=None, score=None):
    """pick lambda by k-fold CV on (X, Y); score(Yhat, Y) is minimised (default MSE)."""
    rng = rng or np.random.default_rng(0)
    idx = rng.permutation(len(X))
    parts = np.array_split(idx, folds)
    score = score or (lambda P, T: float(((P - T) ** 2).mean()))
    best = None
    for lam in lams:
        s = 0.0
        for f in range(folds):
            te = parts[f]
            tr = np.concatenate([parts[g] for g in range(folds) if g != f])
            s += score(predict(fit(X[tr], Y[tr], lam), X[te]), Y[te])
        if best is None or s < best[1]:
            best = (lam, s)
    return best[0]
