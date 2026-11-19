"""
readout.py -- Ivanov 2021 output layer (§2.2): dense 10-unit linear layer on liquid spike
counts, softmax cross-entropy + L2 (lambda 5e-10), Adam lr 0.1, batch 250, weights and
biases initialised to 0, trained until validation accuracy peaks (max 5000 epochs). We stop
after `patience` epochs without a new validation best and test the best-validation weights.
"""
import numpy as np
import torch

import spec as S


def train(Xtr, ytr, Xva, yva, Xte, yte, patience=100, max_epochs=S.MAX_EPOCHS, seed=0,
          device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    Xtr, Xva, Xte = f(Xtr), f(Xva), f(Xte)
    ytr = torch.as_tensor(ytr, device=device)
    yva_n, yte_n = np.asarray(yva), np.asarray(yte)
    lin = torch.nn.Linear(Xtr.shape[1], 10).to(device)
    torch.nn.init.zeros_(lin.weight); torch.nn.init.zeros_(lin.bias)
    opt = torch.optim.Adam(lin.parameters(), lr=S.LR)
    n = len(Xtr)
    best, best_ep, state, hist = -1.0, -1, None, []
    for ep in range(max_epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, S.BATCH):
            idx = perm[i:i + S.BATCH]
            loss = torch.nn.functional.cross_entropy(lin(Xtr[idx]), ytr[idx]) \
                + S.L2 / (2 * len(idx)) * (lin.weight ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            va = float((lin(Xva).argmax(1).cpu().numpy() == yva_n).mean())
        hist.append(va)
        if va > best:
            best, best_ep = va, ep
            state = {k: v.clone() for k, v in lin.state_dict().items()}
        elif ep - best_ep >= patience:
            break
    lin.load_state_dict(state)
    with torch.no_grad():
        pte = lin(Xte).argmax(1).cpu().numpy()
        te = float((pte == yte_n).mean())
        tr = float((lin(Xtr).argmax(1).cpu().numpy() == ytr.cpu().numpy()).mean())
    conf = np.zeros((10, 10), np.int64)
    np.add.at(conf, (yte_n.astype(np.int64), pte.astype(np.int64)), 1)
    return dict(val=best, test=te, train=tr, best_epoch=best_ep, epochs=len(hist),
                confusion=conf.tolist(), val_curve=hist)
