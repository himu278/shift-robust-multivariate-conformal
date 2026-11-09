"""Deep mean + local covariance network ([BBJB] Braun, Berta, Jordan & Bach 2026, Appendix A).

p(y | x) = N(y | f(x), Sigma(x)), Sigma(x) = L(x) L(x)^T with L lower triangular, diagonal through
softplus + eps; trained by Gaussian negative log-likelihood; covariance head initialised so that
off-diagonals start near zero and diagonals equal the marginal std of the (standardized) targets.
Same model as notebooks/01_exchangeable_baselines.ipynb.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _softplus_inv(y):
    return math.log(math.expm1(y))


class MeanCovNet(nn.Module):
    def __init__(self, d, k, hidden, n_layers, eps):
        super().__init__()
        layers, din = [], d
        for _ in range(n_layers):
            layers += [nn.Linear(din, hidden), nn.ReLU()]
            din = hidden
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden, k)
        self.chol_head = nn.Linear(hidden, k * (k + 1) // 2)
        self.k, self.eps = k, eps
        rows, cols = torch.tril_indices(k, k)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        with torch.no_grad():
            self.chol_head.weight.normal_(0.0, 1e-3)
            bias = torch.zeros(k * (k + 1) // 2)
            bias[rows == cols] = _softplus_inv(1.0 - eps)
            self.chol_head.bias.copy_(bias)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mean_head(h)
        L = x.new_zeros(x.shape[0], self.k, self.k)
        L[:, self.rows, self.cols] = self.chol_head(h)
        diag = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + self.eps
        return mu, torch.tril(L, diagonal=-1) + torch.diag_embed(diag)


def gaussian_nll(mu, L, y):
    z = torch.linalg.solve_triangular(L, (y - mu).unsqueeze(-1), upper=False).squeeze(-1)
    return (torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1) + 0.5 * (z ** 2).sum(-1)).mean()


def _train(Xtr, Ytr, Xva, Yva, cfg, seed):
    torch.manual_seed(seed)
    model = MeanCovNet(Xtr.shape[1], Ytr.shape[1], cfg['HIDDEN'], cfg['N_LAYERS'], cfg['EPS']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['LR'], weight_decay=cfg['WEIGHT_DECAY'])
    Xt, Yt, Xv, Yv = [torch.tensor(a, dtype=torch.float32, device=DEVICE) for a in (Xtr, Ytr, Xva, Yva)]
    best, best_state, wait, hist = float('inf'), None, 0, []
    for _ in range(cfg['MAX_EPOCHS']):
        model.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), cfg['BATCH']):
            b = perm[i:i + cfg['BATCH']]
            loss = gaussian_nll(*model(Xt[b]), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr = gaussian_nll(*model(Xt), Yt).item()
            va = gaussian_nll(*model(Xv), Yv).item()
        hist.append((tr, va))
        if va < best - 1e-5:
            best, wait = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg['PATIENCE']:
                break
    model.load_state_dict(best_state)
    return model, np.array(hist)


def _group_split(groups, frac, rng):
    """A random fraction `frac` of the groups (patients) goes to the first part (same as
    cpshift.split_indices_grouped; repeated here so deep.py stays self-contained)."""
    g = rng.permutation(np.unique(groups))
    k = max(1, min(len(g) - 1, int(round(len(g) * frac))))
    first = np.isin(groups, g[:k])
    return np.flatnonzero(first), np.flatnonzero(~first)


class DeepLocalCov:
    """Fit on (Xtr, Ytr) with the same split as notebook 01: a fraction DEEP_CAL_FRAC is held out
    for calibration; the rest is split again for early stopping (DEEP_VAL_FRAC).
    Inputs and targets are standardized with the fit-part statistics."""

    def __init__(self, Xtr, Ytr, cfg, rng, seed, groups=None):
        n = len(Ytr)
        if groups is None:
            perm = rng.permutation(n)
            n_fit = int(n * (1 - cfg['DEEP_CAL_FRAC']))
            fit_i, self.cal_idx = perm[:n_fit], perm[n_fit:]
            n_va = int(n_fit * cfg['DEEP_VAL_FRAC'])
            va_i, tr_i = fit_i[:n_va], fit_i[n_va:]
        else:
            # [my choice] patient-grouped split: calibration and early-stopping patients are
            # different from the patients the network is trained on
            fit_i, self.cal_idx = _group_split(groups, 1 - cfg['DEEP_CAL_FRAC'], rng)
            va_sub, tr_sub = _group_split(groups[fit_i], cfg['DEEP_VAL_FRAC'], rng)
            va_i, tr_i = fit_i[va_sub], fit_i[tr_sub]
        self.tr_idx = tr_i
        self.xm, self.xs = Xtr[fit_i].mean(0), Xtr[fit_i].std(0) + 1e-12
        self.ym, self.ys = Ytr[fit_i].mean(0), Ytr[fit_i].std(0)
        self.model, self.hist = _train(self._sx(Xtr[tr_i]), self._sy(Ytr[tr_i]),
                                       self._sx(Xtr[va_i]), self._sy(Ytr[va_i]), cfg, seed)

    def _sx(self, X):
        return (X - self.xm) / self.xs

    def _sy(self, Y):
        return (Y - self.ym) / self.ys

    def outputs(self, X, Y):
        """Network outputs in standardized-Y units, in the format used by cpshift.deep_*."""
        self.model.eval()
        with torch.no_grad():
            mu, L = self.model(torch.tensor(self._sx(X), dtype=torch.float32, device=DEVICE))
        return dict(mu=mu.cpu().numpy().astype(float), L=L.cpu().numpy().astype(float), Y=self._sy(Y))
