"""Conformal prediction regions for multivariate responses, with and without covariate shift.

NumPy-only part of the project code (the neural network lives in deep.py).

References used in the comments
  [GL]   Gan & Liu (2025), J. Multivariate Analysis 210:105494.
  [TBCR] Tibshirani, Barber, Candes & Ramdas (2019), NeurIPS (weighted conformal prediction).
  [BBJB] Braun, Berta, Jordan & Bach (2026), arXiv:2507.20941 (local Mahalanobis score).
  [KBNC] Klein, Bethune, Ndiaye & Cuturi (2026), TMLR (WSC and CEC-X definitions, Appendix A.2).

Anything not taken from these papers is marked [my choice].
"""
import math

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
FEATURES = ['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
            'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
            'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
TARGETS = ['motor_UPDRS', 'total_UPDRS']

# [GL] Sec. 5: X1 = DFA, X2 = Shimmer:APQ5, X3 = MDVP:APQ, X4 = HNR.
# [my reading] MDVP:APQ is the 11-point amplitude perturbation quotient, stored as 'Shimmer:APQ11'.
SHIFT_FEATURES = ['DFA', 'Shimmer:APQ5', 'Shimmer:APQ11', 'HNR']


def load_parkinsons(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}; found {df.columns.tolist()}")
    return df, df[FEATURES].to_numpy(float), df[TARGETS].to_numpy(float)


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------
def unit_ball_volume(q):
    return math.pi ** (q / 2) / math.gamma(q / 2 + 1)


def ellipsoid_volume(t, S):
    """Volume of {v : v^T S^{-1} v <= t}."""
    q = S.shape[0]
    return unit_ball_volume(q) * t ** (q / 2) * math.sqrt(np.linalg.det(S))


def split_quantile(scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score ([BBJB] Eq. 2)."""
    n = len(scores)
    k = min(int(math.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(scores)[k - 1]


def weighted_quantile_inf(scores, w_cal, w_test, alpha):
    """[TBCR] weighted split-conformal threshold, one value per test point.

    Quantile(1-alpha; sum_i p_i(x) delta_{s_i} + p_{n+1}(x) delta_{+inf}),
    p_i(x) = w_i / (sum_j w_j + w(x)). Returns +inf when the test point carries too much weight.
    """
    order = np.argsort(scores)
    s, cw = scores[order], np.cumsum(w_cal[order])
    target = (1 - alpha) * (cw[-1] + np.asarray(w_test))
    idx = np.searchsorted(cw, target, side='left')
    return np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)


def ols_fit(X, Y):
    A = np.hstack([np.ones((len(X), 1)), X])
    return np.linalg.lstsq(A, Y, rcond=None)[0]


def ols_predict(beta, X):
    return np.hstack([np.ones((len(X), 1)), X]) @ beta


def quad(D, Sinv):
    return np.einsum('ij,jk,ik->i', D, Sinv, D)


# ----------------------------------------------------------------------------------------------
# Covariate shift construction  ([GL] Sec. 5)
# ----------------------------------------------------------------------------------------------
class ShiftSampler:
    """Test inputs drawn by importance sampling with w(X) = exp[k {s(X1)+s(X2)+s(X3)+s(X4)}],
    s(.) = min-max scaling to [0, 1] ([GL] Sec. 5).

    [my choice] min/max taken over the full data set.
    [my reading] "draw 1000 observations ... from the rest 4875" = 1000 distinct records, i.e. sampling
    without replacement (replace=False, default). With replacement, a strong shift (large k) repeats a
    handful of extreme records many times. The oracle ratio below is exact only with replacement.
    """

    def __init__(self, X_all, feature_names=FEATURES, shift_features=SHIFT_FEATURES):
        self.idx = [feature_names.index(f) for f in shift_features]
        Z = X_all[:, self.idx]
        self.lo, self.hi = Z.min(0), Z.max(0)

    def log_w(self, X, k):
        return k * ((X[:, self.idx] - self.lo) / (self.hi - self.lo)).sum(1)

    def probs(self, X_all, pool_idx, k):
        lw = self.log_w(X_all[pool_idx], k)
        p = np.exp(lw - lw.max())
        return p / p.sum()

    def sample(self, X_all, pool_idx, k, m, rng, replace=False):
        return rng.choice(pool_idx, size=m, replace=replace, p=self.probs(X_all, pool_idx, k))

    def diagnostics(self, X_all, pool_idx, k, m, rng):
        """Effective sample size of the importance weights over the pool, and the number of distinct
        records in one draw of m points with replacement."""
        p = self.probs(X_all, pool_idx, k)
        return dict(k=k, ESS=1 / np.sum(p ** 2),
                    unique_with_replacement=len(np.unique(rng.choice(len(p), size=m, replace=True, p=p))))

    def oracle_ratio(self, X_train, k):
        """r(x) = w(x) / E_P[w], with E_P[w] estimated on the training inputs."""
        lw = self.log_w(X_train, k)
        c = lw.max() + np.log(np.mean(np.exp(lw - lw.max())))
        return lambda X: np.exp(self.log_w(X, k) - c)


class PCShiftSampler(ShiftSampler):
    """[my choice] Shift design for data sets without a published one: the same exponential tilt as
    [GL] Sec. 5 applied to one direction, the first principal component of the standardized inputs,
    w(X) = exp[k s(PC1(X))] with s(.) min-max scaling over the full data set."""

    def __init__(self, X_all):
        self.mu, self.sd = X_all.mean(0), X_all.std(0) + 1e-12
        Z = (X_all - self.mu) / self.sd
        self.v = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
        pc = Z @ self.v
        self.lo, self.hi = pc.min(), pc.max()

    def log_w(self, X, k):
        return k * (((X - self.mu) / self.sd) @ self.v - self.lo) / (self.hi - self.lo)


def fit_density_ratio(X_tr, X_te, ridge=1e-4, max_iter=100):
    """Density ratio r(x) = dQ(x)/dP(x) by logistic regression of 'test' vs 'train'.

    Gives the parametric model r(x) = exp(a + g^T x), the same form [GL] use with empirical
    likelihood (their 'EL' estimator). [my choice] fitted by Newton's method on standardized
    inputs with a small ridge penalty; r(x) = (n_tr / n_te) * P(test | x) / P(train | x).
    """
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-12
    Z = np.vstack([(X_tr - mu) / sd, (X_te - mu) / sd])
    A = np.hstack([np.ones((len(Z), 1)), Z])
    t = np.r_[np.zeros(len(X_tr)), np.ones(len(X_te))]
    pen = np.full(A.shape[1], ridge)
    pen[0] = 0.0
    beta = np.zeros(A.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
        g = A.T @ (t - p) - pen * beta
        H = A.T @ (A * (p * (1 - p))[:, None]) + np.diag(pen)
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    offset = np.log(len(X_tr) / len(X_te))
    return lambda X: np.exp(np.clip(np.hstack([np.ones((len(X), 1)), (X - mu) / sd]) @ beta, -30, 30)
                            + offset)


def clip_ratio(r_fn, X_ref, q):
    """Weight clipping [my choice]: r_clip(x) = min{r(x), c}, c = q-quantile of r over the
    training inputs X_ref. Applied to calibration AND test weights (the weighted quantile uses both)."""
    c = np.quantile(r_fn(X_ref), q)
    return lambda X: np.minimum(r_fn(X), c)


def ratio_variants(X_tr, X_te, names):
    """Density-ratio estimators for the ablation of notebook 08 [my choice]:
    'logistic'  : fit_density_ratio as everywhere else (ridge 1e-4)
    'ridge10'   : the same logistic model with a strong ridge penalty (10) -> flatter weights
    'clip99' / 'clip95' : 'logistic' clipped at the 99% / 95% quantile of the training weights."""
    base = fit_density_ratio(X_tr, X_te)
    out = {}
    for nm in names:
        if nm == 'logistic':
            out[nm] = base
        elif nm == 'ridge10':
            out[nm] = fit_density_ratio(X_tr, X_te, ridge=10.0)
        elif nm.startswith('clip'):
            out[nm] = clip_ratio(base, X_tr, int(nm[4:]) / 100)
        else:
            raise ValueError(nm)
    return out


# ----------------------------------------------------------------------------------------------
# Baselines with a linear-regression mean
# ----------------------------------------------------------------------------------------------
def split_indices(n, frac, rng):
    perm = rng.permutation(n)
    k = int(n * frac)
    return perm[:k], perm[k:]


def split_indices_grouped(groups, frac, rng):
    """Split record indices so that no group (patient) is on both sides.
    A random fraction `frac` of the groups goes to the first part. [my choice] split by groups, not records."""
    g = rng.permutation(np.unique(groups))
    k = max(1, min(len(g) - 1, int(round(len(g) * frac))))
    first = np.isin(groups, g[:k])
    return np.flatnonzero(first), np.flatnonzero(~first)


class SplitL2LR:
    """Split conformal with the l2 norm of the LR residual (SCP-L2-LR in [GL]).
    Weighted version (WCP-L2-LR) = [TBCR] with the same score, as compared in [GL] Sec. 4.2."""

    def __init__(self, Xtr, Ytr, rng, fit_frac=0.5, groups=None):
        fi, ci = (split_indices(len(Ytr), fit_frac, rng) if groups is None
                  else split_indices_grouped(groups, fit_frac, rng))
        self.beta = ols_fit(Xtr[fi], Ytr[fi])
        self.Xc = Xtr[ci]
        self.sc = np.linalg.norm(Ytr[ci] - ols_predict(self.beta, Xtr[ci]), axis=1)
        self.q = Ytr.shape[1]

    def _out(self, Xte, Yte, thr):
        s = np.linalg.norm(Yte - ols_predict(self.beta, Xte), axis=1)
        thr = np.broadcast_to(thr, s.shape)
        return dict(covered=s <= thr, vol=unit_ball_volume(self.q) * thr ** self.q)

    def unweighted(self, Xte, Yte, alpha):
        return self._out(Xte, Yte, split_quantile(self.sc, alpha))

    def weighted(self, Xte, Yte, alpha, r_fn):
        return self._out(Xte, Yte, weighted_quantile_inf(self.sc, r_fn(self.Xc), r_fn(Xte), alpha))


# --- Euclidean likelihood, no shift ([GL] Eq. 4) --------------------------------------------------
def eur_fit(Y, M):
    q = Y.shape[1]
    S = np.cov(np.hstack([Y, M]), rowvar=False)
    Syy, Sym, Smm = S[:q, :q], S[:q, q:], S[q:, q:]
    Bm = Sym @ np.linalg.pinv(Smm)          # pinv: robust to collinear inputs (same as solve otherwise)
    Syyx = Syy - Bm @ Sym.T
    return dict(Ybar=Y.mean(0), Mbar=M.mean(0), B=Bm, Syyx=Syyx, Sinv=np.linalg.inv(Syyx))


def eur_score(f, Y, M):
    R = np.atleast_2d(Y) - (f['Ybar'] + (np.atleast_2d(M) - f['Mbar']) @ f['B'].T)
    return quad(R, f['Sinv'])


class EUJackknife:
    """[GL] Eq. (4): Z = (Y, X), Jackknife threshold T_(ceil(n(1-alpha)))."""

    def __init__(self, Xtr, Ytr, alpha, groups=None):
        n = len(Ytr)
        T = np.empty(n)
        if groups is None:
            keep = np.ones(n, bool)
            for i in range(n):
                keep[i] = False
                T[i] = eur_score(eur_fit(Ytr[keep], Xtr[keep]), Ytr[i], Xtr[i])[0]
                keep[i] = True
        else:
            # [my choice] leave-one-GROUP-out: the scores of a patient's records come from a fit
            # without that patient (the record-level jackknife is too optimistic for new patients)
            for g in np.unique(groups):
                out = groups == g
                T[out] = eur_score(eur_fit(Ytr[~out], Xtr[~out]), Ytr[out], Xtr[out])
        self.t = np.sort(T)[int(math.ceil(n * (1 - alpha))) - 1]
        self.f = eur_fit(Ytr, Xtr)

    def predict(self, Xte, Yte):
        s = eur_score(self.f, Yte, Xte)
        return dict(covered=s <= self.t,
                    vol=np.full(len(s), ellipsoid_volume(self.t, self.f['Syyx'])))


# --- Euclidean likelihood under covariate shift ([GL] Eq. 8 and Algorithm 2) -----------------------
def euc_stats(E, r):
    """[GL] Eq. (8) ingredients: e_bar_c = (1/n) sum r_i e_i ;
    S_c = sum p_i e_i e_i^T - (sum p_i e_i)(sum p_i e_i)^T with p_i = r_i / sum_j r_j."""
    n = len(E)
    ebar = (r[:, None] * E).sum(0) / n
    p = r / r.sum()
    m = p @ E
    return ebar, (E * p[:, None]).T @ E - np.outer(m, m)


def euc_stats_batch(E, r):
    """Same as euc_stats for a batch: E (B, n, q), r (B, n)."""
    n = E.shape[1]
    ebar = (r[..., None] * E).sum(1) / n
    p = r / r.sum(1, keepdims=True)
    m = np.einsum('bn,bnq->bq', p, E)
    S = np.einsum('bn,bnq,bnr->bqr', p, E, E) - np.einsum('bq,br->bqr', m, m)
    return ebar, S


def draw_resample(n, B, p, rng, scheme):
    """Pseudo-training indices (B, n) and one pseudo-test index per b.

    'text'      : [GL] Sec. 3.1 and the proof of Thm. 3 -- training from P_hat (uniform),
                  test point from Q_hat (prob. p_i = r_i / sum r).
    'algorithm' : [GL] Algorithm 2 as printed -- training from Q_hat, test point from P_hat.
    """
    if scheme == 'text':
        return rng.integers(0, n, (B, n)), rng.choice(n, size=B, p=p)
    if scheme == 'algorithm':
        return rng.choice(n, size=(B, n), p=p), rng.integers(0, n, B)
    raise ValueError(scheme)


class EUcLR:
    """[GL] Algorithm 2 with a linear-regression mean (their 'EU-LR' under shift)."""

    def __init__(self, Xtr, Ytr, r_fn, alpha, B, rng, scheme='text', groups=None):
        n = len(Ytr)
        r = r_fn(Xtr)
        self.beta = ols_fit(Xtr, Ytr)
        self.ebar, self.S = euc_stats(Ytr - ols_predict(self.beta, Xtr), r)
        if groups is not None:
            self._grouped(Xtr, Ytr, r, alpha, B, rng, groups)
            return
        idx, jj = draw_resample(n, B, r / r.sum(), rng, scheme)
        T = np.empty(B)
        for b in range(B):
            I = idx[b]
            bb = ols_fit(Xtr[I], Ytr[I])
            eb, Sb = euc_stats(Ytr[I] - ols_predict(bb, Xtr[I]), r[I])
            j = jj[b]
            T[b] = quad((eb - (Ytr[j] - ols_predict(bb, Xtr[j:j + 1])[0]))[None], np.linalg.inv(Sb))[0]
        self.t = np.sort(T)[int(math.ceil(B * (1 - alpha))) - 1]

    def _grouped(self, Xtr, Ytr, r, alpha, B, rng, groups):
        """[my choice] grouped resampling: pseudo-training = cluster bootstrap of patients (from P_hat);
        pseudo-test record drawn with prob. proportional to r among the records of patients NOT
        drawn in that resample (from Q_hat, out-of-bag), so that its residual is for a new patient."""
        ug = np.unique(groups)
        rows = {g: np.flatnonzero(groups == g) for g in ug}
        T = []
        for _ in range(B):
            gb = rng.choice(ug, size=len(ug), replace=True)
            oob = np.setdiff1d(ug, gb)
            if len(oob) == 0:
                continue
            I = np.concatenate([rows[g] for g in gb])
            cand = np.concatenate([rows[g] for g in oob])
            j = rng.choice(cand, p=r[cand] / r[cand].sum())
            bb = ols_fit(Xtr[I], Ytr[I])
            eb, Sb = euc_stats(Ytr[I] - ols_predict(bb, Xtr[I]), r[I])
            T.append(quad((eb - (Ytr[j] - ols_predict(bb, Xtr[j:j + 1])[0]))[None], np.linalg.inv(Sb))[0])
        T = np.sort(np.asarray(T))
        self.t = T[int(math.ceil(len(T) * (1 - alpha))) - 1]

    def predict(self, Xte, Yte):
        D = self.ebar - (Yte - ols_predict(self.beta, Xte))
        s = quad(D, np.linalg.inv(self.S))
        return dict(covered=s <= self.t, vol=np.full(len(s), ellipsoid_volume(self.t, self.S)))


# ----------------------------------------------------------------------------------------------
# Deep local-covariance scores  (inputs: network outputs on the calibration and test sets)
#   mu (n, q) and lower-triangular L (n, q, q) in standardized-Y units; yscale = std of Y.
# ----------------------------------------------------------------------------------------------
def whiten(mu, L, Y):
    return np.linalg.solve(L, (Y - mu)[..., None])[..., 0]


def det_L(L):
    return np.prod(np.diagonal(L, axis1=1, axis2=2), axis=1)


def deep_split(cal, test, alpha, yscale):
    """[BBJB]: local Mahalanobis score, split-conformal threshold (no shift correction)."""
    q = cal['Y'].shape[1]
    s_c = (whiten(cal['mu'], cal['L'], cal['Y']) ** 2).sum(1)
    thr = split_quantile(s_c, alpha)
    s_t = (whiten(test['mu'], test['L'], test['Y']) ** 2).sum(1)
    vol = unit_ball_volume(q) * thr ** (q / 2) * det_L(test['L']) * np.prod(yscale)
    return dict(covered=s_t <= thr, vol=vol)


def deep_weighted(cal, test, alpha, yscale, r_cal, r_test):
    """[ours] local Mahalanobis score + [TBCR] weighted calibration."""
    q = cal['Y'].shape[1]
    s_c = (whiten(cal['mu'], cal['L'], cal['Y']) ** 2).sum(1)
    thr = weighted_quantile_inf(s_c, r_cal, r_test, alpha)
    s_t = (whiten(test['mu'], test['L'], test['Y']) ** 2).sum(1)
    vol = unit_ball_volume(q) * thr ** (q / 2) * det_L(test['L']) * np.prod(yscale)
    return dict(covered=s_t <= thr, vol=vol)


def deep_euc(cal, test, alpha, yscale, r_cal, B, rng, scheme='text'):
    """[ours] [GL] Eq. (8) score built on the whitened residuals u = L(x)^{-1}(y - f(x)) of the
    calibration set, calibrated by the [GL] resampling scheme. The network is trained on separate
    data and kept fixed (a pre-trained model, which [GL] allow); only u_bar_c and S_c are
    recomputed on each resample. Region: ellipse centred at f(x) + L(x) u_bar_c with shape
    L(x) S_c L(x)^T, so it adapts to x like [BBJB] and is shift-corrected like [GL]."""
    U = whiten(cal['mu'], cal['L'], cal['Y'])
    n, q = U.shape
    ubar, S = euc_stats(U, r_cal)
    idx, jj = draw_resample(n, B, r_cal / r_cal.sum(), rng, scheme)
    ub, Sb = euc_stats_batch(U[idx], r_cal[idx])
    D = ub - U[jj]
    T = np.einsum('bq,bqr,br->b', D, np.linalg.inv(Sb), D)
    thr = np.sort(T)[int(math.ceil(B * (1 - alpha))) - 1]
    V = whiten(test['mu'], test['L'], test['Y'])
    s = quad(ubar - V, np.linalg.inv(S))
    vol = unit_ball_volume(q) * thr ** (q / 2) * math.sqrt(np.linalg.det(S)) * det_L(test['L']) * np.prod(yscale)
    return dict(covered=s <= thr, vol=vol)


# ----------------------------------------------------------------------------------------------
# Conditional-coverage metrics ([KBNC] Appendix A.2)
# ----------------------------------------------------------------------------------------------
def standardize_from(Xref, X):
    return (X - Xref.mean(0)) / (Xref.std(0) + 1e-12)


def random_directions(n_dir, d, rng):
    v = rng.normal(size=(n_dir, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def wsc(Xs, covered_list, directions, delta, grid):
    """Worst-slab coverage. [my choice] slab end points restricted to a quantile grid."""
    n = len(Xs)
    m = int(math.ceil(delta * n))
    cut = np.unique(np.linspace(0, n, grid + 1).round().astype(int))
    st, en = np.meshgrid(cut, cut, indexing='ij')
    ok = (en - st) >= m
    st, en = st[ok], en[ok]
    best = np.ones(len(covered_list))
    for v in directions:
        order = np.argsort(Xs @ v)
        for k, cov in enumerate(covered_list):
            P = np.concatenate([[0], np.cumsum(cov[order])])
            best[k] = min(best[k], ((P[en] - P[st]) / (en - st)).min())
    return best


def kmeanspp(X, J, rng, n_iter=100):
    C = [X[rng.integers(len(X))]]
    for _ in range(1, J):
        d2 = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(n_iter):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(J)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def assign(X, C):
    return np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)


def cec_x(labels, covered, J, alpha):
    """[my reading of KBNC Eq. 12] mean over test points of (cluster coverage - (1 - alpha))^2."""
    cov_j = np.array([covered[labels == j].mean() if np.any(labels == j) else np.nan for j in range(J)])
    return np.nanmean((cov_j[labels] - (1 - alpha)) ** 2)
