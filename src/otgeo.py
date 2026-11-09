"""Two baselines, both split-conformal with a vector residual score
z = y - f(x) (in standardized-Y units; f = mean head of the deep network):

  OTCP  [KBNC] Klein, Bethune, Ndiaye & Cuturi (2026), TMLR.
        Entropic OT map T_eps from hold-out residuals to the spherical uniform distribution on the
        unit ball (direction uniform on the sphere, radius uniform on [0, 1]; their Sec. 2.3);
        T_eps(z) = sum_j p_j(z) u_j (their Eq. 6); scalar score ||T_eps(z)||; split-conformal
        calibration on a separate calibration split (their Sec. 3.1, three-way split).
        eps = 0.1 relative to the std of the cost matrix (their Sec. 4.2 default).

  GRCP  [CMP] Conrad, Moulines & Perez (2026), ICML.
        Rank-first kernel estimator of the conditional spatial rank
        R(x, z) = sum_j w_j(x) (z - Z_j) / ||z - Z_j||  (their Eq. 16), score ||R(x, z)|| (Eq. 21),
        split-conformal threshold (Eq. 22). "In all GRCP experiments, we estimate the conditional
        rank map using the rank-first estimator."

Choices not fixed by the papers are marked [my choice].
"""
import numpy as np


def _logsumexp(A, axis):
    m = A.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def sample_spherical_uniform(m, d, rng):
    v = rng.normal(size=(m, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * rng.uniform(0, 1, size=(m, 1))


class OTCP:
    def __init__(self, Z_hold, rng, m=1024, eps_rel=0.1, n_iter=1000, tol=1e-6):
        """[my choice] m = 1024 reference points (the paper's Fig. 7 studies m; larger is better)."""
        n, d = Z_hold.shape
        self.U = sample_spherical_uniform(m, d, rng)
        C = ((Z_hold[:, None, :] - self.U[None]) ** 2).sum(-1)
        self.eps = eps_rel * C.std()
        la, lb = -np.log(n), -np.log(m)
        f, g = np.zeros(n), np.zeros(m)
        for _ in range(n_iter):                     # log-domain Sinkhorn on the dual (their Eq. 4)
            f_new = -self.eps * _logsumexp((g[None, :] - C) / self.eps + lb, axis=1)
            g = -self.eps * _logsumexp((f_new[:, None] - C) / self.eps + la, axis=0)
            if np.max(np.abs(f_new - f)) < tol * self.eps:
                f = f_new
                break
            f = f_new
        self.g = g

    def transport(self, Z, chunk=4096):
        out = np.empty((len(Z), self.U.shape[1]))
        for i in range(0, len(Z), chunk):
            z = Z[i:i + chunk]
            L = -(((z[:, None, :] - self.U[None]) ** 2).sum(-1) - self.g[None, :]) / self.eps
            L -= L.max(1, keepdims=True)
            P = np.exp(L)
            out[i:i + chunk] = (P / P.sum(1, keepdims=True)) @ self.U
        return out

    def score(self, Z):
        return np.linalg.norm(self.transport(Z), axis=1)


class GRCP:
    def __init__(self, X_hold, Z_hold, bandwidth=None):
        """[my choice] Gaussian kernel on inputs standardized with hold-out statistics; bandwidth by
        the median heuristic (median pairwise distance) when not given."""
        self.xm, self.xs = X_hold.mean(0), X_hold.std(0) + 1e-12
        self.Xh = (X_hold - self.xm) / self.xs
        self.Zh = Z_hold
        if bandwidth is None:
            D = np.sqrt(((self.Xh[:, None] - self.Xh[None]) ** 2).sum(-1))
            bandwidth = np.median(D[np.triu_indices(len(D), 1)])
        self.h = bandwidth

    def weights(self, X):
        Xs = (X - self.xm) / self.xs
        L = -((Xs[:, None, :] - self.Xh[None]) ** 2).sum(-1) / (2 * self.h ** 2)
        L -= L.max(1, keepdims=True)
        W = np.exp(L)
        return W / W.sum(1, keepdims=True)

    def score(self, X, Z, chunk=2048):
        out = np.empty(len(Z))
        for i in range(0, len(Z), chunk):
            W = self.weights(X[i:i + chunk])
            D = Z[i:i + chunk, None, :] - self.Zh[None]
            U = D / (np.linalg.norm(D, axis=-1, keepdims=True) + 1e-12)
            out[i:i + chunk] = np.linalg.norm(np.einsum('nj,njd->nd', W, U), axis=1)
        return out

    def score_grid(self, x, G):
        """Scores of grid points G (g, d) for one input x."""
        w = self.weights(x[None])[0]
        D = G[:, None, :] - self.Zh[None]
        U = D / (np.linalg.norm(D, axis=-1, keepdims=True) + 1e-12)
        return np.linalg.norm(np.einsum('j,gjd->gd', w, U), axis=1)


def residual_grid(Z_ref, n_side=150, pad=6.0):
    """[my choice] square grid in residual space: mean +/- pad * std of the reference residuals.
    Returns grid points, area of one cell, and a mask of boundary cells."""
    c, s = Z_ref.mean(0), Z_ref.std(0)
    axes = [np.linspace(c[i] - pad * s[i], c[i] + pad * s[i], n_side) for i in range(Z_ref.shape[1])]
    mesh = np.meshgrid(*axes, indexing='ij')
    G = np.stack([m.ravel() for m in mesh], 1)
    cell = np.prod([a[1] - a[0] for a in axes])
    edge = np.zeros(mesh[0].shape, bool)
    for dim in range(len(axes)):
        idx = [slice(None)] * len(axes)
        idx[dim] = 0
        edge[tuple(idx)] = True
        idx[dim] = -1
        edge[tuple(idx)] = True
    return G, cell, edge.ravel()


def area_from_mask(inside, cell, edge):
    """Area of a region given membership on the grid; flags regions that reach the grid boundary."""
    return inside.sum() * cell, bool(inside[edge].any())
