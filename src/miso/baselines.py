import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from .data import PILOTS


OBSERVED = np.array([a * 16 + k for a in range(8) for k in PILOTS], dtype=np.int64)
OBSERVED_RI = np.concatenate([OBSERVED, 128 + OBSERVED])


def interpolate(observed, frequencies):
    if observed.shape[1:] != (8, 3):
        raise ValueError("Only the 24 observed complex CSI values may be supplied")
    rows = observed.reshape(-1, 3)
    real = np.stack([np.interp(frequencies, frequencies[PILOTS], r.real) for r in rows])
    imag = np.stack([np.interp(frequencies, frequencies[PILOTS], r.imag) for r in rows])
    return (real + 1j * imag).reshape(len(observed), 8, 16)


class LMMSE:
    def fit(self, h, ridge_ratio=0.0):
        h = np.asarray(h, dtype=np.complex128).reshape(len(h), 128)
        self.mean = h.mean(axis=0)
        centered = h - self.mean
        covariance = centered.T @ centered.conj() / (len(h) - 1)
        yy = covariance[np.ix_(OBSERVED, OBSERVED)]
        loading = ridge_ratio * np.trace(yy).real / len(OBSERVED)
        yy = yy + loading * np.eye(len(OBSERVED))
        np.linalg.cholesky(yy)
        self.gain = np.linalg.solve(yy.T, covariance[:, OBSERVED].T).T
        self.info = {"ridge_ratio": ridge_ratio, "absolute_loading": float(loading),
                     "condition_number": float(np.linalg.cond(yy)), "raw_complex_csi": True,
                     "normalization": False, "pseudoinverse": False}
        return self

    def predict(self, observed):
        if observed.shape[1:] != (8, 3):
            raise ValueError("Only pilot CSI may be supplied")
        y = observed.reshape(len(observed), 24)
        h = self.mean + (y - self.mean[OBSERVED]) @ self.gain.T
        h = h.reshape(-1, 8, 16)
        h[:, :, PILOTS] = observed
        return h


class GMMEstimator:
    def fit(self, h, coordinates, scale, components=4, ridge=0.001, position=False, seed=2026):
        self.scale, self.position = scale, position
        flat = h.reshape(len(h), 128) / scale
        x = np.concatenate([flat.real, flat.imag], axis=1).astype(np.float64)
        self.coordinate_mean = coordinates.mean(axis=0)
        self.coordinate_std = coordinates.std(axis=0) + 1e-12
        self.observed_indices = OBSERVED_RI
        if position:
            x = np.concatenate([x, (coordinates - self.coordinate_mean) / self.coordinate_std], axis=1)
            self.observed_indices = np.concatenate([OBSERVED_RI, np.arange(256, 259)])
        self.gmm = GaussianMixture(n_components=components, covariance_type="full", reg_covar=ridge,
                                   random_state=seed, n_init=1, max_iter=100, tol=1e-3).fit(x)
        self.components = []
        for mean, covariance, weight in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
            yy = covariance[np.ix_(self.observed_indices, self.observed_indices)]
            cholesky = np.linalg.cholesky(yy)
            inverse = np.linalg.solve(yy, np.eye(len(yy)))
            gain = covariance[:256, self.observed_indices] @ inverse
            constant = np.log(weight) - np.log(np.diag(cholesky)).sum() - len(yy) * np.log(2 * np.pi) / 2
            self.components.append((mean[self.observed_indices], mean[:256], gain, inverse, constant))
        self.info = {"components": components, "ridge_scaled_coordinates": ridge, "position": position,
                     "seed": seed, "converged": bool(self.gmm.converged_), "iterations": int(self.gmm.n_iter_),
                     "feature_dimension": x.shape[1], "training_only_common_csi_scale": float(scale)}
        return self

    def predict(self, observed, coordinates, batch_size=256):
        if observed.shape[1:] != (8, 3):
            raise ValueError("Missing CSI cannot be used in the posterior")
        y = observed.reshape(len(observed), 24) / self.scale
        y = np.concatenate([y.real, y.imag], axis=1).astype(np.float64)
        if self.position:
            y = np.concatenate([y, (coordinates - self.coordinate_mean) / self.coordinate_std], axis=1)
        outputs = []
        for start in range(0, len(y), batch_size):
            batch = y[start:start + batch_size]
            logp, means = [], []
            for mu_y, mu_h, gain, inverse, constant in self.components:
                delta = batch - mu_y
                logp.append(constant - np.sum((delta @ inverse) * delta, axis=1) / 2)
                means.append(mu_h + delta @ gain.T)
            logp = np.stack(logp, axis=1)
            weights = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
            outputs.append(sum(weights[:, i:i + 1] * v for i, v in enumerate(means)))
        x = np.concatenate(outputs)
        h = ((x[:, :128] + 1j * x[:, 128:256]) * self.scale).reshape(-1, 8, 16)
        h[:, :, PILOTS] = observed
        return h
