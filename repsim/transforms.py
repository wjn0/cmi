"""Cross-model representation transforms and their evaluation.

Two families of map from a source representation to a target representation are
fitted on training pairs and scored on held-out pairs:

* ``linear`` -- an affine least-squares map (handles differing dimensions).
* ``rigid`` -- a similarity transform (orthogonal map + translation + a single
  global scale) found via orthogonal Procrustes. Because Procrustes requires equal
  dimensions, both representations are first reduced to their common minimum
  dimension by PCA fitted on the training pairs.

Quality is reported as the coefficient of determination R^2, pooled across all
output dimensions (1 - SS_res / SS_tot).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from sklearn.decomposition import PCA


def r2_score(
    pred: Float[np.ndarray, "n d"], target: Float[np.ndarray, "n d"]
) -> float:
    """Return the pooled R^2 of ``pred`` against ``target`` over all dimensions."""
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean(axis=0, keepdims=True)) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class Whitening:
    """A PCA-whitening map ``x -> (x - mean) @ transform``.

    Centres the data and rotates it into its PCA basis with each component scaled
    to unit variance, so the whitened representation has (approximately) zero mean
    and identity covariance.
    """

    mean: Float[np.ndarray, "d"]
    transform: Float[np.ndarray, "d d"]

    def apply(self, x: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n d"]:
        """Whiten a batch of representations."""
        return (x - self.mean) @ self.transform


def fit_whitening(
    x: Float[np.ndarray, "n d"], eps: float = 1e-3
) -> Whitening:
    """Fit PCA whitening on ``x``.

    Each principal component is scaled to unit variance. ``eps`` floors the
    per-component variance *relative to the largest component* before inverting it,
    so near-degenerate directions (e.g. MAE's tiny-variance dims) are not blown up
    into amplified noise.
    """
    mean = x.mean(axis=0)
    _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    variance = singular**2 / x.shape[0]
    scale = 1.0 / np.sqrt(variance + eps * variance.max())
    return Whitening(mean=mean, transform=vt.T * scale)


@dataclass(frozen=True)
class LinearTransform:
    """An affine map ``x -> x @ weight + bias``."""

    weight: Float[np.ndarray, "d_in d_out"]
    bias: Float[np.ndarray, "d_out"]

    def apply(self, x: Float[np.ndarray, "n d_in"]) -> Float[np.ndarray, "n d_out"]:
        """Apply the affine map to a batch of source representations."""
        return x @ self.weight + self.bias


def fit_linear(
    source: Float[np.ndarray, "n d_in"], target: Float[np.ndarray, "n d_out"]
) -> LinearTransform:
    """Fit an affine least-squares map from ``source`` to ``target``.

    Uses SVD-based ``lstsq`` (with ``rcond`` truncation), not the normal
    equations. Some embeddings (notably MAE) have near-collinear / tiny-variance
    directions, so ``A^T A`` is severely ill-conditioned; a direct
    ``solve(A^T A, ...)`` returns garbage there (in-sample R^2 went negative,
    which is impossible for a correct least-squares fit). ``lstsq`` truncates the
    offending singular values and stays stable.
    """
    augmented = np.hstack([source, np.ones((source.shape[0], 1))])
    solution, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    return LinearTransform(weight=solution[:-1], bias=solution[-1])


@dataclass(frozen=True)
class RigidTransform:
    """A similarity map in a shared PCA subspace.

    The source and target are each projected with their fitted PCA, then the
    source projection is mapped to the target projection via
    ``z -> scale * (z @ rotation) + translation``. Predictions and R^2 are
    therefore expressed in the target's PCA subspace.
    """

    pca_source: PCA
    pca_target: PCA
    rotation: Float[np.ndarray, "k k"]
    scale: float
    translation: Float[np.ndarray, "k"]

    def project_target(
        self, target: Float[np.ndarray, "n d_out"]
    ) -> Float[np.ndarray, "n k"]:
        """Project target representations into the shared PCA subspace."""
        return self.pca_target.transform(target)

    def apply(self, source: Float[np.ndarray, "n d_in"]) -> Float[np.ndarray, "n k"]:
        """Map source representations into the target's PCA subspace."""
        z = self.pca_source.transform(source)
        return self.scale * (z @ self.rotation) + self.translation


def fit_rigid(
    source: Float[np.ndarray, "n d_in"], target: Float[np.ndarray, "n d_out"]
) -> RigidTransform:
    """Fit a similarity transform between PCA-reduced source and target reps.

    Both inputs are reduced to ``k = min(d_in, d_out)`` PCA components fitted on
    the training pairs, then aligned with the optimal rotation and global scale
    (Umeyama's solution, with reflections disallowed).
    """
    k = min(source.shape[1], target.shape[1])
    pca_source = PCA(n_components=k).fit(source)
    pca_target = PCA(n_components=k).fit(target)
    x = pca_source.transform(source)
    y = pca_target.transform(target)

    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mu_x, y - mu_y
    covariance = (xc.T @ yc) / x.shape[0]
    u, singular, vt = np.linalg.svd(covariance)

    # Full orthogonal Procrustes (reflections allowed): forcing a proper
    # rotation is meaningless here because PCA fixes each component's sign
    # arbitrarily, which can flip a rotation into a reflection.
    rotation = u @ vt
    var_x = (xc**2).sum() / x.shape[0]
    scale = float(singular.sum() / var_x)
    translation = mu_y - scale * (mu_x @ rotation)
    return RigidTransform(pca_source, pca_target, rotation, scale, translation)


def evaluate_transform(
    kind: str,
    source_train: Float[np.ndarray, "n d_in"],
    target_train: Float[np.ndarray, "n d_out"],
    source_eval: Float[np.ndarray, "m d_in"],
    target_eval: Float[np.ndarray, "m d_out"],
) -> tuple[float, float]:
    """Fit a transform of type ``kind`` on train pairs and score R^2.

    Returns:
        A tuple ``(r2_train, r2_eval)`` of in-sample and held-out R^2. Comparing
        the two exposes overfitting (a large train-eval gap).
    """
    if kind == "linear":
        transform = fit_linear(source_train, target_train)
        r2_train = r2_score(transform.apply(source_train), target_train)
        r2_eval = r2_score(transform.apply(source_eval), target_eval)
        return r2_train, r2_eval
    if kind == "rigid":
        transform = fit_rigid(source_train, target_train)
        r2_train = r2_score(transform.apply(source_train), transform.project_target(target_train))
        r2_eval = r2_score(transform.apply(source_eval), transform.project_target(target_eval))
        return r2_train, r2_eval
    raise ValueError(f"Unknown transform kind {kind!r}")
