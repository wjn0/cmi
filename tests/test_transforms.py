"""Unit tests for cross-model transforms and R^2 scoring."""

import numpy as np
import pytest

from repsim.transforms import (
    evaluate_transform,
    fit_linear,
    fit_rigid,
    fit_whitening,
    r2_score,
    rbf_cka,
)


def test_r2_perfect_and_mean_baseline():
    target = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert r2_score(target, target) == pytest.approx(1.0)
    mean_pred = np.broadcast_to(target.mean(0), target.shape)
    assert r2_score(mean_pred, target) == pytest.approx(0.0)


def test_linear_recovers_affine_map():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(200, 5))
    weight = rng.normal(size=(5, 3))
    bias = rng.normal(size=3)
    target = source @ weight + bias

    transform = fit_linear(source, target)
    assert r2_score(transform.apply(source), target) == pytest.approx(1.0, abs=1e-8)


def test_linear_handles_different_dimensions():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(100, 8))
    target = rng.normal(size=(100, 4))
    transform = fit_linear(source, target)
    assert transform.apply(source).shape == (100, 4)


def test_rigid_recovers_rotation_and_scale():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(300, 4))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    target = 2.5 * (source @ rotation) + np.array([1.0, -2.0, 0.5, 3.0])

    transform = fit_rigid(source, target)
    assert transform.scale == pytest.approx(2.5, rel=1e-2)
    pred = transform.apply(source)
    assert r2_score(pred, transform.project_target(target)) == pytest.approx(1.0, abs=1e-6)


def test_rigid_projects_to_min_dimension():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(120, 6))
    target = rng.normal(size=(120, 3))
    transform = fit_rigid(source, target)
    assert transform.apply(source).shape == (120, 3)


def test_evaluate_transform_returns_train_and_eval_r2():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(200, 5))
    weight = rng.normal(size=(5, 3))
    target = source @ weight + rng.normal(size=3)
    half = 100
    r2_train, r2_eval = evaluate_transform(
        "linear", source[:half], target[:half], source[half:], target[half:]
    )
    # A noiseless affine relation generalises perfectly on both splits.
    assert r2_train == pytest.approx(1.0, abs=1e-8)
    assert r2_eval == pytest.approx(1.0, abs=1e-8)


def test_whitening_yields_zero_mean_identity_covariance():
    rng = np.random.default_rng(5)
    # Correlated, anisotropic data: a random linear mix of standard normals.
    base = rng.normal(size=(500, 4))
    mixed = base @ rng.normal(size=(4, 4)) + np.array([10.0, -3.0, 7.0, 0.0])

    whitened = fit_whitening(mixed, eps=0.0).apply(mixed)
    assert np.allclose(whitened.mean(axis=0), 0.0, atol=1e-8)
    cov = np.cov(whitened, rowvar=False, bias=True)  # bias=True: ÷n, as fit_whitening uses
    assert np.allclose(cov, np.eye(4), atol=1e-6)


def test_whitening_floors_degenerate_directions():
    rng = np.random.default_rng(6)
    # Last column is (near-)constant: zero variance that naive whitening blows up.
    x = rng.normal(size=(200, 3))
    x = np.hstack([x, np.full((200, 1), 2.0)])

    whitened = fit_whitening(x).apply(x)
    assert np.isfinite(whitened).all()


def test_rbf_cka_is_one_for_identical_representations():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(150, 6))
    assert rbf_cka(x, x) == pytest.approx(1.0, abs=1e-10)


def test_rbf_cka_invariant_to_orthogonal_transform_and_scale():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(150, 5))
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    y = 3.0 * (x @ rotation)  # isotropic scale + rotation leave RBF CKA unchanged
    assert rbf_cka(x, y) == pytest.approx(1.0, abs=1e-6)


def test_rbf_cka_is_symmetric_and_low_for_independent_data():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(200, 4))
    y = rng.normal(size=(200, 7))
    assert rbf_cka(x, y) == pytest.approx(rbf_cka(y, x), abs=1e-10)
    assert rbf_cka(x, y) < 0.1  # independent representations align weakly


def test_evaluate_transform_rbf_cka_returns_split_similarities():
    rng = np.random.default_rng(10)
    source = rng.normal(size=(120, 5))
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    target = source @ rotation
    half = 60
    cka_train, cka_eval = evaluate_transform(
        "rbf_cka", source[:half], target[:half], source[half:], target[half:]
    )
    assert cka_train == pytest.approx(1.0, abs=1e-6)
    assert cka_eval == pytest.approx(1.0, abs=1e-6)


def test_evaluate_transform_unknown_kind():
    x = np.zeros((4, 2))
    with pytest.raises(ValueError):
        evaluate_transform("nonsense", x, x, x, x)
