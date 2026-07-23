"""CPU tests for the information-destroying (nullspace) defense."""

import numpy as np
import torch

from patchguard.defense.nullspace import NullspaceRedaction, _top_dirs, pii_directions


def test_top_dirs_orthonormal_shape():
    M = torch.randn(50, 16)
    D = _top_dirs(M, 4)
    assert D.shape == (16, 4)
    # columns orthonormal
    assert torch.allclose(D.T @ D, torch.eye(4), atol=1e-4)


def test_top_dirs_zero_rank():
    D = _top_dirs(torch.randn(10, 8), 0)
    assert D.shape == (8, 0)


def test_pii_directions_orthonormal():
    rng = np.random.default_rng(0)
    name = rng.standard_normal((120, 32)).astype(np.float32)
    topic = rng.standard_normal((60, 32)).astype(np.float32)
    D = pii_directions(name, topic, k=6, r_topic=4)
    assert D.shape == (32, 6)
    assert torch.allclose(D.T @ D, torch.eye(6), atol=1e-4)


def test_redaction_normalized_and_removes_subspace():
    rng = np.random.default_rng(1)
    name = rng.standard_normal((120, 16)).astype(np.float32)
    topic = rng.standard_normal((60, 16)).astype(np.float32)
    D = pii_directions(name, topic, k=4, r_topic=4)
    P = NullspaceRedaction(D).eval()
    x = torch.randn(20, 16)
    y = P(x)
    # unit-norm output (patches live on the sphere)
    assert torch.allclose(y.norm(dim=-1), torch.ones(20), atol=1e-5)
    # the removed component is annihilated: y has ~zero projection onto D
    proj = (y @ D)  # (20, k)
    assert float(proj.abs().max()) < 1e-4


def test_information_destroyed_on_span_D():
    # a vector lying entirely in span(D) must be crushed toward zero before renormalization
    rng = np.random.default_rng(2)
    name = rng.standard_normal((80, 12)).astype(np.float32)
    topic = rng.standard_normal((40, 12)).astype(np.float32)
    D = pii_directions(name, topic, k=3, r_topic=3)
    v = D[:, 0].clone()  # purely in the removed subspace
    removed = v - (v @ D) @ D.T
    assert float(removed.norm()) < 1e-4  # nothing left -> non-invertible on this direction


def test_k_property():
    D = pii_directions(np.random.randn(50, 8).astype(np.float32),
                       np.random.randn(30, 8).astype(np.float32), k=2, r_topic=2)
    assert NullspaceRedaction(D).k == 2
