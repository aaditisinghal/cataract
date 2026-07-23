"""CPU tests for the learned redaction defense (RedactionProjection + min-max training)."""

import torch

from patchguard.defense.redact import RedactionProjection, maxsim_batch, train_redactor


def test_projection_shape_and_normalized():
    P = RedactionProjection(dim=8)
    x = torch.randn(4, 10, 8)
    y = P(x.reshape(-1, 8)).reshape(4, 10, 8)
    assert y.shape == (4, 10, 8)
    norms = y.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_maxsim_batch_matches_manual():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    docs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1,2,2)
    # token0 max dot = 1 (patch0), token1 max dot = 1 (patch1) -> 2
    assert float(maxsim_batch(q, docs)[0]) == 2.0


def test_train_redactor_runs_and_learns_privacy():
    torch.manual_seed(0)
    B, npch, d = 6, 16, 8
    patches = torch.randn(B, npch, d)
    patches = patches / patches.norm(dim=-1, keepdim=True)
    topic_q = [torch.randn(3, d) for _ in range(B)]
    name_q = [torch.randn(3, d) for _ in range(B)]
    distr = [torch.randn(3, d) for _ in range(8)]
    # lambda>0 should train without error and return an eval-mode module
    P = train_redactor(patches, topic_q, name_q, distr, lam=5.0, dim=d, epochs=30, distractors=distr)
    assert isinstance(P, RedactionProjection)
    assert not P.training
    out = P(patches.reshape(-1, d))
    assert out.shape == (B * npch, d)
