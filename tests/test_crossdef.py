"""CPU tests for the cross-model defense + ablation building blocks (redact_variants).

All on tiny tensors / tiny epochs — no ColPali, no GPU. Exercises the forward contract of the alternative
redactor architectures, the gate config (learned vs fixed), the variant-agnostic trainer, and a minimal
ablation frontier loop that returns (utility, privacy) points the way the experiment does.
"""

import numpy as np
import torch

from patchguard.defense.redact import maxsim_batch
from patchguard.defense.redact_variants import (
    LinearRedaction,
    _parse_gate,
    build_redactor,
    train_variant,
)


def test_linear_redaction_shape_and_normalized():
    P = LinearRedaction(dim=8)
    x = torch.randn(4, 10, 8)
    y = P(x.reshape(-1, 8)).reshape(4, 10, 8)
    assert y.shape == (4, 10, 8)
    norms = y.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_linear_redaction_batched_forward_contract():
    # forward must accept an arbitrary (..., d) shape and normalize the last dim.
    P = LinearRedaction(dim=6)
    x = torch.randn(3, 5, 6)  # (B, np, d) — the shape train_variant feeds
    y = P(x)
    assert y.shape == (3, 5, 6)
    assert torch.allclose(y.norm(dim=-1), torch.ones(3, 5), atol=1e-5)


def test_parse_gate_modes():
    assert _parse_gate("learned") == ("learned", None)
    assert _parse_gate(True) == ("learned", None)
    assert _parse_gate("off") == ("fixed", 1.0)
    assert _parse_gate(False) == ("fixed", 1.0)
    assert _parse_gate(0.5) == ("fixed", 0.5)
    assert _parse_gate("0.25") == ("fixed", 0.25)


def test_linear_gate_learned_vs_fixed():
    learned = LinearRedaction(dim=8, gate="learned")
    assert isinstance(learned.gate, torch.nn.Parameter)
    assert learned.gate.requires_grad
    off = LinearRedaction(dim=8, gate="off")
    assert not isinstance(off.gate, torch.nn.Parameter)  # fixed -> buffer, not a trainable param
    assert float(off.gate) == 1.0
    assert all(p is not off.gate for p in off.parameters())  # buffer, excluded from params
    half = LinearRedaction(dim=8, gate=0.5)
    assert abs(float(half.gate) - 0.5) < 1e-6


def test_build_redactor_variants_forward_and_gate():
    for variant in ("linear", "mlp-depth1", "mlp-depth2", "mlp-depth3"):
        P = build_redactor(variant, dim=8, hidden=16, gate="learned")
        y = P(torch.randn(2, 7, 8))
        assert y.shape == (2, 7, 8)
        assert torch.allclose(y.norm(dim=-1), torch.ones(2, 7), atol=1e-5)
    # depth grows parameter count strictly for the MLP cells
    n1 = sum(p.numel() for p in build_redactor("mlp-depth1", dim=8, hidden=16).parameters())
    n3 = sum(p.numel() for p in build_redactor("mlp-depth3", dim=8, hidden=16).parameters())
    assert n3 > n1
    # gate "off" freezes the MLP gate at full strength and drops it from trainable params
    off = build_redactor("mlp-depth2", dim=8, hidden=16, gate="off")
    assert not off.gate.requires_grad
    assert abs(float(off.gate) - 1.0) < 1e-6


def test_build_redactor_unknown_variant_raises():
    try:
        build_redactor("transformer", dim=8)
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown variant")


def test_train_variant_runs_and_returns_eval_module():
    torch.manual_seed(0)
    B, npch, d = 6, 12, 8
    patches = torch.randn(B, npch, d)
    patches = patches / patches.norm(dim=-1, keepdim=True)
    topic_q = [torch.randn(3, d) for _ in range(B)]
    name_q = [torch.randn(3, d) for _ in range(B)]
    distr = [torch.randn(3, d) for _ in range(6)]
    P = build_redactor("linear", dim=d, gate="learned")
    trained = train_variant(P, patches, topic_q, name_q, distr, lam=5.0, epochs=20)
    assert not trained.training  # returned in eval mode
    out = trained(patches.reshape(-1, d))
    assert out.shape == (B * npch, d)
    assert torch.allclose(out.norm(dim=-1), torch.ones(B * npch), atol=1e-5)


def test_train_variant_skips_frozen_gate():
    # a gate-off MLP has a frozen gate; training must run (optimizer only sees trainable params).
    torch.manual_seed(1)
    B, npch, d = 4, 8, 6
    patches = torch.randn(B, npch, d)
    patches = patches / patches.norm(dim=-1, keepdim=True)
    topic_q = [torch.randn(2, d) for _ in range(B)]
    name_q = [torch.randn(2, d) for _ in range(B)]
    distr = [torch.randn(2, d) for _ in range(4)]
    P = build_redactor("mlp-depth2", dim=d, hidden=12, gate="off")
    gate_before = float(P.gate)
    trained = train_variant(P, patches, topic_q, name_q, distr, lam=2.0, epochs=10)
    assert float(trained.gate) == gate_before  # frozen gate never moved


def _frontier_point(P, docs, topic_q, name_q, distractor_q):
    """Minimal mirror of the experiment's frontier_point on tiny tensors: (utility, privacy)."""
    with torch.no_grad():
        Pt = [P(d) for d in docs]
    hit_u = []
    for i in range(len(docs)):
        scores = torch.stack([maxsim_batch(topic_q[i], dp[None])[0] for dp in Pt])
        hit_u.append(int(torch.argmax(scores).item() == i))
    rec = []
    for i in range(len(docs)):
        cands = [name_q[i]] + distractor_q
        sc = torch.stack([maxsim_batch(c, Pt[i][None])[0] for c in cands])
        rec.append(int(torch.argmax(sc).item() == 0))
    return float(np.mean(hit_u)), 1.0 - float(np.mean(rec))


def test_tiny_ablation_loop_returns_frontier_points():
    torch.manual_seed(2)
    B, npch, d = 5, 10, 8
    patches = torch.randn(B, npch, d)
    patches = patches / patches.norm(dim=-1, keepdim=True)
    docs = [patches[i] for i in range(B)]
    topic_q = [torch.randn(2, d) for _ in range(B)]
    name_q = [torch.randn(2, d) for _ in range(B)]
    distr = [torch.randn(2, d) for _ in range(6)]

    variants = ["linear", "mlp-depth1", "mlp-depth2"]
    frontier = []
    for idx, v in enumerate(variants):
        torch.manual_seed(10 + idx)
        P = build_redactor(v, dim=d, hidden=16, gate="on")
        P = train_variant(P, patches, topic_q, name_q, distr, lam=5.0, epochs=15)
        u, p = _frontier_point(P, docs, topic_q, name_q, distr)
        frontier.append({"variant": v, "utility": u, "privacy": p})

    assert len(frontier) == len(variants)
    for pt in frontier:
        assert 0.0 <= pt["utility"] <= 1.0
        assert 0.0 <= pt["privacy"] <= 1.0
