"""CPU tests for the per-field / combined multi-PII NullspaceRedaction defense (experiments.defense_pii).

Uses the deterministic MockRetriever + tiny tensors — no ColPali, no GPU. Exercises the subspace-union
algebra (combined D annihilates every field subspace) and the {defense}×{field} evaluation harness shape.
"""

import numpy as np
import torch

from experiments.adaptive_attack import Card
from experiments.defense_pii import (
    FIELDS,
    _field_true,
    build_field_subspaces,
    combine_subspaces,
    evaluate_defenses,
)
from patchguard.defense.nullspace import NullspaceRedaction


def _mock_cards(retriever, n, seed):
    """Tiny Cards from the mock retriever: random 3x3 pixel pages + per-field ground truth."""
    rng = np.random.default_rng(seed)
    cards = []
    for i in range(n):
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        enc = retriever.encode_page(img)
        nm = f"NAME{i % 4}"
        cards.append(Card(patches=enc.image_patches().astype(np.float32), name=nm, name_idx=i % 4,
                          topic=f"OFFICE{i % 3}", meta={"id_no": f"{10_000_000 + i}", "dob": f"01/0{i % 9 + 1}/1990"}))
    return cards


def test_field_true_reads_name_and_meta():
    c = Card(patches=np.zeros((2, 4), np.float32), name="JAMES SMITH", name_idx=0, topic="T",
             meta={"id_no": "12345678", "dob": "01/02/1990"})
    assert _field_true(c, "name") == "JAMES SMITH"
    assert _field_true(c, "id") == "12345678"
    assert _field_true(c, "dob") == "01/02/1990"


def test_build_field_subspaces_shapes():
    rng = np.random.default_rng(0)
    d, k, r = 16, 3, 4
    field_tokens = {f: rng.standard_normal((40, d)).astype(np.float32) for f in FIELDS}
    topic = rng.standard_normal((20, d)).astype(np.float32)
    Ds = build_field_subspaces(field_tokens, topic, k=k, r_topic=r)
    assert set(Ds) == set(FIELDS)
    for f in FIELDS:
        assert Ds[f].shape == (d, k)
        assert torch.allclose(Ds[f].T @ Ds[f], torch.eye(k), atol=1e-4)  # orthonormal columns


def test_combined_subspace_orthonormal_and_spans_union():
    rng = np.random.default_rng(1)
    d, k, r = 20, 3, 4
    field_tokens = {f: rng.standard_normal((50, d)).astype(np.float32) for f in FIELDS}
    topic = rng.standard_normal((25, d)).astype(np.float32)
    Ds = build_field_subspaces(field_tokens, topic, k=k, r_topic=r)
    Dc = combine_subspaces([Ds[f] for f in FIELDS])
    # orthonormal columns
    kk = Dc.shape[1]
    assert kk >= k and kk <= 3 * k
    assert torch.allclose(Dc.T @ Dc, torch.eye(kk), atol=1e-4)
    # every per-field direction lies inside span(Dc): projecting it onto Dc leaves it unchanged
    for f in FIELDS:
        v = Ds[f][:, 0]
        proj = Dc @ (Dc.T @ v)
        assert float((proj - v).norm()) < 1e-3


def test_combined_redaction_annihilates_every_field():
    rng = np.random.default_rng(2)
    d, k, r = 24, 2, 3
    field_tokens = {f: rng.standard_normal((40, d)).astype(np.float32) for f in FIELDS}
    topic = rng.standard_normal((20, d)).astype(np.float32)
    Ds = build_field_subspaces(field_tokens, topic, k=k, r_topic=r)
    P = NullspaceRedaction(combine_subspaces([Ds[f] for f in FIELDS])).eval()
    x = torch.randn(10, d)
    y = P(x)
    assert torch.allclose(y.norm(dim=-1), torch.ones(10), atol=1e-5)  # renormalized to the sphere
    # residual projection onto EVERY field subspace is ~zero
    for f in FIELDS:
        assert float((y @ Ds[f]).abs().max()) < 1e-4


def test_combine_subspaces_empty():
    z = torch.zeros(8, 0)
    out = combine_subspaces([z, z, z])
    assert out.shape == (8, 0)


def test_evaluate_defenses_matrix_shape_and_ranges():
    from patchguard.retrievers.mock import MockRetriever

    retriever = MockRetriever(seed=0)
    qcache: dict[str, np.ndarray] = {}

    def q(s):
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    cards = _mock_cards(retriever, 6, seed=3)
    d = cards[0].patches.shape[1]
    # tiny per-field subspaces from mock value-query embeddings
    field_tokens = {
        "name": np.concatenate([q(c.name) for c in cards], axis=0),
        "id": np.concatenate([q(c.meta["id_no"]) for c in cards], axis=0),
        "dob": np.concatenate([q(c.meta["dob"]) for c in cards], axis=0),
    }
    topic_tokens = np.concatenate([q(c.topic) for c in cards], axis=0)
    Ds = build_field_subspaces(field_tokens, topic_tokens, k=2, r_topic=1)
    defenses = {
        "none": None,
        "name": NullspaceRedaction(Ds["name"]).eval(),
        "combined": NullspaceRedaction(combine_subspaces([Ds[f] for f in FIELDS])).eval(),
    }
    field_pools = {
        "name": [f"NAME{i}" for i in range(6)],
        "id": [f"{10_000_000 + i}" for i in range(20)],
        "dob": [f"01/0{i % 9 + 1}/199{i % 10}" for i in range(20)],
    }
    nd = {"name": 3, "id": 5, "dob": 5}
    rng = np.random.default_rng(4)
    matrix, utility = evaluate_defenses(defenses, cards, q, field_pools, nd, "cpu", rng)

    assert set(matrix) == {"none", "name", "combined"}
    for dname in matrix:
        assert set(matrix[dname]) == set(FIELDS)
        for f in FIELDS:
            assert 0.0 <= matrix[dname][f] <= 1.0
        assert 0.0 <= utility[dname] <= 1.0
    # combined removes >= what name removes (rank union), so its own-name recovery is not higher than none
    assert matrix["combined"]["name"] <= matrix["none"]["name"] + 1e-9
