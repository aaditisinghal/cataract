"""CPU tests for the Ghost Vectors soft-delete model (experiments/ghost_vectors.py).

No ColPali / GPU: tiny hand-crafted patches + the mock retriever's query oracle. We assert the SYSTEMS
invariant — that ``SoftDeleteIndex.delete`` tombstones without erasing — plus the security consequence:
a deleted doc is ABSENT from the logical view yet PRESENT and ATTACKABLE in the raw segment view.

Each card's patches are seeded with the mock query embedding of its true label, so the MaxSim
dictionary attack recovers that label deterministically. The labels are equal-length codes over
DISJOINT character sets: MaxSim sums a per-token max-dot, so (a) equal length removes the length bias
and (b) disjoint alphabets mean a distractor's tokens never exactly match the victim's stored tokens —
so the true label uniquely attains the theoretical max (one dot=1.0 self-match per token) and always
wins. (Real English names don't have this clean separation; the actual experiment uses ColPali
embeddings where the separation comes from the model, not the fixture.)
"""

import numpy as np

from patchguard.retrievers.mock import MockRetriever

from experiments.ghost_vectors import SoftDeleteIndex, recover_over_view

DIM = 8
# 8 equal-length (6-char) labels over pairwise-disjoint alphabets (A-F, G-L, ... a-l).
NAMES = ["ABCDEF", "GHIJKL", "MNOPQR", "STUVWX",
         "YZ0123", "456789", "abcdef", "ghijkl"]


def _mk_index(n=8, seed=0):
    """Build an index whose doc i carries the query embedding of NAMES[i] (so it's recoverable)."""
    mock = MockRetriever(dim=DIM)
    rng = np.random.default_rng(seed)
    index = SoftDeleteIndex()
    truths = {}
    originals = {}
    for i in range(n):
        name = NAMES[i % len(NAMES)]
        q = np.asarray(mock.encode_query(name), dtype=np.float32)  # (nq, DIM) unit-norm
        noise = rng.standard_normal((3, DIM)).astype(np.float32)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True) + 1e-8
        patches = np.concatenate([q, noise], axis=0)
        index.insert(i, patches, meta={"name": name})
        truths[i] = name
        originals[i] = patches.copy()
    return index, truths, originals, mock.encode_query


def test_insert_and_raw_read():
    index, truths, originals, _ = _mk_index(6)
    assert index.n_raw() == 6
    assert len(index) == 6  # nothing deleted yet -> logical == raw
    raw = list(index.raw_segment_read())
    assert len(raw) == 6
    assert all(t is False for _, _, t in raw)  # no tombstones yet
    # raw read returns byte-identical vectors
    for doc_id, patches, _ in raw:
        assert np.array_equal(patches, originals[doc_id])


def test_delete_tombstones_without_erasing():
    index, truths, originals, _ = _mk_index(6)
    deleted = [1, 3, 5]
    for d in deleted:
        index.delete(d)

    # logical view drops the deleted docs
    assert index.logical_ids() == [0, 2, 4]
    assert len(index) == 3
    for d in deleted:
        assert index.is_tombstoned(d)
        assert index.logical_get(d) is None            # gone from the query interface
        assert index.raw_get(d) is not None            # STILL physically present
        assert np.array_equal(index.raw_get(d), originals[d])  # bytes untouched (no zeroing)

    # raw segment still enumerates ALL docs, tombstones flagged, bytes intact
    raw = list(index.raw_segment_read())
    assert [d for d, _, _ in raw] == [0, 1, 2, 3, 4, 5]
    tomb = {d: t for d, _, t in raw}
    assert tomb == {0: False, 1: True, 2: False, 3: True, 4: False, 5: True}


def test_deleted_absent_from_logical_but_attackable_in_raw():
    index, truths, originals, q_fn = _mk_index(8, seed=1)
    deleted = [0, 2, 4, 6]
    kept = [1, 3, 5, 7]
    for d in deleted:
        index.delete(d)

    n_distr = len(NAMES) - 1

    # LOGICAL view: deleted docs are simply not present -> attacker cannot reach them
    logical_hits = recover_over_view(index.logical_view(), truths, q_fn, NAMES, n_distr,
                                     np.random.default_rng(0))
    assert set(logical_hits.keys()) == set(kept)
    assert all(d not in logical_hits for d in deleted)

    # RAW view: deleted docs ARE present and the dictionary attack recovers every one of them
    raw_hits = recover_over_view(
        ((doc_id, patches) for doc_id, patches, _ in index.raw_segment_read()),
        truths, q_fn, NAMES, n_distr, np.random.default_rng(0))
    assert set(raw_hits.keys()) == set(range(8))
    deleted_recovery = np.mean([raw_hits[d] for d in deleted])
    assert deleted_recovery == 1.0  # tombstoned PII fully recovered from the raw segment


def test_index_guards():
    index, _, _, _ = _mk_index(2)
    # duplicate insert rejected
    try:
        index.insert(0, np.zeros((2, DIM), dtype=np.float32))
        assert False, "expected KeyError on duplicate doc_id"
    except KeyError:
        pass
    # delete of a missing id rejected
    try:
        index.delete(999)
        assert False, "expected KeyError on missing doc_id"
    except KeyError:
        pass
    # non-2D patches rejected
    try:
        index.insert(7, np.zeros((DIM,), dtype=np.float32))
        assert False, "expected ValueError on 1-D patches"
    except ValueError:
        pass
