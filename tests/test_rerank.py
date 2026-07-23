"""CPU tests for the two-stage positional rerank attack (experiments.retrieval_rerank).

No colpali/torch: uses the MockRetriever + synthetic id cards (PIL only). These assert the rerank
machinery is *well-formed* (valid orderings, recovery numbers in [0, 1]), not that the mock recovers
PII (its embeddings are random). The real recovery/lift numbers come from the GPU run.
"""

import numpy as np

from patchguard.data.synthdoc import generate_id_cards
from patchguard.retrievers.mock import MockRetriever
from experiments.retrieval_rerank import (
    _cell_box,
    _ngrams,
    positional_score,
    rerank_order,
    run_attack,
    two_stage_recover,
)


def _rng_id(rng):
    return f"{int(rng.integers(10_000_000, 99_999_999))}"


def _rng_dob(rng):
    return f"{int(rng.integers(1, 13)):02d}/{int(rng.integers(1, 29)):02d}/{int(rng.integers(1950, 2005))}"


def test_ngrams_positions_and_fallback():
    grams = _ngrams("12345678", 2)
    assert grams[0] == (0, "12")
    assert grams[-1] == (6, "78")
    assert len(grams) == 7
    # too-short string falls back to the whole string at position 0
    assert _ngrams("5", 2) == [(0, "5")]


def test_cell_box_partitions_left_to_right():
    box = (0.0, 0.0, 80.0, 10.0)
    first = _cell_box(box, 8, 0, 0)   # first of 8 cells -> [0,10]
    last = _cell_box(box, 8, 7, 7)    # last cell -> [70,80]
    assert first == (0.0, 0.0, 10.0, 10.0)
    assert last == (70.0, 0.0, 80.0, 10.0)
    # a two-char span is twice as wide and cells advance rightward
    span = _cell_box(box, 8, 2, 3)    # cols 2..3 -> [20,40]
    assert span == (20.0, 0.0, 40.0, 10.0)
    assert last[0] > first[0]


def test_rerank_order_is_valid_permutation():
    rng = np.random.default_rng(0)
    for k in (1, 2, 5, 20):
        pos = rng.random(k)
        s1 = rng.random(k)
        order = rerank_order(pos, s1)
        assert order.shape == (k,)
        assert sorted(order.tolist()) == list(range(k))  # a permutation
        # top of the ordering is the max positional score
        assert np.isclose(pos[order[0]], pos.max())


def test_rerank_order_breaks_ties_by_stage1():
    pos = np.array([1.0, 1.0, 0.0])          # first two tie on positional score
    s1 = np.array([0.2, 0.9, 5.0])           # index 1 wins the tie via higher Stage-1
    order = rerank_order(pos, s1)
    assert order[0] == 1
    assert order[-1] == 2                     # lowest positional score sinks to the bottom


def test_positional_score_is_finite_float():
    cards = generate_id_cards(1, seed=3, value_font_size=34, vary=True)
    im, fs = cards[0]
    r = MockRetriever()
    enc = r.encode_page(im)
    orig_size = (im.shape[1], im.shape[0])
    idf = next(f for f in fs if f.field_type == "id_no")
    q = r.encode_query
    score = positional_score(idf.text, idf.box, enc, q, orig_size, ngram=2)
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_two_stage_recover_returns_valid_ordering():
    cards = generate_id_cards(1, seed=7, value_font_size=34, vary=True)
    im, fs = cards[0]
    r = MockRetriever()
    enc = r.encode_page(im)
    orig_size = (im.shape[1], im.shape[0])
    idf = next(f for f in fs if f.field_type == "id_no")
    rng = np.random.default_rng(1)
    distractors = list({_rng_id(rng) for _ in range(30)})
    res = two_stage_recover(enc, idf.box, idf.text, distractors, r.encode_query, orig_size,
                            topk=10, ngram=2)
    # true value inserted, so before/after are drawn from the lineup
    all_cands = set(distractors) | {idf.text}
    assert res["before"] in all_cands
    assert res["after"] in all_cands
    # in_topk is a well-formed flag (whether the truth survived Stage-1); mock embeddings are random
    # so its value is not guaranteed, only its type.
    assert isinstance(res["in_topk"], bool)
    # reranked is a permutation (as a multiset) of the carried top-K candidates
    assert len(res["reranked"]) == min(10, len(distractors) + 1)
    assert set(res["reranked"]).issubset(all_cands)
    assert isinstance(res["hit_before"], bool)
    assert isinstance(res["hit_after"], bool)


def test_run_attack_metrics_in_unit_range():
    cards = generate_id_cards(5, seed=2, value_font_size=34, vary=True)
    r = MockRetriever()
    rng = np.random.default_rng(9)
    pools = {
        "id_no": list({_rng_id(rng) for _ in range(40)}),
        "dob": list({_rng_dob(rng) for _ in range(40)}),
    }
    summary, rows = run_attack(r, cards, pools, topk=8, ngram=2)
    assert len(rows) == len(cards)
    for ft in ("id_no", "dob"):
        s = summary[ft]
        assert s["n"] == len(cards)
        for key in ("top1_before", "top1_after", "topk_recall"):
            assert 0.0 <= s[key] <= 1.0
        assert -1.0 <= s["lift"] <= 1.0
        assert np.isclose(s["lift"], s["top1_after"] - s["top1_before"])
