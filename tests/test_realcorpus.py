"""CPU tests for the real-corpus transfer helpers (experiments/realcorpus_transfer.py).

All pure/scoring logic runs on the MOCK retriever + a small inline CORD-like ground_truth dict — no
datasets/colpali/GPU. Protects: (1) the gt_parse extractor pulls the right (field_type, text) leaves and
drops trivially-short values; (2) lineups always exclude the truth; (3) the probe returns top1/top5
in [0,1] and a truth-favouring scorer recovers the field.
"""

import numpy as np

from experiments.realcorpus_transfer import (
    _glyph_bucket,
    _length_bucket,
    _lineup,
    _norm,
    _probe,
    extract_cord_fields,
)
from patchguard.retrievers.base import maxsim
from patchguard.retrievers.mock import MockRetriever

# A trimmed but real-shaped CORD gt_parse: menu is a LIST of item dicts; totals are a nested block.
_GT = {
    "menu": [
        {"nm": "Latte", "cnt": "2", "price": "9.00"},
        {"nm": "Croissant", "price": "4.50"},
    ],
    "sub_total": {"subtotal_price": "13.50"},
    "total": {"total_price": "14.85", "cashprice": "20.00"},
}


def test_extractor_pulls_leaf_fields():
    fields = extract_cord_fields(_GT, min_len=3)
    pairs = set(fields)
    assert ("nm", "Latte") in pairs
    assert ("nm", "Croissant") in pairs
    assert ("price", "9.00") in pairs
    assert ("total_price", "14.85") in pairs
    assert ("cashprice", "20.00") in pairs
    assert ("subtotal_price", "13.50") in pairs


def test_extractor_drops_trivially_short():
    # cnt "2" has alnum length 1 < min_len -> dropped.
    fields = extract_cord_fields(_GT, min_len=3)
    assert all(not (ft == "cnt") for ft, _ in fields)
    # with min_len=1 it survives.
    fields2 = extract_cord_fields(_GT, min_len=1)
    assert ("cnt", "2") in set(fields2)


def test_extractor_handles_single_dict_menu():
    gt = {"menu": {"nm": "Espresso", "price": "3.00"}}
    fields = set(extract_cord_fields(gt, min_len=3))
    assert ("nm", "Espresso") in fields and ("price", "3.00") in fields


def test_lineup_excludes_truth_and_bounds_size():
    rng = np.random.default_rng(0)
    pool = ["Latte", "Croissant", "Espresso", "Mocha", "Latte", "latte"]
    lu = _lineup("Latte", pool, k=4, rng=rng)
    assert lu[0] == "Latte"
    assert len(lu) <= 4
    # no distractor matches the truth under normalization (case-folded, incl. the "latte" dup).
    assert all(_norm(x) != _norm("Latte") for x in lu[1:])


def test_lineup_short_pool_degrades_gracefully():
    rng = np.random.default_rng(1)
    lu = _lineup("only", ["only", "only"], k=5, rng=rng)
    assert lu == ["only"]  # no valid distractors -> just the truth


def test_probe_truth_favouring_scorer_recovers():
    def score_of(c):
        return 1.0 if c == "truth" else 0.0

    cands = ["a", "b", "truth", "c", "d", "e"]
    ranked, hit1, hit5 = _probe("truth", cands, score_of)
    assert ranked[0] == "truth"
    assert hit1 is True and hit5 is True


def test_probe_top1_top5_in_unit_interval_on_mock():
    retriever = MockRetriever(grid=(4, 4), dim=8, seed=0)
    rng = np.random.default_rng(3)
    # two fake pages, fields extracted from the CORD-like dict; distractor pool = all texts.
    fields = extract_cord_fields(_GT, min_len=3)
    all_texts = [t for _, t in fields] + ["Water", "Tea", "Juice", "88.00", "01/02/2026"]
    imgs = [(rng.random((32, 32, 3)) * 255).astype(np.uint8) for _ in range(2)]
    h1, h5 = [], []
    for img in imgs:
        enc = retriever.encode_page(img)
        score_of = lambda c: maxsim(retriever.encode_query(c), enc.patches)  # noqa: E731
        for _, truth in fields:
            cands = _lineup(truth, all_texts, k=5, rng=rng)
            assert all(_norm(x) != _norm(truth) for x in cands[1:])  # pool excludes truth
            _, hit1, hit5 = _probe(truth, cands, score_of)
            h1.append(int(hit1))
            h5.append(int(hit5))
    top1 = float(np.mean(h1))
    top5 = float(np.mean(h5))
    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top5 <= 1.0
    assert top5 >= top1  # top-5 recovery can only be >= top-1


def test_bucket_helpers():
    assert _length_bucket("ab") == "short(<=8)"
    assert _length_bucket("abcdefghij") == "long(>8)"
    assert _glyph_bucket("ab") == "glyph<=4"
    assert _glyph_bucket("abcdef") == "glyph5-10"
    assert _glyph_bucket("abcdefghijklm") == "glyph>10"
