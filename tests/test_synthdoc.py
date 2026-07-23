"""CPU tests for the synthetic big-font ID card generator."""

import numpy as np

from patchguard.data.synthdoc import generate_id_card, generate_id_cards


def test_card_shape_and_fields():
    img, fields = generate_id_card(seed=0)
    assert img.shape == (448, 448, 3)
    assert img.dtype == np.uint8
    types = {f.field_type for f in fields}
    assert types == {"name", "dob", "id_no"}
    for f in fields:
        assert f.text  # non-empty ground-truth PII
        x0, y0, x1, y1 = f.box
        assert x1 > x0 and y1 > y0  # real box


def test_cards_are_distinct_and_deterministic():
    a = generate_id_cards(4, seed=1)
    b = generate_id_cards(4, seed=1)
    # deterministic
    assert all(np.array_equal(a[i][0], b[i][0]) for i in range(4))
    # distinct PII across cards (at least the id numbers differ)
    ids = [next(f.text for f in fs if f.field_type == "id_no") for _, fs in a]
    assert len(set(ids)) > 1


def test_boxes_inside_canvas():
    img, fields = generate_id_card(seed=3)
    h, w = img.shape[:2]
    for f in fields:
        x0, y0, x1, y1 = f.box
        assert 0 <= x0 < x1 <= w + 1
        assert 0 <= y0 < y1 <= h + 1
