from patchguard.data.funsd import FUNSD_LABELS, parse_funsd_form

# A minimal FUNSD-shaped annotation (the real schema, trimmed).
_DOC = {
    "form": [
        {
            "id": 0,
            "text": "REGISTRATION NO.",
            "box": [10, 20, 200, 45],
            "label": "question",
            "words": [
                {"text": "REGISTRATION", "box": [10, 20, 120, 45]},
                {"text": "NO.", "box": [125, 20, 200, 45]},
            ],
        },
        {
            "id": 1,
            "text": "4055-XY",
            "box": [210, 20, 320, 45],
            "label": "answer",
            "words": [{"text": "4055-XY", "box": [210, 20, 320, 45]}],
        },
        {"id": 2, "text": "   ", "box": [0, 0, 5, 5], "label": "other", "words": []},  # blank -> dropped
    ]
}


def test_entity_granularity():
    fields = parse_funsd_form(_DOC, granularity="entity")
    # blank element dropped -> 2 fields
    assert len(fields) == 2
    q = [f for f in fields if f.field_type == "question"][0]
    assert q.text == "REGISTRATION NO."
    assert q.box == (10.0, 20.0, 200.0, 45.0)
    a = [f for f in fields if f.field_type == "answer"][0]
    assert a.text == "4055-XY"


def test_word_granularity_splits_boxes():
    fields = parse_funsd_form(_DOC, granularity="word")
    # 2 words + 1 word + 0 = 3 word-fields
    assert len(fields) == 3
    texts = {f.text for f in fields}
    assert "REGISTRATION" in texts and "NO." in texts and "4055-XY" in texts


def test_labels_constant():
    assert FUNSD_LABELS == ("question", "answer", "header", "other")


def test_box_ordering_normalized():
    doc = {"form": [{"text": "x", "label": "other", "box": [300, 45, 210, 20], "words": []}]}
    f = parse_funsd_form(doc)[0]
    assert f.box == (210.0, 20.0, 300.0, 45.0)  # min/max normalized
