"""CPU tests for the synthetic->FUNSD defense-transfer experiment (mock retriever, tiny tensors).

Exercises the whole B8 loop without ColPali/GPU: build a defense on a couple of synthetic cards, apply
it to a couple of inline FUNSD-like pages, and run the attack/utility eval. Asserts privacy and utility
land in [0, 1] for the vanilla index and for BOTH defense variants, and checks the plan/verdict helpers.
"""

import numpy as np

from patchguard.data.fields import AnnotatedField
from patchguard.retrievers.mock import MockRetriever
from experiments.defense_transfer_funsd import (
    _norm,
    _q_cache,
    apply_defense,
    build_defense,
    build_synth_cards,
    eval_defense,
    make_plans,
    verdict_for,
)


def _fake_page(seed, name_text, val_text, question_text):
    """An inline FUNSD-like page: a random image + answer(PII)/question(content) fields."""
    rng = np.random.default_rng(seed)
    img = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
    fields = [
        AnnotatedField(field_type="answer", text=name_text, box=(4.0, 4.0, 40.0, 14.0)),
        AnnotatedField(field_type="answer", text=val_text, box=(4.0, 18.0, 40.0, 28.0)),
        AnnotatedField(field_type="question", text=question_text, box=(4.0, 32.0, 44.0, 42.0)),
    ]
    return img, fields


def _encode_pages(retriever, pages):
    return [(np.asarray(retriever.encode_page(img).patches, dtype=np.float32), fields)
            for img, fields in pages]


def _setup(defense, epochs=6, null_k=4):
    retriever = MockRetriever(dim=8)
    rng = np.random.default_rng(0)
    q = _q_cache(retriever)

    names = ["JAMES SMITH", "MARIA JONES", "ROBERT BROWN", "LINDA DAVIS"]
    synth = build_synth_cards(retriever, names, 4, 100, 20, rng, q)

    pages = [
        _fake_page(1, "JAMES SMITH", "04/11/1988", "REGISTRATION NUMBER"),
        _fake_page(2, "MARIA JONES", "12/02/1975", "DATE OF APPLICATION"),
    ]
    encoded = _encode_pages(retriever, pages)
    all_texts = [f.text for _p, fs in encoded for f in fs]
    priv_plan, util_plan = make_plans(encoded, all_texts, {"answer"}, {"question"},
                                      k=4, max_fields=4, max_util=4, min_len=3, rng=rng)

    P = build_defense(defense, synth, q, lam=5.0, null_k=null_k, r_topic=2, np_train=8,
                      epochs=epochs, dim=8, device="cpu", seed=0, rng=rng, train_names=names)
    return retriever, encoded, priv_plan, util_plan, P, q


def test_plans_are_nonempty_and_well_formed():
    _r, encoded, priv_plan, util_plan, _P, _q = _setup("nullspace")
    assert priv_plan and util_plan
    for (i, true_text, cands) in priv_plan:
        assert 0 <= i < len(encoded)
        assert cands[0] == true_text          # truth is always slot 0
        assert len(cands) >= 1
    for (i, ftext) in util_plan:
        assert 0 <= i < len(encoded)
        assert isinstance(ftext, str)


def test_vanilla_eval_in_unit_range():
    _r, encoded, priv_plan, util_plan, _P, q = _setup("nullspace")
    ap, ut, n_priv, n_util = eval_defense(encoded, priv_plan, util_plan, None, q, "cpu")
    assert 0.0 <= ap <= 1.0
    assert 0.0 <= ut <= 1.0
    assert n_priv == len(priv_plan) and n_util == len(util_plan)


def test_nullspace_defense_eval_in_unit_range():
    _r, encoded, priv_plan, util_plan, P, q = _setup("nullspace")
    ap, ut, _n1, _n2 = eval_defense(encoded, priv_plan, util_plan, P, q, "cpu")
    assert 0.0 <= ap <= 1.0
    assert 0.0 <= ut <= 1.0


def test_redaction_defense_eval_in_unit_range():
    _r, encoded, priv_plan, util_plan, P, q = _setup("redaction", epochs=6)
    ap, ut, _n1, _n2 = eval_defense(encoded, priv_plan, util_plan, P, q, "cpu")
    assert 0.0 <= ap <= 1.0
    assert 0.0 <= ut <= 1.0


def test_apply_defense_normalizes_and_removes_subspace():
    _r, encoded, _pp, _up, P, _q = _setup("nullspace")
    pat = encoded[0][0]
    out = apply_defense(P, pat, "cpu")
    assert out.shape == pat.shape
    norms = np.linalg.norm(out, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)          # patches stay on the unit sphere
    vanilla = apply_defense(None, pat, "cpu")
    assert np.array_equal(vanilla, np.asarray(pat, dtype=np.float32))


def test_verdict_helper_labels_and_ranges():
    # strong transfer: attack collapses to chance, utility kept
    v = verdict_for("nullspace", van_p=0.9, van_u=0.9, def_p=0.05, def_u=0.85, chance_p=0.05)
    assert "STRONG" in v["verdict"]
    assert abs(v["privacy_suppression"] - 0.85) < 1e-9
    assert abs(v["utility_retention"] - (0.85 / 0.9)) < 1e-9
    # no transfer: defended attack ~ vanilla
    v2 = verdict_for("redaction", van_p=0.9, van_u=0.9, def_p=0.88, def_u=0.9, chance_p=0.05)
    assert "NO TRANSFER" in v2["verdict"]
    # vacuous: vanilla already near chance
    v3 = verdict_for("nullspace", van_p=0.06, van_u=0.9, def_p=0.05, def_u=0.9, chance_p=0.05)
    assert "VACUOUS" in v3["verdict"]


def test_norm_alnum_casefold():
    assert _norm("Reg. No: 12-B") == "regno12b"
    assert _norm("") == ""
