from patchguard.eval.pfrr import field_recovery, levenshtein, normalize, pfrr


def test_normalize_folds_ocr_confusions():
    # "O" vs "0", "l" vs "1" should collapse after normalization.
    assert normalize("ACCT-OO12") == normalize("ACCT-0012")
    assert normalize("Ill") == normalize("111")


def test_normalize_strips_punctuation_and_case():
    assert normalize("John Doe") == normalize("johndoe")
    assert normalize("ACC 123-456") == normalize("acc123456")


def test_multichar_rule_rn_to_m():
    assert normalize("rnodern") == normalize("modern")


def test_field_recovery_raw_vs_normalized():
    # Account number off only by OCR-confusable chars: raw miss, normalized hit.
    r = field_recovery("account_no", truth="4055", recovered="4O5S")
    assert r.exact is False
    assert r.normalized_exact is True
    assert r.edit_distance == 0


def test_field_recovery_true_mismatch():
    r = field_recovery("name", truth="Smith", recovered="Jones")
    assert r.exact is False
    assert r.normalized_exact is False
    assert r.edit_distance > 0


def test_levenshtein_basic():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("", "abc") == 3


def test_pfrr_aggregates_by_field_type():
    results = [
        field_recovery("account_no", "4055", "4O5S"),  # normalized hit
        field_recovery("account_no", "1200", "1200"),  # hit
        field_recovery("account_no", "9999", "0000"),  # miss
        field_recovery("name", "Smith", "Smith"),  # hit
    ]
    agg = pfrr(results, normalized=True)
    assert agg["account_no"]["n"] == 3
    assert abs(agg["account_no"]["recovery_rate"] - 2 / 3) < 1e-9
    assert agg["name"]["recovery_rate"] == 1.0


def test_pfrr_raw_is_stricter_than_normalized():
    results = [field_recovery("id_no", "S012", "5O12")]  # only OCR-confusable diffs
    assert pfrr(results, normalized=False)["id_no"]["recovery_rate"] == 0.0
    assert pfrr(results, normalized=True)["id_no"]["recovery_rate"] == 1.0
