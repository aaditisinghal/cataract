"""PII Field Recovery Rate (MASTER_PLAN S4, metric definition RESEARCH_PROTOCOL S5).

The primary metric and itself a contribution: SSIM says whether an image looks similar; PFRR says
whether the attacker got the account number. We OCR a reconstruction, then exact-match each field
against ground truth -- reporting BOTH raw and (OCR-confusion) normalized exact match.

The confusion table mirrors configs/ocr_normalization.yaml; it lives here too so the tested logic
core needs no YAML parser. Keep the two in sync.
"""

from __future__ import annotations

from dataclasses import dataclass

# OCR confuses these; normalized match folds each group to a canonical form. Multi-char rules
# (e.g. "rn"->"m") are applied before single-char rules. Mirror of ocr_normalization.yaml.
DEFAULT_CONFUSIONS: dict[str, str] = {
    # multi-char first
    "rn": "m",
    "cl": "d",
    # single-char folds -> canonical
    "O": "0",
    "o": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "l": "1",
    "|": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "g": "9",
}


def normalize(s: str, confusions: dict[str, str] | None = None) -> str:
    """Casefold, strip non-alphanumerics, then apply OCR-confusion folds.

    Applied identically to truth and recovery so a normalized exact-match credits the attacker when
    the only errors are OCR-ambiguous characters an attacker could resolve with a checksum/DB lookup.
    """
    table = DEFAULT_CONFUSIONS if confusions is None else confusions
    # keep only alphanumerics, casefold spacing/punctuation away
    cleaned = "".join(ch for ch in s if ch.isalnum())
    # multi-char rules (len>1) before single-char, longest-first for determinism
    for src in sorted((k for k in table if len(k) > 1), key=len, reverse=True):
        cleaned = cleaned.replace(src, table[src])
    cleaned = cleaned.casefold()
    # single-char folds applied to the (now casefolded) string; fold keys casefolded too
    single = {k.casefold(): v for k, v in table.items() if len(k) == 1}
    return "".join(single.get(ch, ch) for ch in cleaned)


def levenshtein(a: str, b: str) -> int:
    """Edit distance (pure python; fine for short PII field strings)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class FieldResult:
    field_type: str  # e.g. "name" | "account_no" | "dob" | "address" | "id_no"
    truth: str
    recovered: str
    exact: bool  # raw exact match (after alnum strip + casefold, no confusion folding)
    normalized_exact: bool  # after OCR-confusion folding
    edit_distance: int  # on the normalized strings


def field_recovery(
    field_type: str,
    truth: str,
    recovered: str,
    confusions: dict[str, str] | None = None,
) -> FieldResult:
    raw_t = "".join(ch for ch in truth if ch.isalnum()).casefold()
    raw_r = "".join(ch for ch in recovered if ch.isalnum()).casefold()
    nt = normalize(truth, confusions)
    nr = normalize(recovered, confusions)
    return FieldResult(
        field_type=field_type,
        truth=truth,
        recovered=recovered,
        exact=(raw_t == raw_r),
        normalized_exact=(nt == nr),
        edit_distance=levenshtein(nt, nr),
    )


def pfrr(results: list[FieldResult], normalized: bool = True) -> dict[str, dict[str, float]]:
    """Aggregate per field type. Returns {field_type: {"n", "recovery_rate", "mean_edit"}}.

    ``normalized=True`` scores against normalized exact-match (the headline); False = raw. Report
    both in the paper -- never hide the raw number.
    """
    by_type: dict[str, list[FieldResult]] = {}
    for r in results:
        by_type.setdefault(r.field_type, []).append(r)
    out: dict[str, dict[str, float]] = {}
    for ftype, group in by_type.items():
        hits = sum((r.normalized_exact if normalized else r.exact) for r in group)
        out[ftype] = {
            "n": float(len(group)),
            "recovery_rate": hits / len(group),
            "mean_edit": sum(r.edit_distance for r in group) / len(group),
        }
    return out
