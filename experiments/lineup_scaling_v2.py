"""Corrected lineup-size scaling (fixes the length-bias artifact in ``lineup_scaling.py``; C24).

``lineup_scaling.py`` found recovery collapsing from 1.00 at K=100 to 0.00 at K>=1000. RESULTS.md
diagnosed this as an artifact, not graceful decay: once the 240-name base vocabulary (16 first x 15
last) is exhausted, ``generate_name_candidates`` pads with middle initials ("JAMES A SMITH", "JAMES A
B SMITH") to reach large K. Because MaxSim *sums* over query tokens, a 3- or 4-token distractor
accumulates more max-over-patches terms than the true 2-token name and wins on raw token count, not on
being a harder distractor.

The fix: keep every candidate — true values and distractors alike — at exactly two space-separated
name-like tokens, for any K up to 10^5, by drawing first/last names from a large *procedurally
generated* two-syllable vocabulary (400 first x 400 last = 160,000 >= max K) instead of padding with
initials. This isolates lineup-size difficulty from token-count confound: everything this module adds
lives here; ``lineup_scaling.py`` and its paper-referenced numbers are untouched.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.lineup_scaling import (
    absent_target_eval,
    score_pool_per_card,
    scaling_curve,
)

_CONSONANTS = list("bcdfgklmnprstv")
_VOWELS = list("aeiou")


def _syllable(rng: np.random.Generator) -> str:
    return f"{rng.choice(_CONSONANTS)}{rng.choice(_VOWELS)}"


def _name_token(rng: np.random.Generator, n_syl: int = 2) -> str:
    """A single pronounceable, name-like token, e.g. 'Kavelin' — one word, no internal spaces."""
    s = "".join(_syllable(rng) for _ in range(n_syl))
    return s[0].upper() + s[1:]


def generate_length_matched_candidates(
    n: int, first: list[str], last: list[str], rng: np.random.Generator
) -> list[str]:
    """``n`` unique two-token "FIRST LAST" strings; every candidate is exactly 2 tokens.

    Starts from the real closed vocabulary (``first`` x ``last``, matching the true names actually
    rendered on synthetic cards) so small-K behavior is identical to ``lineup_scaling.py``. Once that
    240-name base is exhausted, extends with procedurally generated two-syllable first/last tokens
    (still exactly 2 words total) rather than appending extra tokens to existing names.
    """
    base = [f"{a} {b}" for a in first for b in last]
    rng.shuffle(base)
    out: list[str] = list(dict.fromkeys(base))
    if len(out) >= n:
        return out[:n]
    seen = set(out)
    # procedurally extend the first/last vocabularies until |F|*|L| >= remaining need
    firsts, lasts = list(first), list(last)
    f_seen, l_seen = set(firsts), set(lasts)
    while (len(firsts) * len(lasts)) < n + len(firsts) + len(lasts):
        f = _name_token(rng)
        if f not in f_seen:
            f_seen.add(f)
            firsts.append(f)
        l = _name_token(rng)
        if l not in l_seen:
            l_seen.add(l)
            lasts.append(l)
    extra = [f"{a} {b}" for a in firsts for b in lasts if f"{a} {b}" not in seen]
    rng.shuffle(extra)
    for s in extra:
        if len(out) >= n:
            break
        out.append(s)
        seen.add(s)
    if len(out) < n:
        raise RuntimeError(f"could not generate {n} unique length-matched candidates, got {len(out)}")
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/lineup_scaling_v2")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=40, help="cards to attack")
    ap.add_argument("--Ks", default="100,1000,10000,100000", help="comma-separated lineup sizes")
    ap.add_argument("--absent-frac", type=float, default=0.5, help="fraction of trials with target removed")
    ap.add_argument("--font-size", type=int, default=34)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Ks = sorted({int(x) for x in str(args.Ks).split(",") if x.strip()})

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    pool = generate_length_matched_candidates(max(Ks), _FIRST, _LAST, rng)
    tok_lens = {len(c.split()) for c in pool}
    print(f"candidate pool: {len(pool)} unique length-matched name strings (target {max(Ks)}); "
          f"token-length set = {tok_lens} (must be {{2}})")
    assert tok_lens == {2}, "length-matching invariant violated"

    retriever = ColPaliRetriever(model_name=args.model)
    cards = generate_id_cards(args.n, seed=args.seed, value_font_size=args.font_size, vary=True)

    per_card = score_pool_per_card(retriever, cards, pool, field="name")
    scaling = scaling_curve(per_card, Ks)
    absent = absent_target_eval(per_card, lineup_size=min(Ks), absent_frac=args.absent_frac, seed=args.seed)

    print("\n=== (1) CORRECTED LINEUP-SIZE SCALING (length-matched, name top-1 recovery) ===")
    for s in scaling:
        print(f"  K={s['K']:>7d}  recovery={s['recovery']:.3f}  lift={s['lift']:.1f}x  (chance {s['chance']:.2e})")
    print("\n=== (2) ABSENT-TARGET / open-world rejection ===")
    print(f"  lineup_size={absent['lineup_size']}  present={absent['n_present']}  absent={absent['n_absent']}")
    print(f"  chosen margin threshold = {absent['threshold']}")
    print(f"  true-accept (present)   = {absent['true_accept']:.3f}")
    print(f"  false-accept (absent)   = {absent['false_accept']:.3f}")
    print(f"  ROC AUC                 = {absent['auc']}")

    payload = {
        "mode": "lineup_scaling_v2_length_matched",
        "model": args.model,
        "n": len(cards),
        "font_size": args.font_size,
        "seed": args.seed,
        "absent_frac": args.absent_frac,
        "Ks": Ks,
        "pool_size": len(pool),
        "pool_token_lengths": sorted(tok_lens),
        "scaling": scaling,
        "absent_target": absent,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "lineup_scaling_v2.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"\nwrote lineup_scaling_v2.json -> {args.out}")


if __name__ == "__main__":
    main()
