"""Arrangement control — Claim 1c NULL test (plan A5): is the leak CONTENT or ARRANGEMENT?

The dictionary name-recovery attack (see ``retrieval_attack.py``) recovers a card's name from its
stored ColPali patch embeddings by ranking the K=240 name vocabulary with MaxSim. A skeptic can ask:
maybe the attack keys on the *spatial arrangement* of the patch grid (where the name sits on the page)
rather than the glyph content inside those patches. This experiment settles that with a controlled
negative.

We run the SAME name-recovery attack on the SAME cards TWICE:
  * ORDERED  — patches in their natural row-major sequence.
  * SHUFFLED — the image-patch block randomly PERMUTED before scoring (prefix/instruction tokens fixed).

Prediction: NULL. MaxSim is a sum-over-query-tokens of a max-over-patches dot product, so it is exactly
invariant to any permutation of the patch axis. Shuffling the arrangement therefore CANNOT change the
score of any candidate, the ranking is identical, and ordered_recovery == shuffled_recovery bit-for-bit
(delta == 0). That is the honest, expected result: it proves the leak is CONTENT (which patches encode)
not ARRANGEMENT (their order), and it doubles as an integrity check on the attack pipeline — a non-zero
delta would reveal an accidental positional dependence or a bug. ``null_confirmed`` is True when the
recovery delta is (numerically) zero.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from patchguard.retrievers.base import maxsim


def permute_image_block(
    patches: np.ndarray, n_prefix: int, n_image: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return a copy of ``patches`` with the [n_prefix : n_prefix+n_image] rows permuted.

    Prefix tokens and any trailing (instruction) tokens keep their positions; only the spatial
    image-patch grid order is scrambled. Also returns the permutation used.
    """
    perm = rng.permutation(n_image)
    out = patches.copy()
    block = patches[n_prefix : n_prefix + n_image]
    out[n_prefix : n_prefix + n_image] = block[perm]
    return out, perm


def _score_lineup(patches: np.ndarray, true_val: str, lineup: list[str], q) -> tuple[str, int]:
    """Rank ``lineup`` by MaxSim against ``patches``; return (top1_string, hit(1/0))."""
    scores = np.array([maxsim(q(c), patches) for c in lineup])
    best = lineup[int(np.argmax(scores))]
    return best, int(best == true_val)


def run_arrangement(
    retriever,
    cards,
    name_pool: list[str],
    n_distractors: int,
    seed: int = 0,
) -> dict:
    """Core (backend-agnostic) arrangement control. Works with ColPali or the mock retriever.

    For each card: build one name lineup (true name + up to ``n_distractors`` distractors drawn from
    ``name_pool``), then recover the name twice — once from the natural patch order and once from a
    random permutation of the image-patch block. The lineup is shared across the two branches so the
    comparison is paired.
    """
    rng = np.random.default_rng(seed)
    qcache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    rows = []
    ordered_hits = 0
    shuffled_hits = 0
    agree = 0
    used_d = 0
    for i, (im, fs) in enumerate(cards):
        enc = retriever.encode_page(im)
        n_prefix = enc.n_prefix_tokens
        gh, gw = enc.grid
        n_image = gh * gw
        true_name = next(f.text for f in fs if f.field_type == "name")

        others = [x for x in name_pool if x != true_name]
        d = min(n_distractors, len(others))
        used_d = d
        lineup = [true_name] + list(rng.choice(others, d, replace=False)) if d > 0 else [true_name]

        rec_o, hit_o = _score_lineup(enc.patches, true_name, lineup, q)
        shuf, _perm = permute_image_block(enc.patches, n_prefix, n_image, rng)
        rec_s, hit_s = _score_lineup(shuf, true_name, lineup, q)

        ordered_hits += hit_o
        shuffled_hits += hit_s
        agree += int(hit_o == hit_s)
        rows.append(
            {
                "card": i,
                "true": true_name,
                "ordered_recovered": rec_o,
                "shuffled_recovered": rec_s,
                "ordered_top1": bool(hit_o),
                "shuffled_top1": bool(hit_s),
            }
        )

    n = len(cards)
    ordered = ordered_hits / n if n else 0.0
    shuffled = shuffled_hits / n if n else 0.0
    delta = ordered - shuffled
    return {
        "n": n,
        "distractors": used_d,
        "ordered_recovery": ordered,
        "shuffled_recovery": shuffled,
        "delta": delta,
        "per_card_agreement": (agree / n) if n else 1.0,
        "null_confirmed": bool(abs(delta) < 1e-9),
        "chance": 1.0 / (used_d + 1),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/arrangement_control")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=40, help="cards to attack")
    ap.add_argument("--distractors", type=int, default=200, help="name lineup distractors (<=239)")
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # K=240
    retriever = ColPaliRetriever(model_name=args.model)
    cards = generate_id_cards(args.n, seed=args.seed, value_font_size=args.font_size, vary=True)

    res = run_arrangement(retriever, cards, name_pool, args.distractors, seed=args.seed)

    for r in res["rows"]:
        print(
            f"card {r['card']:2d} | ordered {r['ordered_recovered']:<20} "
            f"({'OK' if r['ordered_top1'] else 'x'}) | shuffled {r['shuffled_recovered']:<20} "
            f"({'OK' if r['shuffled_top1'] else 'x'})"
        )
    print("\n=== ARRANGEMENT CONTROL (Claim 1c NULL) ===")
    print(f"  ordered  recovery = {res['ordered_recovery']:.3f}")
    print(f"  shuffled recovery = {res['shuffled_recovery']:.3f}")
    print(f"  delta (ord - shuf) = {res['delta']:+.3f}  (chance {res['chance']:.4f})")
    print(f"  per-card agreement = {res['per_card_agreement']:.3f}")
    print(
        "  NULL CONFIRMED: leak is CONTENT not arrangement (MaxSim order-invariant)"
        if res["null_confirmed"]
        else "  !! NON-NULL: arrangement changed recovery -> pipeline bug or positional coupling"
    )

    payload = {
        "mode": "arrangement_control",
        "model": args.model,
        "font_size": args.font_size,
        "seed": args.seed,
        **{k: res[k] for k in (
            "n", "distractors", "ordered_recovery", "shuffled_recovery", "delta",
            "per_card_agreement", "null_confirmed", "chance", "rows",
        )},
        "fingerprint": run_fingerprint(),
    }
    (local_out / "arrangement_control.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote arrangement_control.json -> {args.out}")


if __name__ == "__main__":
    main()
