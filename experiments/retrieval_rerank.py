"""Two-stage positional rerank attack (MASTER_PLAN S6, headline follow-up) — close the
digit-TRANSPOSITION gap on NUMERIC PII.

ColPali late interaction (MaxSim) sums, per query token, the max dot with ANY patch. It is therefore
ORDER-INVARIANT: a query is a *bag* of sub-tokens. For numeric PII this caps exact-string recovery,
because two candidates that are digit transpositions of each other (e.g. ``12345678`` vs ``21345678``,
or ``05/26/1973`` vs ``05/62/1973``) share the same token multiset and score almost identically. The
plain retrieval/dictionary attack (``retrieval_attack.py``) can pick the right digit *set* but ties on
its *order*, so the true value and a transposed twin fight for rank #1.

This experiment adds POSITION back in a second stage:

  Stage 1 (recall)  — MaxSim over the full D+1 candidate lineup (reuse retrieval_attack's spaces:
                      id = 8-digit lineup, dob = date lineup). Keep the top-K by MaxSim.
  Stage 2 (order)   — rerank the top-K by a POSITIONAL-CONSISTENCY score. Using the field box -> patch
                      alignment (``patchguard.data.align``), we split the field box into per-character
                      cells, and score each digit-BIGRAM of a candidate ONLY against the patch window
                      where that position physically sits. A candidate whose digits land in the right
                      cells scores higher; a transposition is penalized because its digits are queried
                      against the wrong spatial windows.

Reports exact-string top-1 recovery BEFORE (Stage-1 argmax) vs AFTER (Stage-2 rerank) for id_no and
dob, plus the LIFT (after - before) and top-K recall. A positive lift means the index leaks not only
the digit multiset but their spatial ORDER — a strictly stronger leak. A null lift means positional
information is not recoverable from the stored patches (bag-of-tokens is all there is).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

# align + maxsim are numpy-only (no torch/colpali) so they are safe at module top level and let the
# pure rerank helpers below be imported/unit-tested on CPU with the mock retriever.
from patchguard.data.align import boxes_to_patch_mask
from patchguard.retrievers.base import maxsim


def _ngrams(s: str, n: int) -> list[tuple[int, str]]:
    """Positional n-grams: ``[(start_index, gram), ...]``. Falls back to the whole string if too short."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(s) < n:
        return [(0, s)]
    return [(k, s[k : k + n]) for k in range(len(s) - n + 1)]


def _cell_box(box, n_chars: int, k0: int, k1: int):
    """Sub-box covering character positions ``k0..k1`` (inclusive) of a text field.

    The field box is split into ``n_chars`` equal-width horizontal cells (monospace-ish assumption:
    good enough to localize which patch column a digit sits in). Height is kept full.
    """
    x0, y0, x1, y1 = box
    n_chars = max(n_chars, 1)
    cw = (x1 - x0) / n_chars
    left = x0 + k0 * cw
    right = x0 + (k1 + 1) * cw
    return (float(left), float(y0), float(right), float(y1))


def positional_score(
    cand: str,
    box,
    enc,
    q_fn,
    orig_size: tuple[int, int],
    ngram: int = 2,
    coverage_threshold: float = 0.0,
) -> float:
    """Positional-consistency score of one candidate string against a page's patches.

    For each positional n-gram of ``cand``, restrict the patch set to the cell where that position
    sits (via the box->patch alignment) and take MaxSim of the n-gram query against ONLY those patches.
    Average over n-grams so the scale is comparable across candidates of the same length. If a cell
    selects no patch (very coarse grids), fall back to the full image-patch set for that n-gram (a weak,
    non-positional term) so the score is always defined.
    """
    img_p = enc.image_patches()
    n_chars = len(cand)
    total = 0.0
    count = 0
    for k0, gram in _ngrams(cand, ngram):
        k1 = k0 + len(gram) - 1
        cbox = _cell_box(box, n_chars, k0, k1)
        mask = boxes_to_patch_mask(
            [cbox], orig_size, enc.grid, enc.input_size,
            resize_policy=enc.resize_policy, coverage_threshold=coverage_threshold, n_prefix_tokens=0,
        )
        sel = img_p[mask]
        if sel.shape[0] == 0:
            sel = img_p  # degrade gracefully to bag-of-tokens for this position
        total += maxsim(q_fn(gram), sel)
        count += 1
    return total / max(count, 1)


def rerank_order(pos_scores: np.ndarray, stage1_scores: np.ndarray) -> np.ndarray:
    """Return indices that sort candidates by positional score desc, tie-broken by Stage-1 desc.

    Always a valid permutation of ``range(len(pos_scores))``.
    """
    pos = np.asarray(pos_scores, dtype=float)
    s1 = np.asarray(stage1_scores, dtype=float)
    if pos.shape != s1.shape or pos.ndim != 1:
        raise ValueError("pos_scores and stage1_scores must be 1-D arrays of equal length")
    # np.lexsort: last key is primary. primary = -pos (desc), tie-break = -s1 (desc).
    return np.lexsort((-s1, -pos))


def two_stage_recover(enc, box, true_val, candidates, q_fn, orig_size,
                      topk: int = 20, ngram: int = 2) -> dict:
    """Run both stages for one field. Returns before/after recovery + the reranked top-K."""
    cands = list(candidates) if true_val in candidates else [true_val, *candidates]
    s1 = np.array([maxsim(q_fn(c), enc.patches) for c in cands])
    order1 = np.argsort(-s1)
    before = cands[int(order1[0])]

    k = min(topk, len(cands))
    top_idx = order1[:k]
    top_cands = [cands[int(i)] for i in top_idx]
    top_s1 = s1[top_idx]
    pos = np.array([positional_score(c, box, enc, q_fn, orig_size, ngram=ngram) for c in top_cands])
    order2 = rerank_order(pos, top_s1)
    reranked = [top_cands[int(i)] for i in order2]
    after = reranked[0]
    return {
        "true": true_val,
        "before": before,
        "after": after,
        "hit_before": before == true_val,
        "hit_after": after == true_val,
        "in_topk": true_val in top_cands,
        "reranked": reranked,
    }


def _make_qcache(encode_query):
    cache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in cache:
            cache[s] = encode_query(s)
        return cache[s]

    return q


def run_attack(retriever, cards, pools: dict[str, list[str]], topk: int = 20, ngram: int = 2,
               q_fn=None, field_types=("id_no", "dob")) -> tuple[dict, list[dict]]:
    """Core attack loop (retriever-agnostic; unit-tested on CPU with the mock retriever).

    ``cards`` is a list of ``(image_np, [AnnotatedField, ...])``. ``pools[ft]`` is the distractor
    candidate space for field type ``ft`` (the true value is inserted per card). Returns
    ``(summary, rows)`` with before/after top-1 accuracy and the lift for each field type.
    """
    if q_fn is None:
        q_fn = _make_qcache(retriever.encode_query)
    agg = {ft: {"before": 0, "after": 0, "in_topk": 0, "n": 0} for ft in field_types}
    rows: list[dict] = []
    for ci, (im, fs) in enumerate(cards):
        enc = retriever.encode_page(im)
        orig_size = (int(im.shape[1]), int(im.shape[0]))  # (width, height)
        truth = {f.field_type: f for f in fs}
        row = {"card": ci}
        for ft in field_types:
            if ft not in truth:
                continue
            f = truth[ft]
            res = two_stage_recover(enc, f.box, f.text, pools[ft], q_fn, orig_size,
                                    topk=topk, ngram=ngram)
            a = agg[ft]
            a["n"] += 1
            a["before"] += int(res["hit_before"])
            a["after"] += int(res["hit_after"])
            a["in_topk"] += int(res["in_topk"])
            row[ft] = {"true": res["true"], "before": res["before"], "after": res["after"],
                       "hit_before": bool(res["hit_before"]), "hit_after": bool(res["hit_after"])}
        rows.append(row)

    summary: dict[str, dict] = {}
    for ft in field_types:
        a = agg[ft]
        n = max(a["n"], 1)
        b, af = a["before"] / n, a["after"] / n
        summary[ft] = {
            "n": a["n"],
            "top1_before": b,
            "top1_after": af,
            "lift": af - b,
            "topk_recall": a["in_topk"] / n,
        }
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/retrieval_rerank")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=40, help="cards to attack")
    ap.add_argument("--distractors", type=int, default=999, help="lineup size D for id/dob")
    ap.add_argument("--topk", type=int, default=20, help="Stage-1 candidates carried into the rerank")
    ap.add_argument("--font-size", type=int, default=34)
    ap.add_argument("--ngram", type=int, default=2, help="positional n-gram width (2 = bigram)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.data.synthdoc import generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    retriever = ColPaliRetriever(model_name=args.model)
    q = _make_qcache(retriever.encode_query)

    def rand_id() -> str:
        return f"{int(rng.integers(10_000_000, 99_999_999))}"

    def rand_dob() -> str:
        return f"{int(rng.integers(1, 13)):02d}/{int(rng.integers(1, 29)):02d}/{int(rng.integers(1950, 2005))}"

    # candidate lineups mirror retrieval_attack.py exactly (id = 8-digit, dob = valid dates).
    id_distractors = list({rand_id() for _ in range(args.distractors * 2)})[: args.distractors]
    dob_distractors = list({rand_dob() for _ in range(args.distractors * 2)})[: args.distractors]
    pools = {"id_no": id_distractors, "dob": dob_distractors}

    cards = generate_id_cards(args.n, seed=42, value_font_size=args.font_size, vary=True)

    # warm the query cache: full distractor strings (Stage 1), true values, and their bigrams (Stage 2).
    for s in id_distractors + dob_distractors:
        q(s)
    for _, fs in cards:
        for f in fs:
            if f.field_type in ("id_no", "dob"):
                q(f.text)
                for _, gram in _ngrams(f.text, args.ngram):
                    q(gram)

    summary, rows = run_attack(retriever, cards, pools, topk=args.topk, ngram=args.ngram, q_fn=q)

    print("\n=== TWO-STAGE POSITIONAL RERANK (exact top-1) ===")
    for ft in ("id_no", "dob"):
        s = summary[ft]
        print(f"  {ft:6s}: before={s['top1_before']:.3f}  after={s['top1_after']:.3f}  "
              f"lift={s['lift']:+.3f}  (topK-recall={s['topk_recall']:.3f}, n={s['n']})")
    lifts = [summary[ft]["lift"] for ft in ("id_no", "dob")]
    verdict = "RERANK RECOVERS DIGIT ORDER" if any(l > 0.02 for l in lifts) else "NO POSITIONAL LIFT"
    print("VERDICT:", verdict)

    payload = {
        "mode": "retrieval_rerank",
        "n": len(cards),
        "distractors": args.distractors,
        "topk": args.topk,
        "ngram": args.ngram,
        "font_size": args.font_size,
        "summary": summary,
        "verdict": verdict,
        "rows": rows,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "retrieval_rerank.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote retrieval_rerank.json -> {args.out}")


if __name__ == "__main__":
    main()
