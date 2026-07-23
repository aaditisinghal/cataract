"""Cross-encoder transfer of the dictionary name-recovery attack (plan A7 — refine the threat model).

The headline retrieval attack (``retrieval_attack.py``) assumes the attacker can run the EXACT encoder
that built the index: it re-encodes candidate names with the SAME ColPali weights the victim used, then
ranks the lineup by MaxSim against the stored patches. A reviewer's obvious escape hatch is that this is
an unrealistically strong, near-white-box attacker — the index encoder is a private asset, so surely the
leak evaporates the moment the adversary only has a *different*, public multi-vector retriever.

This experiment tests exactly that escape hatch. The stored INDEX is encoded by one model, but the
attacker's candidate QUERIES are encoded by a DIFFERENT model (a semi-black-box / cross-encoder threat
model). We compare:

  (a) MATCHED   — same encoder both sides (the white-box upper bound; == the headline attack).
  (b) TRANSFER  — index encoded by model X, candidate queries encoded by model Y, mapped into X's space.
      Run in BOTH directions: index=ColPali/queries=ColQwen2 AND index=ColQwen2/queries=ColPali.

Dim-alignment method. MaxSim needs the two sides in the same d-dimensional coordinate frame; even when
the two encoders share a dimension they do NOT share a basis, so a raw cross-encoder MaxSim is
near-meaningless. We therefore fit a small ridge-regularised LINEAR map ``W`` (shape d_index x d_query)
on a handful of pooled ``(query-encoder, index-encoder)`` anchor embeddings and apply it per query token
before MaxSim. The anchors are GENERIC text (issuing-office / date / id strings) drawn DISJOINT from the
recovered name vocabulary, so the alignment must generalise to names it was never fit on — the honest
semi-black-box cost. When the two dims already match we additionally report the RAW, unaligned
cross-encoder recovery as a floor, to show it is the alignment (not a lucky shared basis) that carries
the transfer. If dims differ, the same linear map bridges them, so no configuration is skipped.

What a result MEANS. TRANSFER recovery that stays clearly above chance is the strong, scary finding: the
attacker does NOT need the victim's exact weights — any sufficiently similar public encoder plus a few
anchor pairs recovers the PII, so the leak is a property of the multi-vector retrieval interface, not of
one secret checkpoint. TRANSFER that collapses to chance while MATCHED stays high is the reassuring
finding: the attack is weight-specific, and keeping the index encoder private is a partial mitigation.
The reported ``degradation`` (matched - transfer) quantifies how much protection the weight gap actually
buys.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.adaptive_attack import _dict_hit, _lineup


# --------------------------------------------------------------------------------------------------
# pure, CPU-testable helpers (no torch/colpali at import time; mock retrievers exercise all of these)
# --------------------------------------------------------------------------------------------------
def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def _pool_norm(qtoks: np.ndarray) -> np.ndarray:
    """Mean-pool a multi-vector query to a single unit vector (the anchor representation)."""
    v = np.asarray(qtoks, dtype=np.float64).mean(axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _page_patches(enc) -> np.ndarray:
    """Model-agnostic patch array — ColPali PageEncoding, ColQwen2 SimpleNamespace and the mock all
    expose ``.patches`` (the full stored multi-vector set the attacker reads from the index)."""
    return np.asarray(enc.patches, dtype=np.float32)


def _cached_encoder(retriever):
    """Wrap ``encode_query`` with a string cache (candidate names are re-queried across every card)."""
    cache: dict[str, np.ndarray] = {}

    def enc(s: str) -> np.ndarray:
        v = cache.get(s)
        if v is None:
            v = np.asarray(retriever.encode_query(s), dtype=np.float32)
            cache[s] = v
        return v

    return enc, cache


def fit_alignment(anchor_texts, encode_src, encode_dst, ridge: float = 1e-2) -> np.ndarray:
    """Fit the linear map W (d_dst x d_src) taking SRC-encoder query space -> DST-encoder query space.

    Ridge-regularised least squares on pooled anchor embeddings; handles n_anchors < d and matched or
    mismatched dims uniformly. ``encode_src``/``encode_dst`` are text -> (nq, d) callables.
    """
    src = np.stack([_pool_norm(encode_src(t)) for t in anchor_texts]).astype(np.float64)  # (n, d_src)
    dst = np.stack([_pool_norm(encode_dst(t)) for t in anchor_texts]).astype(np.float64)  # (n, d_dst)
    d_src = int(src.shape[1])
    gram = src.T @ src + float(ridge) * np.eye(d_src)
    w_t = np.linalg.solve(gram, src.T @ dst)  # (d_src, d_dst)
    return w_t.T.astype(np.float32)  # (d_dst, d_src)


def align_query(qtoks: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Map every query token through W into the index space, then re-normalise (MaxSim scale parity)."""
    mapped = np.asarray(qtoks, dtype=np.float32) @ np.asarray(W, dtype=np.float32).T
    return _l2norm_rows(mapped)


def _recover(idx_cards, q_fn, name_pool, n_distractors: int, seed: int) -> float:
    """Mean top-1 dictionary recovery over the index cards using candidate-query oracle ``q_fn``.

    A fresh rng per call (same seed) means MATCHED / TRANSFER / RAW see IDENTICAL lineups, so their
    difference is a fair measure of the encoder gap and not of lineup luck.
    """
    rng = np.random.default_rng(seed)
    hits = [_dict_hit(p, nm, q_fn, name_pool, n_distractors, rng) for (p, nm) in idx_cards]
    return float(np.mean(hits)) if hits else 0.0


def _beats_chance(rec: float, chance: float) -> bool:
    """Transfer 'works' if recovery is both clearly above chance and not trivially small in absolute terms."""
    return bool(rec >= 3.0 * chance and rec >= 0.15)


def run_transfer_pair(index_ret, query_ret, cards, name_pool, anchor_texts,
                      n_distractors: int, lineup_seed: int = 0, ridge: float = 1e-2) -> dict:
    """One (index-encoder, query-encoder) direction of the transfer attack.

    ``cards`` is a list of ``(image_np, true_name)``. Encodes the index once with ``index_ret``, then
    measures MATCHED recovery (queries from ``index_ret``), TRANSFER recovery (queries from
    ``query_ret`` mapped through a fit linear W) and — when dims match — the RAW unaligned reference.
    Returns recoveries, the degradation, and the COMPUTED ``matched_ge_transfer`` flag (never assumed).
    """
    enc_i, _ = _cached_encoder(index_ret)
    enc_q, _ = _cached_encoder(query_ret)

    idx_cards = [(_page_patches(index_ret.encode_page(im)), nm) for (im, nm) in cards]
    d_index = int(idx_cards[0][0].shape[1]) if idx_cards else 0
    d_query = int(np.asarray(enc_q(anchor_texts[0])).shape[1])

    matched = _recover(idx_cards, lambda c: enc_i(c), name_pool, n_distractors, lineup_seed)

    W = fit_alignment(anchor_texts, enc_q, enc_i, ridge=ridge)
    transfer = _recover(idx_cards, lambda c: align_query(enc_q(c), W), name_pool, n_distractors, lineup_seed)

    chance = 1.0 / len(_lineup(name_pool[0], name_pool, n_distractors, np.random.default_rng(0)))
    out = {
        "matched_recovery": matched,
        "transfer_recovery": transfer,
        "degradation": float(matched - transfer),
        "matched_ge_transfer": bool(matched >= transfer),  # COMPUTED, not asserted as fact
        "transfer_beats_chance": _beats_chance(transfer, chance),
        "chance": chance,
        "d_index": d_index,
        "d_query": d_query,
        "dim_match": bool(d_index == d_query),
        "n_anchors": len(anchor_texts),
        "ridge": float(ridge),
        "alignment": "ridge_linear_map(d_index x d_query)",
    }
    if d_index == d_query:  # only meaningful when MaxSim can consume the query dim without a map
        out["transfer_raw_unaligned_recovery"] = _recover(
            idx_cards, lambda c: enc_q(c), name_pool, n_distractors, lineup_seed
        )
    return out


def _anchor_texts(n: int, rng: np.random.Generator, cities) -> list[str]:
    """Generic, NON-name anchor strings (office / date / id) — disjoint from the recovered name pool."""
    seen: list[str] = []
    seen_set: set[str] = set()
    tries = 0
    while len(seen) < n and tries < n * 50 + 100:
        tries += 1
        r = rng.random()
        if r < 0.6:
            t = f"DISTRICT {int(rng.integers(10, 99))} {rng.choice(cities)} OFFICE"
        elif r < 0.8:
            t = f"{int(rng.integers(1, 13)):02d}/{int(rng.integers(1, 29)):02d}/{int(rng.integers(1950, 2005))}"
        else:
            t = f"{int(rng.integers(10_000_000, 99_999_999))}"
        if t not in seen_set:
            seen_set.add(t)
            seen.append(t)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-encoder transfer of the dictionary name-recovery attack (does the attacker "
                    "need the exact index encoder, or does a different public encoder transfer?)."
    )
    ap.add_argument("--out", default="results/transfer_attack")
    ap.add_argument("--index-model", default="vidore/colpali-v1.3", help="encoder that built the index")
    ap.add_argument("--query-model", default="vidore/colqwen2-v1.0", help="attacker's DIFFERENT encoder")
    ap.add_argument("--n", type=int, default=40, help="victim cards to attack")
    ap.add_argument("--distractors", type=int, default=200, help="dictionary lineup size")
    ap.add_argument("--n-anchors", type=int, default=256, help="shared (query,index) anchor pairs for W")
    ap.add_argument("--ridge", type=float, default=1e-2, help="ridge strength for the alignment fit")
    ap.add_argument("--font-size", type=int, default=34)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.data.synthdoc import _CITIES, _FIRST, _LAST, generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.adaptive_attack import _build_retriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # closed candidate vocabulary K=240
    anchor_texts = _anchor_texts(args.n_anchors, rng, _CITIES)
    if not anchor_texts:
        raise RuntimeError("failed to build anchor texts")

    # victim cards (image + ground-truth name); same synthetic generator as the headline attack.
    raw_cards = generate_id_cards(args.n, seed=args.seed, value_font_size=args.font_size, vary=True)
    cards = [(im, next(f.text for f in fs if f.field_type == "name")) for (im, fs) in raw_cards]

    index_ret = _build_retriever(args.index_model)
    query_ret = _build_retriever(args.query_model)
    print(f"index encoder : {args.index_model}\nquery encoder : {args.query_model}\n"
          f"cards={len(cards)}  distractors={args.distractors}  anchors={len(anchor_texts)}")

    # forward: index=index-model, cross-encoder queries=query-model
    forward = run_transfer_pair(index_ret, query_ret, cards, name_pool, anchor_texts,
                                args.distractors, lineup_seed=args.seed, ridge=args.ridge)
    # reverse (…and vice versa): index=query-model, cross-encoder queries=index-model
    reverse = run_transfer_pair(query_ret, index_ret, cards, name_pool, anchor_texts,
                                args.distractors, lineup_seed=args.seed, ridge=args.ridge)

    chance = forward["chance"]
    transfer_works = forward["transfer_beats_chance"] or reverse["transfer_beats_chance"]
    attacker_needs_exact_weights = not transfer_works

    if transfer_works:
        verdict = (
            f"TRANSFER SUCCEEDS: a DIFFERENT public encoder recovers names above chance without the "
            f"index weights (fwd index={args.index_model} q={args.query_model}: transfer "
            f"{forward['transfer_recovery']:.3f} vs matched {forward['matched_recovery']:.3f}, "
            f"degradation {forward['degradation']:.3f}; rev: transfer {reverse['transfer_recovery']:.3f} "
            f"vs matched {reverse['matched_recovery']:.3f}; chance {chance:.4f}). The attacker does NOT "
            f"need the exact index encoder — a similar public one plus {len(anchor_texts)} anchor pairs "
            f"suffices, so the leak is a property of the retrieval interface, not of one secret checkpoint."
        )
    else:
        verdict = (
            f"TRANSFER FAILS: cross-encoder recovery collapses toward chance while matched stays high "
            f"(fwd matched {forward['matched_recovery']:.3f} -> transfer {forward['transfer_recovery']:.3f}; "
            f"rev matched {reverse['matched_recovery']:.3f} -> transfer {reverse['transfer_recovery']:.3f}; "
            f"chance {chance:.4f}). The attack is weight-specific under this alignment; keeping the index "
            f"encoder private is a partial mitigation, and the leak claim should be scoped to a white-box "
            f"(exact-encoder) adversary."
        )

    dim_note = (
        "MaxSim needs a shared d-dim frame; a ridge-regularised linear map W (d_index x d_query) fit on "
        f"{len(anchor_texts)} pooled (query-encoder, index-encoder) anchor embeddings is applied per query "
        "token then L2-normalised. Anchors are generic office/date/id text DISJOINT from the recovered "
        "name vocabulary (semi-black-box); when dims already match, the raw unaligned recovery is reported "
        "as a floor. The same map bridges mismatched dims, so no configuration is skipped."
    )

    print("\n=== CROSS-ENCODER TRANSFER ATTACK ===")
    print(f"{'configuration':44s} {'recovery':>9s} {'vs chance':>10s}")
    print(f"{'MATCHED  index=q=' + args.index_model:44.44s} {forward['matched_recovery']:9.3f} "
          f"{forward['matched_recovery'] / chance if chance else float('inf'):9.1f}x")
    print(f"{'TRANSFER index=' + args.index_model + ' q=' + args.query_model:44.44s} "
          f"{forward['transfer_recovery']:9.3f} {forward['transfer_recovery'] / chance if chance else 0:9.1f}x")
    if "transfer_raw_unaligned_recovery" in forward:
        print(f"{'  (raw unaligned floor, dims match)':44s} {forward['transfer_raw_unaligned_recovery']:9.3f}")
    print(f"{'MATCHED  index=q=' + args.query_model:44.44s} {reverse['matched_recovery']:9.3f} "
          f"{reverse['matched_recovery'] / chance if chance else float('inf'):9.1f}x")
    print(f"{'TRANSFER index=' + args.query_model + ' q=' + args.index_model:44.44s} "
          f"{reverse['transfer_recovery']:9.3f} {reverse['transfer_recovery'] / chance if chance else 0:9.1f}x")
    if "transfer_raw_unaligned_recovery" in reverse:
        print(f"{'  (raw unaligned floor, dims match)':44s} {reverse['transfer_raw_unaligned_recovery']:9.3f}")
    print(f"chance {chance:.4f} | fwd degradation {forward['degradation']:.3f} | "
          f"rev degradation {reverse['degradation']:.3f}")
    print("VERDICT:", verdict)

    payload = {
        "mode": "transfer_attack",
        "index_model": args.index_model,
        "query_model": args.query_model,
        "n": len(cards),
        "distractors": args.distractors,
        "font_size": args.font_size,
        "n_anchors": len(anchor_texts),
        "ridge": args.ridge,
        "chance": chance,
        "dim_alignment_note": dim_note,
        "configurations": {
            "matched_index": {
                "index": args.index_model, "queries": args.index_model,
                "recovery": forward["matched_recovery"], "kind": "white_box_upper_bound",
            },
            "transfer_index": {
                "index": args.index_model, "queries": args.query_model,
                "recovery": forward["transfer_recovery"],
                "degradation_from_matched": forward["degradation"],
                "raw_unaligned_recovery": forward.get("transfer_raw_unaligned_recovery"),
                "beats_chance": forward["transfer_beats_chance"], "kind": "cross_encoder_transfer",
            },
            "matched_query": {
                "index": args.query_model, "queries": args.query_model,
                "recovery": reverse["matched_recovery"], "kind": "white_box_upper_bound",
            },
            "transfer_query": {
                "index": args.query_model, "queries": args.index_model,
                "recovery": reverse["transfer_recovery"],
                "degradation_from_matched": reverse["degradation"],
                "raw_unaligned_recovery": reverse.get("transfer_raw_unaligned_recovery"),
                "beats_chance": reverse["transfer_beats_chance"], "kind": "cross_encoder_transfer",
            },
        },
        "forward": forward,
        "reverse": reverse,
        "transfer_works": transfer_works,
        "attacker_needs_exact_weights": attacker_needs_exact_weights,
        "verdict": verdict,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "transfer_attack.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"\nwrote transfer_attack.json -> {args.out}")


if __name__ == "__main__":
    main()
