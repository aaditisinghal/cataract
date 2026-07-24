"""Linkage MARGIN under progressive deletion (reviewer Tier-2) — does the HOLOGRAPHIC claim have room?

Claim 4 (erasure.py) showed that deleting a name field's OWN patches, even dilated by a radius, leaves
top-1 dictionary recovery pinned at 1.00 — the field's information has bled across the page. But a top-1
that is stuck at 1.00 is a saturated metric: it cannot tell "smeared everywhere" from "just barely
recovered". A reviewer will (rightly) ask for a MARGIN.

This sweeps a deletion FRACTION (budget = a fraction of the page's image patches). At each fraction we
delete the field's own patches and grow the deletion outward by dilation (the erasure.py mechanism) until
the removed count reaches the budget, then re-run the dictionary attack on the REMAINING patches and
measure:

  * LINKAGE MARGIN = MaxSim(true name) - max_over_distractors MaxSim(distractor)   [averaged over cards]
  * top-1 recovery (the saturated metric, for reference)

Reading the result honestly:
  * If the margin stays LARGE as the deletion fraction grows, the true name still out-scores every
    distractor even after its patches (and their neighbourhood) are gone => the identity is smeared
    across the page: positive, quantitative evidence for the holographic claim.
  * If the margin DECAYS toward 0 while top-1 clings to 1.00, we say so plainly: the leak is real but
    thin, and the holographic claim narrows to "recoverable" rather than "richly redundant".

Reuses erasure.py's deletion/dilation and retrieval_attack.py's MaxSim scoring.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def _ci(vals, n_res: int = 3000, seed: int = 0):
    """Bootstrap mean + 95% CI for a list of (continuous) per-card values."""
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    b = a[rng.integers(0, a.size, (n_res, a.size))].mean(1)
    return float(a.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def _dilate(grid_mask_1d: np.ndarray, grid: tuple[int, int], r: int) -> np.ndarray:
    """Grow a flat (gh*gw,) boolean patch mask by r grid steps (4-neighbourhood). From erasure.py."""
    gh, gw = grid
    m = grid_mask_1d.reshape(gh, gw).copy()
    for _ in range(int(r)):
        nxt = m.copy()
        nxt[1:, :] |= m[:-1, :]
        nxt[:-1, :] |= m[1:, :]
        nxt[:, 1:] |= m[:, :-1]
        nxt[:, :-1] |= m[:, 1:]
        m = nxt
    return m.reshape(-1)


def _deletion_mask(field_mask: np.ndarray, grid: tuple[int, int], target_count: int) -> np.ndarray:
    """Deletion budget as a patch count: start from the field's own patches, then dilate outward until
    at least ``target_count`` patches are removed (or the grid is exhausted).

    target_count <= 0 deletes NOTHING (the fraction=0 baseline). If the field mask already meets the
    budget, only the field patches are deleted (the minimal semantic unit — you cannot delete "half a
    field" here). Monotone in target_count by construction.
    """
    field_mask = np.asarray(field_mask, dtype=bool)
    if target_count <= 0 or not field_mask.any():
        return np.zeros_like(field_mask)
    m = field_mask.copy()
    r = 0
    while m.sum() < target_count:
        r += 1
        nxt = _dilate(field_mask, grid, r)
        if nxt.sum() == m.sum():  # dilation can't grow further (mask spans the grid)
            m = nxt
            break
        m = nxt
    return m


def margin_sweep(retriever, cards, name_pool, fractions, distractors, seed: int = 0):
    """Core sweep (CPU-safe: works with the mock retriever + tiny patches).

    retriever   : anything with encode_page/encode_query (ColPali or MockRetriever).
    cards       : list of (image ndarray, list[AnnotatedField]) — must contain a 'name' field.
    name_pool   : candidate name vocabulary (the dictionary the attacker ranks).
    fractions   : deletion budgets, each a fraction of the page's gh*gw image patches.
    distractors : cap on how many pool names (besides the truth) enter each card's lineup.
    """
    from patchguard.data.align import boxes_to_patch_mask
    from patchguard.retrievers.base import maxsim

    rng = np.random.default_rng(seed)
    qcache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    for s in name_pool:
        q(s)

    fractions = [float(x) for x in fractions]
    F = len(fractions)
    margins: list[list[float]] = [[] for _ in range(F)]
    top1s: list[list[int]] = [[] for _ in range(F)]
    removed: list[list[float]] = [[] for _ in range(F)]

    for img, fields in cards:
        arr = np.asarray(img)
        enc = retriever.encode_page(arr)
        gh, gw = enc.grid
        n_img = gh * gw
        true = next(f.text for f in fields if f.field_type == "name")
        name_box = next(f.box for f in fields if f.field_type == "name")
        wh = (int(arr.shape[1]), int(arr.shape[0]))
        # Field's own patches over the image-patch block (n_prefix=0 -> length gh*gw, aligns to image_patches()).
        field_mask = boxes_to_patch_mask([name_box], wh, enc.grid, enc.input_size, enc.resize_policy, 0.0, 0)
        image_p = enc.image_patches()
        trailing = enc.patches[enc.n_prefix_tokens + n_img:]  # instruction tokens after the image block

        # Per-card lineup: truth + a random sample of the rest of the dictionary (like erasure.py).
        others = [x for x in name_pool if x != true]
        k = min(int(distractors), len(others))
        dist = list(rng.choice(others, k, replace=False)) if k > 0 else []
        cands = [true] + dist
        q(true)

        for fi, frac in enumerate(fractions):
            target = int(round(frac * n_img))
            del_mask = _deletion_mask(field_mask, enc.grid, target)
            keep = image_p[~del_mask]
            remainder = np.concatenate([keep, trailing], axis=0) if trailing.size else keep
            removed[fi].append(float(del_mask.sum()) / n_img if n_img else 0.0)
            if remainder.shape[0] == 0:  # everything deleted (shouldn't happen for frac<=0.33)
                margins[fi].append(0.0)
                top1s[fi].append(0)
                continue
            scores = np.array([maxsim(q(c), remainder) for c in cands], dtype=float)
            margin = float(scores[0] - scores[1:].max()) if scores.size > 1 else float(scores[0])
            margins[fi].append(margin)
            top1s[fi].append(int(np.argmax(scores) == 0))

    per_fraction = []
    for fi, frac in enumerate(fractions):
        mm, lo, hi = _ci(margins[fi], seed=100 + fi)
        per_fraction.append({
            "fraction": frac,
            "removed_frac_mean": float(np.mean(removed[fi])) if removed[fi] else 0.0,
            "margin_mean": mm,
            "margin_ci": [lo, hi],
            "top1": float(np.mean(top1s[fi])) if top1s[fi] else 0.0,
        })
    return {"n": len(cards), "per_fraction": per_fraction, "margins_raw": margins}


def _verdict(per_fraction):
    """Honest read-out: does the margin persist (holographic) or decay (claim narrows)?"""
    if not per_fraction:
        return {"label": "no_data", "margin_retention": 0.0}
    base = per_fraction[0]["margin_mean"]
    final = per_fraction[-1]["margin_mean"]
    final_top1 = per_fraction[-1]["top1"]
    retention = float(final / base) if base > 1e-9 else 0.0
    if final > 1e-6 and retention >= 0.5:
        label = "holographic_margin_persists"
    elif final_top1 >= 0.99:
        label = "narrows_top1_saturated_margin_decays"
    else:
        label = "leak_erodes"
    return {"label": label, "margin_retention": retention,
            "baseline_margin": base, "final_margin": final, "final_top1": final_top1}


def main() -> None:
    ap = argparse.ArgumentParser(description="Linkage-margin sweep under progressive field deletion.")
    ap.add_argument("--out", default="results/margin_analysis")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=40, help="cards to attack")
    ap.add_argument("--distractors", type=int, default=200, help="lineup size (pool names besides truth)")
    ap.add_argument("--fractions", default="0,0.05,0.1,0.2,0.33",
                    help="deletion budgets, each a fraction of the page's image patches")
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    fractions = [float(x) for x in args.fractions.split(",")]
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    retriever = ColPaliRetriever(model_name=args.model)
    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # closed dictionary the attacker ranks
    cards = generate_id_cards(args.n, seed=11, value_font_size=args.font_size, vary=True)

    result = margin_sweep(retriever, cards, name_pool, fractions, args.distractors, seed=args.seed)
    per_fraction = result["per_fraction"]
    verdict = _verdict(per_fraction)

    print(f"=== LINKAGE MARGIN vs DELETION FRACTION (n={result['n']}, distractors={args.distractors}) ===")
    for pf in per_fraction:
        lo, hi = pf["margin_ci"]
        print(f"  frac={pf['fraction']:.2f} (removed~{pf['removed_frac_mean']:.2f}): "
              f"margin={pf['margin_mean']:+.3f}[{lo:+.3f},{hi:+.3f}]  top1={pf['top1']:.2f}")
    print(f"VERDICT: {verdict['label']}  (margin retained {verdict['margin_retention']:.2f} of baseline; "
          f"final top1={verdict['final_top1']:.2f})")

    payload = {"mode": "margin_analysis", "n": result["n"], "distractors": args.distractors,
               "font_size": args.font_size, "fractions": fractions,
               "per_fraction": [{k: v for k, v in pf.items()} for pf in per_fraction],
               "verdict": verdict, "fingerprint": run_fingerprint()}
    (local_out / "margin_analysis.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote margin_analysis.json -> {args.out}")


if __name__ == "__main__":
    main()
