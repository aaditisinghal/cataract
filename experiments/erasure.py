"""Claim 4 — erasure & context bleed (MASTER_PLAN S10), and why naive patch-scoping failed.

The defense frontier showed perturbing a field's own patches gives ZERO privacy. Hypothesis: ColPali's
full-attention encoder spreads a field's information across many patches, so touching only its spatial
patches doesn't remove it. This quantifies it:

  * LOCALITY control: attack using ONLY the field's patches vs the whole page minus those patches.
    If the field recovers from the page WITHOUT its own patches, the info has bled out.
  * DILATION / erasure sweep: DELETE the field's patches dilated by radius r (grid neighbors), re-run
    the attack on the remainder. The radius where recovery finally drops = the erasure radius, and its
    patch cost = the utility price. If r=0 deletion doesn't erase, naive patch deletion does NOT satisfy
    erasure (the paper's legal hook).

Reports recovery vs dilation radius (+ patches removed) with bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def _ci(hits, n_res=3000, seed=0):
    a = np.asarray(hits, float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    b = a[rng.integers(0, a.size, (n_res, a.size))].mean(1)
    return float(a.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def _dilate(grid_mask_1d, grid, r):
    gh, gw = grid
    m = grid_mask_1d.reshape(gh, gw).copy()
    for _ in range(r):
        nxt = m.copy()
        nxt[1:, :] |= m[:-1, :]
        nxt[:-1, :] |= m[1:, :]
        nxt[:, 1:] |= m[:, :-1]
        nxt[:, :-1] |= m[:, 1:]
        m = nxt
    return m.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/erasure")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--radii", default="0,1,2,3,4,6,8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.data.align import boxes_to_patch_mask
    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.base import maxsim
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    radii = [int(x) for x in args.radii.split(",")]
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    retriever = ColPaliRetriever(model_name=args.model)
    qcache: dict[str, np.ndarray] = {}

    def q(s):
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]
    for s in name_pool:
        q(s)

    cards = generate_id_cards(args.n, seed=11, value_font_size=args.font_size, vary=True)

    def attack(patches, true_name):
        cands = [true_name] + list(rng.choice([x for x in name_pool if x != true_name],
                                              min(args.distractors, len(name_pool) - 1), replace=False))
        sc = np.array([maxsim(q(c), patches) for c in cands])
        return int(np.argmax(sc) == 0)

    per_card = []
    for im, fs in cards:
        enc = retriever.encode_page(im)
        name = next(f.text for f in fs if f.field_type == "name")
        name_box = next(f.box for f in fs if f.field_type == "name")
        wh = (im.shape[1], im.shape[0])
        gmask = boxes_to_patch_mask([name_box], wh, enc.grid, enc.input_size, enc.resize_policy, 0.0, 0)
        gh, gw = enc.grid
        # LOCALITY control
        img = enc.image_patches()
        local_only = attack(img[gmask], name) if gmask.any() else 0
        without_field = attack(img[~gmask], name)  # page minus the field's own patches
        # DILATION erasure sweep (delete dilated field patches, attack remainder incl. prefix/trailing)
        rec_by_r, removed_by_r = [], []
        for r in radii:
            dil = _dilate(gmask, enc.grid, r)
            keep_img = img[~dil]
            # remainder = kept image patches + trailing instruction tokens (after the 1024 block)
            trailing = enc.patches[gh * gw:]  # n_prefix=0 for colpali
            remainder = np.concatenate([keep_img, trailing], axis=0) if trailing.size else keep_img
            rec_by_r.append(attack(remainder, name))
            removed_by_r.append(int(dil.sum()))
        per_card.append({"local_only": local_only, "without_field": without_field,
                         "rec_by_r": rec_by_r, "removed_by_r": removed_by_r})

    lo = _ci([c["local_only"] for c in per_card], seed=1)
    wo = _ci([c["without_field"] for c in per_card], seed=2)
    print(f"LOCALITY: recover from field-patches-ONLY = {lo[0]:.2f}[{lo[1]:.2f},{lo[2]:.2f}] | "
          f"recover from page WITHOUT field patches = {wo[0]:.2f}[{wo[1]:.2f},{wo[2]:.2f}]  (bleed if this is high)")
    print("ERASURE sweep (delete field patches dilated by r):")
    sweep = []
    for j, r in enumerate(radii):
        rm, rlo, rhi = _ci([c["rec_by_r"][j] for c in per_card], seed=10 + j)
        avg_removed = float(np.mean([c["removed_by_r"][j] for c in per_card]))
        sweep.append({"radius": r, "recovery": rm, "ci": [rlo, rhi], "patches_removed": avg_removed})
        print(f"  r={r}: recovery={rm:.2f}[{rlo:.2f},{rhi:.2f}]  patches_removed={avg_removed:.0f}/{args.n and 1024}")
    erased = [s for s in sweep if s["recovery"] <= 0.1]
    radius = erased[0]["radius"] if erased else None
    print(f"ERASURE RADIUS (recovery<=0.1): r={radius}" if radius is not None else "NEVER erased in sweep")

    payload = {"mode": "erasure", "n": args.n,
               "locality": {"field_only": {"acc": lo[0], "ci": [lo[1], lo[2]]},
                            "without_field": {"acc": wo[0], "ci": [wo[1], wo[2]]}},
               "sweep": sweep, "erasure_radius": radius, "fingerprint": run_fingerprint()}
    (local_out / "erasure.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote erasure.json -> {args.out}")


if __name__ == "__main__":
    main()
