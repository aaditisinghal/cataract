"""Wrong-page control (MASTER_PLAN S10 validation) — locks down the holographic-bleed finding.

The erasure result (name recoverable even after deleting its own patches) is only meaningful if the
attack genuinely reads the name's PRESENCE on the page, not a universal "the true name always ranks
first" artifact. So attack name_i against a DIFFERENT card j: it must FAIL (~chance). Four cells:

  correct_full   : name_i vs page_i (full)                 -> expect ~1.0
  wrong_full     : name_i vs page_j (full, name_i absent)  -> expect ~chance
  correct_erased : name_i vs page_i MINUS name_i's patches -> expect ~1.0 (the bleed result)
  wrong_erased   : name_i vs page_j MINUS name_j's patches -> expect ~chance

If correct_* are high and wrong_* are at chance, recovery depends on the name being present and the
bleed is real (not an artifact).
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/control_wrongpage")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--font-size", type=int, default=24)
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

    encs, names, remainders = [], [], []
    for im, fs in cards:
        enc = retriever.encode_page(im)
        name = next(f.text for f in fs if f.field_type == "name")
        name_box = next(f.box for f in fs if f.field_type == "name")
        wh = (im.shape[1], im.shape[0])
        gmask = boxes_to_patch_mask([name_box], wh, enc.grid, enc.input_size, enc.resize_policy, 0.0, 0)
        gh, gw = enc.grid
        img = enc.image_patches()
        trailing = enc.patches[gh * gw:]
        rem = np.concatenate([img[~gmask], trailing], axis=0) if trailing.size else img[~gmask]
        encs.append(enc); names.append(name); remainders.append(rem)

    cf, wf, ce, we = [], [], [], []
    N = len(cards)
    for i in range(N):
        j = (i + 1) % N  # a different card (name_i absent from page_j)
        cf.append(attack(encs[i].patches, names[i]))
        wf.append(attack(encs[j].patches, names[i]))
        ce.append(attack(remainders[i], names[i]))
        we.append(attack(remainders[j], names[i]))

    chance = 1.0 / (args.distractors + 1)
    res = {"chance": chance,
           "correct_full": _ci(cf, seed=1), "wrong_full": _ci(wf, seed=2),
           "correct_erased": _ci(ce, seed=3), "wrong_erased": _ci(we, seed=4)}
    for k in ("correct_full", "wrong_full", "correct_erased", "wrong_erased"):
        m, lo, hi = res[k]
        print(f"  {k:15s}: {m:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"  chance = {chance:.4f}")
    verdict = ("HOLOGRAPHIC BLEED CONFIRMED"
               if res["correct_erased"][0] >= 0.8 and res["wrong_full"][0] <= 0.15 and res["wrong_erased"][0] <= 0.15
               else "NOT CLEAN — reassess")
    print(f"  VERDICT: {verdict}")

    payload = {"mode": "control_wrongpage", "n": args.n, "chance": chance,
               "correct_full": res["correct_full"][0], "wrong_full": res["wrong_full"][0],
               "correct_erased": res["correct_erased"][0], "wrong_erased": res["wrong_erased"][0],
               "cis": {k: list(res[k][1:]) for k in ("correct_full", "wrong_full", "correct_erased", "wrong_erased")},
               "verdict": verdict, "fingerprint": run_fingerprint()}
    (local_out / "control_wrongpage.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote control_wrongpage.json -> {args.out}")


if __name__ == "__main__":
    main()
