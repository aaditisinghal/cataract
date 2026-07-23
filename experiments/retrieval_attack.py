"""Retrieval / dictionary attack (MASTER_PLAN S6, headline) — recover PII WITHOUT reconstruction.

Given only the stored ColPali patch embeddings for a page, the attacker enumerates plausible candidate
values for each field and ranks them by MaxSim against the stored patches. The top-1 candidate IS the
recovered value — no pixels, no decoder, no OCR. This is the strongest, cleanest form of the leak the
discriminative probe proved exists: the retrieval interface is itself a PII oracle.

Per field the candidate space is the realistic format space:
  * name  : the closed name vocabulary (K=240) -> argmax = recovered name.
  * id_no : the true 8-digit value hidden among D random 8-digit distractors (a D+1 "lineup").
  * dob   : the true date hidden among D random valid-date distractors.
Recovery succeeds when the true value is ranked #1 (chance = 1/(D+1)). Reports top-1 + top-5 and the
actual recovered strings.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/retrieval")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=40, help="cards to attack")
    ap.add_argument("--distractors", type=int, default=999, help="lineup size D for id/dob")
    ap.add_argument("--font-size", type=int, default=34)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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

    def q(s: str) -> np.ndarray:
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    def rand_id() -> str:
        return f"{int(rng.integers(10_000_000, 99_999_999))}"

    def rand_dob() -> str:
        return f"{int(rng.integers(1,13)):02d}/{int(rng.integers(1,29)):02d}/{int(rng.integers(1950,2005))}"

    # candidate spaces (encode once)
    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # K=240
    id_distractors = list({rand_id() for _ in range(args.distractors * 2)})[: args.distractors]
    dob_distractors = list({rand_dob() for _ in range(args.distractors * 2)})[: args.distractors]
    for s in name_pool + id_distractors + dob_distractors:
        q(s)  # warm cache

    def recover(patches, true_val, candidates, topk=5):
        cands = candidates if true_val in candidates else [true_val, *candidates]
        scores = np.array([maxsim(q(c), patches) for c in cands])
        order = np.argsort(-scores)
        ranked = [cands[i] for i in order]
        return ranked[0], ranked[:topk], (ranked[0] == true_val), (true_val in ranked[:topk])

    cards = generate_id_cards(args.n, seed=42, value_font_size=args.font_size, vary=True)
    rows = []
    agg = {"name": [0, 0], "id_no": [0, 0], "dob": [0, 0]}  # [top1, top5]
    for i, (im, fs) in enumerate(cards):
        enc = retriever.encode_page(im)
        row = {"card": i}
        truth = {f.field_type: f.text for f in fs}
        for ft, cand_pool in (("name", name_pool), ("id_no", id_distractors), ("dob", dob_distractors)):
            rec, top5, hit1, hit5 = recover(enc.patches, truth[ft], cand_pool)
            row[ft] = {"true": truth[ft], "recovered": rec, "top1": hit1, "in_top5": hit5}
            agg[ft][0] += int(hit1)
            agg[ft][1] += int(hit5)
        rows.append(row)
        print(f"card {i:2d} | name {row['name']['recovered']:<20} ({'OK' if row['name']['top1'] else 'x'}) "
              f"| id {row['id_no']['recovered']} ({'OK' if row['id_no']['top1'] else 'x'}) "
              f"| dob {row['dob']['recovered']} ({'OK' if row['dob']['top1'] else 'x'})")

    n = len(cards)
    summary = {ft: {"top1_acc": agg[ft][0] / n, "top5_acc": agg[ft][1] / n,
                    "chance": (1 / 240 if ft == "name" else 1 / (args.distractors + 1))}
               for ft in agg}
    print("\n=== TOP-1 RECOVERY ACCURACY ===")
    for ft, s in summary.items():
        print(f"  {ft:8s}: top1={s['top1_acc']:.3f}  top5={s['top5_acc']:.3f}  (chance {s['chance']:.4f})")

    payload = {"mode": "retrieval_attack", "n": n, "distractors": args.distractors,
               "font_size": args.font_size, "summary": summary, "rows": rows,
               "fingerprint": run_fingerprint()}
    (local_out / "retrieval.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote retrieval.json -> {args.out}")


if __name__ == "__main__":
    main()
