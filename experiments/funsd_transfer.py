"""FUNSD transfer — the REAL-DOCUMENT gate (MASTER_PLAN S6).

The synthetic probe/retrieval killed the layout confound but not the "is it just synthetic?" question.
This runs the same MaxSim retrieval-discrimination on REAL FUNSD field text: for each real form field,
can ColPali's own encode_query retrieve the field's true text over K distractors drawn from OTHER real
fields in the corpus? A hit means the field content is encoded and accessible on real documents — the
leak transfers. Open-vocabulary (real field strings), real layout variance, no synthetic template.

Reports top-1 / top-5 discrimination accuracy, broken out by FUNSD label (answer/header/question) and
by field text length (short fields ~ big fonts, long fields ~ dense text). Also normalized-substring
recovery (does the OCR-normalized true text appear in the top-ranked candidate) for a softer read.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FUNSD root (local or gs://)")
    ap.add_argument("--out", default="results/funsd_transfer")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n-pages", type=int, default=60)
    ap.add_argument("--k", type=int, default=20, help="lineup size (true + k-1 distractors)")
    ap.add_argument("--max-fields", type=int, default=12, help="fields probed per page")
    ap.add_argument("--min-len", type=int, default=3, help="skip trivially-short field text")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image

    from patchguard.data.funsd import iter_funsd
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.base import maxsim
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_download, _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    data_root = args.data
    if str(data_root).startswith("gs://"):
        data_root = str(_gcs_download(data_root, Path(tempfile.mkdtemp())))

    # collect real pages + their fields (image loaded, text + label + box)
    pages = []
    all_texts: list[str] = []
    for ps in iter_funsd(data_root, split="testing_data", granularity="entity"):
        img = np.array(Image.open(ps.image_path).convert("RGB"))
        fields = [f for f in ps.fields if len("".join(ch for ch in f.text if ch.isalnum())) >= args.min_len]
        if fields:
            pages.append((img, fields))
            all_texts.extend(f.text for f in fields)
        if len(pages) >= args.n_pages:
            break
    print(f"real FUNSD pages: {len(pages)} | distractor text pool: {len(all_texts)}")

    retriever = ColPaliRetriever(model_name=args.model)
    qcache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    def norm(s: str) -> str:
        return "".join(ch for ch in s if ch.isalnum()).casefold()

    rows = []
    hits1, hits5 = [], []
    by_label: dict[str, list[int]] = {}
    by_lenbucket: dict[str, list[int]] = {}
    for pi, (img, fields) in enumerate(pages):
        enc = retriever.encode_page(img)
        probe_fields = list(rng.permutation(len(fields)))[: args.max_fields]
        for fi in probe_fields:
            f = fields[fi]
            # distractors: other real field texts (different from truth), from the corpus
            pool = [t for t in all_texts if norm(t) != norm(f.text)]
            distr = list(rng.choice(pool, min(args.k - 1, len(pool)), replace=False))
            cands = [f.text] + distr
            scores = np.array([maxsim(q(c), enc.patches) for c in cands])
            order = np.argsort(-scores)
            ranked = [cands[i] for i in order]
            hit1 = norm(ranked[0]) == norm(f.text)
            hit5 = any(norm(r) == norm(f.text) for r in ranked[:5])
            hits1.append(int(hit1))
            hits5.append(int(hit5))
            by_label.setdefault(f.field_type, []).append(int(hit1))
            bucket = "short(<=8)" if len(norm(f.text)) <= 8 else "long(>8)"
            by_lenbucket.setdefault(bucket, []).append(int(hit1))
            if len(rows) < 40:
                rows.append({"page": pi, "label": f.field_type, "true": f.text,
                             "recovered": ranked[0], "top1": bool(hit1), "in_top5": bool(hit5)})

    n = len(hits1)
    summary = {
        "n_fields": n, "lineup": args.k, "chance": 1.0 / args.k,
        "top1_acc": float(np.mean(hits1)), "top5_acc": float(np.mean(hits5)),
        "by_label_top1": {k: float(np.mean(v)) for k, v in by_label.items()},
        "by_length_top1": {k: float(np.mean(v)) for k, v in by_lenbucket.items()},
    }
    print("\n=== FUNSD REAL-DOC TRANSFER ===")
    print(f"  top1={summary['top1_acc']:.3f}  top5={summary['top5_acc']:.3f}  (chance {summary['chance']:.3f}, n={n})")
    print(f"  by label : {summary['by_label_top1']}")
    print(f"  by length: {summary['by_length_top1']}")
    for r in rows[:12]:
        print(f"  [{r['label']:<8}] true {r['true'][:24]!r:<26} -> {r['recovered'][:24]!r} {'OK' if r['top1'] else 'x'}")

    payload = {"mode": "funsd_transfer", "summary": summary, "rows": rows, "fingerprint": run_fingerprint()}
    (local_out / "funsd_transfer.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote funsd_transfer.json -> {args.out}")


if __name__ == "__main__":
    main()
