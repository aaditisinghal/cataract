"""Real-corpus breadth (MASTER_PLAN A3) — does the MaxSim retrieval leak transfer BEYOND FUNSD?

funsd_transfer.py proved the leak on one real corpus (FUNSD forms). A reviewer's natural retort is
"FUNSD is one narrow, English, form-shaped distribution — is the leak an artifact of that corpus?"
This experiment answers it by running the SAME retrieval-discrimination attack on independent REAL
document distributions loaded from HuggingFace ``datasets``:

  * cord   (PRIMARY): ``naver-clova-ix/cord-v2`` — photographed restaurant/store RECEIPTS. Each example
    carries an ``image`` and a ``ground_truth`` JSON whose ``gt_parse`` lists real field values (menu
    names ``nm``, quantities ``cnt``, prices ``price``, and the ``total``/``sub_total`` blocks). These
    are exactly the financially-sensitive fields — store totals, dates, item lines — a receipt index
    would leak. For each field we ask ColPali's own ``encode_query`` to retrieve the field's TRUE text
    over K-1 distractors drawn from OTHER receipts' fields (a K-way lineup). A hit means the field
    content is encoded and accessible on real receipts with no reconstruction — the leak transfers.
  * doclaynet (BEST-EFFORT): ``ds4sd/DocLayNet`` financial_reports. DocLayNet ships layout boxes and
    category labels but not clean per-cell OCR text in the HF image config, so if the text isn't
    cleanly available we SKIP it with a printed warning rather than blocking — CORD alone is the
    deliverable.

A POSITIVE result (top-1 >> 1/K on a corpus the synthetic pipeline never touched) means the encoding
leak is a property of ColPali, not of any one dataset. A NEGATIVE result (top-1 ~= chance) would bound
the leak to FUNSD-like documents. Reports top-1 / top-5 discrimination overall and broken out by field
type and by the same glyph/length axis funsd_transfer uses, with the corpus name stamped in the payload.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------------------
# pure, CPU-testable helpers (no heavy imports — datasets/PIL/colpali are deferred inside main/loaders)
# --------------------------------------------------------------------------------------------------
def _alnum(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum())


def _norm(s: str) -> str:
    return _alnum(s).casefold()


def _length_bucket(text: str) -> str:
    """The SAME short/long split funsd_transfer reports, on normalized-alnum length."""
    return "short(<=8)" if len(_norm(text)) <= 8 else "long(>8)"


def _glyph_bucket(text: str) -> str:
    """Glyph-count proxy (non-space characters) — finer stand-in for rendered font size."""
    g = len(str(text).replace(" ", ""))
    if g <= 4:
        return "glyph<=4"
    if g <= 10:
        return "glyph5-10"
    return "glyph>10"


def extract_cord_fields(gt_parse, min_len: int = 3) -> list[tuple[str, str]]:
    """Walk a CORD ``gt_parse`` dict -> list of (field_type, text).

    ``field_type`` is the leaf key (``nm``, ``price``, ``cnt``, ``total_price``, ``cashprice``, ...).
    ``menu`` may be a single dict OR a list of item dicts; both are handled. Leaf values whose
    alphanumeric length is below ``min_len`` (e.g. a bare quantity "2") are dropped as too trivial
    to constitute a discriminative lineup.
    """
    out: list[tuple[str, str]] = []

    def walk(node, key: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, str(k))
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, key)
        else:
            text = str(node).strip()
            if len(_alnum(text)) >= min_len:
                out.append((str(key), text))

    walk(gt_parse, "root")
    return out


def _lineup(truth: str, pool, k: int, rng) -> list[str]:
    """A K-way recovery lineup: [truth, *distractors]; distractors EXCLUDE anything matching truth."""
    others = [t for t in pool if _norm(t) != _norm(truth)]
    m = int(min(max(k - 1, 0), len(others)))
    distr = list(rng.choice(others, m, replace=False)) if m > 0 else []
    return [truth, *distr]


def _probe(truth: str, cands, score_of):
    """Rank ``cands`` by ``score_of(text)->float``; return (ranked, hit1, hit5) vs ``truth``."""
    scores = np.array([float(score_of(c)) for c in cands])
    order = np.argsort(-scores)
    ranked = [cands[i] for i in order]
    hit1 = _norm(ranked[0]) == _norm(truth)
    hit5 = any(_norm(r) == _norm(truth) for r in ranked[:5])
    return ranked, bool(hit1), bool(hit5)


# --------------------------------------------------------------------------------------------------
# concrete corpus loaders — each yields (image_np HxWx3 uint8, fields=list[(field_type, text)]).
# Heavy imports (datasets, PIL) live INSIDE so the module imports and --help works with none installed.
# --------------------------------------------------------------------------------------------------
def _iter_cord(n_pages: int, min_len: int, seed: int):
    from datasets import load_dataset  # deferred: GPU-container-only dep

    ds = load_dataset("naver-clova-ix/cord-v2", split="test")
    n = 0
    for ex in ds:
        try:
            img = np.array(ex["image"].convert("RGB"))
            gt = json.loads(ex["ground_truth"])
            gp = gt.get("gt_parse", gt) if isinstance(gt, dict) else gt
            fields = extract_cord_fields(gp, min_len)
        except Exception:
            continue
        if fields:
            yield img, fields
            n += 1
        if n >= n_pages:
            break


def _iter_doclaynet(n_pages: int, min_len: int, seed: int):
    # Best-effort per the plan: DocLayNet's HF image config exposes layout boxes + a doc_category, but
    # not clean per-cell OCR text, so there is nothing to build a text lineup from. Warn and skip
    # rather than block — CORD is the primary deliverable.
    print(
        "[doclaynet] best-effort SKIP: ds4sd/DocLayNet (financial_reports) provides layout boxes and "
        "category labels but not clean per-cell OCR text, so no text lineup can be formed. "
        "Run --corpus cord for the real-doc transfer number."
    )
    return
    yield  # pragma: no cover — keeps this a generator


_CORPORA = {"cord": _iter_cord, "doclaynet": _iter_doclaynet}


def _collect(corpus: str, n_pages: int, min_len: int, seed: int):
    """Return (pages, all_texts, status). Never raises: an unavailable dataset yields status only."""
    loader = _CORPORA[corpus]
    pages: list[tuple[np.ndarray, list[tuple[str, str]]]] = []
    all_texts: list[str] = []
    try:
        for img, fields in loader(n_pages, min_len, seed):
            pages.append((img, fields))
            all_texts.extend(t for _, t in fields)
            if len(pages) >= n_pages:
                break
        status = "ok" if pages else "empty (no usable pages/fields)"
    except ImportError as e:
        status = f"unavailable: missing dependency ({e})"
    except Exception as e:  # dataset offline / auth / schema drift — degrade to empty payload
        status = f"unavailable: {type(e).__name__}: {e}"
    return pages, all_texts, status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="cord", choices=sorted(_CORPORA),
                    help="real corpus to transfer the retrieval attack onto (cord is primary)")
    ap.add_argument("--out", default="results/realcorpus_transfer")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n-pages", type=int, default=60)
    ap.add_argument("--k", type=int, default=20, help="lineup size (true + k-1 distractors)")
    ap.add_argument("--max-fields", type=int, default=12, help="fields probed per page")
    ap.add_argument("--min-len", type=int, default=3, help="skip trivially-short field text")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    pages, all_texts, status = _collect(args.corpus, args.n_pages, args.min_len, args.seed)
    print(f"[{args.corpus}] status={status} | pages={len(pages)} | distractor text pool={len(all_texts)}")

    rows: list[dict] = []
    hits1: list[int] = []
    hits5: list[int] = []
    by_field: dict[str, list[int]] = {}
    by_length: dict[str, list[int]] = {}
    by_glyph: dict[str, list[int]] = {}

    if pages:
        from patchguard.retrievers.base import maxsim
        from patchguard.retrievers.colpali import ColPaliRetriever

        retriever = ColPaliRetriever(model_name=args.model)
        qcache: dict[str, np.ndarray] = {}

        def q(s: str) -> np.ndarray:
            if s not in qcache:
                qcache[s] = retriever.encode_query(s)
            return qcache[s]

        for pi, (img, fields) in enumerate(pages):
            enc = retriever.encode_page(img)
            score_of = lambda c: maxsim(q(c), enc.patches)  # noqa: E731
            probe_idx = list(rng.permutation(len(fields)))[: args.max_fields]
            for fi in probe_idx:
                ftype, truth = fields[fi]
                cands = _lineup(truth, all_texts, args.k, rng)
                if len(cands) < 2:  # nothing to discriminate against
                    continue
                ranked, hit1, hit5 = _probe(truth, cands, score_of)
                hits1.append(int(hit1))
                hits5.append(int(hit5))
                by_field.setdefault(ftype, []).append(int(hit1))
                by_length.setdefault(_length_bucket(truth), []).append(int(hit1))
                by_glyph.setdefault(_glyph_bucket(truth), []).append(int(hit1))
                if len(rows) < 40:
                    rows.append({"page": pi, "field": ftype, "true": truth,
                                 "recovered": ranked[0], "top1": hit1, "in_top5": hit5})

    n = len(hits1)
    summary = {
        "n_fields": n,
        "lineup": args.k,
        "chance": 1.0 / args.k,
        "top1_acc": float(np.mean(hits1)) if n else 0.0,
        "top5_acc": float(np.mean(hits5)) if n else 0.0,
        "by_field_top1": {k: float(np.mean(v)) for k, v in by_field.items()},
        "by_length_top1": {k: float(np.mean(v)) for k, v in by_length.items()},
        "by_glyph_top1": {k: float(np.mean(v)) for k, v in by_glyph.items()},
    }

    print(f"\n=== REAL-CORPUS TRANSFER ({args.corpus}) ===")
    print(f"  status   : {status}")
    if n:
        print(f"  top1={summary['top1_acc']:.3f}  top5={summary['top5_acc']:.3f}  "
              f"(chance {summary['chance']:.3f}, n={n})")
        print(f"  by field : {summary['by_field_top1']}")
        print(f"  by length: {summary['by_length_top1']}")
        print(f"  by glyph : {summary['by_glyph_top1']}")
        for r in rows[:12]:
            print(f"  [{r['field']:<12}] true {r['true'][:22]!r:<24} -> "
                  f"{r['recovered'][:22]!r} {'OK' if r['top1'] else 'x'}")
    else:
        print("  no fields probed (corpus unavailable or empty) — wrote an empty-but-valid payload.")

    payload = {
        "mode": "realcorpus_transfer",
        "corpus": args.corpus,
        "model": args.model,
        "n_pages": len(pages),
        "k": args.k,
        "status": status,
        "summary": summary,
        "rows": rows,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "realcorpus_transfer.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote realcorpus_transfer.json -> {args.out}")


if __name__ == "__main__":
    main()
