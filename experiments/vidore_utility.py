"""Retrieval-utility cost of the redaction defense (MASTER_PLAN B2) — what does P actually cost?

A privacy defense that destroys the index is worthless. This experiment measures the REAL retrieval
utility of three indices over the SAME corpus, so the utility price of privacy is explicit:

  * vanilla    : stored patches unchanged (the undefended, maximal-utility upper bound).
  * P          : stored patches passed through a trained RedactionProjection (the anisotropic defense).
  * flat-noise : stored patches + isotropic Gaussian, at the sigma whose PRIVACY MATCHES P's privacy.

Utility is standard ranked-retrieval quality (NDCG@5, Recall@1, MRR) of legitimate topic queries against
the corpus. The scientific claim: if P retains far more utility than flat-noise AT THE SAME PRIVACY, the
learned defense buys privacy cheaply; if P's utility collapses to flat-noise, the defense is no better
than blurring the whole index. A positive result = P >> flat-noise utility at matched privacy.

Data: a ViDoRe-style (query, page) set via HuggingFace ``datasets`` when available; otherwise (or with
``--synthetic``) a multi-topic synthetic corpus of ``generate_id_cards(with_topic=True)`` where the
issuing-office line is the legitimate, non-PII retrieval target and the name is the PII the attacker wants.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------------------
# ranked-retrieval metrics (single relevant doc per query -> binary graded relevance)
# --------------------------------------------------------------------------------------------------
def _rank_of_gold(scores: np.ndarray, gold: int) -> int:
    """1-indexed rank of the gold doc under descending score (ties broken by argsort order)."""
    order = np.argsort(-scores)
    return int(np.where(order == gold)[0][0]) + 1


def _ndcg_at_k(rank: int, k: int = 5) -> float:
    # one relevant doc with gain 1 -> IDCG = 1/log2(2) = 1; DCG = 1/log2(rank+1) if in top-k else 0.
    return float(1.0 / np.log2(rank + 1)) if rank <= k else 0.0


def compute_retrieval_metrics(queries: list[np.ndarray], docs: list[np.ndarray],
                              golds: list[int], k: int = 5) -> dict:
    """queries: list of (nq,d); docs: list of (Np,d); golds: index into docs per query.

    Returns {"ndcg@5", "recall@1", "mrr"} — each a mean over queries, all in [0,1].
    """
    from patchguard.retrievers.base import maxsim

    if not queries:
        return {"ndcg@5": 0.0, "recall@1": 0.0, "mrr": 0.0}
    ndcg, recall1, rr = [], [], []
    for qv, g in zip(queries, golds):
        scores = np.array([maxsim(qv, d) for d in docs], dtype=np.float64)
        rank = _rank_of_gold(scores, g)
        ndcg.append(_ndcg_at_k(rank, k))
        recall1.append(1.0 if rank == 1 else 0.0)
        rr.append(1.0 / rank)
    return {"ndcg@5": float(np.mean(ndcg)), "recall@1": float(np.mean(recall1)),
            "mrr": float(np.mean(rr))}


# --------------------------------------------------------------------------------------------------
# defense transforms (numpy, applied to the STORED patches only)
# --------------------------------------------------------------------------------------------------
def _apply_P_np(P, patches_np: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return P(torch.tensor(patches_np, dtype=torch.float32)).cpu().numpy().astype(np.float32)


def _flat_noise_np(patches_np: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Isotropic Gaussian at relative magnitude sigma, renormalized to the unit sphere."""
    if sigma <= 0:
        return patches_np
    noise = rng.standard_normal(patches_np.shape).astype(np.float32)
    p = patches_np + sigma * np.linalg.norm(patches_np, axis=1, keepdims=True) * noise
    return (p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def _name_privacy(qfn, docs: list[np.ndarray], names: list[str], pool: list[str],
                  rng: np.random.Generator, distractors: int) -> float:
    """Privacy = 1 - name dictionary-attack top-1. Attacker uses vanilla name queries on stored docs."""
    from patchguard.retrievers.base import maxsim

    hit = []
    for i, nm in enumerate(names):
        others = [x for x in pool if x != nm]
        cands = [nm] + list(rng.choice(others, min(distractors, len(others)), replace=False))
        sc = np.array([maxsim(qfn(c), docs[i]) for c in cands])
        hit.append(int(np.argmax(sc) == 0))
    return 1.0 - float(np.mean(hit)) if hit else 1.0


def _interp_at_priv(frontier: list[dict], target: float, key: str) -> float:
    """Interpolate a utility metric of the flat-noise frontier at a target privacy."""
    if not frontier:
        return 0.0
    P = np.array([f["privacy"] for f in frontier])
    U = np.array([f["metrics"][key] for f in frontier])
    o = np.argsort(P)
    return float(np.interp(target, P[o], U[o]))


# --------------------------------------------------------------------------------------------------
# synthetic multi-topic corpus path (full-featured: trains P, measures privacy, matched flat-noise)
# --------------------------------------------------------------------------------------------------
def run_synthetic(retriever, *, n_corpus: int, n_queries: int, n_train: int, lam: float,
                  epochs: int, seed: int, font_size: int = 24, dim: int = 128,
                  np_train: int = 32, distractors_priv: int = 64,
                  sigmas: tuple[float, ...] = (0.1, 0.2, 0.35, 0.5, 0.75)) -> dict:
    """Build vanilla / P / flat-noise indices over a synthetic office-topic corpus and score utility."""
    import torch

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_card
    from patchguard.defense.redact import train_redactor

    rng = np.random.default_rng(seed)
    pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # K=240
    rng.shuffle(pool)
    train_names, corpus_names = pool[:180], pool[180:]  # disjoint -> open-set privacy

    qcache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in qcache:
            qcache[s] = retriever.encode_query(s).astype(np.float32)
        return qcache[s]

    def gen(names, k, seed0):
        cards = []
        for i in range(k):
            nm = names[int(rng.integers(0, len(names)))]
            im, fs = generate_id_card(seed0 + i, value_font_size=font_size, vary=True,
                                      fixed_name=nm, with_topic=True)
            enc = retriever.encode_page(im)
            topic = next((f.text for f in fs if f.field_type == "office"), "OFFICE")
            cards.append({"patches": enc.image_patches().astype(np.float32), "name": nm, "topic": topic})
        return cards

    train = gen(train_names, n_train, 1000)
    corpus = gen(corpus_names, n_corpus, 5000)

    # queries: a subset of corpus docs, each retrieved by its own topic line (gold = itself).
    n_q = min(n_queries, len(corpus))
    query_idx = list(range(n_q))
    q_topic = [q(corpus[i]["topic"]) for i in query_idx]
    corpus_names_list = [c["name"] for c in corpus]

    docs_van = [c["patches"] for c in corpus]
    vanilla = compute_retrieval_metrics(q_topic, docs_van, query_idx)

    # --- train the anisotropic defense P (topic = utility target, name = privacy target) ---
    def sub(p, kk):
        idx = rng.choice(p.shape[0], min(kk, p.shape[0]), replace=False)
        return p[idx]

    tr_patches = torch.tensor(np.stack([sub(c["patches"], np_train) for c in train]))
    tr_topic = [torch.tensor(q(c["topic"])) for c in train]
    tr_name = [torch.tensor(q(c["name"])) for c in train]
    distr_q = [torch.tensor(q(n)) for n in rng.choice(train_names, min(16, len(train_names)), replace=False)]
    P = train_redactor(tr_patches, tr_topic, tr_name, distr_q, lam=lam, dim=dim,
                       epochs=epochs, seed=seed, distractors=distr_q)

    docs_P = [_apply_P_np(P, c["patches"]) for c in corpus]
    p_metrics = compute_retrieval_metrics(q_topic, docs_P, query_idx)
    p_privacy = _name_privacy(q, docs_P, corpus_names_list, pool, rng, distractors_priv)

    # --- flat-noise frontier + its utility interpolated at P's privacy (matched-privacy comparison) ---
    flat = []
    for sigma in sigmas:
        docs_n = [_flat_noise_np(c["patches"], float(sigma), rng) for c in corpus]
        u = compute_retrieval_metrics(q_topic, docs_n, query_idx)
        pv = _name_privacy(q, docs_n, corpus_names_list, pool, rng, distractors_priv)
        flat.append({"sigma": float(sigma), "privacy": pv, "metrics": u})

    flat_matched = {m: _interp_at_priv(flat, p_privacy, m) for m in ("ndcg@5", "recall@1", "mrr")}
    utility_cost = {m: vanilla[m] - p_metrics[m] for m in vanilla}
    p_vs_flat = {m: p_metrics[m] - flat_matched[m] for m in flat_matched}
    verdict = ("P BEATS FLAT NOISE at matched privacy"
               if p_vs_flat["recall@1"] > 0.05 else "NO CLEAR UTILITY WIN over flat noise")

    return {"mode": "vidore_utility", "source": "synthetic", "n_corpus": len(corpus),
            "n_queries": n_q, "lam": lam, "open_set_names": True,
            "vanilla": vanilla, "P": {"metrics": p_metrics, "privacy": p_privacy},
            "flat_frontier": flat, "matched_privacy": p_privacy,
            "flat_at_matched_privacy": flat_matched, "utility_cost_of_P": utility_cost,
            "P_minus_flat_at_matched_privacy": p_vs_flat, "verdict": verdict}


# --------------------------------------------------------------------------------------------------
# real ViDoRe path (best-effort; utility only — no PII labels, so privacy is not measurable here)
# --------------------------------------------------------------------------------------------------
def _load_vidore(dataset: str, n_corpus: int, n_queries: int, seed: int):
    """Load a ViDoRe-style (query, image) HF dataset. Corpus = images; gold = same-row image."""
    from datasets import load_dataset  # guarded: only present in the GPU container

    ds = load_dataset(dataset, split="test")
    rng = np.random.default_rng(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[: max(n_corpus, n_queries)]
    imgs, queries, golds = [], [], []
    for pos, row_i in enumerate(idx):
        row = ds[int(row_i)]
        img = row["image"]
        arr = np.asarray(img.convert("RGB")) if hasattr(img, "convert") else np.asarray(img)
        imgs.append(arr.astype(np.uint8))
        qtext = row.get("query") or row.get("question")
        if qtext and len(queries) < n_queries:
            queries.append(str(qtext))
            golds.append(pos)
    return imgs[:n_corpus], queries, golds


def run_real(retriever, corpus_imgs, queries, golds, *, sigmas, dim: int, p_ckpt: str | None,
             seed: int) -> dict:
    rng = np.random.default_rng(seed)
    docs_van = [retriever.encode_page(im).image_patches().astype(np.float32) for im in corpus_imgs]
    q_np = [retriever.encode_query(t).astype(np.float32) for t in queries]
    vanilla = compute_retrieval_metrics(q_np, docs_van, golds)

    p_block = None
    if p_ckpt:
        import torch

        from patchguard.defense.redact import RedactionProjection

        P = RedactionProjection(dim=dim)
        P.load_state_dict(torch.load(p_ckpt, map_location="cpu"))
        P.eval()
        docs_P = [_apply_P_np(P, d) for d in docs_van]
        p_block = compute_retrieval_metrics(q_np, docs_P, golds)

    flat = []
    for sigma in sigmas:
        docs_n = [_flat_noise_np(d, float(sigma), rng) for d in docs_van]
        flat.append({"sigma": float(sigma), "metrics": compute_retrieval_metrics(q_np, docs_n, golds)})

    return {"mode": "vidore_utility", "source": "vidore", "n_corpus": len(corpus_imgs),
            "n_queries": len(queries), "vanilla": vanilla,
            "P": ({"metrics": p_block} if p_block else None), "flat_frontier": flat,
            "note": "real ViDoRe corpus has no PII labels -> privacy not measured here"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/vidore_utility")
    ap.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    ap.add_argument("--synthetic", action="store_true", help="force the synthetic office-topic corpus")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--retriever", default="colpali", choices=["colpali", "colqwen2"])
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-corpus", type=int, default=60)
    ap.add_argument("--n-queries", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--distractors", type=int, default=64, help="lineup size for the name privacy attack")
    ap.add_argument("--p-ckpt", default=None, help="optional RedactionProjection state_dict for the real path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    if args.retriever == "colqwen2":
        from patchguard.retrievers.colqwen2 import ColQwen2Retriever
        retriever = ColQwen2Retriever(model_name=args.model or "vidore/colqwen2-v1.0")
    else:
        from patchguard.retrievers.colpali import ColPaliRetriever
        retriever = ColPaliRetriever(model_name=args.model)

    sigmas = (0.1, 0.2, 0.35, 0.5, 0.75)
    payload = None
    if not args.synthetic:
        try:
            imgs, queries, golds = _load_vidore(args.dataset, args.n_corpus, args.n_queries, args.seed)
            payload = run_real(retriever, imgs, queries, golds, sigmas=sigmas, dim=128,
                               p_ckpt=args.p_ckpt, seed=args.seed)
        except Exception as e:  # datasets missing / schema mismatch -> honest fallback
            print(f"[vidore_utility] real dataset unavailable ({e!r}); falling back to --synthetic")
    if payload is None:
        payload = run_synthetic(retriever, n_corpus=args.n_corpus, n_queries=args.n_queries,
                                n_train=args.n_train, lam=args.lam, epochs=args.epochs, seed=args.seed,
                                font_size=args.font_size, dim=128, distractors_priv=args.distractors,
                                sigmas=sigmas)

    payload["fingerprint"] = run_fingerprint()

    print(f"\n=== VIDORE UTILITY ({payload['source']}) ===")
    v = payload["vanilla"]
    print(f"  vanilla : ndcg@5={v['ndcg@5']:.3f}  recall@1={v['recall@1']:.3f}  mrr={v['mrr']:.3f}")
    if payload.get("P"):
        pm = payload["P"]["metrics"]
        priv = payload["P"].get("privacy")
        print(f"  P       : ndcg@5={pm['ndcg@5']:.3f}  recall@1={pm['recall@1']:.3f}  mrr={pm['mrr']:.3f}"
              + (f"  privacy={priv:.3f}" if priv is not None else ""))
    if "flat_at_matched_privacy" in payload:
        fm = payload["flat_at_matched_privacy"]
        print(f"  flat@P-privacy: ndcg@5={fm['ndcg@5']:.3f}  recall@1={fm['recall@1']:.3f}  mrr={fm['mrr']:.3f}")
        print("VERDICT:", payload["verdict"])

    (local_out / "vidore_utility.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote vidore_utility.json -> {args.out}")


if __name__ == "__main__":
    main()
