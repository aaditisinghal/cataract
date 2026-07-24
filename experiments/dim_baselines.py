"""Dimensionality-reduction baselines vs Cataract (reviewer Tier-2: "why not just smaller embeddings?").

A natural reviewer objection to Cataract (``NullspaceRedaction``): if the win is "remove the PII
subspace", couldn't you get the same privacy for free just by storing SMALLER embeddings? This
experiment answers it directly by putting two content-agnostic dimensionality reducers on the SAME
privacy/utility frontier and the SAME threat model as ``learned_defense.py`` / ``certified_defense.py``
(the index-time transform touches the STORED patches; the query is encoded by the vanilla frozen
encoder and then reduced the SAME way, so scoring stays consistent in the reduced space):

  * RANDOM PROJECTION to 32-d — a fixed Gaussian matrix R (d=128 -> 32) applied to index AND query
    patches, then unit-normalized and scored with MaxSim in 32-d. Compression that is oblivious to
    which directions carry PII vs content.
  * PCA to 32-d — fit the top-32 principal components on TRAIN patches, project index+query onto them.
    Compression that keeps the highest-variance directions (which need NOT be the content directions).

  utility = topic-query Recall@1 over the held-out corpus (retrieval preserved?)
  privacy = 1 - name dictionary-attack top-1 (PII query suppressed?)

Both reducers are UNINFORMED about the PII-vs-content split, so the expectation (which isolates
subspace-removal from mere compression) is that they cannot reach Cataract's operating point: at a
compression heavy enough to suppress the name attack they also crush topic utility, or they preserve
utility but leak PII holographically just like the full-dim index. Cataract's k=96 point (privacy 0.90 /
utility 0.875 — ``certified_defense.py`` k-sweep, RESULTS.md §4.13–4.16) is cited as the reference the
reducers must match to defeat the objection.

The eval math is numpy MaxSim only, so the reducers and the frontier point are CPU/torch-free and
unit-testable with the mock retriever; only the ColPali encode step in ``main`` needs the GPU stack.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

# Cataract's reported operating point (NullspaceRedaction k-sweep, certified_defense.py; RESULTS.md
# §4.13–4.16). Cited, not recomputed here — this experiment only needs the reducer points to contrast.
CATARACT_REF = {
    "method": "Cataract (NullspaceRedaction)",
    "k": 96,
    "privacy": 0.90,
    "utility": 0.875,
    "source": "certified_defense.py k-sweep; RESULTS.md §4.13–4.16",
}


# --------------------------------------------------------------------------------------------------
# Reusable, retriever-agnostic reducers + eval (imported by tests with the mock retriever; no colpali).
# Each reducer is a callable proj: (..., d_in) -> (..., d_out) unit-normalized, applied to BOTH the
# stored patches and the query tokens so MaxSim is scored consistently in the reduced space.
# --------------------------------------------------------------------------------------------------
def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def random_projection_matrix(d_in: int, d_out: int, rng) -> np.ndarray:
    """Fixed Gaussian projection R (d_in -> d_out), scaled 1/sqrt(d_out) (JL-style norm preservation)."""
    d_out = min(int(d_out), int(d_in))
    R = rng.standard_normal((int(d_in), d_out)).astype(np.float32) / np.sqrt(d_out)
    return R


def make_random_proj(R: np.ndarray):
    """Reducer from a fixed random matrix: x -> l2normalize(x @ R)."""
    def proj(x: np.ndarray) -> np.ndarray:
        return _l2(np.asarray(x, dtype=np.float32) @ R)

    return proj


def fit_pca(patches: np.ndarray, d_out: int):
    """Fit top-``d_out`` PCA on stacked TRAIN patches (N, d). Returns (mean (1,d), comps (d_out,d))."""
    X = np.asarray(patches, dtype=np.float32).reshape(-1, np.asarray(patches).shape[-1])
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(int(d_out), Vt.shape[0])
    return mean.astype(np.float32), Vt[:k].astype(np.float32)


def make_pca_proj(mean: np.ndarray, comps: np.ndarray):
    """Reducer onto fitted PCA components: x -> l2normalize((x - mean) @ comps^T)."""
    def proj(x: np.ndarray) -> np.ndarray:
        return _l2((np.asarray(x, dtype=np.float32) - mean) @ comps.T)

    return proj


def dimreduce_frontier_point(test, proj, q, pool, distractors, rng):
    """One (utility, privacy) point for a dimensionality reducer applied to index AND query.

    utility = topic-query Recall@1 (each card's topic query must rank its own doc #1 in reduced space).
    privacy = 1 - name dictionary-attack top-1 (true name vs ``distractors`` decoys, all reduced).
    """
    from patchguard.retrievers.base import maxsim

    stored = [proj(c["patches"]) for c in test]
    # utility: topic query (reduced) ranks its own doc first
    hit_u = []
    for i, c in enumerate(test):
        tq = proj(q(c["topic"]))
        scores = [maxsim(tq, d) for d in stored]
        hit_u.append(int(int(np.argmax(scores)) == i))
    # privacy: name dictionary attack (true name vs decoys), all queries reduced consistently
    rec = []
    for i, c in enumerate(test):
        decoys = list(rng.choice([x for x in pool if x != c["name"]], distractors, replace=False))
        cands = [c["name"], *decoys]
        sc = [maxsim(proj(q(cc)), stored[i]) for cc in cands]
        rec.append(int(int(np.argmax(sc)) == 0))
    return float(np.mean(hit_u)), 1.0 - float(np.mean(rec))


def matches_cataract(point, ref=CATARACT_REF, tol=0.02) -> bool:
    """Does a reducer point reach BOTH Cataract's privacy and utility (within ``tol``)?"""
    return (point["privacy"] >= ref["privacy"] - tol) and (point["utility"] >= ref["utility"] - tol)


# --------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Random-projection / PCA compression vs Cataract on the privacy/utility frontier."
    )
    ap.add_argument("--out", default="results/dim_baselines")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n-train", type=int, default=64, help="cards for fitting PCA / random matrix")
    ap.add_argument("--n-test", type=int, default=40, help="held-out open-set victim/eval cards")
    ap.add_argument("--distractors", type=int, default=200, help="name dictionary lineup size")
    ap.add_argument("--reduce-dim", type=int, default=32, help="target dimensionality (d -> reduce-dim)")
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # heavy imports INSIDE main (colpali/torch live in the GPU container only)
    from experiments.baseline_frontier import gen_cards, make_qcache
    from patchguard.data.synthdoc import _FIRST, _LAST
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # K=240
    rng.shuffle(pool)
    train_names, test_names = pool[:180], pool[180:]  # DISJOINT -> open-set privacy

    retriever = ColPaliRetriever(model_name=args.model)
    q = make_qcache(retriever)
    for s in pool:  # warm the full name lineup (true + distractors)
        q(s)

    train = gen_cards(retriever, train_names, args.n_train, 1000, args.font_size, rng)
    test = gen_cards(retriever, test_names, args.n_test, 5000, args.font_size, rng)
    for c in train + test:
        q(c["topic"])

    d_in = int(train[0]["patches"].shape[-1])
    d_out = min(int(args.reduce_dim), d_in)
    print(f"reducing d={d_in} -> {d_out} | n_train={args.n_train} n_test={args.n_test} "
          f"distractors={args.distractors}")

    # --- RANDOM PROJECTION (fixed Gaussian, oblivious to PII vs content) ---
    R = random_projection_matrix(d_in, d_out, rng)
    rp_proj = make_random_proj(R)
    rp_u, rp_p = dimreduce_frontier_point(test, rp_proj, q, pool, args.distractors, rng)
    random_proj = {"dim": d_out, "utility": rp_u, "privacy": rp_p}
    print(f"RANDOM-PROJ ({d_out}d): utility={rp_u:.3f} privacy={rp_p:.3f}")

    # --- PCA (top components fit on TRAIN patches) ---
    train_stack = np.concatenate([c["patches"] for c in train], axis=0)
    mean, comps = fit_pca(train_stack, d_out)
    pca_proj = make_pca_proj(mean, comps)
    pca_u, pca_p = dimreduce_frontier_point(test, pca_proj, q, pool, args.distractors, rng)
    pca32 = {"dim": int(comps.shape[0]), "utility": pca_u, "privacy": pca_p}
    print(f"PCA-{comps.shape[0]}:      utility={pca_u:.3f} privacy={pca_p:.3f}")

    # --- contrast with Cataract's cited k=96 operating point ---
    ref = CATARACT_REF
    rp_match = matches_cataract(random_proj)
    pca_match = matches_cataract(pca32)
    for tag, pt, m in (("RANDOM-PROJ", random_proj, rp_match), ("PCA", pca32, pca_match)):
        du = pt["utility"] - ref["utility"]
        dp = pt["privacy"] - ref["privacy"]
        print(f"vs CATARACT k={ref['k']} (priv {ref['privacy']:.2f}/util {ref['utility']:.3f}): "
              f"{tag} dutil={du:+.3f} dpriv={dp:+.3f} matches={m}")

    if rp_match or pca_match:
        verdict = ("A DIMENSIONALITY REDUCER REACHES CATARACT'S POINT — the 'smaller embeddings' "
                   "objection is NOT dismissed by this run; report which and revisit subspace-removal.")
    else:
        verdict = ("DIMENSIONALITY REDUCTION ALONE CANNOT MATCH CATARACT: neither random projection nor "
                   f"PCA to {d_out}-d reaches privacy {ref['privacy']:.2f} AT utility {ref['utility']:.3f} "
                   "(each tanks utility or leaks PII). Compression is content-agnostic; Cataract removes "
                   "the PII SUBSPACE while sparing content — that is the difference, not the dimension.")
    print("VERDICT:", verdict)

    payload = {
        "mode": "dim_baselines",
        "model": args.model,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "distractors": args.distractors,
        "reduce_dim": d_out,
        "d_in": d_in,
        "font_size": args.font_size,
        "open_set_names": True,
        "random_proj": random_proj,
        "pca32": pca32,
        "cataract_ref": ref,
        "random_proj_matches_cataract": rp_match,
        "pca32_matches_cataract": pca_match,
        "verdict": verdict,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "dim_baselines.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote dim_baselines.json -> {args.out}")


if __name__ == "__main__":
    main()
