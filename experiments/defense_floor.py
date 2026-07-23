"""The FUNDAMENTAL FLOOR of the redaction defense (MASTER_PLAN B3) — when can NO P help?

RedactionProjection works only if the PII directions and the legitimate-retrieval directions are
SEPARABLE in embedding space. That assumption fails at one extreme: if the legitimate query IS the
name ("find the card belonging to JOHN SMITH"), then the PII value is literally the retrieval content
— suppressing it for privacy necessarily destroys utility. There is no anisotropic transform that keeps
a direction usable for retrieval while making it un-rankable for an attacker who queries the SAME
direction. This experiment maps exactly where that wall is.

We parameterize retrieval intent by an entanglement knob ``alpha in [0,1]``. The legitimate per-doc
utility query is a weighted concatenation of the topic tokens and the name tokens::

    q_util(alpha) = concat( (1 - alpha) * topic_tokens ,  alpha * name_tokens )

  * alpha = 0 : retrieval is purely about the topic; the name is incidental -> P should keep utility high
                while driving name-attack privacy up (a real, open frontier).
  * alpha = 1 : retrieval is find-by-name; the name IS the content -> the utility query and the attacker
                query coincide, so the frontier must collapse to the leak (privacy XOR utility).

For each alpha we sweep the privacy weight lambda, trace the (privacy, utility) frontier, and read off
the best utility still achievable at a target privacy. The ``crossover alpha`` is the smallest alpha at
which that best-achievable utility falls below the collapse threshold — the empirical location of the
floor. This turns "the defense can't be perfect" into a measured boundary.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def _blend_query(topic_np: np.ndarray, name_np: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted token concatenation. MaxSim sums max-dot over query tokens, so a per-token scale is a
    smooth mixing knob: alpha=0 -> pure topic, alpha=1 -> pure name (the zeroed side contributes 0)."""
    return np.concatenate([(1.0 - alpha) * topic_np, alpha * name_np], axis=0).astype(np.float32)


def _apply_P_np(P, patches_np: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return P(torch.tensor(patches_np, dtype=torch.float32)).cpu().numpy().astype(np.float32)


def _self_retrieval_recall1(queries: list[np.ndarray], docs: list[np.ndarray]) -> float:
    """Utility = fraction of docs retrieved rank-1 by their own (blended) legitimate query."""
    from patchguard.retrievers.base import maxsim

    if not docs:
        return 0.0
    hit = []
    for i, qv in enumerate(queries):
        scores = np.array([maxsim(qv, d) for d in docs])
        hit.append(int(np.argmax(scores) == i))
    return float(np.mean(hit))


def _name_privacy(qfn, docs: list[np.ndarray], names: list[str], pool: list[str],
                  rng: np.random.Generator, distractors: int) -> float:
    """Privacy = 1 - name dictionary-attack top-1 (attacker queries the pure name direction)."""
    from patchguard.retrievers.base import maxsim

    hit = []
    for i, nm in enumerate(names):
        others = [x for x in pool if x != nm]
        cands = [nm] + list(rng.choice(others, min(distractors, len(others)), replace=False))
        sc = np.array([maxsim(qfn(c), docs[i]) for c in cands])
        hit.append(int(np.argmax(sc) == 0))
    return 1.0 - float(np.mean(hit)) if hit else 1.0


def _interp_util_at_priv(frontier: list[dict], target: float) -> float:
    """Best-achievable utility at a target privacy, interpolated over the lambda frontier."""
    if not frontier:
        return 0.0
    P = np.array([f["privacy"] for f in frontier])
    U = np.array([f["utility"] for f in frontier])
    o = np.argsort(P)
    return float(np.interp(target, P[o], U[o]))


def run_floor(retriever, *, alphas: list[float], n_train: int, n_test: int, epochs: int,
              lams: tuple[float, ...] = (0.0, 1.0, 5.0, 20.0), seed: int = 0, font_size: int = 24,
              dim: int = 128, priv_target: float = 0.8, collapse_threshold: float = 0.5,
              np_train: int = 32, distractors_priv: int = 64) -> dict:
    """Sweep entanglement alpha; per alpha trace the (privacy, utility) frontier of trained P."""
    import torch

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_card
    from patchguard.defense.redact import train_redactor

    rng = np.random.default_rng(seed)
    pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # K=240
    rng.shuffle(pool)
    train_names, test_names = pool[:180], pool[180:]  # disjoint -> open-set privacy

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
            cards.append({"patches": enc.image_patches().astype(np.float32), "name": nm,
                          "topic_q": q(topic), "name_q": q(nm)})
        return cards

    train = gen(train_names, n_train, 1000)
    test = gen(test_names, n_test, 5000)
    test_names_list = [c["name"] for c in test]

    def sub(p, kk):
        idx = rng.choice(p.shape[0], min(kk, p.shape[0]), replace=False)
        return p[idx]

    tr_patches = torch.tensor(np.stack([sub(c["patches"], np_train) for c in train]))
    tr_name = [torch.tensor(c["name_q"]) for c in train]
    distr_q = [torch.tensor(q(n)) for n in rng.choice(train_names, min(16, len(train_names)), replace=False)]

    per_alpha = []
    for alpha in alphas:
        a = float(alpha)
        # utility target = alpha-blended (topic, name) query, for both training and eval.
        tr_util = [torch.tensor(_blend_query(c["topic_q"], c["name_q"], a)) for c in train]
        te_util = [_blend_query(c["topic_q"], c["name_q"], a) for c in test]

        frontier = []
        for lam in lams:
            P = train_redactor(tr_patches, tr_util, tr_name, distr_q, lam=float(lam), dim=dim,
                               epochs=epochs, seed=seed, distractors=distr_q)
            docs_P = [_apply_P_np(P, c["patches"]) for c in test]
            utility = _self_retrieval_recall1(te_util, docs_P)
            privacy = _name_privacy(q, docs_P, test_names_list, pool, rng, distractors_priv)
            frontier.append({"lambda": float(lam), "privacy": privacy, "utility": utility})

        util_at_target = _interp_util_at_priv(frontier, priv_target)
        collapsed = bool(util_at_target < collapse_threshold)
        per_alpha.append({"alpha": a, "frontier": frontier,
                          "util_at_priv_target": util_at_target, "collapsed": collapsed})
        print(f"alpha={a:.2f} | util@priv>={priv_target:.2f} = {util_at_target:.3f}"
              f"  {'COLLAPSED (floor)' if collapsed else 'open frontier'}")

    crossover = next((p["alpha"] for p in per_alpha if p["collapsed"]), None)
    print(f"CROSSOVER alpha (frontier collapses to the leak): {crossover}")

    return {"mode": "defense_floor", "priv_target": priv_target,
            "collapse_threshold": collapse_threshold, "lams": [float(x) for x in lams],
            "n_train": n_train, "n_test": n_test, "open_set_names": True,
            "per_alpha": per_alpha, "crossover_alpha": crossover}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/defense_floor")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--retriever", default="colpali", choices=["colpali", "colqwen2"])
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--lams", default="0,1,5,20")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--distractors", type=int, default=64)
    ap.add_argument("--priv-target", type=float, default=0.8)
    ap.add_argument("--collapse-threshold", type=float, default=0.5)
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

    alphas = [float(x) for x in args.alphas.split(",")]
    lams = tuple(float(x) for x in args.lams.split(","))

    payload = run_floor(retriever, alphas=alphas, n_train=args.n_train, n_test=args.n_test,
                        epochs=args.epochs, lams=lams, seed=args.seed, font_size=args.font_size,
                        dim=128, priv_target=args.priv_target,
                        collapse_threshold=args.collapse_threshold, distractors_priv=args.distractors)
    payload["fingerprint"] = run_fingerprint()

    (local_out / "defense_floor.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote defense_floor.json -> {args.out}")


if __name__ == "__main__":
    main()
