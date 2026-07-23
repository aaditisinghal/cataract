"""Baseline frontiers vs our learned RedactionProjection (MASTER_PLAN S8, Claim 2 — dominance).

Claim 2 is a DOMINANCE claim: our learned, anisotropic index-time transform P beats the field's
embedding-privacy defenses on the same privacy/utility frontier and the same threat model (transform the
STORED patches only; query with the vanilla frozen encoder). This experiment ports three baselines
(EntroGuard, PRESS, KOGA — see ``patchguard/defense/baselines.py``) as index-time patch transforms and
sweeps each one's strength knob over the SAME held-out synthetic ID cards used by ``learned_defense.py``:

    utility  = topic-query Recall@1 over the held-out corpus (retrieval preserved?)
    privacy  = 1 - name dictionary-attack top-1 (PII query suppressed?)

It also trains our learned P at a small lambda sweep to lay down OUR frontier as the reference, then
reports a dominance summary: at matched privacy (0.5 / 0.8 / 0.9) how much more utility does learned P
retain than each baseline's best. A positive result — learned P above every baseline frontier at matched
privacy — is Claim 2. If any baseline meets or beats P, the claim is not yet supported and says so.

Names are OPEN-SET (train/test name pools are disjoint) exactly as in learned_defense.py, so privacy is
measured against names P never saw. Eval uses the numpy MaxSim late-interaction so the sweep helpers are
CPU/torch-free and unit-testable with the mock retriever; only the learned-P reference needs torch.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------------------
# Reusable, retriever-agnostic helpers (imported by tests with the mock retriever; no colpali/torch).
# --------------------------------------------------------------------------------------------------
def make_qcache(retriever):
    """A memoized query-encoder: str -> (nq, d) np array."""
    cache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in cache:
            cache[s] = np.asarray(retriever.encode_query(s), dtype=np.float32)
        return cache[s]

    return q


def gen_cards(retriever, names, k, seed0, font_size, rng):
    """Encode k ID cards (fixed names drawn from ``names``) into stored image patches + topic line."""
    from patchguard.data.synthdoc import generate_id_card

    cards = []
    for i in range(k):
        nm = names[int(rng.integers(0, len(names)))]
        im, fs = generate_id_card(seed0 + i, value_font_size=font_size, vary=True,
                                  fixed_name=nm, with_topic=True)
        enc = retriever.encode_page(im)
        topic = next((f.text for f in fs if f.field_type == "office"), "OFFICE")
        cards.append({"patches": enc.image_patches().astype(np.float32), "name": nm, "topic": topic})
    return cards


def frontier_point(test, transform, q, pool, distractors, rng):
    """One (utility, privacy) point for a stored-patch ``transform`` (patches_np -> patches_np)."""
    from patchguard.retrievers.base import maxsim

    stored = [np.asarray(transform(c["patches"]), dtype=np.float32) for c in test]
    # utility: topic query ranks its own doc #1
    hit_u = []
    for i, c in enumerate(test):
        tq = q(c["topic"])
        scores = [maxsim(tq, d) for d in stored]
        hit_u.append(int(int(np.argmax(scores)) == i))
    # privacy: name dictionary attack (true name vs `distractors` decoys) top-1
    rec = []
    for i, c in enumerate(test):
        decoys = list(rng.choice([x for x in pool if x != c["name"]], distractors, replace=False))
        cands = [c["name"], *decoys]
        sc = [maxsim(q(cc), stored[i]) for cc in cands]
        rec.append(int(int(np.argmax(sc)) == 0))
    return float(np.mean(hit_u)), 1.0 - float(np.mean(rec))


def baseline_frontier(name, test, q, pool, strengths, distractors, rng, contrast=None):
    """Sweep one baseline's strength knob -> list of {strength, utility, privacy}."""
    from patchguard.defense.baselines import apply_baseline

    pts = []
    for s in strengths:
        def tf(p, s=s):
            return apply_baseline(name, p, s, rng=rng, contrast=contrast)

        u, pv = frontier_point(test, tf, q, pool, distractors, rng)
        pts.append({"strength": float(s), "utility": u, "privacy": pv})
    return pts


def util_at_priv(pts, target):
    """Interpolate utility at a target privacy along a frontier (monotone-sorted by privacy)."""
    if not pts:
        return float("nan")
    P = np.array([x["privacy"] for x in pts], dtype=float)
    U = np.array([x["utility"] for x in pts], dtype=float)
    o = np.argsort(P)
    return float(np.interp(target, P[o], U[o]))


# --------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/baseline_frontier")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--np-train", type=int, default=128)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--strengths", default="0.05,0.1,0.2,0.35,0.5",
                    help="baseline strength knob sweep (identity -> heavy)")
    ap.add_argument("--lams", default="0,2,5,10", help="lambda sweep for our learned-P reference frontier")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from patchguard.data.synthdoc import _FIRST, _LAST
    from patchguard.defense.baselines import BASELINE_NAMES
    from patchguard.defense.redact import train_redactor
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strengths = [float(x) for x in args.strengths.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
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

    # PII-vs-content contrast handed to PRESS: per-train-card (mean name-query dir - mean topic-query dir).
    contrast = np.stack([q(c["name"]).mean(0) - q(c["topic"]).mean(0) for c in train]).astype(np.float32)

    # --- baseline frontiers ---
    baselines = {}
    for name in BASELINE_NAMES:
        c = contrast if name == "press" else None
        pts = baseline_frontier(name, test, q, pool, strengths, args.distractors, rng, contrast=c)
        baselines[name] = pts
        for p in pts:
            print(f"{name:>10} s={p['strength']:>5}: utility={p['utility']:.3f} privacy={p['privacy']:.3f}")

    # --- OUR learned-P reference frontier ---
    def sub(p, k):
        idx = rng.choice(p.shape[0], min(k, p.shape[0]), replace=False)
        return p[idx]

    tr_patches = torch.tensor(np.stack([sub(c["patches"], args.np_train) for c in train]))
    tr_topic = [torch.tensor(q(c["topic"])) for c in train]
    tr_name = [torch.tensor(q(c["name"])) for c in train]
    distr_q = [torch.tensor(q(n)) for n in list(rng.choice(train_names, 16, replace=False))]

    learned = []
    for lam in lams:
        P = train_redactor(tr_patches, tr_topic, tr_name, distr_q, lam=lam, dim=128,
                           epochs=args.epochs, device=device, seed=args.seed, distractors=distr_q)

        def tf_P(patches_np, P=P):
            with torch.no_grad():
                return P(torch.tensor(patches_np).to(device)).cpu().numpy()

        u, pv = frontier_point(test, tf_P, q, pool, args.distractors, rng)
        learned.append({"lambda": lam, "utility": u, "privacy": pv})
        print(f"  learned lam={lam:>4}: utility={u:.3f} privacy={pv:.3f}")

    # --- dominance summary: learned-P utility minus each baseline's utility at matched privacy ---
    targets = (0.5, 0.8, 0.9)
    dominance = {}
    for name, pts in baselines.items():
        row = {}
        for tp in targets:
            lu, bu = util_at_priv(learned, tp), util_at_priv(pts, tp)
            row[str(tp)] = {"learned_util": lu, "baseline_util": bu, "delta": lu - bu}
        row["dominated_by_learned"] = bool(all(row[str(tp)]["delta"] > 0.0 for tp in targets))
        dominance[name] = row
        print(f"DOMINANCE vs {name}: " + ", ".join(
            f"@{tp}:{row[str(tp)]['delta']:+.3f}" for tp in targets)
            + f"  dominated={row['dominated_by_learned']}")

    all_dominated = all(dominance[n]["dominated_by_learned"] for n in baselines)
    verdict = ("LEARNED P DOMINATES ALL BASELINES" if all_dominated
               else "LEARNED P DOES NOT DOMINATE ALL BASELINES")
    print("VERDICT:", verdict)

    payload = {
        "mode": "baseline_frontier",
        "model": args.model,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "distractors": args.distractors,
        "font_size": args.font_size,
        "open_set_names": True,
        "strengths": strengths,
        "lams": lams,
        "baseline_frontiers": baselines,
        "learned_frontier": learned,
        "dominance": dominance,
        "all_dominated": all_dominated,
        "verdict": verdict,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "baseline_frontier.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote baseline_frontier.json -> {args.out}")


if __name__ == "__main__":
    main()
