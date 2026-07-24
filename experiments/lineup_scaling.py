"""Lineup-size scaling + absent-target rejection (reviewer Tier-1 minimum, MASTER_PLAN S6).

The dictionary/linkage attack (``retrieval_attack.py``) recovers a card's name by ranking a candidate
"lineup" (true value + distractors) by MaxSim against the stored patch embeddings. Two reviewer-mandated
robustness questions this file answers:

(1) LINEUP-SIZE SCALING — does the leak survive a realistically LARGE candidate space, and does it decay
    gracefully? We sweep the lineup size K in {100, 1000, 10000, 100000}. Distractors are name-like
    strings: the closed FIRST x LAST vocabulary (240) for the small end, extended with deterministic
    middle-initial variants ("JAMES A SMITH", "JAMES A B SMITH") to reach 100k unique synthetic names.
    At each K we report top-1 recovery and its LIFT over chance (chance = 1/K). Because larger lineups are
    strict supersets of smaller ones (per card, adding distractors can only break a hit, never create one),
    recovery is monotone non-increasing in K -> graceful decay is structural, not luck.

(2) ABSENT-TARGET / open-world rejection — the attack must NOT be a machine that always emits a candidate.
    For a fraction of trials we REMOVE the true value from the lineup (open world). A rejection rule fires
    on the top-1 MaxSim MARGIN (top1 - top2): accept ("present, here it is") only when margin >= threshold.
    We report, at a chosen threshold, the FALSE-ACCEPT rate (emits a wrong candidate as "present" when the
    true value is actually absent) and the TRUE-ACCEPT rate (correctly emits the true value when present),
    plus the full ROC and its AUC. A high margin genuinely signals presence -> the attacker can abstain.

Core scoring is backend-agnostic (ColPali or the mock retriever) so the CPU tests exercise the exact same
functions on tiny synthetic cards. Heavy imports live inside ``main`` so ``--help`` works with no GPU.
"""

from __future__ import annotations

import argparse
import json
import string
import tempfile
from pathlib import Path

import numpy as np

from patchguard.retrievers.base import maxsim


# ----------------------------------------------------------------------------- candidate pool
def generate_name_candidates(
    n: int, first: list[str], last: list[str], rng: np.random.Generator
) -> list[str]:
    """Return ``n`` unique name-like strings.

    Base = FIRST x LAST (shuffled so a small subsample isn't alphabetically biased). If more are needed,
    extend deterministically with 1- then 2-letter middle initials ("JAMES A SMITH", "JAMES A B SMITH").
    Capacity with |FIRST|=16, |LAST|=15 is 240 + 240*26 + 240*676 = 168720 >> 100000.
    """
    base = [f"{a} {b}" for a in first for b in last]
    rng.shuffle(base)
    out: list[str] = list(dict.fromkeys(base))
    if len(out) >= n:
        return out[:n]
    seen = set(out)
    letters = list(string.ascii_uppercase)
    for m in letters:  # single middle initial
        for a in first:
            for b in last:
                s = f"{a} {m} {b}"
                if s not in seen:
                    seen.add(s)
                    out.append(s)
                    if len(out) >= n:
                        return out[:n]
    for m1 in letters:  # two middle initials
        for m2 in letters:
            for a in first:
                for b in last:
                    s = f"{a} {m1} {m2} {b}"
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
                        if len(out) >= n:
                            return out[:n]
    return out[:n]


# ----------------------------------------------------------------------------- scoring
def build_qfn(retriever):
    """A memoized ``text -> query encoding`` closure so each unique candidate is encoded once."""
    cache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        v = cache.get(s)
        if v is None:
            v = retriever.encode_query(s)
            cache[s] = v
        return v

    return q


def score_pool_per_card(retriever, cards, pool: list[str], field: str = "name", qfn=None) -> list[dict]:
    """Score every card's true value and every pool distractor by MaxSim, ONCE.

    Returns per card ``{"true", "true_score", "dist_scores"}`` where ``dist_scores`` is the pool with the
    card's own true value removed, in pool order. Both experiments derive from this: recovery at any K is
    ``true_score > max(dist_scores[:K-1])`` and every margin is a top-2 gap of a slice of these scores. So
    the O(cards x pool) encode/score cost is paid exactly once, not per K.
    """
    if qfn is None:
        qfn = build_qfn(retriever)
    results: list[dict] = []
    for im, fs in cards:
        enc = retriever.encode_page(im)
        true = next(f.text for f in fs if f.field_type == field)
        dist = [c for c in pool if c != true]
        ds = np.array([maxsim(qfn(c), enc.patches) for c in dist], dtype=float)
        ts = float(maxsim(qfn(true), enc.patches))
        results.append({"true": true, "true_score": ts, "dist_scores": ds})
    return results


# ----------------------------------------------------------------------------- (1) scaling
def scaling_curve(per_card: list[dict], Ks: list[int]) -> list[dict]:
    """Top-1 recovery + lift-over-chance at each lineup size K (true value + K-1 distractors)."""
    out: list[dict] = []
    n = len(per_card)
    for K in Ks:
        hits = 0
        for r in per_card:
            d = r["dist_scores"][: max(0, K - 1)]
            top = float(d.max()) if d.size else -np.inf
            hits += int(r["true_score"] > top)
        rec = hits / n if n else 0.0
        chance = 1.0 / K if K > 0 else 0.0
        out.append(
            {
                "K": int(K),
                "recovery": rec,
                "chance": chance,
                "lift": (rec / chance) if chance > 0 else 0.0,
            }
        )
    return out


# ----------------------------------------------------------------------------- (2) absent-target
def _roc(present: list[tuple[float, int]], absent: list[float]) -> dict:
    """ROC of the margin-threshold rejection rule. Positive class = target PRESENT.

    ``present`` = (margin, correct) per present trial; accept counts only when the emitted top-1 is the
    true value. ``absent`` = margin per absent trial; any accept is a false accept. Sweeps every observed
    margin as a threshold, returns the point list, the AUC (upper-envelope trapezoid), and the Youden-J
    optimal finite threshold with its (false_accept, true_accept).
    """
    n_p, n_a = len(present), len(absent)
    if n_p == 0 or n_a == 0:
        return {"roc": [], "auc": None, "threshold": None, "false_accept": 0.0, "true_accept": 0.0}
    pm = np.array([m for m, _ in present], dtype=float)
    pc = np.array([c for _, c in present], dtype=float)
    am = np.array(absent, dtype=float)
    finite = np.unique(np.concatenate([pm, am]))
    thrs = np.concatenate([[np.inf], finite[::-1], [-np.inf]])  # high -> low: reject-all first

    pts: list[dict] = []
    best_j, best = -np.inf, {"threshold": None, "false_accept": 0.0, "true_accept": 0.0}
    for t in thrs:
        tpr = float(np.mean((pm >= t) & (pc == 1)))  # true-accept: accepted AND correct
        fpr = float(np.mean(am >= t))  # false-accept: accepted a wrong candidate as present
        pts.append({"threshold": (float(t) if np.isfinite(t) else None), "fpr": fpr, "tpr": tpr})
        if np.isfinite(t) and (tpr - fpr) > best_j:
            best_j = tpr - fpr
            best = {"threshold": float(t), "false_accept": fpr, "true_accept": tpr}

    # AUC over the upper envelope (max tpr per fpr), with (0,0)/(1,1) anchored.
    env: dict[float, float] = {0.0: 0.0, 1.0: 1.0}
    for p in pts:
        env[p["fpr"]] = max(env.get(p["fpr"], 0.0), p["tpr"])
    xs = sorted(env)
    ys = [env[x] for x in xs]
    auc = float(np.trapz(ys, xs))
    return {"roc": pts, "auc": auc, **best}


def absent_target_eval(
    per_card: list[dict], lineup_size: int, absent_frac: float, seed: int = 0
) -> dict:
    """Open-world rejection ROC. A fixed fraction of trials have the true value removed from the lineup."""
    rng = np.random.default_rng(seed)
    n = len(per_card)
    min_dist = min((r["dist_scores"].size for r in per_card), default=0)
    L = max(2, min(int(lineup_size), min_dist))  # need >=2 candidates for a top1-top2 margin

    idx = rng.permutation(n)
    n_absent = int(round(absent_frac * n))
    absent_set = set(idx[:n_absent].tolist())

    present: list[tuple[float, int]] = []
    absent: list[float] = []
    for i, r in enumerate(per_card):
        ds = r["dist_scores"]
        if i in absent_set:  # true value removed -> lineup is L distractors only
            cand = np.sort(ds[:L])[::-1]
            if cand.size >= 2:
                absent.append(float(cand[0] - cand[1]))
        else:  # true value present -> lineup is true + (L-1) distractors
            d = ds[: L - 1]
            allc = np.concatenate([[r["true_score"]], d])
            s = np.sort(allc)[::-1]
            margin = float(s[0] - s[1]) if s.size >= 2 else float(s[0])
            correct = int(r["true_score"] >= float(allc.max()))
            present.append((margin, correct))

    res = _roc(present, absent)
    res.update({"lineup_size": L, "n_present": len(present), "n_absent": len(absent)})
    return res


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/lineup_scaling")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=40, help="cards to attack")
    ap.add_argument("--Ks", default="100,1000,10000,100000", help="comma-separated lineup sizes")
    ap.add_argument("--absent-frac", type=float, default=0.5, help="fraction of trials with target removed")
    ap.add_argument("--font-size", type=int, default=34)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Ks = sorted({int(x) for x in str(args.Ks).split(",") if x.strip()})

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    pool = generate_name_candidates(max(Ks), _FIRST, _LAST, rng)
    print(f"candidate pool: {len(pool)} unique name-like strings (target {max(Ks)})")

    retriever = ColPaliRetriever(model_name=args.model)
    cards = generate_id_cards(args.n, seed=args.seed, value_font_size=args.font_size, vary=True)

    per_card = score_pool_per_card(retriever, cards, pool, field="name")
    scaling = scaling_curve(per_card, Ks)
    absent = absent_target_eval(per_card, lineup_size=min(Ks), absent_frac=args.absent_frac, seed=args.seed)

    print("\n=== (1) LINEUP-SIZE SCALING (name top-1 recovery) ===")
    for s in scaling:
        print(f"  K={s['K']:>7d}  recovery={s['recovery']:.3f}  lift={s['lift']:.1f}x  (chance {s['chance']:.2e})")
    print("\n=== (2) ABSENT-TARGET / open-world rejection ===")
    print(f"  lineup_size={absent['lineup_size']}  present={absent['n_present']}  absent={absent['n_absent']}")
    print(f"  chosen margin threshold = {absent['threshold']}")
    print(f"  true-accept (present)   = {absent['true_accept']:.3f}")
    print(f"  false-accept (absent)   = {absent['false_accept']:.3f}")
    print(f"  ROC AUC                 = {absent['auc']}")

    payload = {
        "mode": "lineup_scaling",
        "model": args.model,
        "n": len(cards),
        "font_size": args.font_size,
        "seed": args.seed,
        "absent_frac": args.absent_frac,
        "Ks": Ks,
        "pool_size": len(pool),
        "scaling": scaling,
        "absent_target": absent,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "lineup_scaling.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"\nwrote lineup_scaling.json -> {args.out}")


if __name__ == "__main__":
    main()
