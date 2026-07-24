"""Per-field NullspaceRedaction: extend the name-centric defense (§4.14 Cataract) to id_no + dob.

Reviewer Tier-1 objection: the certified NullspaceRedaction result is NAME-centric. A multi-vector VLM
retriever leaks EVERY field holographically (retrieval_attack recovers name AND id_no AND dob at ~1.0),
so a defense that only annihilates the name-discriminative subspace protects one field and leaves the
id/dob channels wide open. This experiment builds a SEPARATE nullspace subspace ``D`` for each PII field
type — from that field's value-query embeddings vs the topic queries (``pii_directions``) — and ALSO a
COMBINED ``D`` that removes all three subspaces at once (orthonormalised union of the three).

On held-out open-set cards we measure, for five index-time defenses
``{none, name, id, dob, combined}`` × three fields ``{name, id, dob}``:
  * per-field linkage recovery — the retrieval_attack dictionary attack (MaxSim vs stored patches over
    the field's realistic candidate space: 240-name pool / random 8-digit ids / random dates); and
  * topic-retrieval utility — held-out issuing-office Recall@1 on the same defended index.

The three things a reviewer wants to see fall out of the matrix:
  (a) a name-ONLY projection does NOT suppress id/dob (the leak is per-field, so single-field defenses
      are insufficient) — recovery[name][id] and [name][dob] stay near the undefended level;
  (b) the COMBINED projection suppresses all three fields at once; and
  (c) the utility COST of protecting more fields — topic Recall@1 falls monotonically as we remove more
      subspace (none >= single-field >= combined), quantifying the price of full multi-PII redaction.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

# field name -> the ground-truth key on a Card (name lives on Card.name; id/dob in Card.meta)
FIELDS = ("name", "id", "dob")
_META_KEY = {"name": "name", "id": "id_no", "dob": "dob"}


def _field_true(card, field: str) -> str:
    """The ground-truth value of ``field`` for one card (name attribute, else meta)."""
    if field == "name":
        return card.name
    return card.meta[_META_KEY[field]]


def build_field_subspaces(field_tokens: dict, topic_tokens, k: int, r_topic: int) -> dict:
    """Per-field orthonormal (d,k) nullspace basis: ``pii_directions(value_queries, topic_queries)``.

    ``field_tokens[f]`` are the concatenated VALUE-query embeddings for field ``f`` (name strings,
    id strings, dob strings); ``topic_tokens`` the topic/office queries spared as content.
    """
    from patchguard.defense.nullspace import pii_directions

    return {f: pii_directions(field_tokens[f], topic_tokens, k=k, r_topic=r_topic) for f in FIELDS}


def combine_subspaces(Ds, tol: float = 1e-6):
    """Orthonormal basis for the UNION of the per-field subspaces (columns of a (d, r<=sum_k) matrix).

    Each ``D`` already has orthonormal columns; their union generally does not, so we re-orthonormalise
    the horizontal stack via SVD and keep the directions with non-negligible singular value. The result
    is a proper projection basis for ``NullspaceRedaction`` that annihilates every field's subspace.
    """
    import torch

    cols = [D for D in Ds if D.shape[1] > 0]
    if not cols:
        d = Ds[0].shape[0] if Ds else 0
        return torch.zeros(d, 0)
    Dcat = torch.cat(cols, dim=1)  # (d, sum_k)
    U, S, _ = torch.linalg.svd(Dcat, full_matrices=False)  # U: (d, sum_k) orthonormal columns
    if S.numel() == 0:
        return U[:, :0].contiguous()
    keep = int((S > tol * float(S[0])).sum())
    return U[:, :keep].contiguous()


def field_recovery(cards, P, q_fn, field, pool, n_distractors, device, rng) -> float:
    """Dictionary-attack recovery for one field over ``cards`` under defense ``P`` (None = undefended)."""
    from experiments.adaptive_attack import _apply_P, _dict_hit

    hits = []
    for c in cards:
        pts = (_apply_P(P, c.patches, device).cpu().numpy() if P is not None
               else np.asarray(c.patches, np.float32))
        hits.append(_dict_hit(pts, _field_true(c, field), q_fn, pool, n_distractors, rng))
    return float(np.mean(hits)) if hits else 0.0


def topic_recall(cards, P, q_fn, device) -> float:
    """Held-out topic (issuing-office) retrieval Recall@1 on the defended index — the utility metric."""
    from experiments.adaptive_attack import _apply_P
    from patchguard.retrievers.base import maxsim

    Pt = [(_apply_P(P, c.patches, device).cpu().numpy() if P is not None
           else np.asarray(c.patches, np.float32)) for c in cards]
    hits = []
    for i, c in enumerate(cards):
        sc = [maxsim(np.asarray(q_fn(c.topic), np.float32), Pt[j]) for j in range(len(cards))]
        hits.append(int(np.argmax(sc) == i))
    return float(np.mean(hits)) if hits else 0.0


def evaluate_defenses(defenses, cards, q_fn, field_pools, nd_by_field, device, rng):
    """The {defense}×{field} recovery matrix + per-defense topic utility on ``cards``.

    ``defenses``: {name: P|None}. Returns ``(matrix, utility)`` where ``matrix[d][f]`` is field ``f``
    recovery under defense ``d`` and ``utility[d]`` is topic Recall@1 under defense ``d``.
    """
    matrix, utility = {}, {}
    for dname, P in defenses.items():
        matrix[dname] = {
            f: field_recovery(cards, P, q_fn, f, field_pools[f], nd_by_field[f], device, rng)
            for f in FIELDS
        }
        utility[dname] = topic_recall(cards, P, q_fn, device)
    return matrix, utility


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-field vs combined NullspaceRedaction over name/id_no/dob (multi-PII defense).")
    ap.add_argument("--out", default="results/defense_pii")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n-train", type=int, default=64, help="cards for building the per-field subspaces")
    ap.add_argument("--n-test", type=int, default=40, help="held-out open-set victim/eval cards")
    ap.add_argument("--k", type=int, default=96, help="removed rank per field (Cataract operating point)")
    ap.add_argument("--r-topic", type=int, default=8, help="topic directions spared before extracting PII dirs")
    ap.add_argument("--distractors", type=int, default=200, help="dictionary lineup size per field")
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from experiments.adaptive_attack import Card, _build_retriever
    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_card
    from patchguard.defense.nullspace import NullspaceRedaction
    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    # ---- candidate spaces (retrieval_attack): closed 240-name pool + random id/dob lineups ----
    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # K=240
    name_to_idx = {nm: i for i, nm in enumerate(name_pool)}
    pool_shuf = list(name_pool)
    rng.shuffle(pool_shuf)
    train_names, test_names = pool_shuf[:180], pool_shuf[180:]  # DISJOINT -> open-set names

    def rand_id() -> str:
        return f"{int(rng.integers(10_000_000, 99_999_999))}"

    def rand_dob() -> str:
        return (f"{int(rng.integers(1, 13)):02d}/{int(rng.integers(1, 29)):02d}/"
                f"{int(rng.integers(1950, 2005))}")

    pool_sz = max(args.distractors * 2, args.distractors + 1)
    id_pool = list({rand_id() for _ in range(pool_sz * 2)})[:pool_sz]
    dob_pool = list({rand_dob() for _ in range(pool_sz * 2)})[:pool_sz]

    retriever = _build_retriever(args.model)
    qcache: dict[str, np.ndarray] = {}

    def q(s):
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    for s in name_pool + id_pool + dob_pool:  # warm the candidate lineups
        q(s)

    def gen(names, k, seed0):
        cards = []
        for i in range(k):
            nm = names[int(rng.integers(0, len(names)))]
            im, fs = generate_id_card(seed0 + i, value_font_size=args.font_size, vary=True,
                                      fixed_name=nm, with_topic=True)
            enc = retriever.encode_page(im)
            truth = {f.field_type: f.text for f in fs}
            topic = truth.get("office", "OFFICE")
            cards.append(Card(patches=enc.image_patches().astype(np.float32), name=nm,
                              name_idx=name_to_idx[nm], topic=topic,
                              meta={"id_no": truth["id_no"], "dob": truth["dob"]}))
            q(topic); q(truth["id_no"]); q(truth["dob"])  # warm value + topic queries
        return cards

    train = gen(train_names, args.n_train, 1000)   # subspace estimation
    test = gen(test_names, args.n_test, 5000)       # held-out open-set victims
    topic_pool = sorted({c.topic for c in (train + test)})
    for t in topic_pool:
        q(t)

    # ---- per-field VALUE-query tokens from TRAIN (public encoder; attacker-agnostic) ----
    field_tokens = {
        "name": np.concatenate([q(c.name) for c in train], axis=0),
        "id": np.concatenate([q(c.meta["id_no"]) for c in train], axis=0),
        "dob": np.concatenate([q(c.meta["dob"]) for c in train], axis=0),
    }
    topic_tokens = np.concatenate([q(t) for t in topic_pool], axis=0)

    Ds = build_field_subspaces(field_tokens, topic_tokens, k=args.k, r_topic=args.r_topic)
    D_combined = combine_subspaces([Ds[f] for f in FIELDS])
    subspace_rank = {f: int(Ds[f].shape[1]) for f in FIELDS}
    subspace_rank["combined"] = int(D_combined.shape[1])

    defenses = {
        "none": None,
        "name": NullspaceRedaction(Ds["name"].to(device)).to(device).eval(),
        "id": NullspaceRedaction(Ds["id"].to(device)).to(device).eval(),
        "dob": NullspaceRedaction(Ds["dob"].to(device)).to(device).eval(),
        "combined": NullspaceRedaction(D_combined.to(device)).to(device).eval(),
    }

    field_pools = {"name": name_pool, "id": id_pool, "dob": dob_pool}
    nd_by_field = {
        "name": int(min(args.distractors, len(name_pool) - 1)),
        "id": int(min(args.distractors, len(id_pool) - 1)),
        "dob": int(min(args.distractors, len(dob_pool) - 1)),
    }
    chance = {f: 1.0 / (nd_by_field[f] + 1) for f in FIELDS}

    matrix, utility = evaluate_defenses(defenses, test, q, field_pools, nd_by_field, device, rng)

    # ---- reviewer readings (a)/(b)/(c) ----
    def near_chance(rec, f):
        return bool(rec <= 3.0 * chance[f])

    # (a) name-only does NOT suppress id/dob: those channels stay well above chance under the name defense
    name_leaves_id = bool(matrix["name"]["id"] >= 0.5 * matrix["none"]["id"] and not near_chance(matrix["name"]["id"], "id"))
    name_leaves_dob = bool(matrix["name"]["dob"] >= 0.5 * matrix["none"]["dob"] and not near_chance(matrix["name"]["dob"], "dob"))
    # (b) combined suppresses ALL three fields toward chance
    combined_suppresses_all = bool(all(near_chance(matrix["combined"][f], f) for f in FIELDS))
    # each single-field defense at least suppresses its OWN field
    own_field_suppressed = {f: bool(near_chance(matrix[f][f], f)) for f in FIELDS}
    # (c) utility cost: more subspace removed -> lower topic Recall@1
    util_cost_combined = float(utility["none"] - utility["combined"])
    single_utils = [utility[f] for f in FIELDS]
    utility_monotone = bool(utility["none"] + 1e-9 >= max(single_utils)
                            and min(single_utils) + 1e-9 >= utility["combined"] - 1e-9)

    analysis = {
        "a_name_only_leaves_id": name_leaves_id,
        "a_name_only_leaves_dob": name_leaves_dob,
        "b_combined_suppresses_all": combined_suppresses_all,
        "own_field_suppressed": own_field_suppressed,
        "c_utility_cost_of_combined": util_cost_combined,
        "c_utility_monotone_none_ge_single_ge_combined": utility_monotone,
    }

    if combined_suppresses_all and (name_leaves_id or name_leaves_dob):
        verdict = (
            f"MULTI-PII: a name-only nullspace leaves id/dob recoverable "
            f"(id {matrix['name']['id']:.3f}, dob {matrix['name']['dob']:.3f} vs chance "
            f"{chance['id']:.3f}/{chance['dob']:.3f}); the COMBINED subspace (rank {subspace_rank['combined']}) "
            f"suppresses all three (name {matrix['combined']['name']:.3f}, id {matrix['combined']['id']:.3f}, "
            f"dob {matrix['combined']['dob']:.3f}) at a topic-utility cost of {util_cost_combined:.3f} "
            f"(Recall@1 {utility['none']:.3f} -> {utility['combined']:.3f}). Per-field leakage demands a "
            f"per-field-union defense.")
    elif combined_suppresses_all:
        verdict = (
            f"COMBINED subspace suppresses all three fields (name {matrix['combined']['name']:.3f}, "
            f"id {matrix['combined']['id']:.3f}, dob {matrix['combined']['dob']:.3f}) at utility cost "
            f"{util_cost_combined:.3f}; single-field defenses did not clearly leave the others exposed in "
            f"this run — inspect the matrix for cross-field overlap.")
    else:
        verdict = (
            "COMBINED subspace did NOT drive all fields to chance in this run — the per-field PII subspaces "
            "may need a larger k, or id/dob share directions with the content subspace here; report the "
            "matrix and the frontier.")

    payload = {
        "mode": "defense_pii",
        "model": args.model, "n_train": args.n_train, "n_test": args.n_test, "k": args.k,
        "r_topic": args.r_topic, "distractors": args.distractors, "font_size": args.font_size,
        "seed": args.seed, "fields": list(FIELDS),
        "chance": chance, "subspace_rank": subspace_rank,
        "matrix": matrix, "utility": utility, "analysis": analysis, "verdict": verdict,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "defense_pii.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)

    # ---- summary ----
    print(f"=== defense_pii  k={args.k} r_topic={args.r_topic} distractors={args.distractors} ===")
    print(f"subspace ranks: name={subspace_rank['name']} id={subspace_rank['id']} "
          f"dob={subspace_rank['dob']} combined={subspace_rank['combined']}")
    print(f"chance: name={chance['name']:.4f} id={chance['id']:.4f} dob={chance['dob']:.4f}")
    hdr = f"{'defense':10s} {'rec:name':>9s} {'rec:id':>9s} {'rec:dob':>9s} {'util':>7s}"
    print(hdr)
    for d in ("none", "name", "id", "dob", "combined"):
        print(f"{d:10s} {matrix[d]['name']:9.3f} {matrix[d]['id']:9.3f} {matrix[d]['dob']:9.3f} "
              f"{utility[d]:7.3f}")
    print("VERDICT:", verdict)
    print(f"\nwrote defense_pii.json -> {args.out}")


if __name__ == "__main__":
    main()
