"""Ghost Vectors — soft-delete does NOT erase ColPali page embeddings (MASTER_PLAN A6, legal hook).

Production vector databases (Qdrant, Milvus, Weaviate, pgvector-on-segments, ...) implement
``delete`` as a TOMBSTONE, not a byte-erase: the point is flagged invisible to queries, but its raw
vectors remain in the on-disk segment until an *optional, asynchronous* compaction/optimizer pass
rewrites the segment. Between the delete call and that compaction — which may never be triggered, or
may be indefinitely deferred under load — the "deleted" vectors are physically present and readable by
anyone with storage / segment-file / backup access. For a multi-vector page index this is acute: an
entire document's per-patch embedding grid persists, and the retrieval/dictionary attack (which the
discriminative probe proved recovers PII from ColPali patches) still runs against it verbatim.

What a POSITIVE result means here: a document the query interface reports as *deleted* (absent from the
logical view, recovery = 0 through the API) is nonetheless fully recovered from the RAW segment view
(recovery ~ 1.0). That is a right-to-erasure violation — GDPR Article 17 / India DPDP right to erasure
require the personal data to be *rendered irrecoverable*, and a tombstone does not do that. A NEGATIVE
result (raw-view recovery collapses to chance) would mean the backend genuinely zeroed the bytes on
delete — i.e. erasure was honoured — which production soft-delete backends demonstrably do not.

This is deliberately a *systems* claim, not a new ML claim: the attack is the already-established MaxSim
name-dictionary attack; the novelty is grounding the threat model in deployed vector-DB delete
semantics via an in-process ``SoftDeleteIndex`` that faithfully models tombstone-without-erase, with an
optional in-memory Qdrant cross-check of the logical delete semantics.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

# CPU-safe: adaptive_attack's module top imports only argparse/json/tempfile/dataclass/pathlib/numpy.
from experiments.adaptive_attack import _dict_hit


# --------------------------------------------------------------------------------------------------
# SoftDeleteIndex — faithful model of production tombstone-without-erase delete semantics.
# Dependency-free (numpy only); this is the PRIMARY, always-runnable path.
# --------------------------------------------------------------------------------------------------
class SoftDeleteIndex:
    """An in-process multi-vector index with production soft-delete semantics.

    * ``insert(doc_id, patches)`` stores one page's per-patch multi-vector grid in an append-only
      segment (insertion order preserved, mirroring how a segment is laid out on disk).
    * ``delete(doc_id)`` sets a per-doc TOMBSTONE flag and DELIBERATELY leaves the vector bytes in
      place — no zeroing, no removal. This is what Qdrant/Milvus/etc. do until a later, optional
      optimizer/compaction pass rewrites the segment.
    * The LOGICAL view (``logical_view`` / ``logical_get`` — what a normal query sees) hides
      tombstoned docs. The RAW SEGMENT read (``raw_segment_read`` / ``raw_get`` — what an attacker
      with storage / segment-file / backup access sees) enumerates EVERY vector, tombstones ignored.

    The whole point of the class is the gap between those two views for a deleted document.
    """

    def __init__(self) -> None:
        # doc_id -> {"patches": (Np, d) float32, "tombstone": bool, "meta": dict}
        self._seg: dict[int, dict] = {}
        self._order: list[int] = []  # append-only segment layout (insertion order)

    def insert(self, doc_id: int, patches: np.ndarray, meta: dict | None = None) -> None:
        if doc_id in self._seg:
            raise KeyError(f"doc_id {doc_id} already present")
        arr = np.asarray(patches, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"patches must be 2-D (n_patches, d); got {arr.shape}")
        self._seg[doc_id] = {"patches": arr, "tombstone": False, "meta": dict(meta or {})}
        self._order.append(doc_id)

    def delete(self, doc_id: int) -> None:
        """Tombstone the doc. Does NOT touch the vector bytes (that is the entire finding)."""
        if doc_id not in self._seg:
            raise KeyError(f"doc_id {doc_id} not present")
        self._seg[doc_id]["tombstone"] = True

    # ---- introspection ----
    def is_tombstoned(self, doc_id: int) -> bool:
        return bool(self._seg[doc_id]["tombstone"])

    def logical_ids(self) -> list[int]:
        return [d for d in self._order if not self._seg[d]["tombstone"]]

    def raw_ids(self) -> list[int]:
        return list(self._order)

    def n_raw(self) -> int:
        return len(self._order)

    def __len__(self) -> int:  # logical size — what `count` reports through the query API
        return len(self.logical_ids())

    # ---- views ----
    def logical_get(self, doc_id: int) -> np.ndarray | None:
        """Query-interface read: None if the doc has been (soft-)deleted."""
        rec = self._seg.get(doc_id)
        if rec is None or rec["tombstone"]:
            return None
        return rec["patches"]

    def raw_get(self, doc_id: int) -> np.ndarray | None:
        """Storage-level read: the physical bytes, tombstone or not."""
        rec = self._seg.get(doc_id)
        return None if rec is None else rec["patches"]

    def logical_view(self) -> Iterator[tuple[int, np.ndarray]]:
        """(doc_id, patches) for every NON-tombstoned doc — the normal retrieval surface."""
        for d in self._order:
            rec = self._seg[d]
            if not rec["tombstone"]:
                yield d, rec["patches"]

    def raw_segment_read(self) -> Iterator[tuple[int, np.ndarray, bool]]:
        """(doc_id, patches, tombstone) for EVERY vector in the segment — the attacker's view."""
        for d in self._order:
            rec = self._seg[d]
            yield d, rec["patches"], bool(rec["tombstone"])


# --------------------------------------------------------------------------------------------------
# attack driver — the established MaxSim name-dictionary attack, run over an arbitrary view
# --------------------------------------------------------------------------------------------------
def recover_over_view(view: Iterable[tuple[int, np.ndarray]], truths: dict[int, str], q_fn,
                      name_pool: list[str], distractors: int, rng) -> dict[int, int]:
    """Run the dictionary attack over (doc_id, patches) pairs -> {doc_id: 1 if name recovered else 0}."""
    hits: dict[int, int] = {}
    for doc_id, patches in view:
        hits[doc_id] = _dict_hit(patches, truths[doc_id], q_fn, name_pool, distractors, rng)
    return hits


# --------------------------------------------------------------------------------------------------
# optional real-DB cross-check — in-memory Qdrant multi-vector collection
# --------------------------------------------------------------------------------------------------
def _qdrant_crosscheck(doc_patches: dict[int, np.ndarray], deleted_ids: list[int]) -> dict:
    """Replicate insert->delete->read in an in-memory Qdrant multi-vector collection.

    Cross-checks that a REAL vector DB's client API exhibits the same LOGICAL delete semantics the
    ``SoftDeleteIndex`` models (deleted points vanish from queries/retrieve). Skipped — never fails the
    run — if qdrant-client is not importable. Note: the physical segment persistence (the Ghost
    Vectors) is not observable through the client API without segment-file access; that is exactly what
    the in-process index demonstrates.
    """
    try:
        from qdrant_client import QdrantClient, models
    except Exception as e:  # noqa: BLE001 — optional dependency
        return {"skipped": True, "reason": f"qdrant-client unavailable ({type(e).__name__})"}
    try:
        dim = int(next(iter(doc_patches.values())).shape[1])
        client = QdrantClient(location=":memory:")
        coll = "ghost_pages"
        client.create_collection(
            collection_name=coll,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM),
            ),
        )
        points = [models.PointStruct(id=int(i), vector=np.asarray(p, np.float32).tolist())
                  for i, p in doc_patches.items()]
        client.upsert(collection_name=coll, points=points)
        count_before = int(client.count(coll, exact=True).count)
        client.delete(collection_name=coll,
                      points_selector=models.PointIdsList(points=[int(i) for i in deleted_ids]))
        count_after = int(client.count(coll, exact=True).count)
        retrieved = client.retrieve(collection_name=coll, ids=[int(i) for i in deleted_ids])
        return {
            "skipped": False,
            "count_before": count_before,
            "count_after": count_after,
            "n_deleted": len(deleted_ids),
            "deleted_retrievable_via_api": len(retrieved),
            "logical_delete_matches": bool(count_after == count_before - len(deleted_ids)
                                           and len(retrieved) == 0),
            "note": ("Qdrant's client API models the LOGICAL view: deleted points vanish from "
                     "count/retrieve. The physical persistence of tombstoned multi-vectors in the "
                     "on-disk segment until optimizer compaction — the Ghost Vectors — is what the "
                     "in-process SoftDeleteIndex demonstrates and is not observable through the "
                     "client API without segment-file access."),
        }
    except Exception as e:  # noqa: BLE001 — cross-check must never break the primary result
        return {"skipped": True, "reason": f"qdrant cross-check error: {type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ghost Vectors: soft-delete tombstones leave ColPali page embeddings recoverable.")
    ap.add_argument("--out", default="results/ghost_vectors")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=40, help="pages to insert into the index")
    ap.add_argument("--distractors", type=int, default=200, help="dictionary lineup size for the name attack")
    ap.add_argument("--font-size", type=int, default=34)
    ap.add_argument("--delete-frac", type=float, default=0.5, help="fraction of docs to (soft-)delete")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.adaptive_attack import _build_retriever

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    retriever = _build_retriever(args.model)
    qcache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # closed vocabulary K=240
    for s in name_pool:  # warm cache
        q(s)

    # ---- encode N pages and INSERT them into the index ----
    cards = generate_id_cards(args.n, seed=42, value_font_size=args.font_size, vary=True)
    index = SoftDeleteIndex()
    truths: dict[int, str] = {}
    originals: dict[int, np.ndarray] = {}
    for i, (im, fs) in enumerate(cards):
        enc = retriever.encode_page(im)
        patches = np.asarray(enc.patches, dtype=np.float32)
        name = next(f.text for f in fs if f.field_type == "name")
        index.insert(i, patches, meta={"name": name})
        truths[i] = name
        originals[i] = patches.copy()  # keep a pristine copy to prove bytes are untouched by delete
    n = index.n_raw()

    # ---- DELETE (tombstone) a fraction of the docs ----
    n_del = max(0, min(n, int(round(args.delete_frac * n))))
    perm = rng.permutation(n)
    deleted_ids = sorted(int(x) for x in perm[:n_del])
    kept_ids = sorted(int(x) for x in perm[n_del:])
    for d in deleted_ids:
        index.delete(d)
    n_kept = len(kept_ids)

    chance = 1.0 / (min(args.distractors, len(name_pool) - 1) + 1)

    # ---- attack the LOGICAL view (query interface: deleted docs are absent) ----
    logical_hits = recover_over_view(index.logical_view(), truths, q, name_pool, args.distractors, rng)
    kept_recovery_logical = float(np.mean([logical_hits[i] for i in kept_ids])) if kept_ids else 0.0
    deleted_visible_logical = sum(1 for d in deleted_ids if d in logical_hits)  # expected 0

    # ---- attack the RAW SEGMENT view (attacker w/ storage access: tombstoned vectors recovered) ----
    raw_hits = recover_over_view(
        ((doc_id, patches) for doc_id, patches, _tomb in index.raw_segment_read()),
        truths, q, name_pool, args.distractors, rng)
    deleted_recovery_raw = float(np.mean([raw_hits[i] for i in deleted_ids])) if deleted_ids else 0.0
    kept_recovery_raw = float(np.mean([raw_hits[i] for i in kept_ids])) if kept_ids else 0.0

    # deletion did not touch the bytes: raw view is byte-identical to what was inserted
    bytes_preserved = bool(all(np.array_equal(index.raw_get(d), originals[d]) for d in deleted_ids))

    # ---- optional Qdrant cross-check of the logical delete semantics ----
    qd = _qdrant_crosscheck(originals, deleted_ids)

    # ---- verdict ----
    ghost_recovered = (deleted_visible_logical == 0 and bytes_preserved
                       and deleted_recovery_raw >= max(0.8, 3.0 * chance))
    erased = deleted_recovery_raw <= max(2.0 * chance, 0.1)
    if ghost_recovered:
        verdict = (
            f"GHOST VECTORS CONFIRMED: {n_del} tombstoned documents are ABSENT from the logical query "
            f"interface (recovery 0.000) yet FULLY recovered from the raw segment (name recovery "
            f"{deleted_recovery_raw:.3f} vs chance {chance:.4f}), with vector bytes byte-identical to "
            f"insertion. Soft-delete satisfies the API contract but does NOT render the personal data "
            f"irrecoverable — a right-to-erasure violation (GDPR Art.17 / DPDP)."
        )
    elif erased:
        verdict = (
            f"NO ghost leak: raw-segment recovery on deleted docs collapsed to ~chance "
            f"({deleted_recovery_raw:.3f} vs {chance:.4f}) — the backend appears to have zeroed the "
            f"bytes on delete. (Production soft-delete backends do not do this.)"
        )
    else:
        verdict = (
            f"PARTIAL: raw-segment recovery on deleted docs = {deleted_recovery_raw:.3f} "
            f"(chance {chance:.4f}); logical deleted-doc visibility = {deleted_visible_logical}; "
            f"bytes_preserved = {bytes_preserved}. Deletion is not a clean erasure but the leak is not maximal."
        )

    # ---- human summary ----
    print(f"inserted N={n} pages | tombstoned {n_del} ({args.delete_frac:.0%}) | kept {n_kept}")
    print(f"LOGICAL view: {len(kept_ids)} docs visible, {deleted_visible_logical} deleted docs visible "
          f"| name recovery on kept = {kept_recovery_logical:.3f}")
    print(f"RAW segment : {index.n_raw()} docs visible (incl. {n_del} tombstoned) | bytes_preserved="
          f"{bytes_preserved} | name recovery on DELETED = {deleted_recovery_raw:.3f} (chance {chance:.4f})")
    if qd.get("skipped"):
        print(f"Qdrant cross-check: SKIPPED ({qd.get('reason')})")
    else:
        print(f"Qdrant cross-check: count {qd['count_before']}->{qd['count_after']}, deleted retrievable "
              f"via API = {qd['deleted_retrievable_via_api']} (logical_delete_matches={qd['logical_delete_matches']})")
    print("VERDICT:", verdict)

    payload = {
        "mode": "ghost_vectors",
        "model": args.model,
        "n": n, "n_deleted": n_del, "n_kept": n_kept, "delete_frac": args.delete_frac,
        "font_size": args.font_size, "distractors": args.distractors,
        "name_pool_size": len(name_pool), "chance": chance,
        "logical_view": {
            "n_visible": len(kept_ids),
            "deleted_docs_visible": deleted_visible_logical,
            "kept_recovery": kept_recovery_logical,
            "deleted_recovery": 0.0,  # deleted docs are unreachable through the query interface
        },
        "raw_view": {
            "n_visible": index.n_raw(),
            "deleted_docs_visible": len(deleted_ids),
            "bytes_preserved": bytes_preserved,
            "deleted_recovery": deleted_recovery_raw,
            "kept_recovery": kept_recovery_raw,
        },
        "deleted_recovery_logical_vs_raw": [0.0, deleted_recovery_raw],
        "ghost_vectors_confirmed": bool(ghost_recovered),
        "verdict": verdict,
        "qdrant_crosscheck": qd,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "ghost_vectors.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        from experiments.train_funsd import _gcs_upload

        _gcs_upload(local_out, args.out)
    print(f"wrote ghost_vectors.json -> {args.out}")


if __name__ == "__main__":
    main()
