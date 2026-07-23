"""How much redactor machinery is load-bearing? — the defense ABLATION (MASTER_PLAN S8, plan item B7).

The learned defense (RedactionProjection) is a residual GELU-MLP with a learnable gate. This sweep asks
which of those ingredients the privacy/utility frontier actually needs, at a FIXED privacy weight lambda:

  architecture depth : {linear (depth 0), mlp-depth1, mlp-depth2, mlp-depth3}
  gate               : {learned scalar (init 0.1), fixed full-strength}
  hidden width       : a couple of sizes (ignored for the linear cell)

For each cell we train the redactor with the identical min-max objective and report the achieved
(privacy = 1 - name-attack top-1, utility = topic Recall@1) plus its parameter count. Interpretation:

  * If the LINEAR cell matches the best MLP cell, the PII and content directions are ~linearly separable
    in the embedding — the defense collapses to one interpretable matrix W (the paper's cleanest claim).
  * If privacy only appears with depth / gating, that quantifies exactly the capacity the defense requires
    and rules out the trivial linear explanation.

Names are OPEN-SET (disjoint train/test pools). An identity (no-defense) row anchors the top of the
frontier. Runs on ColPali by default; ``--retriever colqwen2`` reuses the same grid on the other backbone.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def _build_cells(variants: list[str], gates: list[str], hiddens: list[int]) -> list[dict]:
    """Cartesian grid of (arch x gate x hidden); the linear cell has no hidden width, so it's deduped."""
    cells = []
    for arch in variants:
        a = arch.strip().lower()
        hs = [None] if a in ("linear", "mlp-depth0", "depth0") else hiddens
        for gate in gates:
            for h in hs:
                cells.append({"arch": a, "gate": gate.strip(), "hidden": h})
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/defense_ablation")
    ap.add_argument("--retriever", default="colpali", choices=["colpali", "colqwen2"])
    ap.add_argument("--model", default=None, help="override backbone checkpoint (else backend default)")
    ap.add_argument("--variants", default="linear,mlp-depth1,mlp-depth2,mlp-depth3")
    ap.add_argument("--gates", default="on,off")
    ap.add_argument("--hiddens", default="128,256")
    ap.add_argument("--lam", type=float, default=5.0, help="fixed privacy weight for the whole sweep")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--np-train", type=int, default=128)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_card
    from patchguard.defense.redact import maxsim_batch
    from patchguard.defense.redact_variants import build_redactor, train_variant
    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    if args.retriever == "colqwen2":
        from patchguard.retrievers.colqwen2 import ColQwen2Retriever
        model_name = args.model or "vidore/colqwen2-v1.0"
        retriever = ColQwen2Retriever(model_name=model_name)
    else:
        from patchguard.retrievers.colpali import ColPaliRetriever
        model_name = args.model or "vidore/colpali-v1.3"
        retriever = ColPaliRetriever(model_name=model_name)

    pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # K=240
    rng.shuffle(pool)
    train_names, test_names = pool[:180], pool[180:]  # DISJOINT -> open-set privacy test

    qcache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in qcache:
            qcache[s] = np.asarray(retriever.encode_query(s), dtype=np.float32)
        return qcache[s]

    for s in pool:
        q(s)

    def gen(names, k, seed0):
        cards = []
        for i in range(k):
            nm = names[int(rng.integers(0, len(names)))]
            im, fs = generate_id_card(seed0 + i, value_font_size=args.font_size, vary=True,
                                      fixed_name=nm, with_topic=True)
            enc = retriever.encode_page(im)
            topic = next((f.text for f in fs if f.field_type == "office"), "OFFICE")
            cards.append({"patches": np.asarray(enc.patches, dtype=np.float32), "name": nm, "topic": topic})
            q(topic)
        return cards

    train = gen(train_names, args.n_train, 1000)
    test = gen(test_names, args.n_test, 5000)
    dim = int(train[0]["patches"].shape[-1])
    npt = min(args.np_train, min(int(c["patches"].shape[0]) for c in train))

    def sub(p, k):
        idx = rng.choice(p.shape[0], min(k, p.shape[0]), replace=False)
        return p[idx]

    tr_patches = torch.tensor(np.stack([sub(c["patches"], npt) for c in train]))
    tr_topic = [torch.tensor(q(c["topic"])) for c in train]
    tr_name = [torch.tensor(q(c["name"])) for c in train]
    distr_q = [torch.tensor(q(n)) for n in list(rng.choice(train_names, 16, replace=False))]

    def apply_P(P, patches_np):
        with torch.no_grad():
            t = torch.tensor(patches_np).to(device)
            return P(t) if P is not None else t

    def frontier_point(P=None):
        Pt = [apply_P(P, c["patches"]) for c in test]
        hit_u = []
        for i, c in enumerate(test):
            tq = torch.tensor(q(c["topic"])).to(device)
            scores = torch.stack([maxsim_batch(tq, d[None])[0] for d in Pt])
            hit_u.append(int(torch.argmax(scores).item() == i))
        rec = []
        for i, c in enumerate(test):
            cands = [c["name"]] + list(rng.choice([x for x in pool if x != c["name"]],
                                                  min(args.distractors, len(pool) - 1), replace=False))
            sc = torch.stack([maxsim_batch(torch.tensor(q(cc)).to(device), Pt[i][None])[0] for cc in cands])
            rec.append(int(torch.argmax(sc).item() == 0))
        return float(np.mean(hit_u)), 1.0 - float(np.mean(rec))

    variants = [x for x in args.variants.split(",") if x.strip()]
    gates = [x for x in args.gates.split(",") if x.strip()]
    hiddens = [int(x) for x in args.hiddens.split(",") if x.strip()]
    cells = _build_cells(variants, gates, hiddens)

    print(f"{args.retriever} ({model_name}): {len(train)} train / {len(test)} test docs, dim={dim}, "
          f"lam={args.lam}, {len(cells)} ablation cells")

    # identity (no-defense) anchor
    u0, p0 = frontier_point(P=None)
    print(f"  identity            : utility={u0:.3f}  privacy={p0:.3f}")

    grid = []
    for idx, cell in enumerate(cells):
        torch.manual_seed(args.seed + idx)  # reproducible per-cell weight init
        hidden = cell["hidden"] if cell["hidden"] is not None else 256
        P = build_redactor(cell["arch"], dim=dim, hidden=hidden, gate=cell["gate"])
        n_params = int(sum(p.numel() for p in P.parameters() if p.requires_grad))
        P = train_variant(P, tr_patches, tr_topic, tr_name, distr_q, lam=args.lam,
                          epochs=args.epochs, device=device)
        u, p = frontier_point(P=P)
        row = {"arch": cell["arch"], "gate": cell["gate"], "hidden": cell["hidden"],
               "trainable_params": n_params, "utility": u, "privacy": p}
        grid.append(row)
        tag = f"{cell['arch']}/gate={cell['gate']}/h={cell['hidden']}"
        print(f"  {tag:<32}: utility={u:.3f}  privacy={p:.3f}  (params {n_params})")

    def sort_key(r):
        return (r["privacy"] + r["utility"], r["privacy"])

    lin = [r for r in grid if r["arch"] in ("linear", "mlp-depth0", "depth0")]
    mlp = [r for r in grid if r not in lin]
    best_linear = max(lin, key=sort_key) if lin else None
    best_mlp = max(mlp, key=sort_key) if mlp else None
    # "linear suffices" if the best linear cell is within 0.05 of the best MLP cell on BOTH axes.
    linear_suffices = bool(
        best_linear and best_mlp
        and best_linear["privacy"] >= best_mlp["privacy"] - 0.05
        and best_linear["utility"] >= best_mlp["utility"] - 0.05
    )
    print(f"BEST LINEAR: {best_linear}")
    print(f"BEST MLP   : {best_mlp}")
    print("LINEAR SUFFICES (within 0.05 of best MLP on both axes):", linear_suffices)

    payload = {"mode": "defense_ablation", "retriever": args.retriever, "model": model_name,
               "lam": args.lam, "n_train": args.n_train, "n_test": args.n_test, "epochs": args.epochs,
               "open_set_names": True, "dim": dim, "distractors": args.distractors,
               "identity": {"utility": u0, "privacy": p0}, "grid": grid,
               "best_linear": best_linear, "best_mlp": best_mlp, "linear_suffices": linear_suffices,
               "fingerprint": run_fingerprint()}
    (local_out / "defense_ablation.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote defense_ablation.json -> {args.out}")


if __name__ == "__main__":
    main()
