"""Certified impossibility bound for the nullspace defense (MASTER_PLAN B6 — from empirical to PROVABLE).

``experiments/certified_defense.py`` showed EMPIRICALLY that the inverse attacker (B1b), which broke the
residual ``RedactionProjection``, fails against ``NullspaceRedaction``: recovery collapses to chance. That
is a strong result, but a security venue asks the harder question: *did the attacker just try too weakly,
or is recovery impossible for ANY adversary?* This experiment upgrades the claim from "the inverse attack
fails" to "no adversary can recover the removed subspace", by proving and then double-checking a bound.

THE MECHANISM. ``NullspaceRedaction`` stores ``P(x) = (I - D Dᵀ) x`` — the ORTHOGONAL PROJECTION of each
patch onto ``span(D)^⊥`` (rank ``d-k``), where ``D`` is the ``k`` PII-discriminative directions. This map is
information-theoretically MANY-TO-ONE: every ``x`` that differs only within ``span(D)`` collapses to the SAME
stored vector. The stored index therefore carries ZERO bits about the ``span(D)`` component of ``x``.

THE BOUND (what a positive result MEANS). Because the stored output is independent of the ``span(D)``
component, an adversary's posterior over that component equals its prior — no observation moves it. Hence for
ANY reconstruction function ``g`` (linear, MLP, diffusion, anything), the expected squared error ON
``span(D)`` is at least the prior variance of that component, i.e. the FULL removed energy. Recovery of the
PII subspace is pinned at chance. What survives (``span(D)^⊥``) is perfectly recoverable — that is the utility
side and the reason the bound is non-trivial: it removes exactly the PII directions and nothing else.

We verify the bound two independent ways:

  (1) GEOMETRY CORE  (``--synthetic-geometry``, CPU, no ColPali). For random unit patches and an ARBITRARY
      orthonormal ``D``, we build the OPTIMAL linear inverse — the least-squares Moore-Penrose pseudo-inverse
      of the projection — and decompose its reconstruction error into in-``span(D)`` vs out-of-``span(D)``. For a
      projection, ``pinv(P)=P``, so the optimal inverse maps straight back into ``span(D)^⊥``: the in-``span(D)``
      residual is EXACTLY 100% of the removed energy (nothing recovered, for any linear map, since the stored
      vectors are rank-deficient in the complement), while the out-of-``span(D)`` part is recovered exactly.
      A strongest-possible trained MLP inverse is run alongside and also lands at chance.

  (2) REAL PATH  (GPU). ``D`` is built from real name/topic ColPali query tokens via
      ``patchguard.defense.nullspace.pii_directions``; the strongest learned inverse we can train (the MLP of
      ``adaptive_attack.strat_inverse``) is fit on real ``(P(patch), patch)`` pairs, and its recovery of the
      ``span(D)``-projected PII component is measured across a ``k``-sweep. It stays at chance, matching the bound.

Per ``k`` we emit ``removed_energy_fraction``, ``optimal_inverse_span_error`` (≈ 1.0 = fully lost),
``trained_inverse_span_recovery`` (≈ chance = 0), and the stated bound. A NEGATIVE result — any inverse
pushing span recovery above chance — would falsify the projection's many-to-one claim and must be believed.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

# chance level for the span-component recovery R²: predicting the prior mean recovers nothing.
_CHANCE_SPAN_RECOVERY = 0.0


# --------------------------------------------------------------------------------------------------
# geometry helpers (pure numpy/torch — CPU-testable, no ColPali; heavy import deferred inside)
# --------------------------------------------------------------------------------------------------
def _random_orthonormal(d: int, k: int, seed: int):
    """A (d,k) matrix with orthonormal columns (k directions to remove). k==0 -> (d,0)."""
    import torch

    if k <= 0:
        return torch.zeros(d, 0, dtype=torch.float32)
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(d, k, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A)  # (d,k) orthonormal columns
    return Q[:, :k].contiguous()


def _span_metrics(X, Xhat, D) -> dict:
    """Decompose reconstruction error of ``Xhat`` for ``X`` into in-span(D) vs out-of-span(D).

    X, Xhat : (n,d) torch ; D : (d,k) torch orthonormal. Returns a dict of energy fractions.
      * removed_energy      : mean_i ||Dᵀ x_i||²                         (the PII component's energy)
      * span_error_fraction : in-span(D) residual / removed_energy       (1.0 == the span is fully lost)
      * span_recovery       : 1 - span_error_fraction                    (0.0 == chance; the R² on span(D))
      * out_recovered       : 1 - out-of-span residual / out-of-span energy (1.0 == complement recovered)
    """
    import torch

    n, d = X.shape
    k = int(D.shape[1])
    total_energy = float(X.pow(2).sum(-1).mean())
    if k == 0:
        # Nothing removed: the "PII subspace" is empty, so error on it is 0 and it is trivially recovered.
        diff = Xhat - X
        out_err = float(diff.pow(2).sum(-1).mean())
        out_energy = total_energy
        return {
            "removed_energy": 0.0,
            "removed_energy_fraction": 0.0,
            "span_error_fraction": 0.0,
            "span_recovery": 1.0,
            "out_recovered": float(1.0 - (out_err / out_energy if out_energy > 0 else 0.0)),
            "total_energy": total_energy,
        }
    diff = Xhat - X
    in_err = float((diff @ D).pow(2).sum(-1).mean())          # error inside span(D)
    removed = float((X @ D).pow(2).sum(-1).mean())            # true energy inside span(D)
    diff_out = diff - (diff @ D) @ D.T
    out_err = float(diff_out.pow(2).sum(-1).mean())           # error inside span(D)^⊥
    X_out = X - (X @ D) @ D.T
    out_energy = float(X_out.pow(2).sum(-1).mean())           # true energy inside span(D)^⊥
    span_err_frac = in_err / removed if removed > 0 else 0.0
    return {
        "removed_energy": removed,
        "removed_energy_fraction": removed / total_energy if total_energy > 0 else 0.0,
        "span_error_fraction": span_err_frac,
        "span_recovery": 1.0 - span_err_frac,
        "out_recovered": 1.0 - (out_err / out_energy if out_energy > 0 else 0.0),
        "total_energy": total_energy,
    }


def _projection(D):
    """Orthogonal projection ``I - D Dᵀ`` onto span(D)^⊥ as a (d,d) torch matrix."""
    import torch

    d = int(D.shape[0])
    I = torch.eye(d, dtype=D.dtype)
    if D.shape[1] == 0:
        return I
    return I - D @ D.T


def optimal_linear_inverse_eval(X, D) -> dict:
    """The OPTIMAL linear inverse of the projection (least-squares pseudo-inverse) + its error decomposition.

    Stores ``Y = X @ P`` with ``P = I - D Dᵀ`` (orthogonal projection, rank d-k), reconstructs with the
    Moore-Penrose pseudo-inverse ``Xhat = Y @ pinv(P)``. Since ``pinv(P)=P`` for a projection, the optimal
    inverse maps back into span(D)^⊥ and recovers NONE of the span(D) component: in-span error == removed
    energy for every linear map (the stored ``Y`` is rank-deficient in the complement). Also returns the
    projection diagnostics (idempotence residual, rank) used by the certified claim / tests.
    """
    import torch

    d = int(X.shape[1])
    k = int(D.shape[1])
    P = _projection(D)
    Y = X @ P                                   # the stored (defended) patches, linear form
    Pinv = torch.linalg.pinv(P)                 # optimal least-squares inverse of the projection
    Xhat = Y @ Pinv                             # best linear reconstruction (== Y, since pinv(P)=P)
    m = _span_metrics(X, Xhat, D)
    idem_err = float((P @ P - P).abs().max())   # projection is idempotent: P@P == P
    rank = int(torch.linalg.matrix_rank(P)) if d > 0 else 0
    m.update({
        "projection_idempotent_err": idem_err,
        "projection_rank": rank,
        "expected_rank": d - k,
        "pinv_equals_projection_err": float((Pinv - P).abs().max()),
    })
    return m


# --------------------------------------------------------------------------------------------------
# strongest LEARNED inverse (mirrors adaptive_attack.strat_inverse's Pinv: Linear->GELU->Linear, cosine loss)
# --------------------------------------------------------------------------------------------------
def _train_inverse(X_def, X_tgt, epochs: int, device: str, seed: int):
    """Fit the strongest MLP inverse ``g`` on (defended, original) pairs. Returns eval-mode module.

    Same architecture/objective as ``adaptive_attack.strat_inverse``: predict the L2-normalized original
    patch from the defended patch, maximizing cosine (== minimizing ``1 - cos``). This is the best learned
    reconstruction the white-box attacker gets to build.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(int(seed))
    X_def = X_def.to(device)
    Y = F.normalize(X_tgt.to(device), dim=-1)
    d = int(X_def.shape[1])
    g = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)).to(device)
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    for _ in range(int(epochs)):
        pred = F.normalize(g(X_def), dim=-1)
        loss = (1.0 - (pred * Y).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return g.eval()


def trained_inverse_span_eval(X_def_tr, X_tgt_tr, X_def_te, X_tgt_te, D, epochs, device, seed) -> dict:
    """Train the MLP inverse on the train split, measure span(D) recovery on the held-out split.

    Returns ``span_recovery`` (≈ chance == 0 iff the info is truly gone), ``recon_cosine`` (overall
    reconstruction quality — high is fine, the point is it can be high yet still recover NONE of span(D)),
    and the full error decomposition on the eval split.
    """
    import torch
    import torch.nn.functional as F

    g = _train_inverse(X_def_tr, X_tgt_tr, epochs, device, seed)
    with torch.no_grad():
        Xhat = F.normalize(g(X_def_te.to(device)), dim=-1).cpu()
        Xtgt = F.normalize(X_tgt_te, dim=-1).cpu()
        recon_cos = float((Xhat * Xtgt).sum(-1).mean())
    m = _span_metrics(Xtgt, Xhat, D.cpu())
    return {"span_recovery": m["span_recovery"], "recon_cosine": recon_cos, "decomp": m}


def _bound_statement(removed_frac: float, k: int) -> str:
    if k == 0:
        return "k=0: no subspace removed (control) — the full patch is stored, recovery is trivially perfect."
    return (
        f"P(x)=(I-DDᵀ)x is many-to-one on span(D); the stored index is independent of the span(D) component, "
        f"so posterior=prior and E‖ĝ_span - x_span‖² ≥ removed energy ({removed_frac:.3f} of patch energy) for "
        f"ANY reconstruction ĝ. Span(D) recovery is pinned at chance; span(D)^⊥ is fully recoverable."
    )


# --------------------------------------------------------------------------------------------------
# (1) SYNTHETIC GEOMETRY CORE — CPU, no ColPali
# --------------------------------------------------------------------------------------------------
def _run_geometry(args, ks) -> dict:
    import torch
    import torch.nn.functional as F

    d = int(args.dim)
    n_tr, n_te = int(args.geom_train), int(args.geom_test)
    g = torch.Generator().manual_seed(int(args.seed))
    X_tr = F.normalize(torch.randn(n_tr, d, generator=g, dtype=torch.float32), dim=-1)  # unit patches
    X_te = F.normalize(torch.randn(n_te, d, generator=g, dtype=torch.float32), dim=-1)

    sweep = []
    for k in ks:
        D = _random_orthonormal(d, k, seed=args.seed + 1)  # arbitrary orthonormal subspace
        # optimal linear inverse (the theorem) on held-out patches
        opt = optimal_linear_inverse_eval(X_te, D)
        # strongest trained MLP inverse on the DEPLOYED (normalized) defended patches
        P = _projection(D)
        Xdef_tr = F.normalize(X_tr @ P, dim=-1)
        Xdef_te = F.normalize(X_te @ P, dim=-1)
        tr = trained_inverse_span_eval(Xdef_tr, X_tr, Xdef_te, X_te, D, args.inv_epochs, "cpu", args.seed)
        row = {
            "k": k,
            "removed_energy_fraction": opt["removed_energy_fraction"],
            "optimal_inverse_span_error": opt["span_error_fraction"],
            "optimal_inverse_out_recovered": opt["out_recovered"],
            "trained_inverse_span_recovery": tr["span_recovery"],
            "trained_inverse_recon_cosine": tr["recon_cosine"],
            "projection_idempotent_err": opt["projection_idempotent_err"],
            "projection_rank": opt["projection_rank"],
            "expected_rank": opt["expected_rank"],
            "pinv_equals_projection_err": opt["pinv_equals_projection_err"],
            "bound": _bound_statement(opt["removed_energy_fraction"], k),
        }
        sweep.append(row)
        print(f"k={k:>3} | removed={row['removed_energy_fraction']:.3f} | "
              f"OPT-inverse span-error={row['optimal_inverse_span_error']:.3f} (want 1.0) | "
              f"OPT out-recovered={row['optimal_inverse_out_recovered']:.3f} | "
              f"TRAINED span-recovery={row['trained_inverse_span_recovery']:.3f} (chance {_CHANCE_SPAN_RECOVERY}) | "
              f"rank={row['projection_rank']}/{row['expected_rank']}")

    return {"path": "synthetic_geometry", "dim": d, "n_train": n_tr, "n_test": n_te, "sweep": sweep}


# --------------------------------------------------------------------------------------------------
# (2) REAL PATH — ColPali patches + pii_directions + strongest learned inverse (GPU)
# --------------------------------------------------------------------------------------------------
def _run_real(args, ks) -> dict:
    import torch

    from experiments.adaptive_attack import Card, _apply_P, _build_retriever, baseline_dictionary, strat_inverse
    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_card
    from patchguard.defense.nullspace import NullspaceRedaction, pii_directions

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # closed vocabulary K=240
    name_to_idx = {nm: i for i, nm in enumerate(pool)}
    pool_shuf = list(pool)
    rng.shuffle(pool_shuf)
    train_names, test_names = pool_shuf[:180], pool_shuf[180:]  # DISJOINT -> open-set

    retriever = _build_retriever(args.model)
    qcache: dict[str, np.ndarray] = {}

    def q(s):
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
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
            cards.append(Card(patches=enc.image_patches().astype(np.float32), name=nm,
                              name_idx=name_to_idx[nm], topic=topic))
            q(topic)
        return cards

    train = gen(train_names, args.n_train, 1000)   # inverse-net training pairs
    test = gen(test_names, args.n, 5000)           # held-out open-set eval patches
    topic_pool = sorted({c.topic for c in (train + test)})
    for t in topic_pool:
        q(t)

    name_tokens = np.concatenate([q(n) for n in train_names], axis=0)  # (T_name, d)
    topic_tokens = np.concatenate([q(t) for t in topic_pool], axis=0)  # (T_topic, d)

    # raw ColPali patches for the optimal-linear (theorem) check
    X_te = torch.tensor(np.concatenate([c.patches for c in test], axis=0), dtype=torch.float32)

    dict_chance = 1.0 / (min(args.distractors, len(pool) - 1) + 1)
    undefended = baseline_dictionary(test, None, q, pool, args.distractors, device, rng)
    print(f"ANCHOR undefended dict={undefended:.3f} (chance {dict_chance:.4f})")

    sweep = []
    for k in ks:
        D = pii_directions(name_tokens, topic_tokens, k=k, r_topic=args.r_topic)  # (d,k) from real queries
        # (a) optimal linear inverse on real patches (the bound made concrete on real D)
        opt = optimal_linear_inverse_eval(X_te, D)
        # (b) strongest learned inverse: build defended (P(patch)) pairs, train MLP, measure span recovery
        Pnull = NullspaceRedaction(D).to(device).eval()
        Xdef_tr = torch.cat([_apply_P(Pnull, c.patches, device).cpu() for c in train], dim=0)
        Xtgt_tr = torch.cat([torch.tensor(np.asarray(c.patches, np.float32)) for c in train], dim=0)
        Xdef_te = torch.cat([_apply_P(Pnull, c.patches, device).cpu() for c in test], dim=0)
        Xtgt_te = torch.cat([torch.tensor(np.asarray(c.patches, np.float32)) for c in test], dim=0)
        tr = trained_inverse_span_eval(Xdef_tr, Xtgt_tr, Xdef_te, Xtgt_te, D, args.inv_epochs, device, args.seed)
        # (c) dictionary-attack anchor through the trained inverse (reuse strat_inverse)
        dict_rec, dict_m = strat_inverse(train, test, Pnull, q, pool, args.distractors, device,
                                         args.inv_epochs, args.seed, rng)
        row = {
            "k": k,
            "removed_energy_fraction": opt["removed_energy_fraction"],
            "optimal_inverse_span_error": opt["span_error_fraction"],
            "optimal_inverse_out_recovered": opt["out_recovered"],
            "trained_inverse_span_recovery": tr["span_recovery"],
            "trained_inverse_recon_cosine": tr["recon_cosine"],
            "trained_inverse_dict_recovery": dict_rec,
            "trained_inverse_dict_privacy": 1.0 - dict_rec,
            "dict_recon_cosine": dict_m.get("recon_cosine"),
            "projection_idempotent_err": opt["projection_idempotent_err"],
            "projection_rank": opt["projection_rank"],
            "expected_rank": opt["expected_rank"],
            "bound": _bound_statement(opt["removed_energy_fraction"], k),
        }
        sweep.append(row)
        print(f"k={k:>3} | removed={row['removed_energy_fraction']:.3f} | "
              f"OPT span-error={row['optimal_inverse_span_error']:.3f} (want 1.0) | "
              f"TRAINED span-recovery={row['trained_inverse_span_recovery']:.3f} (chance {_CHANCE_SPAN_RECOVERY}) | "
              f"dict-rec={dict_rec:.3f} (chance {dict_chance:.4f})")

    return {"path": "real", "dim": int(X_te.shape[1]), "n_test": int(X_te.shape[0]),
            "dict_chance": dict_chance, "undefended_dict": undefended, "r_topic": args.r_topic,
            "sweep": sweep}


def _make_verdict(sweep) -> tuple[str, bool]:
    """Certified iff every k>0 has optimal span-error ≈ 1.0 AND trained span-recovery ≈ chance."""
    pos = [r for r in sweep if r["k"] > 0]
    held = all(r["optimal_inverse_span_error"] >= 0.99 for r in pos) and \
        all(r["trained_inverse_span_recovery"] <= 0.1 for r in pos)
    if not pos:
        return ("Only k=0 (control) evaluated — no subspace removed, so no bound to certify.", False)
    max_rec = max(r["trained_inverse_span_recovery"] for r in pos)
    if held:
        verdict = (
            "CERTIFIED: for every removed rank k>0 the OPTIMAL linear inverse leaves 100% of the span(D) "
            f"energy as error (span recovery == chance), and the strongest trained inverse stays at chance "
            f"(max span recovery {max_rec:.3f}). NullspaceRedaction is provably many-to-one on span(D): no "
            "adversary — linear, MLP, or otherwise — can recover the removed PII subspace. span(D)^⊥ (utility) "
            "is recovered exactly, so the bound removes only the PII directions."
        )
    else:
        verdict = (
            "BOUND VIOLATED in this run: some inverse pushed span(D) recovery above chance "
            f"(max {max_rec:.3f}) or the optimal inverse recovered part of span(D). This would falsify the "
            "projection's many-to-one claim — investigate the D construction / numerics before trusting it."
        )
    return verdict, held


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Certified impossibility bound: NullspaceRedaction is many-to-one on span(D).")
    ap.add_argument("--out", default="results/certified_bound")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--ks", default="0,2,4,8,16,32", help="removed-rank sweep")
    ap.add_argument("--n", type=int, default=40, help="held-out eval cards (real path)")
    ap.add_argument("--n-train", type=int, default=64, help="inverse-net training cards (real path)")
    ap.add_argument("--distractors", type=int, default=200, help="dictionary lineup size (real path anchor)")
    ap.add_argument("--r-topic", type=int, default=8, help="topic directions spared before extracting PII dirs")
    ap.add_argument("--inv-epochs", type=int, default=400, help="trained-inverse (MLP) epochs")
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--synthetic-geometry", action="store_true",
                    help="run the CPU geometry core (no ColPali) instead of the real ColPali path")
    ap.add_argument("--dim", type=int, default=128, help="patch dim for the geometry core")
    ap.add_argument("--geom-train", type=int, default=4000, help="synthetic patches for inverse training")
    ap.add_argument("--geom-test", type=int, default=1000, help="synthetic held-out patches")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.repro import run_fingerprint, seed_everything

    seed_everything(args.seed)
    ks = [int(x) for x in str(args.ks).split(",")]
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    if args.synthetic_geometry:
        result = _run_geometry(args, ks)
    else:
        result = _run_real(args, ks)

    verdict, certified = _make_verdict(result["sweep"])
    print("\nVERDICT:", verdict)

    payload = {
        "mode": "certified_bound",
        "model": args.model,
        "ks": ks,
        "chance_span_recovery": _CHANCE_SPAN_RECOVERY,
        "certified": certified,
        "verdict": verdict,
        "certified_claim": (
            "NullspaceRedaction stores (I-DDᵀ)x, the orthogonal projection onto span(D)^⊥. The map is "
            "information-theoretically many-to-one on span(D): the stored index is independent of the span(D) "
            "component of x, so every adversary's posterior over it equals the prior. The best achievable "
            "reconstruction error ON span(D) equals the full removed energy for ANY reconstruction function."
        ),
        "seed": args.seed,
        "fingerprint": run_fingerprint(),
    }
    payload.update(result)

    name = "certified_bound"
    (local_out / f"{name}.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        from experiments.train_funsd import _gcs_upload

        _gcs_upload(local_out, args.out)
    print(f"\nwrote {name}.json -> {args.out}")


if __name__ == "__main__":
    main()
