"""Adaptive white-box attacker vs the RedactionProjection defense (MASTER_PLAN B1 — the untested claim).

The learned defense ``P`` currently reports privacy ~= 1.00, but ONLY against a NON-ADAPTIVE attacker:
one who runs the vanilla dictionary attack ``MaxSim(name_query, P(patch))`` and is blind to ``P``. A
security venue rejects that threat model outright — the defense is public (index-time transform), so a
real adversary KNOWS ``P`` and adapts to it. This experiment mounts the Kerckhoffs-compliant attack: the
attacker has white-box access to ``P`` and the defended index, and we honestly measure how much field
PII survives under four INDEPENDENT adaptive strategies.

  (B1a) DISTILLATION PROBE  — the likely breaker / headline. Train an MLP + linear classifier mapping the
        DEFENDED, pooled embedding ``pool(P(patch)) -> name`` (240-way closed vocabulary) on the attacker's
        own labelled cards, then read the name straight off HELD-OUT defended cards. If this beats chance
        (1/240), ``P`` only broke MaxSim *matching* — it did not remove the information. Privacy claims that
        rest on a broken matcher are not privacy at all.
  (B1b) INVERSE LEARNING    — the attacker fits an inverse net ``Pinv`` on ``(P(patch), patch)`` pairs it
        generates by running the PUBLIC ``P`` on its OWN cards, then runs the vanilla dictionary attack on
        ``Pinv(P(patch))``. Measures whether ``P`` is (approximately) invertible.
  (B1c) ADAPTIVE QUERY      — ``P`` is known and differentiable, so per candidate the attacker takes a few
        anchored gradient steps on the query to maximise ``maxsim_batch(q*, P(patch))``, then ranks the
        lineup by the achieved score. Measures whether the PII direction ``P`` hides is gradient-reachable.
  (B1d) CORRELATION EXPLOIT — ``P`` is *tuned* to preserve topic/office retrieval, so the retained content
        is itself recoverable through ``P``; we measure how far a co-occurring retained field leaks the
        name indirectly (name-from-topic channel + a direct topic-recovery probe of the mechanism).

Every strategy is compared to two anchors: the NON-ADAPTIVE dictionary baseline on ``P(patch)`` (which
``P`` defeats, privacy ~1.0) and the UNDEFENDED index (recovery ~1.0). The emitted ``verdict`` states,
honestly, whether ``P`` survives the adaptive attacker and — if not — which strategy breaks it and what
the defensible, scoped claim becomes.

Threat-model note on the name spaces: ``P`` is trained on OPEN-SET names (train/test name-disjoint, exactly
as ``learned_defense.py``) so its privacy objective must GENERALISE, not memorise. The MaxSim strategies
(B1b/B1c/baseline) are evaluated on those genuinely held-out names — recovering one is recovering a name
``P`` never saw. The distillation/correlation probes are closed-vocabulary classifiers, so they require a
shared 240-name label space between fit and eval; they therefore use the attacker's own labelled cards
drawn from the FULL pool, split by TEMPLATE (novel renderings), which is the decisive info-presence test.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Card:
    """One encoded document as the attacker sees it: defended-able patches + ground truth."""

    patches: np.ndarray  # (Np, d) image patches (the thing P transforms), float32
    name: str
    name_idx: int  # index into the closed 240-name vocabulary (for the classifier probes)
    topic: str
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------------
# small shared helpers (kept pure + CPU-testable; heavy imports are deferred inside each function)
# --------------------------------------------------------------------------------------------------
def _apply_P(P, patches_np, device="cpu"):
    """Run the (white-box, public) defense P over one card's patches -> torch (Np, d). No grad."""
    import torch

    x = torch.tensor(np.asarray(patches_np, dtype=np.float32)).to(device)
    if P is None:
        return x
    with torch.no_grad():
        return P(x)


def _lineup(true_name, pool, n_distractors, rng):
    """A recovery lineup: [true, *distractors] sampled from the candidate pool."""
    others = [x for x in pool if x != true_name]
    k = int(min(n_distractors, len(others)))
    distr = list(rng.choice(others, k, replace=False)) if k > 0 else []
    return [true_name, *distr]


def _dict_hit(patches_np, true_val, q_fn, pool, n_distractors, rng):
    """Vanilla MaxSim dictionary attack against one (possibly defended/reconstructed) patch set."""
    from patchguard.retrievers.base import maxsim

    cands = _lineup(true_val, pool, n_distractors, rng)
    sc = np.array([maxsim(np.asarray(q_fn(c), dtype=np.float32), np.asarray(patches_np, dtype=np.float32))
                   for c in cands])
    return int(np.argmax(sc) == 0)


def _probe(Xtr, ytr, Xte, yte, n_classes, epochs, seed, device="cpu"):
    """Train an MLP probe and a linear probe on (X, y); return (mlp_top1, linear_top1) on the held-out split."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    Xtr = torch.tensor(np.asarray(Xtr, dtype=np.float32)).to(device)
    ytr = torch.tensor(np.asarray(ytr)).long().to(device)
    Xte = torch.tensor(np.asarray(Xte, dtype=np.float32)).to(device)
    yte = torch.tensor(np.asarray(yte)).long().to(device)
    d = int(Xtr.shape[1])

    def train_eval(net):
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(net(Xtr), ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return float((net(Xte).argmax(1) == yte).float().mean())

    mlp = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                        nn.Linear(256, n_classes)).to(device)
    lin = nn.Linear(d, n_classes).to(device)
    return train_eval(mlp), train_eval(lin)


# --------------------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------------------
def baseline_dictionary(cards, P, q_fn, pool, n_distractors, device, rng):
    """Non-adaptive dictionary attack. P=None -> undefended index (~1.0); P!=None -> what P defeats (~0)."""
    hits = []
    for c in cards:
        pts = _apply_P(P, c.patches, device).cpu().numpy() if P is not None else np.asarray(c.patches, np.float32)
        hits.append(_dict_hit(pts, c.name, q_fn, pool, n_distractors, rng))
    return float(np.mean(hits)) if hits else 0.0


# --------------------------------------------------------------------------------------------------
# (B1a) DISTILLATION PROBE on the defended embedding
# --------------------------------------------------------------------------------------------------
def strat_distillation(train, test, P, n_classes, device="cpu", epochs=300, seed=0):
    """Read the name off pool(P(patch)) with a trained probe. Returns (recovery_top1, metrics)."""
    import torch

    def feat(c):
        pt = _apply_P(P, c.patches, device)  # (Np, d)
        return torch.cat([pt.mean(0), pt.max(0).values]).cpu().numpy()  # mean+max pooled -> (2d,)

    Xtr = [feat(c) for c in train]
    ytr = [c.name_idx for c in train]
    Xte = [feat(c) for c in test]
    yte = [c.name_idx for c in test]
    mlp_acc, lin_acc = _probe(Xtr, ytr, Xte, yte, n_classes, epochs, seed, device)
    rec = float(max(mlp_acc, lin_acc))
    return rec, {"mlp_top1": mlp_acc, "linear_top1": lin_acc, "n_fit": len(train), "n_eval": len(test)}


# --------------------------------------------------------------------------------------------------
# (B1b) INVERSE LEARNING
# --------------------------------------------------------------------------------------------------
def strat_inverse(pair_cards, test, P, q_fn, pool, n_distractors, device="cpu", epochs=300, seed=0, rng=None):
    """Fit Pinv on (P(patch), patch) pairs, then dictionary-attack Pinv(P(patch)). Returns (recovery, metrics)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if rng is None:
        rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    d = int(np.asarray(pair_cards[0].patches).shape[1])

    with torch.no_grad():
        X = torch.cat([_apply_P(P, c.patches, device) for c in pair_cards], dim=0)  # (M, d) defended
        Y = torch.cat([torch.tensor(np.asarray(c.patches, np.float32)).to(device) for c in pair_cards], dim=0)
        Y = F.normalize(Y, dim=-1)

    Pinv = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)).to(device)
    opt = torch.optim.Adam(Pinv.parameters(), lr=1e-3)
    for _ in range(epochs):
        pred = F.normalize(Pinv(X), dim=-1)
        loss = (1.0 - (pred * Y).sum(-1)).mean()  # 1 - cosine reconstruction
        opt.zero_grad()
        loss.backward()
        opt.step()
    Pinv.eval()

    hits = []
    for c in test:
        with torch.no_grad():
            recon = F.normalize(Pinv(_apply_P(P, c.patches, device)), dim=-1).cpu().numpy()
        hits.append(_dict_hit(recon, c.name, q_fn, pool, n_distractors, rng))
    with torch.no_grad():
        final_cos = float((F.normalize(Pinv(X), dim=-1) * Y).sum(-1).mean())
    return (float(np.mean(hits)) if hits else 0.0), {"n_pairs": int(X.shape[0]), "recon_cosine": final_cos}


# --------------------------------------------------------------------------------------------------
# (B1c) ADAPTIVE QUERY — gradient through the known, differentiable P
# --------------------------------------------------------------------------------------------------
def strat_adaptive_query(test, P, q_fn, pool, n_distractors, device="cpu", epochs=60, lr=0.05,
                         anchor=0.5, seed=0, rng=None):
    """Per-candidate anchored gradient ascent on maxsim(q*, P(patch)); rank lineup by achieved score."""
    import torch
    import torch.nn.functional as F

    from patchguard.defense.redact import maxsim_batch

    if rng is None:
        rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    hits = []
    for c in test:
        docs = _apply_P(P, c.patches, device).detach()[None]  # (1, Np, d) — index is FIXED, only query moves
        cands = _lineup(c.name, pool, n_distractors, rng)
        scores = []
        for cand in cands:
            q0 = torch.tensor(np.asarray(q_fn(cand), np.float32)).to(device)
            q = q0.clone().requires_grad_(True)
            opt = torch.optim.Adam([q], lr=lr)
            for _ in range(epochs):
                s = maxsim_batch(F.normalize(q, dim=-1), docs)[0]
                loss = -s + anchor * ((q - q0) ** 2).sum()  # anchor: only exploit directions truly present
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                scores.append(float(maxsim_batch(F.normalize(q, dim=-1), docs)[0]))
        hits.append(int(np.argmax(scores) == 0))
    return (float(np.mean(hits)) if hits else 0.0), {"epochs": epochs, "lr": lr, "anchor": anchor}


# --------------------------------------------------------------------------------------------------
# (B1d) CORRELATION EXPLOIT
# --------------------------------------------------------------------------------------------------
def strat_correlation(train, test, P, q_fn, topic_pool, name_pool, n_distractors, n_classes,
                      device="cpu", epochs=300, seed=0, rng=None):
    """Measure indirect name leakage via the retained (topic) channel. Returns (name_via_topic, metrics)."""
    if rng is None:
        rng = np.random.default_rng(seed)

    # (i) mechanism check — does the retained topic itself survive P (dictionary attack over topics)?
    topic_hits = []
    for c in test:
        pts = _apply_P(P, c.patches, device).cpu().numpy()
        topic_hits.append(_dict_hit(pts, c.topic, q_fn, topic_pool, n_distractors, rng))
    topic_rec = float(np.mean(topic_hits)) if topic_hits else 0.0

    # (ii) can the name be inferred from the topic embedding? (chance iff name ⟂ topic in the corpus)
    def tfeat(c):
        return np.asarray(q_fn(c.topic), np.float32).mean(0)  # pooled topic-query embedding

    Xtr = [tfeat(c) for c in train]
    ytr = [c.name_idx for c in train]
    Xte = [tfeat(c) for c in test]
    yte = [c.name_idx for c in test]
    mlp_acc, lin_acc = _probe(Xtr, ytr, Xte, yte, n_classes, epochs, seed, device)
    name_via_topic = float(max(mlp_acc, lin_acc))
    return name_via_topic, {"topic_recovery": topic_rec, "name_from_topic_mlp": mlp_acc,
                            "name_from_topic_linear": lin_acc}


def _breaks(rec, chance):
    """A strategy 'breaks' P if recovery is both absolutely high and clearly above chance."""
    return bool(rec >= 0.5 and rec >= 3.0 * chance)


def _build_retriever(model):
    if "qwen" in model.lower():
        from patchguard.retrievers.colqwen2 import ColQwen2Retriever

        return ColQwen2Retriever(model_name=model)
    from patchguard.retrievers.colpali import ColPaliRetriever

    return ColPaliRetriever(model_name=model)


def main() -> None:
    ap = argparse.ArgumentParser(description="Adaptive white-box attacker vs the RedactionProjection defense.")
    ap.add_argument("--out", default="results/adaptive_attack")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n-train", type=int, default=64, help="cards for P-training AND attacker labelled fit")
    ap.add_argument("--n-test", type=int, default=40, help="held-out victim/eval cards")
    ap.add_argument("--distractors", type=int, default=200, help="dictionary lineup size for MaxSim strategies")
    ap.add_argument("--lam", type=float, default=5.0, help="privacy weight of the P we attack")
    ap.add_argument("--epochs", type=int, default=300, help="P-training epochs (train_redactor)")
    ap.add_argument("--np-train", type=int, default=128, help="patches/card subsampled for P training")
    ap.add_argument("--probe-epochs", type=int, default=300, help="distillation/correlation probe epochs")
    ap.add_argument("--inv-epochs", type=int, default=400, help="inverse-net Pinv training epochs")
    ap.add_argument("--aq-epochs", type=int, default=60, help="adaptive-query gradient steps per candidate")
    ap.add_argument("--aq-lr", type=float, default=0.05)
    ap.add_argument("--aq-anchor", type=float, default=0.5)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_card
    from patchguard.defense.redact import train_redactor
    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # closed vocabulary K=240
    name_to_idx = {nm: i for i, nm in enumerate(pool)}
    n_classes = len(pool)
    rng.shuffle(pool_shuf := list(pool))
    train_names, test_names = pool_shuf[:180], pool_shuf[180:]  # DISJOINT -> open-set for P + MaxSim strategies

    retriever = _build_retriever(args.model)
    qcache: dict[str, np.ndarray] = {}

    def q(s):
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    for s in pool:  # warm all name queries (candidates + truths)
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

    # card sets (see docstring for why the probes need full-pool labelled cards)
    train = gen(train_names, args.n_train, 1000)       # P-training + attacker inverse pairs
    test = gen(test_names, args.n_test, 5000)          # held-out OPEN-SET victims for MaxSim strategies
    p_train = gen(pool, args.n_train, 20000)           # attacker labelled fit (full closed vocabulary)
    p_test = gen(pool, args.n_test, 40000)             # attacker held-out eval (novel templates)

    topic_pool = sorted({c.topic for c in (train + test + p_train + p_test)})
    for t in topic_pool:
        q(t)

    # ---- train the defense P we are attacking (open-set, exactly like learned_defense.py) ----
    def sub(p, kk):
        idx = rng.choice(p.shape[0], min(kk, p.shape[0]), replace=False)
        return p[idx]

    tr_patches = torch.tensor(np.stack([sub(c.patches, args.np_train) for c in train]))
    tr_topic = [torch.tensor(q(c.topic)) for c in train]
    tr_name = [torch.tensor(q(c.name)) for c in train]
    distr_q = [torch.tensor(q(n)) for n in list(rng.choice(train_names, 16, replace=False))]
    P = train_redactor(tr_patches, tr_topic, tr_name, distr_q, lam=args.lam, dim=128,
                       epochs=args.epochs, device=device, seed=args.seed, distractors=distr_q)
    print(f"trained P (lam={args.lam}) on {len(train)} open-set cards")

    # ---- anchors ----
    undefended = baseline_dictionary(test, None, q, pool, args.distractors, device, rng)
    non_adaptive = baseline_dictionary(test, P, q, pool, args.distractors, device, rng)
    dict_chance = 1.0 / (min(args.distractors, len(pool) - 1) + 1)
    print(f"ANCHOR undefended dict={undefended:.3f}  |  non-adaptive dict on P={non_adaptive:.3f} "
          f"(privacy={1 - non_adaptive:.3f}, chance {dict_chance:.4f})")

    # ---- (B1a) distillation ----
    b1a, m1a = strat_distillation(p_train, p_test, P, n_classes, device, args.probe_epochs, args.seed)
    print(f"B1a DISTILLATION   name top1={b1a:.3f} (chance {1/n_classes:.4f})  {m1a}")
    # ---- (B1b) inverse ----
    b1b, m1b = strat_inverse(train, test, P, q, pool, args.distractors, device, args.inv_epochs, args.seed, rng)
    print(f"B1b INVERSE        recovery={b1b:.3f} (chance {dict_chance:.4f})  {m1b}")
    # ---- (B1c) adaptive query ----
    b1c, m1c = strat_adaptive_query(test, P, q, pool, args.distractors, device, args.aq_epochs,
                                    args.aq_lr, args.aq_anchor, args.seed, rng)
    print(f"B1c ADAPTIVE-QUERY recovery={b1c:.3f} (chance {dict_chance:.4f})  {m1c}")
    # ---- (B1d) correlation ----
    b1d, m1d = strat_correlation(p_train, p_test, P, q, topic_pool, pool, args.distractors, n_classes,
                                 device, args.probe_epochs, args.seed, rng)
    print(f"B1d CORRELATION    name-via-topic={b1d:.3f} (chance {1/n_classes:.4f})  "
          f"topic_recovery={m1d['topic_recovery']:.3f}")

    strategies = {
        "B1a_distillation": {"recovery": b1a, "chance": 1.0 / n_classes, "metrics": m1a,
                             "desc": "trained probe name<-pool(P(patch))"},
        "B1b_inverse": {"recovery": b1b, "chance": dict_chance, "metrics": m1b,
                        "desc": "dictionary attack on Pinv(P(patch))"},
        "B1c_adaptive_query": {"recovery": b1c, "chance": dict_chance, "metrics": m1c,
                               "desc": "anchored gradient query vs known P"},
        "B1d_correlation": {"recovery": b1d, "chance": 1.0 / n_classes, "metrics": m1d,
                            "desc": "name inferred via retained topic channel"},
    }
    for k, v in strategies.items():
        v["lift_over_chance"] = float(v["recovery"] / v["chance"]) if v["chance"] > 0 else float("inf")
        v["breaks_defense"] = _breaks(v["recovery"], v["chance"])

    breakers = [k for k, v in strategies.items() if v["breaks_defense"]]
    survives = len(breakers) == 0
    if survives:
        verdict = (f"P SURVIVES the adaptive attacker: none of distillation/inverse/adaptive-query/"
                   f"correlation exceed chance meaningfully (max recovery "
                   f"{max(v['recovery'] for v in strategies.values()):.3f}). The privacy=~{1-non_adaptive:.2f} "
                   f"claim holds under a white-box adversary.")
    else:
        head = "B1a_distillation" if "B1a_distillation" in breakers else breakers[0]
        verdict = (f"P BROKEN by {breakers}. The residual PII is still decodable from P(patch) "
                   f"(headline breaker: {head}, recovery {strategies[head]['recovery']:.3f} vs chance "
                   f"{strategies[head]['chance']:.4f}). Scoped claim: P defeats only the NON-ADAPTIVE MaxSim "
                   f"dictionary attack (privacy ~{1-non_adaptive:.2f} in that model); it does NOT remove the "
                   f"information, so privacy MUST be reported attack-relative, not absolute.")

    print("\n=== ADAPTIVE ATTACK SUMMARY ===")
    print(f"{'strategy':22s} {'recovery':>9s} {'chance':>9s} {'lift':>7s}  breaks")
    for k, v in strategies.items():
        print(f"{k:22s} {v['recovery']:9.3f} {v['chance']:9.4f} {v['lift_over_chance']:7.1f}  {v['breaks_defense']}")
    print(f"{'baseline:non_adaptive':22s} {non_adaptive:9.3f} {dict_chance:9.4f}")
    print(f"{'baseline:undefended':22s} {undefended:9.3f} {dict_chance:9.4f}")
    print("VERDICT:", verdict)

    payload = {
        "mode": "adaptive_attack",
        "model": args.model, "lam": args.lam, "n_train": args.n_train, "n_test": args.n_test,
        "distractors": args.distractors, "font_size": args.font_size, "open_set_names": True,
        "n_classes": n_classes,
        "baselines": {"undefended_dict": undefended, "non_adaptive_dict_on_P": non_adaptive,
                      "non_adaptive_privacy": 1.0 - non_adaptive, "dict_chance": dict_chance},
        "strategies": strategies,
        "breakers": breakers, "survives": survives, "verdict": verdict,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "adaptive_attack.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"\nwrote adaptive_attack.json -> {args.out}")


if __name__ == "__main__":
    main()
