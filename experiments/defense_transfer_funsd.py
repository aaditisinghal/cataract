"""Defense transfer: does a defense TRAINED on synthetic cards protect REAL FUNSD docs? (plan B8).

Every defense in this paper is *fit* on synthetic id-cards — synthetic name/topic queries, synthetic
glyph renderings. A deployment, though, indexes REAL documents the defense never saw. The open question a
security reviewer will ask is therefore: is the redaction a synthetic-template artifact, or does it remove
a model-intrinsic PII direction that survives the domain shift to real forms?

This experiment answers it end-to-end. We (1) train a defense purely on synthetic cards — either the
learned residual ``RedactionProjection`` (via ``train_redactor``) or the information-destroying
``NullspaceRedaction`` (via ``pii_directions`` estimated from synthetic name/topic query tokens) — then
(2) FREEZE it and apply it at index time to REAL FUNSD page embeddings, and (3) re-run the FUNSD
retrieval-discrimination attack (``funsd_transfer``-style: recover a real field's text over a lineup of
real distractor fields by MaxSim) on the vanilla vs the defended index. Alongside privacy we measure a
utility proxy on the SAME real pages: legitimate page-retrieval Recall@1 of held-out real content fields
(question/header boilerplate), so a defense that simply destroys the embedding is not rewarded.

What a result MEANS:
  * privacy transfers  => defended attack top-1 collapses toward chance on REAL docs — the removed
                          direction was model-intrinsic PII, not a synthetic-template crutch.
  * utility preserved  => real content queries still fetch their own page => the index is still usable.
  * NO transfer        => the synthetic-trained transform does nothing on real forms (attack top-1
                          unchanged) — the defense is a synthetic artifact and the paper must say so.
Transfer may be PARTIAL (privacy drops but utility drops too, or privacy drops only modestly); the
emitted verdict reports whichever of these is true, honestly, for each defense.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------------------
# small pure helpers (CPU-testable; heavy torch/patchguard imports are DEFERRED inside each function
# so that `python3 -m experiments.defense_transfer_funsd --help` works without torch/colpali present)
# --------------------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    """OCR-ish normalization: alnum-only, casefolded (matches funsd_transfer's soft match)."""
    return "".join(ch for ch in s if ch.isalnum()).casefold()


def _q_cache(retriever):
    """A memoized query encoder: text -> (nq, d) float32, cached by string."""
    cache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in cache:
            cache[s] = np.asarray(retriever.encode_query(s), dtype=np.float32)
        return cache[s]

    return q


def build_synth_cards(retriever, names, k, seed0, font_size, rng, q):
    """Encode ``k`` synthetic id-cards (fixed name + non-PII issuing-office topic) — the defense's
    ONLY training data. Returns dicts of {image patches, true name, topic}."""
    from patchguard.data.synthdoc import generate_id_card

    cards = []
    for i in range(k):
        nm = names[int(rng.integers(0, len(names)))]
        im, fs = generate_id_card(seed0 + i, value_font_size=font_size, vary=True,
                                  fixed_name=nm, with_topic=True)
        enc = retriever.encode_page(im)
        topic = next((f.text for f in fs if f.field_type == "office"), "OFFICE")
        cards.append({"patches": np.asarray(enc.image_patches(), dtype=np.float32),
                      "name": nm, "topic": topic})
        q(topic)
    return cards


def build_defense(defense, synth_cards, q, *, lam, null_k, r_topic, np_train, epochs, dim,
                  device, seed, rng, train_names):
    """Fit the chosen defense on the SYNTHETIC cards only. Returns a frozen (eval) torch module
    whose forward is the index-time transform ((...,d) -> normalized (...,d))."""
    import torch

    if defense == "nullspace":
        from patchguard.defense.nullspace import NullspaceRedaction, pii_directions

        # PII subspace estimated from synthetic name-value query tokens, sparing topic directions.
        name_tokens = np.concatenate([q(c["name"]) for c in synth_cards], axis=0)
        topic_tokens = np.concatenate([q(c["topic"]) for c in synth_cards], axis=0)
        D = pii_directions(name_tokens, topic_tokens, k=null_k, r_topic=r_topic)
        return NullspaceRedaction(D).eval().to(device)

    if defense == "redaction":
        from patchguard.defense.redact import train_redactor

        def sub(p, kk):
            idx = rng.choice(p.shape[0], min(kk, p.shape[0]), replace=False)
            return p[idx]

        npt = min(np_train, min(int(c["patches"].shape[0]) for c in synth_cards))
        tr_patches = torch.tensor(np.stack([sub(c["patches"], npt) for c in synth_cards]))
        tr_topic = [torch.tensor(q(c["topic"])) for c in synth_cards]
        tr_name = [torch.tensor(q(c["name"])) for c in synth_cards]
        ndist = min(16, len(train_names))
        distr_q = [torch.tensor(q(n)) for n in list(rng.choice(train_names, ndist, replace=False))]
        return train_redactor(tr_patches, tr_topic, tr_name, distr_q, lam=lam, dim=dim,
                              epochs=epochs, device=device, seed=seed, distractors=distr_q)

    raise ValueError(f"unknown defense {defense!r}")


def apply_defense(P, patches_np, device="cpu"):
    """Run the (synthetic-trained, frozen) defense over one real page's stored patches -> np (Np, d).
    ``P is None`` is the vanilla index (patches already unit-norm from the encoder)."""
    if P is None:
        return np.asarray(patches_np, dtype=np.float32)
    import torch

    with torch.no_grad():
        t = torch.tensor(np.asarray(patches_np, dtype=np.float32)).to(device)
        return P(t).cpu().numpy()


def make_plans(encoded, all_texts, pii_labels, content_labels, *, k, max_fields, max_util,
               min_len, rng):
    """Fix the probe plan ONCE so vanilla and every defense are attacked on identical fields+lineups.

    privacy plan: (page_idx, true_text, [true, *k-1 distractors]) over PII-ish (answer) fields.
    utility plan: (page_idx, field_text) over held-out content (question/header) fields.
    """
    def elig(text):
        return len(_norm(text)) >= min_len

    priv_plan: list[tuple[int, str, list[str]]] = []
    util_plan: list[tuple[int, str]] = []
    for i, (_pat, fields) in enumerate(encoded):
        pii = [f for f in fields if f.field_type in pii_labels and elig(f.text)]
        for j in list(rng.permutation(len(pii)))[:max_fields]:
            f = pii[j]
            pool = [t for t in all_texts if _norm(t) != _norm(f.text)]
            distr = list(rng.choice(pool, min(k - 1, len(pool)), replace=False)) if pool else []
            priv_plan.append((i, f.text, [f.text, *distr]))
        content = [f for f in fields if f.field_type in content_labels and elig(f.text)]
        for j in list(rng.permutation(len(content)))[:max_util]:
            util_plan.append((i, content[j].text))
    return priv_plan, util_plan


def eval_defense(encoded, priv_plan, util_plan, P, q, device="cpu"):
    """Apply ``P`` to the whole real index, then measure (attack_top1, utility_recall@1).

    attack_top1 : fraction of PII fields whose true text is ranked #1 in its lineup (privacy leak).
    utility_r1  : fraction of content-field queries that retrieve their OWN page over the corpus.
    """
    from patchguard.retrievers.base import maxsim

    defended = [apply_defense(P, pat, device) for pat, _ in encoded]

    hits1 = []
    for (i, true_text, cands) in priv_plan:
        scores = np.array([maxsim(q(c), defended[i]) for c in cands])
        ranked = [cands[j] for j in np.argsort(-scores)]
        hits1.append(int(_norm(ranked[0]) == _norm(true_text)))
    attack_top1 = float(np.mean(hits1)) if hits1 else 0.0

    uhits = []
    for (i, ftext) in util_plan:
        scores = np.array([maxsim(q(ftext), d) for d in defended])
        uhits.append(int(int(np.argmax(scores)) == i))
    utility_r1 = float(np.mean(uhits)) if uhits else 0.0
    return attack_top1, utility_r1, len(hits1), len(uhits)


def verdict_for(defense, van_p, van_u, def_p, def_u, chance_p):
    """Honest per-defense verdict from the vanilla vs defended (privacy, utility) pair."""
    suppression = van_p - def_p                                  # absolute drop in attack success
    retention = (def_u / van_u) if van_u > 0 else float("nan")   # fraction of legit retrieval kept
    strong_priv = def_p <= max(1.5 * chance_p, 0.15)
    meaningful_priv = van_p > 0 and def_p <= 0.7 * van_p
    good_util = (van_u <= 0) or (retention >= 0.7)

    if van_p <= 2.0 * chance_p:
        tag = "VACUOUS (vanilla attack already near chance — nothing to suppress on these pages)"
    elif strong_priv and good_util:
        tag = "STRONG TRANSFER (attack collapses to ~chance, utility preserved)"
    elif meaningful_priv and good_util:
        tag = "PARTIAL TRANSFER (attack meaningfully suppressed, utility preserved)"
    elif meaningful_priv:
        tag = "PRIVACY-ONLY TRANSFER (attack suppressed but real-doc utility harmed)"
    else:
        tag = "NO TRANSFER (defended attack ~ vanilla; synthetic-trained transform does nothing on real forms)"
    return {"defense": defense, "attack_top1": def_p, "utility_r1": def_u,
            "privacy_suppression": suppression, "utility_retention": retention, "verdict": tag}


def main() -> None:
    ap = argparse.ArgumentParser(description="Does a synthetic-trained defense protect REAL FUNSD docs?")
    ap.add_argument("--data", required=True, help="FUNSD root (local or gs://)")
    ap.add_argument("--out", default="results/defense_transfer_funsd")
    ap.add_argument("--defense", default="nullspace", choices=["redaction", "nullspace", "both"])
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n-train", type=int, default=64, help="synthetic cards the defense is trained on")
    ap.add_argument("--n-pages", type=int, default=40, help="real FUNSD pages indexed + attacked")
    ap.add_argument("--k", type=int, default=20, help="attack lineup size (true + k-1 real distractors)")
    ap.add_argument("--lam", type=float, default=5.0, help="privacy weight for the redaction defense")
    ap.add_argument("--null-k", type=int, default=32, help="PII directions removed by the nullspace defense")
    ap.add_argument("--r-topic", type=int, default=8, help="topic directions spared by the nullspace defense")
    ap.add_argument("--np-train", type=int, default=128, help="patches/card subsampled for redaction training")
    ap.add_argument("--epochs", type=int, default=300, help="redaction training epochs")
    ap.add_argument("--font-size", type=int, default=24, help="synthetic value font size")
    ap.add_argument("--max-fields", type=int, default=8, help="PII fields probed per page (privacy)")
    ap.add_argument("--max-util", type=int, default=6, help="content fields probed per page (utility)")
    ap.add_argument("--min-len", type=int, default=3, help="skip trivially-short field text")
    ap.add_argument("--pii-labels", default="answer",
                    help="FUNSD labels treated as PII targets for the privacy attack (comma list)")
    ap.add_argument("--content-labels", default="question,header",
                    help="FUNSD labels used as legitimate retrieval queries for the utility proxy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image

    from patchguard.data.funsd import iter_funsd
    from patchguard.data.synthdoc import _FIRST, _LAST
    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.adaptive_attack import _build_retriever
    from experiments.train_funsd import _gcs_download, _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    pii_labels = {t.strip().lower() for t in args.pii_labels.split(",") if t.strip()}
    content_labels = {t.strip().lower() for t in args.content_labels.split(",") if t.strip()}
    defenses = ["redaction", "nullspace"] if args.defense == "both" else [args.defense]

    data_root = args.data
    if str(data_root).startswith("gs://"):
        data_root = str(_gcs_download(data_root, Path(tempfile.mkdtemp())))

    retriever = _build_retriever(args.model)
    q = _q_cache(retriever)

    # ---- (1) train the defense(s) on SYNTHETIC cards ONLY ----
    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # K=240 synthetic names
    synth = build_synth_cards(retriever, name_pool, args.n_train, 1000, args.font_size, rng, q)
    dim = int(synth[0]["patches"].shape[-1])
    print(f"trained-on: {len(synth)} synthetic cards (dim={dim}); defense(s)={defenses}")

    trained = {}
    for d in defenses:
        trained[d] = build_defense(d, synth, q, lam=args.lam, null_k=args.null_k, r_topic=args.r_topic,
                                    np_train=args.np_train, epochs=args.epochs, dim=dim, device=device,
                                    seed=args.seed, rng=rng, train_names=name_pool)
        extra = f"k={trained[d].k}" if d == "nullspace" else f"lam={args.lam}"
        print(f"  built {d} defense ({extra})")

    # ---- (2) index REAL FUNSD pages ----
    encoded = []
    all_texts: list[str] = []
    for ps in iter_funsd(data_root, split="testing_data", granularity="entity"):
        img = np.array(Image.open(ps.image_path).convert("RGB"))
        fields = [f for f in ps.fields if len(_norm(f.text)) >= args.min_len]
        if fields:
            enc = retriever.encode_page(img)
            encoded.append((np.asarray(enc.patches, dtype=np.float32), fields))
            all_texts.extend(f.text for f in fields)
        if len(encoded) >= args.n_pages:
            break
    print(f"real FUNSD index: {len(encoded)} pages | distractor text pool: {len(all_texts)}")

    priv_plan, util_plan = make_plans(encoded, all_texts, pii_labels, content_labels, k=args.k,
                                      max_fields=args.max_fields, max_util=args.max_util,
                                      min_len=args.min_len, rng=rng)
    # fall back to probing ALL labels if the requested split is empty on this corpus
    if not priv_plan:
        all_labels = {f.field_type for _p, fs in encoded for f in fs}
        priv_plan, _ = make_plans(encoded, all_texts, all_labels, content_labels, k=args.k,
                                   max_fields=args.max_fields, max_util=args.max_util,
                                   min_len=args.min_len, rng=rng)
    if not util_plan:
        all_labels = {f.field_type for _p, fs in encoded for f in fs}
        _, util_plan = make_plans(encoded, all_texts, pii_labels, all_labels, k=args.k,
                                  max_fields=args.max_fields, max_util=args.max_util,
                                  min_len=args.min_len, rng=rng)
    for _i, _t, cands in priv_plan:  # warm attack candidate queries
        for c in cands:
            q(c)
    for _i, ftext in util_plan:      # warm utility queries
        q(ftext)

    # ---- (3) attack vanilla vs each defended index ----
    chance_p = 1.0 / args.k
    chance_u = 1.0 / max(1, len(encoded))
    van_p, van_u, n_priv, n_util = eval_defense(encoded, priv_plan, util_plan, None, q, device)
    print(f"\nVANILLA (no defense): attack_top1={van_p:.3f} (chance {chance_p:.3f}, n={n_priv})  "
          f"utility_r1={van_u:.3f} (chance {chance_u:.3f}, n={n_util})")

    per_defense = {}
    for d in defenses:
        dp, du, _, _ = eval_defense(encoded, priv_plan, util_plan, trained[d], q, device)
        v = verdict_for(d, van_p, van_u, dp, du, chance_p)
        per_defense[d] = v
        print(f"DEFENDED [{d:9s}]: attack_top1={dp:.3f}  utility_r1={du:.3f}  "
              f"suppression={v['privacy_suppression']:+.3f}  utility_retention={v['utility_retention']:.3f}")
        print(f"    -> {v['verdict']}")

    primary = args.defense if args.defense != "both" else defenses[0]
    overall = "; ".join(f"{d}: {per_defense[d]['verdict']}" for d in defenses)

    payload = {
        "mode": "defense_transfer_funsd",
        "model": args.model, "defense": args.defense, "primary_defense": primary,
        "n_train_synth": len(synth), "n_pages": len(encoded), "dim": dim,
        "k": args.k, "lam": args.lam, "null_k": args.null_k, "r_topic": args.r_topic,
        "pii_labels": sorted(pii_labels), "content_labels": sorted(content_labels),
        "chance": {"attack_top1": chance_p, "utility_r1": chance_u},
        "vanilla": {"attack_top1": van_p, "utility_r1": van_u, "n_priv": n_priv, "n_util": n_util},
        "defenses": per_defense,
        "verdict": overall,
        "fingerprint": run_fingerprint(),
    }
    (local_out / "defense_transfer_funsd.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"\nVERDICT: {overall}")
    print(f"wrote defense_transfer_funsd.json -> {args.out}")


if __name__ == "__main__":
    main()
