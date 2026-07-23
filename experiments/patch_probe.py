"""Discriminative patch probe — the DECISIVE info-presence instrument (design-panel result).

Answers "is field-level PII IN ColPali's stored patches?" WITHOUT generation / OCR / VAE / GAN, so a
NULL is a valid proof of ABSENCE (a generative failure never is). Two instruments + controls, font sweep:

  1. MaxSim retrieval-discrimination (TRAINING-FREE): encode the true field value + K-1 same-format
     distractors with ColPali's own encode_query; score each against the stored page patches (MaxSim).
     Accuracy = fraction where the true value is argmax. Chance = 1/K. No sample-size/capacity confound.
  2. Trained closed-set MLP probe: predict the closed-set name (K=240) from field-token-pooled features.

CONTROLS (make a null interpretable):
  * positive control — predict FIELD-TYPE (K=3) from the same tokens. If THIS works but name-identity
    fails, the probe has power and glyph info is genuinely absent. If even this fails, probe is
    underpowered and the null is meaningless.
  * shuffle control — shuffle name labels; must collapse to chance (proves it reads content not position).

KILL CRITERIA are pre-registered below (committed before the run). Font sweep exposes the predicted
size dependence (large glyphs span many 14px patches; 8px text lives inside one).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

# ---- PRE-REGISTERED THRESHOLDS (do not edit after the run) -----------------------------------------
GREENLIGHT_ACC = 0.40          # held-out name accuracy >= this at font >= 24px => info PRESENT (thesis)
STRUCTURE_MAX_RATIO = 1.5      # name acc <= 1.5x chance at EVERY font incl 48px => info absent...
POSITIVE_CONTROL_MIN = 0.80    # ...ONLY valid if field-type control >= this (probe has power)
MAXSIM_GREENLIGHT = 0.40       # MaxSim discrimination (K=8, chance 0.125) >= this => retrievable
# ---------------------------------------------------------------------------------------------------


def _train_probe(X, y, n_classes, epochs=300, seed=0, hidden=256):
    """2-layer MLP closed-set probe. Returns held-out accuracy (80/20 split)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    y = torch.tensor(np.asarray(y), dtype=torch.long)
    n = len(y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    X, y = X[perm], y[perm]
    ntr = int(0.8 * n)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    d = X.shape[1]
    net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((net(Xte).argmax(1) == yte).float().mean())
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/patch_probe")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=300, help="cards per font size")
    ap.add_argument("--k", type=int, default=8, help="MaxSim closed-set size (true + k-1 distractors)")
    ap.add_argument("--fonts", default="12,24,48")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch  # noqa: F401

    from patchguard.data.align import boxes_to_patch_mask
    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.base import maxsim
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    fonts = [int(f) for f in args.fonts.split(",")]
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    retriever = ColPaliRetriever(model_name=args.model)

    # --- alignment sanity on the REAL processor (the critique's mandatory gate) ---
    demo_img, _ = generate_id_cards(1, seed=7, value_font_size=34)[0]
    e0 = retriever.encode_page(demo_img)
    align_info = {"n_patches": e0.n_patches, "grid": list(e0.grid),
                  "n_prefix_tokens": e0.n_prefix_tokens, "resize_policy": e0.resize_policy,
                  "dim": e0.dim}
    print(f"ALIGNMENT: {align_info}")

    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # closed set K=240
    name_to_idx = {nm: i for i, nm in enumerate(name_pool)}
    qcache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    for nm in name_pool:
        q(nm)  # warm cache

    def field_feat(enc, box, img_wh):
        # grid-only mask (length gh*gw) aligned with image_patches(); avoids the trailing
        # instruction tokens (ColPali: 1030 total = 1024 image + 6 suffix).
        mask = boxes_to_patch_mask([box], img_wh, enc.grid, enc.input_size, enc.resize_policy, 0.0, 0)
        imgp = enc.image_patches()
        return imgp[mask].mean(0) if mask.any() else imgp.mean(0)

    per_font = {}
    for font in fonts:
        # vary=True: font family/size jitter, position jitter, SHUFFLED field order (decouples
        # field-type from position), bg tint, pixel noise -> kills the fixed-template confounds.
        cards = generate_id_cards(args.n, seed=1000 + font, value_font_size=font, vary=True)
        Xname, yname = [], []              # trained probe: name identity from name-field tokens
        Xtype, ytype = [], []              # positive control: field-type from field tokens
        maxsim_hit = {"name": [], "id_no": [], "dob": []}
        for im, fs in cards:
            enc = retriever.encode_page(im)
            wh = (im.shape[1], im.shape[0])
            for f in fs:
                feat = field_feat(enc, f.box, wh)
                Xtype.append(feat)
                ytype.append({"name": 0, "dob": 1, "id_no": 2}[f.field_type])
                if f.field_type == "name":
                    Xname.append(feat)
                    yname.append(name_to_idx[f.text])
                # MaxSim discrimination: true vs k-1 same-format distractors
                if f.field_type == "name":
                    pool = [n for n in name_pool if n != f.text]
                    distr = list(rng.choice(pool, args.k - 1, replace=False))
                elif f.field_type == "id_no":
                    distr = [f"{int(rng.integers(10_000_000, 99_999_999))}" for _ in range(args.k - 1)]
                else:  # dob
                    distr = [f"{int(rng.integers(1,13)):02d}/{int(rng.integers(1,29)):02d}/{int(rng.integers(1950,2005))}"
                             for _ in range(args.k - 1)]
                cands = [f.text] + distr
                scores = [maxsim(q(c), enc.patches) for c in cands]
                maxsim_hit[f.field_type].append(int(np.argmax(scores) == 0))

        acc_name = _train_probe(Xname, yname, 240, seed=args.seed)
        acc_type = _train_probe(Xtype, ytype, 3, seed=args.seed)  # positive control
        yshuf = list(np.random.default_rng(args.seed).permutation(yname))
        acc_shuffle = _train_probe(Xname, yshuf, 240, seed=args.seed)  # must be ~chance

        per_font[font] = {
            "n": args.n,
            "chance_name": 1.0 / 240,
            "mlp_name_acc": acc_name,
            "positive_control_type_acc": acc_type,
            "shuffle_name_acc": acc_shuffle,
            "maxsim_acc": {ft: float(np.mean(v)) for ft, v in maxsim_hit.items()},
            "maxsim_chance": 1.0 / args.k,
        }
        print(f"font {font}: name_classify_acc={acc_name:.3f} (chance {1/240:.4f}) | pos_ctrl(type,shuffled-order)={acc_type:.3f} "
              f"| label_shuffle={acc_shuffle:.3f} | maxsim_discrim={per_font[font]['maxsim_acc']}")

    # --- same-name-different-template control: does MaxSim recover a FIXED name across varied layouts? ---
    from patchguard.data.synthdoc import generate_same_name_cards

    ctrl_names = list(rng.choice(name_pool, 3, replace=False))
    same_name_hits = []
    for ci, nm in enumerate(ctrl_names):
        for im, _ in generate_same_name_cards(nm, 20, seed=7000 + ci):
            enc = retriever.encode_page(im)
            distr = list(rng.choice([x for x in name_pool if x != nm], args.k - 1, replace=False))
            scores = [maxsim(q(c), enc.patches) for c in [nm] + distr]
            same_name_hits.append(int(np.argmax(scores) == 0))
    same_name_maxsim = float(np.mean(same_name_hits))
    print(f"SAME-NAME-DIFFERENT-TEMPLATE control: MaxSim recovers fixed name across varied layouts = {same_name_maxsim:.3f} (chance {1/args.k:.3f})")

    # verdict against pre-registered thresholds
    greenlight = any(v["mlp_name_acc"] >= GREENLIGHT_ACC or max(v["maxsim_acc"].values()) >= MAXSIM_GREENLIGHT
                     for f, v in per_font.items() if f >= 24)
    powered = all(v["positive_control_type_acc"] >= POSITIVE_CONTROL_MIN for v in per_font.values())
    max_font = max(fonts)
    structure_kill = (powered and
                      all(v["mlp_name_acc"] <= STRUCTURE_MAX_RATIO * v["chance_name"] for v in per_font.values()) and
                      all(max(v["maxsim_acc"].values()) <= STRUCTURE_MAX_RATIO * v["maxsim_chance"] for v in per_font.values()))
    decision = "GREENLIGHT_ATTACK" if greenlight else ("STRUCTURE_PAPER" if structure_kill else "INCONCLUSIVE")

    payload = {"mode": "patch_probe", "alignment": align_info, "fonts": fonts,
               "varied_templates": True, "same_name_diff_template_maxsim": same_name_maxsim,
               "per_font": {str(k): v for k, v in per_font.items()},
               "greenlight": greenlight, "probe_powered": powered, "structure_kill": structure_kill,
               "decision": decision, "fingerprint": run_fingerprint()}
    (local_out / "patch_probe.json").write_text(json.dumps(payload, indent=2))
    print(f"\nDECISION: {decision}  (greenlight={greenlight} powered={powered} structure_kill={structure_kill})")
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote patch_probe.json -> {args.out}")


if __name__ == "__main__":
    main()
