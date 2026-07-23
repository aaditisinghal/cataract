"""Recovery vs glyph-height property curve + anagram negative control (MASTER_PLAN S6 headline).

Scopes the leak by DOCUMENT PROPERTY, not corpus. Both corpora collapse onto one axis = glyph height
in post-resize pixels (what ColPali actually sees). We own the renderer, so we sweep it: a FIXED set of
PII values, held constant, rendered across a font-size sweep on a fixed template — so the only variable
is glyph height. FUNSD is then one measured point (real fields, glyph height computed through the same
resize). The deliverable is a threshold: recovery collapses below ~N px of glyph height.

ANAGRAM NEGATIVE CONTROL (gates the verb): MaxSim is a bag-of-tokens (sum of per-token maxes), so it may
recover a character MULTISET, not a string. For each field we also rank the true value against
permutations of its own characters (same multiset, same format). If the true value does NOT beat its
anagrams, "recovered PII" is the wrong verb — it's multiset/anagram-class recovery. Reported per field.

All accuracies carry n and a bootstrap 95% CI.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def _boot_ci(hits, n_res=4000, seed=0):
    a = np.asarray(hits, dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    b = a[rng.integers(0, a.size, (n_res, a.size))].mean(1)
    return float(a.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def _anagrams(value: str, m: int, rng: np.random.Generator):
    """Permutations of value's alphanumerics, keeping non-alnum (space/slash) positions fixed."""
    idx = [i for i, c in enumerate(value) if c.isalnum()]
    chars = [value[i] for i in idx]
    out, seen, tries = [], {value}, 0
    while len(out) < m and tries < m * 40:
        tries += 1
        perm = list(chars)
        rng.shuffle(perm)
        s = list(value)
        for j, i in enumerate(idx):
            s[i] = perm[j]
        cand = "".join(s)
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/property_curve")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=30, help="fixed PII tuples, rendered at every font size")
    ap.add_argument("--fonts", default="8,10,12,14,16,20,24,32,40,48")
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--anagrams", type=int, default=8)
    ap.add_argument("--funsd", default="gs://patchguard-reakon-data/funsd")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    from patchguard.data.synthdoc import _font, _pii, generate_id_card
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.base import maxsim
    from patchguard.retrievers.colpali import ColPaliRetriever

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    fonts = [int(f) for f in args.fonts.split(",")]
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    retriever = ColPaliRetriever(model_name=args.model)
    qcache: dict[str, np.ndarray] = {}

    def q(s: str):
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    def glyph_h(font_size):
        d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        bb = d.textbbox((0, 0), "8W8W8W8W", font=_font(font_size, 1))
        return bb[3] - bb[1]

    # fixed PII set (held constant across all font sizes -> only glyph height varies)
    pii_set = [_pii(np.random.default_rng(1000 + i)) for i in range(args.n)]

    # shared random-distractor pools (same format), encoded once
    def rid():
        return f"{int(rng.integers(10_000_000, 99_999_999))}"

    def rdob():
        return f"{int(rng.integers(1,13)):02d}/{int(rng.integers(1,29)):02d}/{int(rng.integers(1950,2005))}"

    from patchguard.data.synthdoc import _FIRST, _LAST
    distr = {
        "name": [f"{a} {b}" for a in _FIRST for b in _LAST],
        "id_no": list({rid() for _ in range(args.distractors * 2)})[: args.distractors],
        "dob": list({rdob() for _ in range(args.distractors * 2)})[: args.distractors],
    }
    for pool in distr.values():
        for s in pool:
            q(s)

    def rank_top1(patches, true, pool):
        cands = [true] + list(rng.choice([x for x in pool if x != true], min(args.distractors, len(pool) - 1), replace=False))
        sc = np.array([maxsim(q(c), patches) for c in cands])
        return int(np.argmax(sc) == 0)

    def beats_anagrams(patches, true):
        ana = _anagrams(true, args.anagrams, rng)
        if not ana:
            return None
        sc = np.array([maxsim(q(c), patches) for c in [true] + ana])
        return int(np.argmax(sc) == 0)

    curve = []
    for font in fonts:
        gh = glyph_h(font)
        hits = {"name": [], "id_no": [], "dob": []}
        ana = {"name": [], "id_no": [], "dob": []}
        for i, pii in enumerate(pii_set):
            img, fields = generate_id_card(2000 + i, value_font_size=font, vary=False, fixed_pii=pii)
            enc = retriever.encode_page(img)
            for f in fields:
                hits[f.field_type].append(rank_top1(enc.patches, f.text, distr[f.field_type]))
                b = beats_anagrams(enc.patches, f.text)
                if b is not None:
                    ana[f.field_type].append(b)
        point = {"font": font, "glyph_px": gh}
        for ft in ("name", "id_no", "dob"):
            m, lo, hi = _boot_ci(hits[ft], seed=font)
            am, alo, ahi = _boot_ci(ana[ft], seed=font + 1)
            point[ft] = {"top1": m, "ci": [lo, hi], "n": len(hits[ft]),
                         "anagram_beat": am, "anagram_ci": [alo, ahi]}
        curve.append(point)
        print(f"font {font:>2} (glyph {gh:>2}px): "
              + " | ".join(f"{ft} top1={point[ft]['top1']:.2f}[{point[ft]['ci'][0]:.2f},{point[ft]['ci'][1]:.2f}] "
                           f"ana-beat={point[ft]['anagram_beat']:.2f}" for ft in ("name", "id_no", "dob")))

    # threshold: smallest glyph height where name top1 >= 0.5 (interp on the swept points)
    gh_axis = [p["glyph_px"] for p in curve]
    for ft in ("name", "id_no", "dob"):
        accs = [p[ft]["top1"] for p in curve]
        above = [gh for gh, a in zip(gh_axis, accs) if a >= 0.5]
        thr = min(above) if above else None
        print(f"THRESHOLD {ft}: top1>=0.5 above ~{thr}px glyph height" if thr else f"THRESHOLD {ft}: never >=0.5 in sweep")

    # FUNSD as points on the SAME glyph-height axis (real fields, glyph px through the resize)
    funsd_points = _funsd_points(args, retriever, q, maxsim, rng)

    payload = {"mode": "property_curve", "curve": curve, "funsd_points": funsd_points,
               "fonts": fonts, "n_per_point": args.n, "fingerprint": run_fingerprint()}
    (local_out / "property_curve.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        from experiments.train_funsd import _gcs_upload
        _gcs_upload(local_out, args.out)
    print(f"wrote property_curve.json -> {args.out}")


def _funsd_points(args, retriever, q, maxsim, rng):
    """Place real FUNSD fields on the glyph-height axis (box height through squash->448)."""
    import numpy as np
    from pathlib import Path
    import tempfile
    from PIL import Image
    from patchguard.data.funsd import iter_funsd

    root = args.funsd
    if str(root).startswith("gs://"):
        from experiments.train_funsd import _gcs_download
        root = str(_gcs_download(root, Path(tempfile.mkdtemp())))

    pages, texts = [], []
    for ps in iter_funsd(root, split="testing_data", granularity="entity"):
        img = np.array(Image.open(ps.image_path).convert("RGB"))
        fs = [f for f in ps.fields if len("".join(c for c in f.text if c.isalnum())) >= 3]
        if fs:
            pages.append((img, fs, ps.size))
            texts.extend(f.text for f in fs)
        if len(pages) >= 50:
            break

    def norm(s):
        return "".join(c for c in s if c.isalnum()).casefold()

    buckets: dict[str, list[int]] = {}
    for img, fs, (ow, oh) in pages:
        enc = retriever.encode_page(img)
        scale = 448.0 / oh  # squash vertical scale
        for f in rng.permutation(len(fs))[:12]:
            fd = fs[f]
            gh = (fd.box[3] - fd.box[1]) * scale  # glyph height in encoded px
            distr = list(rng.choice([t for t in texts if norm(t) != norm(fd.text)], 19, replace=False))
            sc = np.array([maxsim(q(c), enc.patches) for c in [fd.text] + distr])
            hit = int(np.argmax(sc) == 0)
            b = "gh<=10" if gh <= 10 else ("gh10-16" if gh <= 16 else "gh>16")
            buckets.setdefault(b, []).append(hit)
    out = {}
    for b, v in buckets.items():
        a = np.asarray(v, float)
        rb = np.random.default_rng(0)
        boot = a[rb.integers(0, a.size, (2000, a.size))].mean(1)
        out[b] = {"top1": float(a.mean()), "ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], "n": a.size}
    print("FUNSD (real) by glyph-height bucket:", {b: f"{o['top1']:.2f}[{o['ci'][0]:.2f},{o['ci'][1]:.2f}] n={o['n']}" for b, o in out.items()})
    return out


if __name__ == "__main__":
    main()
