# The Persistence of Vision — Complete Results & Findings

**Consolidated empirical record of the build session (2026-07-23).** Every finding + metric. This is
the source of truth for the paper's results section, figures, and `reproduce.sh`. Target: IEEE S&P
(primary) · USENIX Security (secondary) · NeurIPS D&B (fallback).

> **Title:** *The Persistence of Vision: PII Reconstruction and Patch-Scoped Privacy in Multi-Vector
> Vision-Language Retrieval.* (The "reconstruction/patch-scoped" framing is now partly superseded — see
> the honest thesis below; a retitle toward "retrieval leakage / holographic PII" is likely.)

---

## 0. The honest one-paragraph thesis (final)

The stored **multi-vector** patch embeddings of visual-document retrievers (ColPali, ColQwen2) are a
**field-level PII oracle**: an attacker with read access to the index recovers names/IDs/dates by
**ranking candidate values via the model's own MaxSim** — no image reconstruction. The leak is
**holographically distributed** (info in every patch via full attention), **capacity-driven** (comes
from storing many per-patch vectors — the same thing that gives SOTA retrieval), and **generalizes
across backbones**. Consequently PII **cannot be spatially redacted, erased, or locally defended** —
only globally scrubbed. Naive global noise wrecks utility; a **learned anisotropic index-time
projection** suppresses PII while preserving retrieval and **dominates the noise baseline** (in the
separable-PII regime; a fundamental floor exists where PII *is* the retrieval content). *Not yet tested:
adaptive attacker, real-retrieval utility (ViDoRe).*

---

## 1. Infrastructure & reproducibility (all live)

- **GCP project:** `patchguard-reakon` (org profitwise.app), billing = $25k credit acct `014F80-4CDA1C-D6AEC2`.
- **Buckets:** `gs://patchguard-reakon-data` (FUNSD staged), `gs://patchguard-reakon-artifacts` (all run JSON + logs).
- **Repro image:** built by `scripts/10_build_image.sh` → Artifact Registry `…/patchguard/repro:<git_sha>`.
  Base `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` (Python 3.10). **Pinned `colpali-engine==0.3.5`**
  (→ transformers 4.46.x; NOT 5.x which breaks on torch 2.3.1) + pillow, google-cloud-storage,
  pytesseract, diffusers==0.31.0, accelerate, qwen-vl-utils, fonts-dejavu-core, tesseract-ocr.
- **Ephemeral GPU runs:** `scripts/12_train_vm.sh ZONE PROFILE JOB` — GCE VM, self-deletes, 65-min hard cap.
- **Warm-VM (fast iteration):** `scripts/20_warm_vm.sh` + `21_exec.sh` (sudo docker over SSH, ColPali
  cached in `/opt/hfcache`): cold-start once (~15 min), then **~3.5 min/experiment** vs ~20 min ephemeral.
- **Provenance:** `repro.py` stamps git sha (build-stamped `_buildinfo.py` inside container) + dirty flag
  + torch/CUDA/GPU into every `metrics.json`. Determinism scoped `warn_only`, `CUBLAS_WORKSPACE_CONFIG`.
- **Cost of the whole session:** well under ~$15 total (many $0.5–1 runs, all auto-torn-down).

**Fresh-project gotchas hit & fixed (recorded in scripts):** Vertex `run_module.py` entrypoint (→ use
`containerSpec.command`); Cloud Build compute-SA IAM roles; Python 3.11→3.10 floor; L4/A100 STOCKOUT
(sweep zones, fail-fast); DLVM has driver but not docker (install in startup); `.gitignore data/`
silently excluded source pkg `patchguard/data/` (anchored to `/data/` + added `.gcloudignore`); the
tee-to-serial `exec` broke docker install (reverted).

---

## 2. Model & alignment ground truth (verified on the LIVE processor)

- **ColPali** (`vidore/colpali-v1.3`, SigLIP-So400m / PaliGemma): 448×448 input → **32×32 = 1024 image
  patches**, each **128-dim**, **squash** resize. Emits **1030 tokens = 1024 image (FIRST, `n_prefix=0`)
  + 6 instruction-suffix tokens**. Late interaction = MaxSim. **Storage = 131,840 floats/page (1030×128).**
- **BiPali** (control): same backbone, mean-pooled to **1 vector = 128 floats/page** (represented as a
  1×1 grid — no locality). ~1030× less storage than ColPali.
- **ColQwen2** (`vidore/colqwen2-v1.0`, Qwen2-VL): dynamic resolution, **~267 patches/page** (variable),
  128-dim, MaxSim.
- **Alignment confirmed:** the FUNSD validator + real-processor dump both show patches land on fields;
  `n_prefix=0`, image-first, squash — so the defense/attack numbers riding on the mapping are valid.

---

## 3. Attack evolution (chronological — what failed, why, and the pivot)

| # | Attack | Result | Why it mattered |
|---|---|---|---|
| v0 | ConvTranspose decoder (L1+LPIPS+field-weighted) | blurry **structure**, no legible text; PFRR ≈ 0.008 | over-optimizes blurry mean |
| ★ | Generative **kill test** ColPali vs BiPali | **STOP** — both PFRR ≈ 0.008 (identical, floor) | a *confounded* null |
| v1 | SD-VAE latent + PatchGAN | **collapsed to blank white** | docs ~90% white → loss rewards blank; discriminator saturated d≈0.006, g_adv≈3.68 |
| v2 | + ink-weighted loss + R1 GAN stabilization | GAN healthy (d=2.06, g_adv=0.03); **ink blobs, still no glyphs** | white-collapse fixed; SD VAE 8× downsample too coarse for 1–2px strokes |
| — | **Design-panel workflow** | verdict: *stop generating pixels to answer an information question; a generative failure can never prove absence* | drove the pivot |
| ✅ | **Discriminative probe** | **GREENLIGHT** | proved the info IS there |
| ✅ | **Retrieval / dictionary attack** | the real, working attack | rank candidates by MaxSim, no reconstruction |

**Key realization:** generative reconstruction is the wrong instrument. SD's VAE is text-destroying
(8×8px latent cell vs 1–2px strokes) and its natural-image prior hallucinates. The *retrieval*
interface is the attack.

---

## 4. COMPLETE RESULTS (every metric; bootstrap 95% CI; chance noted)

### 4.1 The retrieval/dictionary attack — recovery (varied templates, 1000-candidate lineup, chance 0.001)
| field | top-1 | top-5 |
|---|---|---|
| **name** | **1.00** | 1.00 |
| **id_no** | 0.725 | 0.90 |
| **dob** | 0.40 | 0.80 |
- id/dob misses = MaxSim **bag-of-tokens digit TRANSPOSITION** (right multiset, wrong order).

### 4.2 Verb validation — anagram control (property_curve; per-field, over glyph sweep)
- **name** beats own char-permutations = **1.00** (order preserved → genuinely "recovered")
- **dob** anagram-beat = **0.93–1.00**
- **id_no** anagram-beat = **0.77–0.87** ("recovered ~80% exact, else transposed")
→ MaxSim is **not** pure multiset matching; contextual query tokens preserve order (esp. names).

### 4.3 Discriminative probe (de-confounded, varied templates)
- MaxSim name discrimination (K=8, chance 0.125) = **1.00**
- MLP 240-way name **classification** = **0.33** (chance 0.004 → ~80×; *classification, not open recovery*)
- positive control (field-type, shuffled order) = **1.00** (probe has genuine content-reading power)
- label-shuffle = **0.00** (reads content, not position)
- **same-name-different-template MaxSim = 1.00** (the fixed-layout confound is refuted; glyph content)
- *(first, fixed-layout probe reported MLP 0.53–0.78 — INFLATED by deterministic-render duplicate
  leakage; superseded by the de-confounded 0.33.)*

### 4.4 Claim 1 — multi-vector vs pooled (retrieval attack, n=60, chance 0.005)
| field | ColPali (multi-vector) | BiPali (pooled) | Δ | McNemar (ColPali-only / BiPali-only) |
|---|---|---|---|---|
| name | **1.000** | 0.083 | +0.917 | **55 / 0** |
| id_no | 0.850 | 0.033 | +0.817 | 49 / 0 |
| dob | 0.817 | 0.017 | +0.800 | 48 / 0 |
→ perfectly one-directional; a **qualitative** architecture difference.

### 4.5 Claim 1b — matched-bytes control (HONEST: leak is capacity-driven)
| ColPali subsample | floats | recovery | | BiPali | floats | recovery |
|---|---|---|---|---|---|---|
| K=1 | 128 | **0.10** | | pooled | 128 | **0.20** |
| K=2 | 256 | 0.26 | | | | |
| K=4 | 512 | 0.36 | | | | |
| K=8 | 1024 | 0.70 | | | | |
| K=16 | 2048 | 0.78 | | | | |
| K=64 | 8192 | 1.00 | | | | |
| K=256 | 32768 | 1.00 | | | | |
→ **At matched 128 floats, ColPali-1-patch (0.10) < BiPali (0.20).** The multi-vector advantage is
CAPACITY (storing many patches), NOT late-interaction magic. Claim 1 reframed: leak ∝ #stored patches,
which is inseparable from retrieval quality.

### 4.6 Attack is genuine — wrong-page control (n=50, chance 0.005)
| cell | recovery |
|---|---|
| correct_full (name_i vs page_i) | **1.000** |
| wrong_full (name_i vs page_j) | **0.000** |
| correct_erased (page_i minus name_i's patches) | **1.000** |
| wrong_erased (page_j minus name_j's patches) | **0.000** |
→ recovery depends on the name's **presence** (not "true name always wins"); validates ALL recovery
numbers. `VERDICT: HOLOGRAPHIC BLEED CONFIRMED`.

### 4.7 Erasure & holographic bleed (n=50)
- recover from field-patches-**ONLY** = **1.00**; recover from page **WITHOUT** field patches = **1.00**
- Dilation-deletion sweep (delete name patches dilated by grid radius r; attack remainder):
  | r | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
  |---|---|---|---|---|---|---|---|
  | recovery | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
  | patches removed /1024 | 31 | 62 | 97 | 133 | 172 | 254 | **341 (33%)** |
- **NEVER ERASED** — deleting a third of all patches still recovers. → **naive patch deletion does not
  satisfy Article 17; redaction is provably insufficient.**

### 4.8 Naive patch-scoped DEFENSE fails (defense_frontier, n=50)
| noise σ | FLAT priv / util | PATCH-SCOPED priv / util |
|---|---|---|
| 0.0 | 0.00 / 0.86 | 0.00 / 0.86 |
| 0.1 | 0.04 / 0.76 | **0.00** / 0.86 |
| 0.2 | 0.16 / 0.48 | **0.00** / 0.86 |
| 0.35 | 0.70 / 0.30 | **0.00** / 0.86 |
| 0.5 | 0.92 / 0.26 | **0.00** / 0.86 |
| 0.75 | 0.98 / 0.22 | **0.00** / 0.86 |
→ patch-scoped gives **ZERO privacy at any noise** (info bled out of its patches). Original protocol
Claim 3 ("patch-scoped dominates") is **REFUTED by our data**. Flat noise works but craters utility.

### 4.9 Cross-model generalization — ColQwen2 (n=40, ~267 patches/page, chance 0.005)
- LEAK (name recovery) = **0.975** ; WRONG-PAGE = **0.000** (genuine)
- BLEED (random-remove patches): 25%→**1.00**, 50%→**0.98**, 75%→**1.00**, 90%→**0.83**
- `GENERALIZES: True` → leak + holographic bleed are a **general multi-vector-VLM property**, not
  ColPali-specific. Paper scope = "multi-vector visual document retrievers."

### 4.10 Glyph-height property curve (clean synthetic; recovery vs glyph px; anagram-controlled)
- **name recovery FLAT at 1.00 from 6px → 37px** (no size threshold on clean docs!)
- id_no ≈ 0.70–0.83 ; dob ≈ 0.73–0.90 across the sweep
→ **REFUTES a simple glyph-height threshold** for clean docs; the clean→real gap is document *realism*
(noise/density/contrast), not font size alone.

### 4.11 Real-document transfer — FUNSD (n=600 fields, 20-lineup, chance 0.05)
- overall top-1 = **0.19**, top-5 = **0.68**
- by label: answer 0.26, other 0.39, question 0.09, header 0.10 (boilerplate labels recover worst)
- by glyph height: **≤10px → 0.11 [0.08,0.14]**, 10–16px → 0.32 [0.19,0.45], **>16px → 0.46 [0.34,0.58]**
→ transfers (above chance) but **weak** on real dense small-font forms; recovery rises with text
distinctiveness/size. Caveat: FUNSD probes mix real PII with form boilerplate — answer-only rerun owed.

### 4.12 ★ LEARNED DEFENSE beats flat noise (learned_defense; HELD-OUT cards + HELD-OUT names, open-set)
**RedactionProjection** = per-patch anisotropic residual MLP, index-time, min-max trained
(utility=InfoNCE topic ranking; privacy=push true-PII MaxSim below distractors).
| Learned P (λ) | utility | privacy | | Flat noise (σ) | utility | privacy |
|---|---|---|---|---|---|---|
| 0.0 | 0.825 | 0.425 | | 0.0 | 0.925 | 0.000 |
| 1.0 | 0.850 | **1.000** | | 0.1 | 0.775 | 0.050 |
| 2.0 | 0.875 | **1.000** | | 0.2 | 0.550 | 0.225 |
| **5.0** | **0.950** | **1.000** | | 0.35 | 0.150 | 0.800 |
| 10.0 | 0.925 | **1.000** | | 0.5 | 0.100 | 0.925 |
| 20.0 | 0.950 | **1.000** | | 0.75 | 0.050 | 0.975 |
- **DOMINANCE (learned util − flat util at matched privacy): +0.47 @0.5, +0.69 @0.8, +0.74 @0.9.**
- `VERDICT: LEARNED DEFENSE BEATS FLAT NOISE.`
- **Mechanism:** name-value queries and topic queries occupy **separable subspaces**; global P collapses
  the PII direction, spares the content direction (isotropic noise can't). Reconciles with holographic
  bleed: local defense impossible, but a **global learned anisotropic transform** works.

---

## 5. Claims — final state

| Claim | Verdict | Key metric |
|---|---|---|
| 1 — multi-vector leaks, pooled doesn't | ✅ strong | 1.00 vs 0.08; McNemar 55/0 |
| 1b — …driven by capacity (per-patch storage), not late-interaction magic | ✅ honest | K=1 (128fl) 0.10 < BiPali 0.20 |
| Attack genuine (not artifact) | ✅ airtight | wrong-page 0.000 |
| Holographic bleed / info non-local | ✅ airtight | delete 33% → still 1.00 |
| Erasure by patch deletion impossible | ✅ strong negative | never erased; Article 17 |
| Local (patch-scoped) defense impossible | ✅ (refutes orig Claim 3) | patch-scoped priv 0.00 |
| Generalizes across backbones | ✅ | ColQwen2 leak 0.975, bleed 90%→0.83 |
| Learned global anisotropic defense works & beats noise | ✅ (separable regime) | priv 1.00 @ util 0.95; dom +0.74 |

---

## 6. Repo map (experiments — all in the repro image)

`experiments/`: `kill_test.py` (mock + real path), `train_funsd.py` (v0 decoder), `train_diffusion.py`
(v1/v2), `overfit_probe.py` (deprecated diagnostic), `bigfont_probe.py`, `patch_probe.py`
(discriminative), `retrieval_attack.py`, `property_curve.py` (glyph curve + anagram), `funsd_transfer.py`,
`claim1.py`, `claim1b.py`, `control_wrongpage.py`, `erasure.py`, `cross_model.py`, `defense_frontier.py`,
`learned_defense.py`.
`patchguard/`: `retrievers/{base,colpali,bipali,colqwen2,mock}.py`, `data/{align,fields,funsd,synthdoc}.py`,
`attack/{decoder,diffusion,train}.py`, `defense/{perturb,localize,redact}.py`, `eval/{pfrr,frontier,
killgate,reconstruct}.py`, `repro.py`. ~90 CPU tests.

---

## 7. Open questions / NOT yet tested (the honest to-do)

1. **⚠️ ADAPTIVE attacker vs RedactionProjection** — privacy=1.00 is against a NON-adaptive attacker.
   An attacker who KNOWS P (invert P, adapt queries, exploit content↔PII correlation) is the security-
   venue bar. **Decisive untested experiment.** Until this, the defense claim is unproven.
2. **Real-retrieval utility** — replace synthetic topic-Recall@1 with **ViDoRe NDCG@5** on a ColPali+P index.
3. **The fundamental floor** — the defense works where PII is *incidental*; characterize the regime where
   PII IS the retrieval content (find-by-name) and no learned defense can help.
4. **Baselines (Claim 2)** — EntroGuard / PRESS / Koga vs the retrieval attack (likely reduce to global noise).
5. **FUNSD answer-fields-only** — the fair real-doc number (strip boilerplate).
6. **Multi-seed + Holm–Bonferroni** across the claim family.
7. **Fix 7 citation errors** (see `RELATED_WORK_VERIFIED.md`).
8. **Writeup + figures + reproduce.sh + artifact-eval + responsible disclosure.**

---

## 8. Numbers to remember (headline)

- Attack: **name recovery 1.00** from a 1000-lineup; ColQwen2 **0.975** (generalizes).
- Multi-vector vs pooled: **1.00 vs 0.08**, McNemar **55/0**; but matched-bytes → **capacity-driven**.
- Holographic: delete **33% of patches → still 1.00**; erasure impossible.
- Naive local defense: **0.00 privacy**. Learned defense: **priv 1.00 @ util 0.95**, dominates noise by **+0.74**.
- Real docs (FUNSD): weak (**0.19** top-1), scales with glyph size (≤10px 0.11 → >16px 0.46).
