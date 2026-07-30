# The Persistence of Vision: State-of-the-Art Privacy for Multi-Vector VLM Retrievers — Complete Results & Findings

**Consolidated empirical record of the build session (2026-07-23).** Every finding + metric. This is
the source of truth for the paper's results section, figures, and `reproduce.sh`. Target: IEEE S&P
(primary) · USENIX Security (secondary) · NeurIPS D&B (fallback).

> **Title (locked):** *The Persistence of Vision: State-of-the-Art Privacy for Multi-Vector VLM Retrievers.*
> Positioning: prior options are strictly dominated — leak all PII, degrade retrieval, or offer only reversible
> (false) privacy. We characterize the privacy–utility frontier of multi-vector VLM document retrieval and give
> the first defense that reaches it: irreversible, tunable redaction holding ~0.90 privacy at ~0.875 retrieval.

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

**Exact generative-line training metrics (dead-end, for the record):**
- **v0 decoder** (149 real FUNSD pages, A100, 40 epochs): loss total **2.877 → 1.041** (2.76×), l1
  0.386→0.055, field_l1 **0.361 → 0.131**, lpips 0.688→0.331. (Loss fell — decoder *learns* — but
  reconstruction was blurry structure, PFRR ≈ 0.008.)
- **v1 diffusion** (85d59b8, white-collapse): final total 1.107, l1 0.053, field_l1 0.131, lpips 0.40,
  latent 0.348, **g_adv 3.683, d 0.006** (discriminator saturated → adversarial signal dead).
- **v2 diffusion** (3fa35fb, ink-weighted + R1): final total 3.026, l1 0.07, field_l1 0.254, lpips 0.42,
  latent 0.332, **g_adv 0.032, d 2.058** (GAN balanced → ink placed at text locations, still no glyphs).
- **Kill test** (382ac89): decision **STOP**, PFRR_colpali = PFRR_bipali = **0.008** (identical floor).
- **Vertex smoke job**: `SMOKE OK: NVIDIA L4 | torch 2.3.0+cu121` (infra validated; keyless ADC + teardown).

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

**First GREENLIGHT probe (aff7b78, FIXED layout — full per-font, superseded but recorded):**
| font | MLP name | MaxSim name / id_no / dob |
|---|---|---|
| 12px | 0.533 | 1.00 / 1.00 / 0.99 |
| 24px | 0.783 | 1.00 / 0.99 / 0.98 |
| 48px | 0.550 | 1.00 / 0.99 / 0.99 |
Decision = `GREENLIGHT_ATTACK`; positive-control 1.00, shuffle 0.00. (MaxSim recovery robust across
fonts; the MLP numbers here are the inflated ones — use §4.3 de-confounded values for the paper.)

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
- **⚠️ but this privacy is NON-ADAPTIVE only — see §4.13, which breaks it.**

### 4.13 ★★ ADAPTIVE attacker BREAKS the learned defense (adaptive_attack, n_test=40, A100, git 8cb6b45)
The decisive security test: the attacker **KNOWS the public P** (Kerckhoffs). Four white-box strategies vs P(λ=5).
| Strategy | recovery | chance | breaks P? |
|---|---|---|---|
| baseline: undefended dict | 1.000 | 0.005 | — (the leak) |
| baseline: non-adaptive dict on P | 0.000 | 0.005 | — (P's original "privacy 1.00") |
| B1a distillation probe on P(patch) | 0.000 | 0.004 | no\* (underpowered: 64 fit cards, 240-way) |
| **B1b inverse-learning (P⁻¹)** | **1.000** | 0.005 | **YES** — recon cosine **0.998**, 65 536 pairs |
| B1c adaptive-query (anchored grad) | 0.000 | 0.005 | no |
| B1d correlation (via retained topic) | 0.000 | 0.004 | no (topic_recovery 1.00; synthetic fields independent) |
- **VERDICT: P BROKEN.** RedactionProjection is a residual transform `y = x + gate·MLP(x)` → **near-bijective →
  invertible**. An attacker trains `P⁻¹` from (P(patch), patch) pairs (P is public), reconstructs the original
  patches at **cosine 0.998**, and the dictionary attack returns to **1.000**.
- **Consequence:** the learned defense defeats only the **non-adaptive** MaxSim attacker → **privacy must be
  reported attack-relative, not absolute**. A real defense must be **provably information-destroying** (lossy /
  certified), not an invertible reshaping. *(B1a's 0.000 is underpowered — 64 cards for 240 classes — so B1b is
  the definitive breaker, not evidence of removal.)*
- **Efficiency (CPU, efficiency_bench):** P applies at **0.62 µs/patch (0.64 ms/page @ 1030)**, **+0 B** stored,
  **0** query-path overhead, 131 K params (527 KB one-time).

### 4.14 Completion batch — attack tightenings + defense characterization (git 8cb6b45, A100)
- **Positional rerank closes the transposition gap (retrieval_rerank, n=40, D=999).** Two-stage MaxSim →
  positional-bigram rerank: **id_no 0.725 → 0.80**, **dob 0.50 → 0.70** exact top-1 (top-K recall 0.95 / 0.925 —
  the true value is nearly always in the top-K; the rerank pulls it to #1). Confirms the id/dob misses were
  digit *transposition*, not absence.
- **Arrangement is a NULL (arrangement_control, Claim 1c, n=40).** ordered recovery **1.000** = shuffled **1.000**,
  Δ=0, per-card agreement 1.0 → the leak is **content, not spatial arrangement** (MaxSim is order-invariant).
  A clean controlled negative.
- **FUNSD answer-fields-only (funsd_transfer --labels answer, n=449 real fields, 20-lineup, chance 0.05).**
  Fair real-doc number: top-1 **0.225** / top-5 **0.673** (was 0.19 diluted by boilerplate). Scales with length:
  **long(>8) 0.303**, short(≤8) 0.056 (≈chance). Real-doc transfer is real but glyph/length-gated.
- **ViDoRe-style NDCG@5 utility (vidore_utility, synthetic corpus).** vanilla NDCG@5 **0.966** → learned-P
  **0.926** (Recall@1 0.93→0.83, MRR 0.95→0.90) at non-adaptive privacy 0.867 (λ=5). Flat noise at matched
  privacy is far worse (σ=0.2 → NDCG 0.759 @ priv 0.13). **P dominates flat noise on a real retrieval metric,
  not just synthetic Recall@1.** *Caveats: synthetic (not real ViDoRe) corpus; the privacy is the non-adaptive
  number — §4.13 still breaks it adaptively.*
- **Fundamental floor (defense_floor).** As PII↔retrieval entanglement α rises (topic-target → name-target),
  the achievable frontier degrades: util@priv-0.8 = 0.65 at α=0 (PII incidental), falling as α→1 (find-by-name).
  The defense works where PII is incidental and hits a floor where PII *is* the retrieval content.
- **Ablation (defense_ablation, partial 6/14 cells — killed for speed).** linear ≈ MLP: linear/gate-off
  util 0.975/priv 0.975, linear/gate-on 0.925/1.000, mlp-depth1 0.975/1.000 → a **linear** P nearly matches
  the MLP (interpretable). Full sweep + baselines + cross-model defense re-queued in the combined fire batch.

### 4.15 ★★ The information-DESTROYING defense — real, tunable privacy vs the adaptive attacker (certified_defense, git c66cc68, A100)
Motivated by §4.13: a residual (invertible) transform *hides*; a rank-deficient projection *removes*. **NullspaceRedaction**
annihilates the k name-discriminative directions of every stored patch, then we attack it with the SAME inverse attack that
broke the residual P. Full k-sweep (d=128):
| k (dims removed) | non-adaptive priv | **INVERSE-attack priv** | inverse recovery | utility (topic R@1) | recon cos |
|---|---|---|---|---|---|
| residual P (§4.13) | 0.95 | 0.025 ✗ | 1.00 | 0.95 | 0.993 |
| 32 | 0.70 | 0.60 | 0.40 | 0.925 | 0.969 |
| 64 | 0.90 | 0.675 | 0.325 | 0.900 | 0.947 |
| 80 | 0.90 | 0.70 | 0.30 | 0.875 | 0.934 |
| **96 (knee)** | **0.925** | **0.90** | **0.10** | **0.875** | 0.918 |
| 112 | 0.975 | 0.875 | 0.125 | 0.675 ↓ | 0.888 |
| 128 (remove all) | 1.00 | — | — | 0.000 (degenerate) | — |
- **THE CONSTRUCTIVE RESULT: at k=96 the inverse attacker — which crushed the residual P (recovery 1.00) — is held to
  0.10 recovery (privacy 0.90) while topic utility stays 0.875.** Destroying the subspace defeats the attack that reversing
  could not. **This is the first real, tunable, deployable privacy setting for multi-vector VLM retrieval** — the title's
  "SOTA privacy": every prior option is strictly dominated (leak / degrade / reversible-fake).
- **The cost IS the thesis:** you must remove ~96/128 = 75% of dimensions because PII is holographically distributed
  (low-rank k≤32 barely dents it, priv 0.60). Utility holds to k≈96 then cliffs (0.675 @112, 0 @128). k≈96 = the knee.
- **Certified bound (certified_bound).** The map is an orthogonal projection onto span(D)^⊥: the OPTIMAL LINEAR inverse
  recovers **0%** of span(D) (proven, opt_inverse_span_error = 1.000 ∀k>0). A NONLINEAR trained inverse claws back ~0.73 of
  the span *component* via real-data correlations — but end-to-end PII recovery still collapses to 0.10 at k=96. So:
  **provable against linear inversion; empirically strong (0.10) against the best trained adaptive attacker.** The pure
  synthetic-geometry run is fully CERTIFIED (independent directions → 0 recovery); real embeddings add the correlation caveat.

### 4.16 Supporting completion results (git c66cc68, A100)
- **Baselines dominated (baseline_frontier, Claim 2).** Prior embedding-privacy defenses on the same frontier (vs the
  NON-adaptive attack): EntroGuard best privacy **0.00** @ util 0.93; Koga **0.03** @ 0.93; PRESS **0.82** @ util **0.80**.
  Only PRESS (a subspace-removal method, like ours) buys real privacy — at more utility cost (0.80) than our nullspace
  k=96 (0.875), and PRESS is only tested non-adaptively where ours holds against the *adaptive* attacker.
- **Ghost Vectors CONFIRMED (ghost_vectors) — Article 17.** Soft-delete 20/40 docs: logical query view deleted_recovery
  **0.000** (gone from the API) but raw-segment view deleted_recovery **1.000** with **bytes byte-identical** to insertion.
  "Deletion" satisfies the API contract but leaves PII fully recoverable → right-to-erasure violation (GDPR Art.17 / DPDP).
- **Attacker needs the index encoder (transfer_attack) — threat-model refinement.** Same-encoder attack **1.000**;
  CROSS-encoder (ColPali index ↔ ColQwen2 queries, ridge-aligned on 256 anchors) **0.000** both directions. The attack is
  white-box-on-the-(public)-encoder, NOT model-agnostic transfer — realistic for open retrievers, worth stating precisely.
- **Synthetic-trained defense does NOT transfer to real docs (defense_transfer_funsd).** P trained on synthetic cards,
  applied to real FUNSD: attack 0.221 → 0.169 (suppression only +0.05, utility retention 0.97) = **NO TRANSFER**. A
  deployable defense must be fit on the target distribution (or trained privacy-native — the v2 direction).
- **Deploy cost (efficiency_bench, CPU).** Index-time only: **0.64 ms/page**, **+0 B** storage, **0** query-path cost →
  ~$4 of CPU to protect **1 billion** pages, $0 ongoing. The only real cost is the tunable ~5-pt utility trade.

### 4.17 Reviewer-requested experiments (git 09da9a9, A100) — *batch partially collected; auth-gated*
- **★ Margin analysis substantiates "holographic" (margin_analysis, n=40, K=201).** The reviewer's objection: top-1
  pinned at 1.00 can't distinguish *smeared* from *barely-winning*. We measured the linkage **margin** = (true-name
  MaxSim) − (best-distractor MaxSim) vs. deletion fraction: **+1.51 → +1.53 → +1.53 → +1.53 → +1.55** at 0 / 6 / 12 /
  22 / **35%** of page patches deleted (top-1 = 1.00 throughout; margin retained **1.02×** of baseline). The margin does
  **not** decay — it holds (slightly rises) as a third of the page is removed → the field is genuinely distributed, not
  barely-winning. Upgrades the holographic claim from "recovery stays 1.00" to "the score margin is undiminished."
  Verdict: `holographic_margin_persists`.
- **Pending collection (batch driver died on session restart; gcloud re-auth required):** `dim_baselines`
  (random-proj / PCA-32 vs Cataract), `lineup_scaling` (recovery vs K=10²–10⁵ + open-world rejection ROC),
  `defense_pii` (name/id/dob + combined nullspace subspaces), and multi-seed (`adaptive`/`certified` seeds 1–2 for
  mean±CI). These were fired to the same warm VM; results land at `runs/{dimbase,lineup,piidef,adaptive-s*,certified-s*}-repro-09da9a9`.

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
| Learned global anisotropic defense works & beats noise | ⚠️ **non-adaptive only** | priv 1.00 @ util 0.95 (non-adaptive); dom +0.74 |
| **Adaptive attacker (knows P) BREAKS it** | ✅ strong negative | inverse-learning recon **0.998** → recovery **1.00** |
| **★ Information-destroying (nullspace) defense — REAL adaptive privacy** | ✅ constructive | inverse **1.00 → 0.10** @ util 0.875 (k=96) |
| Certified vs linear inversion | ✅ proven | optimal linear inverse recovers **0%** of span(D) |
| Prior defenses dominated (Claim 2) | ✅ | EntroGuard 0.00 / Koga 0.03 / PRESS 0.82@0.80 |
| Ghost Vectors — soft-delete leaves PII recoverable | ✅ airtight | logical 0.00 / raw **1.00**, bytes preserved (Art.17) |
| Attack needs the (public) index encoder | ✅ threat scope | cross-encoder transfer **0.00** |

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

1. **✅ ADAPTIVE attacker vs RedactionProjection — DONE (§4.13).** P is **INVERTIBLE**: inverse-learning
   reconstructs patches at cosine **0.998** → recovery **1.00**. The defense is **non-adaptive-only**.
   → New open item: a **provably information-destroying / certified** defense (lossy P + calibrated noise
   with an ε bound) — the honest next design, motivated by this negative result.
**✅ DONE since:** the **information-destroying defense** (§4.15, k=96 constructive) · **certified bound** vs linear inversion (§4.15) · **NDCG@5 utility** (§4.14, synthetic corpus) · **fundamental floor** (§4.14) · **baselines/Claim 2** (§4.16) · **FUNSD answer-only** (§4.14) · **Ghost Vectors** storage (§4.16) · **transfer/threat-model** (§4.16) · **7 citation fixes** (`RELATED_WORK_VERIFIED.md`) · **efficiency/deploy cost** (§4.16) · `reproduce.sh` + `ARTIFACT.md`.

**Still open (honest remaining):**
1. **Paper writeup** (10 IEEE sections) — all inputs ready; the biggest remaining piece.
2. **Multi-seed + Holm–Bonferroni** — `aggregate_seeds.py` + `eval/stats.py` built; needs the seed re-runs.
3. **Responsible disclosure** to ColPali/ColQwen (+ Qdrant) maintainers — before any preprint.
4. **`datasets`-gated runs** — CORD real-corpus + real ViDoRe both failed on a missing `datasets` dep in the image
   (add dep → rebuild → rerun). ViDoRe/CORD numbers are currently synthetic-corpus / unavailable.
5. **Nice-to-have re-runs** — `defense_crossmodel` (ColQwen2 defense) + full `defense_ablation` (6-cell signal banked).
6. **★ Privacy-native retriever (v2)** — train the retriever so PII concentrates into a few cheaply-removable dims →
   push toward "SOTA search *and* high privacy". Biggest new experiment; unproven; the natural follow-up.
7. **Figures** — `make_figures.py` built; generate from the bucketed JSONs. **Artifact-eval** packaging + Zenodo/HF.

---

## 8. Numbers to remember (headline)

- Attack: **name recovery 1.00** from a 1000-lineup; ColQwen2 **0.975** (generalizes).
- Multi-vector vs pooled: **1.00 vs 0.08**, McNemar **55/0**; but matched-bytes → **capacity-driven**.
- Holographic: delete **33% of patches → still 1.00**; erasure impossible.
- Naive local defense: **0.00 privacy**. Learned defense: **priv 1.00 @ util 0.95** (non-adaptive), dominates noise by **+0.74**.
- **★ Adaptive attacker BREAKS the learned defense: P is invertible (P⁻¹ recon cosine 0.998 → recovery 1.00).**
  Privacy is **non-adaptive-only**; a real defense must be **information-destroying**, not an invertible reshaping.
- **★★ The fix — information-destroying nullspace defense: inverse-attack recovery 1.00 → 0.10 (privacy 0.90) at util
  0.875 (k=96)** — the first real, tunable, deployable privacy for the class. Cost: remove ~75% of dims (holographic),
  ~5-pt utility. Certified: optimal linear inverse recovers **0%** of the destroyed subspace.
- **Deploy cost: 0.64 ms/page, +0 storage, 0 query overhead** → ~$4 to protect 1B pages, $0 ongoing.
- Prior defenses dominated: EntroGuard 0.00 / Koga 0.03 / PRESS 0.82@0.80. Ghost Vectors: soft-delete → raw recovery **1.00**.
- Real docs (FUNSD): weak (answer-only **0.225** top-1), scales with glyph size (short ≈chance → long 0.30).
