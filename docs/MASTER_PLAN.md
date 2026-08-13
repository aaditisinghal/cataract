# PatchGuard — Master Plan (0 → 100)

> Execution roadmap for *The Persistence of Vision: State-of-the-Art Privacy for Multi-Vector VLM
> Retrievers* — see the [paper](../paper/PersistenceOfVision.pdf) and [README](../README.md) for the
> finished result this plan built toward.

**v1.0** · the single execution roadmap. Read this first; it points to the other two docs.

- **Science / what's true:** [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md)
- **Compute / where it runs:** [`GCP_COMPUTE_PLAN.md`](GCP_COMPUTE_PLAN.md)
- **This doc / the order of operations:** what to build, when, and the gate that says "keep going or stop."

> **The one rule that overrides everything:** build the ruler before the thing you measure, and prove
> the premise before you build. Order is: *verify citations → build eval harness → dumb attack →
> KILL TEST → everything else.* Nothing in Stages 4–10 starts until the Stage 3 gate passes.

---

## The arc on one screen (0 → 100)

| % | Stage | One-line goal | Gate to pass before moving on |
|---|---|---|---|
| 0–5 | **S0 Premise** | Verify the related-work gap is real | Every cited paper exists & says what we claim; intersection empty |
| 5–10 | **S1 Ground** | GCP project + repo skeleton + determinism | Vertex smoke job runs; CI green; `repro.py` stamps fingerprints |
| 10–20 | **S2 Retrievers** | One clean interface over ColPali/BiPali/… | 4 backends encode a page; BiPali confirmed available; golden-file test |
| 20–30 | **S3 Alignment** | Map field boxes → patches, correctly | Visual validator: patches sit on fields for 50 docs/dataset, by eye |
| 30–35 | **S4 Eval harness** | PFRR + utility + frontier plotting | Metrics run on synthetic fake results; two OCR engines wired |
| 35–45 | **S5 Attack v0** | Dumbest decoder that could work | OCR-readable text on held-out FUNSD |
| **45** | **★ KILL TEST** | ColPali vs BiPali, one frontier plot + 4a probe | **≥15pp PFRR delta AND frontier-AUC>0 at 95%.** Fail → retitle/stop |
| 45–60 | **S6 Attack at scale** | Diffusion + training-free + cross-corpus | IDNet legible; DocLayNet→IDNet transfer works (the headline) |
| 60–70 | **S7 Baselines** | 5 defenses, equal tuning budget | All ported, public W&B sweep, adaptations documented |
| 70–80 | **S8 Defense** | Patch-scoped perturbation + 3 localizers | Dominates frontier; oracle→deployable gap reported |
| 80–85 | **S9 Adaptive** | Attacker who knows the defense | denoise-then-invert + inpaint, in main body |
| 85–90 | **S10 Erasure** | Full 4a/4b + storage-layer test | Erasure radius + utility cost curve; Qdrant residual check |
| 90–100 | **S11 Ship** | Write, package, disclose, submit | `reproduce.sh` regenerates every table on a clean machine |

---

## The gates (the only places the project can change shape)

1. **S0 premise gate.** If a "closest precursor" (e.g. Ghost Vectors) doesn't exist as described, or a
   real neighbor was missed → re-frame novelty *before* writing code. Cheapest possible pivot.
2. **★ KILL TEST gate (45%).** The whole project's go/no-go.
   - *Both thresholds pass* → title earned, execute S6–S11.
   - *Delta passes, frontier doesn't* → **measurement paper**: keep attack, demote defense to
     preliminary, retitle, aim NeurIPS D&B.
   - *Delta fails* → architecture story is wrong. **Stop.** Reassess before spending more.
3. **S6 legibility gate.** If IDNet reconstructions are blurry not OCR-readable → **retitle in month 2**,
   not month 6. PFRR is the arbiter, not SSIM.
4. **S9 adaptive gate.** If inpaint attack defeats the defense → that's not failure, it's **Claim 4a
   arriving early**; the defense then needs dilation (S10). A better paper, not a worse one.

---

## Stage detail — objective · tasks · deliverable · gate · GCP

### S0 — Premise (0–5%) · week 0–1 · **no code**
- **Objective:** prove the paper's premise (empty four-literature intersection) before investing.
- **Tasks:**
  1. For every arXiv ID in `RESEARCH_PROTOCOL.md §10`, confirm it exists and says what we attribute
     to it. Split into *confident-real* (spot-check) and *must-verify* (Ghost Vectors, LeakyCLIP,
     TrustCLIP, EntroGuard, PRESS, Koga, PRAG, P²RAG, Implicit Inversion, NinjaDesc, GEIA).
  2. For each *must-verify*: real? correct claim? does it already do what we claim is novel?
  3. Write `docs/RELATED_WORK_VERIFIED.md`: per-paper {exists, correct-ID, what-it-actually-does,
     overlap-with-us, verdict}.
  4. Decide: intersection still empty? If not, adjust framing in the protocol.
- **Deliverable:** `RELATED_WORK_VERIFIED.md` + a go/adjust decision.
- **Gate:** every relied-upon citation verified; novelty statement survives.
- **GCP:** none.

### S1 — Ground (5–10%) · ~3 days
- **Objective:** a deterministic, CI-guarded skeleton and a proven GCP path — before science.
- **Tasks (code):**
  - `patchguard/repro.py` — `seed_everything()` (+ `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
    `use_deterministic_algorithms(warn_only=True)`, `cudnn.benchmark=False`) and `run_fingerprint()`
    (git sha, dirty flag, torch/CUDA/GPU). Imported everywhere; dirty → results provisional.
  - `pyproject.toml` (uv, committed lockfile) · `Dockerfile` (CUDA/torch pinned **by digest**) ·
    `Makefile` (`test|kill-test|paper`) · `configs/` (hydra) · `tests/` (4-page fixture, <60s CPU).
  - GitHub Actions: lint + `mypy --strict patchguard/` + CPU tests on every PR.
- **Tasks (GCP — see compute plan §1):** create `patchguard` project; request L4×4 + A100×2 quota
  (Compute **and** Vertex buckets); `gcloud auth application-default login`; run a 1-GPU Vertex
  smoke job; create `patchguard-data` / `patchguard-artifacts` buckets; budget alerts.
- **Deliverable:** green CI, a Vertex job that ran on a GPU and wrote a file to the bucket.
- **Gate:** smoke job succeeded under keyless ADC; fingerprint stamping works.

### S2 — Retrievers (10–20%) · ~3 days
- **Objective:** one interface over the whole ColPali family; everything downstream depends on it.
- **Tasks:**
  - `patchguard/retrievers/base.py` — `PageEncoding{patches:(n,d), grid, input_size, model_id}` +
    `Retriever` protocol (`encode_page`, `encode_query`, `score`=MaxSim).
  - Backends: `colpali.py`, `colqwen3.py`, `colmodernvbert.py` (dev default, 250M, ~$0 GPU),
    `bipali.py`. **BiPali is not optional — it's Claim 1.** If absent from the release, reconstruct:
    same backbone, mean-pool patches, retrain projection head (budget 3 days). Confirm *now*.
  - Golden-file test: encode a fixed page, assert patch-tensor hash == committed reference.
- **Deliverable:** 4 backends behind one protocol; golden test in CI.
- **Gate:** BiPali confirmed usable; hashes stable.

### S3 — Alignment (20–30%) · ~4 days · **the #1 silent risk**
- **Objective:** map dataset boxes (original px) → patch indices through the model's *exact* transform.
- **Tasks:**
  - `patchguard/data/align.py::boxes_to_patch_mask(boxes, orig_size, enc, resize_policy,
    coverage_threshold=0.0) -> (n_patches,) bool`. Handle: aspect-ratio policy (letterbox vs squash —
    **read the actual `image_processor` config, don't assume**), partial coverage (state the
    threshold; appendix sensitivity 0.0/0.25/0.5), grid row/col order + prepended special tokens.
  - `experiments/validate_alignment.py` — render page + box overlay + highlighted patch mask,
    side by side, 50 docs/dataset. **Eyeball all 50.** This gates every downstream number.
- **Deliverable:** validator images for FUNSD/DocLayNet/IDNet; alignment asserted vs preprocessor.
- **Gate:** patches visibly sit on fields. If not, stop and fix — a bug here corrupts everything.

### S4 — Eval harness (30–35%) · ~3 days · **before any real attack**
- **Objective:** build the ruler.
- **Tasks:**
  - `patchguard/eval/pfrr.py` — OCR recon → exact-match vs truth, per field type. Report **raw AND
    normalized** exact-match. `configs/ocr_normalization.yaml` (0/O, 1/l/I, 5/S, 8/B, rn/m), versioned.
    **Two OCR engines** (Tesseract + PaddleOCR or a small VLM); report both.
  - `patchguard/eval/utility.py` — NDCG@5 (ViDoRe) + Recall@1 + MRR.
  - `patchguard/eval/frontier.py` — AUC + bootstrap CIs (10k), dominance at 1%/3%/5% utility.
  - Test all three on *synthetic fake* results so plotting is correct before real data.
- **Deliverable:** metrics + frontier plotter, unit-tested on fixtures.
- **Gate:** fake-data frontier plot renders correctly; both OCR engines wired.

### S5 — Attack v0 (35–45%) · ~1 week
- **Objective:** the dumbest decoder that produces a signal in week one.
- **Tasks:**
  - `patchguard/attack/decoder.py` — `(n_patches,128) → (3,448,448)`: reshape to grid, 5× ConvTranspose
    (32→64→112→224→448). Loss = L1 + LPIPS + **text-region-weighted L1 (5–10× on field-box patches)** —
    the trick that makes text legible instead of a gray smear.
  - `patchguard/attack/optim.py` — training-free variant (freq-aware INR + Procrustes align +
    natural-image blend loss). Weaker, but it's the realistic no-corpus adversary.
  - Train on DocLayNet, eval on held-out FUNSD/IDNet.
- **Deliverable:** first reconstructions + PFRR numbers.
- **Gate (S6 legibility):** OCR recovers fields on held-out FUNSD. Blurry → retitle now.

### ★ KILL TEST (45%) · ~2 days · **stop everything else**
- **Run exactly:** 200 FUNSD docs · ColPali vs BiPali at matched bytes · Attack v0 only ·
  patch-scoped vs flat Gaussian × 6 noise levels · 5 seeds.
- **Pre-register (commit thresholds *before* running):** PFRR delta (ColPali−BiPali) **≥15pp** AND
  frontier-AUC difference **excludes 0 at 95%**.
- **Immediately after (cheap):** minimal **Claim-4a probe** — delete a field's patches, reconstruct
  from remainder on ~50 docs. De-risks the best result early.
- **Deliverable:** one frontier plot + the go/no-go decision, both fingerprinted.
- **Gate:** see [gate #2](#the-gates-the-only-places-the-project-can-change-shape). ~$30 GCP, 6 weeks in.

### S6 — Attack at scale (45–60%) · ~2 weeks
- **Objective:** the headline attack.
- **Tasks:**
  - `patchguard/attack/diffusion.py` — project patch grid into frozen SD conditioning (IP-Adapter-style)
    + LeakyCLIP refinements (adversarial fine-tune, linear embedding align, SD refinement).
  - Ship 5 attack rows: `decoder` (upper bound) · `optim` (training-free floor) · `diffusion`
    (headline) · **`joint` vs independent (Claim 1c — same capacity/budget, only cross-patch attention
    differs)** · `adaptive` (S9).
  - **Cross-corpus headline:** train decoder on public DocLayNet → attack IDNet. This answers "where
    does the attacker get training data?"
  - Real-doc validity: confirm on FUNSD/CORD/DocLayNet-finance, not just synthetic IDNet.
- **Deliverable:** attack table across variants + corpora, per-field PFRR.
- **GCP:** ~120 A100-hr on-demand + IDNet encode on L4 Spot; **cache encoded patches** to the bucket.

### S7 — Baselines (60–70%) · ~2 weeks · feels wasted, decides acceptance
- **Objective:** beat defenses that were given their *strongest* visual adaptation.
- **Tasks:** `defense/{gaussian,entroguard,press,koga,trustclip}.py`. **One config-driven sweep, equal
  budget for all methods including ours**, every run to a **public** W&B project. Document each
  visual-adaptation decision in an appendix; sanity-check a port with a baseline author if reachable.
- **Deliverable:** baseline frontier curves + public sweep link + adaptation appendix.
- **Gate:** predicted failure mechanism (no locality → degrade all 1024 to protect 40) confirmed/refuted.

### S8 — Defense (70–80%) · ~2 weeks
- **Objective:** patch-scoped perturbation that dominates the frontier.
- **Tasks:**
  - `defense/patchguard.py::PatchGuard.protect(enc, scope)` — localize PII patches, perturb only those,
    keep entitled scope. Calibrate noise to **local embedding-norm distribution**, not globally.
  - 3 localizers = 3 rows: `OracleLocalizer` (boxes, upper bound) · `DetectorLocalizer` (trained on
    DocLayNet) · `OCRNERLocalizer` (OCR+Presidio-style NER, deployable today).
  - `defense/accountant.py` — privacy budget bookkeeping.
- **Deliverable:** frontier dominance (AUC + 1/3/5% dominance), **oracle→deployable gap reported**,
  per-document variance.
- **GCP:** ~100 A100-hr.

### S9 — Adaptive (80–85%) · ~1 week · **main body, not appendix**
- **Objective:** attacker knows the defense, ε, and which patches were perturbed.
- **Tasks:** `attack/adaptive.py` — (i) denoise-then-invert (decoder trained on perturbed embeddings);
  (ii) **inpaint** (protected patches = missing, reconstruct from spatial context).
- **Deliverable:** adaptive rows on the frontier.
- **Gate:** [gate #4](#the-gates-the-only-places-the-project-can-change-shape) — inpaint success ⇒ pull
  dilation forward from S10.

### S10 — Erasure (85–90%) · ~1 week
- **Objective:** naive patch deletion doesn't erase; give operators a radius.
- **Tasks:**
  - `defense/erase.py::erase(index, subject_id) -> ErasureCertificate` — delete field patches, dilate
    by radius r, re-invert, assert PFRR < threshold, return signed cert w/ fingerprint + residual.
  - 4a full (reconstruct from remainder) · 4b (sweep dilation → residual-vs-NDCG curve).
  - **Storage-layer test:** mirror Ghost Vectors on **Qdrant** — confirm multi-vectors persist through
    deletion (else fall back to backup-leak instantiation). Validate this early if S1 left slack.
- **Deliverable:** erasure-radius curve + Qdrant residual result + certificate format.

### S11 — Ship (90–100%) · ~2 weeks
- **Objective:** reproducible, disclosed, submitted.
- **Tasks:**
  - `patchguard/api.py` (6-line public interface) · Qdrant deployment adapter (multi-vector + MaxSim).
  - `reproduce.sh` regenerates **every** table/figure from GCS `metrics.json` on a clean machine;
    `experiments/make_tables.py` (no hand-copied numbers).
  - Docker image + checkpoints on HuggingFace; artifact-eval README (USENIX template).
  - **Responsible disclosure** to ColPali/ColQwen maintainers + Qdrant/Milvus **before preprint**.
  - Write paper; submit; `gcloud projects delete patchguard` after artifact mirror is pushed.
- **Deliverable = "100":** submitted paper + released repo + passing `reproduce.sh` + disclosure sent.

---

## The complete file tree (what to code, by stage)

```
patchguard/
├── Makefile                         S1
├── Dockerfile                       S1  (pinned by digest)
├── pyproject.toml                   S1  (uv, locked)
├── reproduce.sh                     S11
├── configs/                         S1+  (hydra; one experiment = one YAML)
│   ├── ocr_normalization.yaml       S4
│   └── <experiment>.yaml            per claim
├── patchguard/
│   ├── repro.py                     S1  ← written first, imported everywhere
│   ├── retrievers/base.py           S2
│   │   ├── colpali.py colqwen3.py colmodernvbert.py bipali.py
│   ├── data/align.py                S3  ← the crux
│   ├── attack/
│   │   ├── decoder.py               S5
│   │   ├── optim.py                 S5  (training-free)
│   │   ├── diffusion.py             S6  (headline)
│   │   └── adaptive.py              S9
│   ├── defense/
│   │   ├── gaussian/entroguard/press/koga/trustclip.py   S7 (baselines)
│   │   ├── patchguard.py            S8
│   │   ├── localize.py              S8  (oracle/detector/ocrner)
│   │   ├── erase.py                 S10
│   │   └── accountant.py            S8
│   ├── eval/{pfrr,utility,frontier}.py   S4
│   └── api.py                       S11
├── experiments/
│   ├── validate_alignment.py        S3
│   ├── kill_test.py                 ★  (~30 lines: read config, call library)
│   ├── <one per claim>.py
│   └── make_tables.py               S11
└── tests/                           S1+  (4-page fixture, CPU, <60s)
```

---

## Milestone map (protocol timeline ↔ stages)

| Month | Protocol milestone | Stages | Cumulative % |
|---|---|---|---|
| Wk0 | citation verification | S0 | 5 |
| M1 | kill test | S1–S5 + ★ | 45 |
| M2 | attack at scale | S6 | 60 |
| M3 | baselines | S7 | 70 |
| M4 | defense + frontier + adaptive | S8–S9 | 85 |
| M5 | erasure + packaging | S10 (+ S11 start) | 90 |
| M6 | write, release, disclose, submit | S11 | 100 |

---

## Definition of "100" (all must hold)

- [ ] Every paper table/figure regenerated by `reproduce.sh` on a clean machine from bucketed `metrics.json`.
- [ ] No number hand-copied into LaTeX; all carry a run fingerprint; no `git_dirty` result in any table.
- [ ] Kill-test thresholds were committed *before* the run (timestamped in git).
- [ ] Adaptive adversary evaluated in the main body.
- [ ] Baselines tuned with equal budget; public W&B link in the paper.
- [ ] Real-document validity shown, not only synthetic IDNet.
- [ ] Cross-corpus (DocLayNet→IDNet) transfer reported as the headline attack.
- [ ] Erasure radius + utility cost curve + Qdrant residual result.
- [ ] Responsible disclosure sent before preprint.
- [ ] Artifact-evaluation package submitted; GCP project deletable in one command.

---

## Working conventions (for whoever executes this — future me included)

- **Update the `%` in the arc table** as stages complete; this doc is the single source of progress.
- **Every stage ends with a fingerprinted artifact in the bucket** — not a screenshot, not a terminal
  scroll. Compute is ephemeral; results are files. (See GCP plan: "screenshot then kill" is the
  anti-pattern; auto-write `metrics.json` then kill.)
- **Do not start a Stage 4–10 task before the ★ kill-test gate passes.** If tempted, re-read gate #2.
- **When a gate redirects the project** (measurement paper / stop / early-4a), edit
  `RESEARCH_PROTOCOL.md §0` with the decision and date so the pivot is recorded, not lost.
- **One config = one experiment.** A reviewer's "try variant X" should be a YAML change, not a refactor.
```
