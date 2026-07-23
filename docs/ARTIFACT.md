# Artifact Appendix — *The Persistence of Vision*

*PII Reconstruction and Patch-Scoped Privacy in Multi-Vector Vision-Language Retrieval*

USENIX Security / IEEE S&P artifact-evaluation appendix. This document is self-contained: it lets an
evaluator obtain the image + data, reproduce every headline number, and check each against the paper's
claims. All result numbers below are the source-of-truth values recorded in
[`docs/RESULTS.md`](RESULTS.md); the reproduction regenerates them into fingerprinted JSON.

---

## A. Abstract

The stored **multi-vector** patch embeddings of visual-document retrievers (ColPali, ColQwen2) are a
**field-level PII oracle**: an attacker with read access to the index recovers names/IDs/dates by ranking
candidate values with the model's own MaxSim — no image reconstruction. The leak is **holographically
distributed** (deleting a third of all patches still recovers PII), **capacity-driven** (it comes from
storing many per-patch vectors — the same property that gives SOTA retrieval), and **generalizes across
backbones**. Consequently PII **cannot be spatially redacted, erased, or locally defended** — only
globally scrubbed. A learned anisotropic index-time projection suppresses the non-adaptive attacker while
preserving retrieval and dominating flat noise, **but an adaptive attacker who knows the (public) defense
inverts it** (recon cosine 0.998 → recovery 1.00); only an **information-destroying (rank-deficient /
nullspace) transform** survives that adaptive attack.

The artifact is a single Docker image plus a set of `experiments.*` modules and a driver
(`reproduce.sh`). Each experiment writes a fingerprinted `*.json` to a GCS bucket; `make_figures.py`
renders the paper's seven figures from those JSONs. Everything is CPU-importable (`--help` works
anywhere); the numbers require one NVIDIA A100 (or L4) for the ColPali/ColQwen2 forward passes.

**Artifacts available / functional / reproduced** — we claim all three badges: the image + code + result
JSONs are archived (Zenodo/HF, §C), each module runs end-to-end from `--help` to fingerprinted JSON, and
`reproduce.sh` regenerates every table/figure in the paper.

---

## B. Description & requirements

### B.1 Security / privacy / ethics
All PII in the synthetic experiments is **generated** (`patchguard/data/synthdoc.py`, a closed 240-name
vocabulary + random IDs/dates) — no real personal data. FUNSD/CORD are public research corpora used
read-only. The attack recovers PII from an *index the attacker already has read access to*; it does not
break confidentiality of the source images. Responsible disclosure to the ColPali/ColQwen and
Qdrant/Milvus maintainers precedes any preprint (COMPLETION_PLAN D6).

### B.2 How to access
- **Code:** this repository (tag the AE commit; the repro image is built from it).
- **Repro image:** `us-central1-docker.pkg.dev/patchguard-reakon/patchguard/repro:<git-sha>`
  (Artifact Registry). The headline results in `docs/RESULTS.md` were produced with the pinned image
  **`repro:8cb6b45`** (adaptive-attack commit; recorded in `.last_image`). A fresh build re-tags with the
  current short git sha via `scripts/10_build_image.sh` so every run pins the exact code that produced it.
- **Archival copy:** image `.tar` + result JSONs + figures are bundled by
  `scripts/package_artifact.sh` into `dist/patchguard-artifact-<sha>.tar.gz` and uploaded to Zenodo/HF
  (§C).

### B.3 Hardware dependencies
- **1× NVIDIA A100 40 GB** (`a2-highgpu-1g`) — the profile used for every headline number. An **NVIDIA
  L4** (`g2-standard-8`) also works and is cheaper; select it with the `l4` profile argument.
- ~150 GB boot disk (image + HF cache). No multi-GPU, no special interconnect.
- The two CPU-only steps (`efficiency_bench`, `make_figures`) run on any workstation — no GPU.

### B.4 Software dependencies
Everything is baked into the repro image; an evaluator needs **only** Docker + an NVIDIA driver (the GCE
`common-cu129` deep-learning VM image provides the driver, `scripts/*_vm.sh` install the container
toolkit). Image internals (pinned for the reproducibility claim):

| Component | Version | Note |
|---|---|---|
| Base image | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` | Python **3.10**, torch **2.3.1**, CUDA 12.1 |
| `colpali-engine` | **0.3.5** | pins transformers 4.46.x (transformers 5.x breaks on torch 2.3.1) |
| transformers / peft | resolved by colpali-engine | do **not** pin separately |
| pillow, google-cloud-storage, pytesseract, accelerate, qwen-vl-utils | latest compatible | image IO / GCS IO / OCR / ColQwen2 |
| diffusers | 0.31.0 | dead-end generative line only (not on the reproduction path) |
| numpy | 1.26 | the one hard dep of the `patchguard` package itself |

Determinism: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `seed_everything(seed)`, `np.random.default_rng(seed)`;
`repro.py` stamps the git sha (build-time `_buildinfo.py`), dirty flag, torch/CUDA/GPU into every JSON's
`"fingerprint"`.

### B.5 Benchmarks / data
- **FUNSD** (199 real forms, ~50 MB) — staged to `gs://patchguard-reakon-data/funsd/{training_data,testing_data}`
  by `scripts/03_prep_funsd.sh` (downloads the public zip, uploads). Referenced by experiments as
  `--data gs://patchguard-reakon-data/funsd`.
- **CORD** (~1k receipts, financial PII) — pulled from **HuggingFace** (`naver-clova-ix/cord-v2`) by the
  `realcorpus_transfer` loader and placed on the shared glyph-height axis.
- **Synthetic ID cards** — generated on the fly (`patchguard/data/synthdoc.py`), no download.
- **Models** — `vidore/colpali-v1.3`, `vidore/colqwen2-v1.0` from HuggingFace; the warm VM pre-caches
  ColPali into `/opt/hfcache` so subsequent runs skip the download.

---

## C. Obtaining the image, data, and archival bundle

```bash
# 0) auth (keyless ADC; the GCP org disables API/SA keys — see MEMORY)
gcloud auth login && gcloud auth application-default login
gcloud config set project patchguard-reakon

# 1) build + push the repro image (tags with the current short git sha, writes .last_image)
scripts/10_build_image.sh

# 2) stage FUNSD into the data bucket (CORD is pulled from HF by the loader at run time)
scripts/03_prep_funsd.sh

# 3) (evaluators) fetch the archived artifact bundle instead of rebuilding:
#    gcloud storage cp gs://patchguard-reakon-artifacts/dist/patchguard-artifact-<sha>.tar.gz .
#    tar xzf patchguard-artifact-<sha>.tar.gz    # -> ARTIFACT.md, reproduce.sh, results/*.json, figures/*
#    Assemble/refresh that bundle yourself with:  scripts/package_artifact.sh
```

The Zenodo/HuggingFace upload of the bundle is documented (as commented commands) at the tail of
`scripts/package_artifact.sh`.

---

## D. Claim → experiment → expected-result map

Each **paper claim** maps to exactly one experiment **module** and a **headline number** (source:
`docs/RESULTS.md` §4–§5). Chance is noted where relevant. Run any single row with the warm-VM one-liner
in §E.2. "PHASE-3" rows are the newly wired completion experiments; their numbers are directional
expectations (not yet in RESULTS.md) — the reproduction produces the real values.

### D.1 Attack / findings

| # | Claim | Module | Headline (expected) |
|---|---|---|---|
| C1 | The retrieval/dictionary attack recovers PII with no reconstruction | `experiments.retrieval_attack` | name **1.00**, id_no 0.725, dob 0.40 top-1 (1000-lineup, chance 0.001) |
| C2 | Multi-vector leaks, pooled (BiPali) does not | `experiments.claim1` | **1.00 vs 0.08**; McNemar **55/0** (n=60, chance 0.005) |
| C3 | …the advantage is **capacity-driven**, not late-interaction magic | `experiments.claim1b` | K=1 (128 fl) **0.10 < BiPali 0.20** at matched bytes |
| C4 | The attack is genuine (not an artifact) | `experiments.control_wrongpage` | correct 1.00 / **wrong-page 0.000** (n=50) |
| C5 | Holographic bleed: PII is non-local | `experiments.erasure` | delete **33 % of patches → still 1.00** |
| C6 | Erasure by patch deletion is impossible (Article 17) | `experiments.erasure` | recovery **1.00 at every dilation radius** |
| C7 | Verb validation — order preserved, not bag-of-tokens | `experiments.property_curve` | name beats own char-permutations **1.00** |
| C8 | No simple glyph-height threshold on clean docs | `experiments.property_curve` | name **flat 1.00 from 6→37 px** |
| C9 | id/dob transposition closed by positional rerank (A4) | `experiments.retrieval_rerank` | exact id/dob top-1 rises toward name's 1.00 |
| C10 | Arrangement adds nothing beyond content (A5, a null) | `experiments.arrangement_control` | **NULL** — recovery unchanged under patch shuffle |
| C11 | Generalizes across backbones (ColQwen2) | `experiments.cross_model` | leak **0.975**, bleed 90 %→0.83 (n=40) |
| C12 | Real-document transfer (FUNSD) | `experiments.funsd_transfer` | top-1 **0.19**, ≤10 px 0.11 → >16 px 0.46; `--labels answer` for the fair number (A1) |
| **P3-A3** | Real-corpus transfer to CORD receipts (financial PII) | `experiments.realcorpus_transfer` | above chance, **scales with glyph size**; financial fields recover better than dense FUNSD |
| **P3-A6** | Ghost vectors — soft-deleted multi-vectors stay attackable | `experiments.ghost_vectors` | soft-deleted vectors **physically persist + recover** |
| **P3-A7** | Transfer attack with a proxy (different) encoder | `experiments.transfer_attack` | same-encoder near-perfect; **cross-encoder degrades but partially transfers** |

### D.2 Defense

| # | Claim | Module | Headline (expected) |
|---|---|---|---|
| C13 | Local (patch-scoped) defense is impossible | `experiments.defense_frontier` | patch-scoped privacy **0.00 at any noise**; flat noise craters utility |
| C14 | Learned global anisotropic P beats flat noise (**non-adaptive only**) | `experiments.learned_defense` | priv **1.00 @ util 0.95**; dominance **+0.74** @ priv 0.9 |
| C15 | **Adaptive attacker who knows P breaks it** | `experiments.adaptive_attack` | inverse-learning recon **0.998 → recovery 1.00**; P is invertible |
| C16 | Baselines (EntroGuard/PRESS/Koga) reduce to noise; P dominates (B5) | `experiments.baseline_frontier` | P frontier dominates the ported baselines |
| C17 | The fundamental floor where PII **is** the retrieval target (B3) | `experiments.defense_floor` | frontier collapses to the leak in the find-by-name regime |
| C18 | Cross-model defense — P on ColQwen2 (B4) | `experiments.defense_crossmodel` | P works on ColQwen2 (same subspace-separability) |
| C19 | Architecture ablation of P (B7) | `experiments.defense_ablation` | linear ≈ residual-MLP (subspaces ~linear) |
| C20 | Real-retrieval utility — ViDoRe NDCG@5 (B2) | `experiments.vidore_utility` | small NDCG@5 drop where PII is incidental; gap vs noise persists |
| C21 | Deploy cost of P is negligible (C1) | `experiments.efficiency_bench` | **0.62 µs/patch (0.64 ms/page), +0 B stored, 0 query overhead** |
| **P3-B6a** | Constructive answer: **information-destroying (nullspace) P survives the inverse attack** | `experiments.certified_defense` | a rank **k** where **inverse recovery collapses** while topic utility ≥ 0.8, unlike the residual P (§4.13) |
| **P3-B6b** | A **provable (ε) privacy bound** on the nullspace defense | `experiments.certified_bound` | certified privacy floor vs k, with its utility cost |
| **P3-B8** | Real-doc defense transfer — nullspace P (synthetic) on real FUNSD | `experiments.defense_transfer_funsd` | partial→full protection transfer to real FUNSD field text |

### D.3 Statistics
Headline families (`retrieval_attack`, `claim1`, `control_wrongpage`, `erasure`, `cross_model`,
`learned_defense`) run at **5 seeds**; `experiments.aggregate_seeds` reduces each family to mean ± 95 %
bootstrap CI and applies **Holm–Bonferroni** across the primary claim p-values (COMPLETION_PLAN A2). Most
headline cells are 1.00 / 0.00 with tight CIs.

---

## E. Reproduction

`reproduce.sh` regenerates **every** JSON in §D and then all figures, in dependency order. Two run paths:
a **warm VM** (pay the cold start once, ~3.5 min/experiment) or **ephemeral one-shot** VMs (self-deleting,
one per job). Both use the same image and write to `gs://patchguard-reakon-artifacts/runs/`.

### E.1 Full reproduction — warm-VM fast path (recommended)

```bash
# bring the warm A100 up ONCE (installs docker+toolkit, pulls the image, pre-caches ColPali ~15 min)
scripts/20_warm_vm.sh us-central1-a a100        # writes .warm_vm / .warm_zone

# regenerate EVERY table + figure (headline experiments at 5 seeds; ~1-2 h wall on the warm VM)
bash reproduce.sh

# tear the warm VM down when finished (it does NOT self-delete)
gcloud compute instances delete "$(cat .warm_vm)" --zone="$(cat .warm_zone)" --quiet
```

`reproduce.sh` runs Phase 1 (attack), Phase 2 (defense), the multi-seed aggregation, then the
CPU-local `efficiency_bench` + `make_figures`, and finally **Phase 3** (the newly wired experiments:
`certified_defense`, `realcorpus_transfer`, `ghost_vectors`, `certified_bound`, `defense_transfer_funsd`,
`transfer_attack`) followed by a figure refresh so `paper/figures/*.{pdf,png}` include them.

Override the seed set for a quick single-seed smoke of the headline experiments:

```bash
SEEDS=0 bash reproduce.sh
```

### E.2 One experiment at a time — warm VM

`scripts/21_exec.sh <module> <out_subdir> [args…]` SSHes into the warm VM, runs the module in the image,
and uploads to `gs://patchguard-reakon-artifacts/runs/<out_subdir>-<image_tag>/<name>.json`:

```bash
# a claim (C2): multi-vector vs pooled
scripts/21_exec.sh experiments.claim1                 claim1          --n 60 --distractors 200 --font-size 24
# the decisive adaptive test (C15)
scripts/21_exec.sh experiments.adaptive_attack        adaptive        --lam 5.0 --n-train 64 --n-test 40 --epochs 300
# the constructive nullspace defense (P3-B6a)
scripts/21_exec.sh experiments.certified_defense      certified       --ks 0,2,4,8,16,32 --n-train 64 --n-test 40
# real-corpus CORD transfer (P3-A3)
scripts/21_exec.sh experiments.realcorpus_transfer    realcorpus-cord --corpus cord --n-pages 60 --k 20

# read a result back:
gcloud storage cat gs://patchguard-reakon-artifacts/runs/claim1-*/claim1.json
```

### E.3 Ephemeral one-shot alternative (no warm VM)

Each `<job>` provisions a GPU VM that runs the container and **self-deletes** on completion (45-min hard
cap, on-demand so no preemption; worst-case ~$0.65 L4 / ~$1 A100):

```bash
scripts/12_train_vm.sh us-central1-a a100 claim1        # -> runs/claim1-<tag>/claim1.json
scripts/12_train_vm.sh us-central1-a a100 retrieval
scripts/12_train_vm.sh us-central1-a a100 defense2      # learned_defense
# jobs: retrieval claim1 claim1b wrongpage erasure crossmodel defense defense2 funsd curve killtest …
```

### E.4 CPU-only steps (no GPU, run anywhere)

```bash
python3 -m experiments.efficiency_bench --out results/efficiency_bench --n-patches 1030 --iters 200
python3 -m experiments.make_figures --runs gs://patchguard-reakon-artifacts/runs --figdir paper/figures
```

`make_figures` renders the seven paper figures (`fig1`…`fig7`) into `paper/figures/` as PDF (vector) +
PNG (preview); each figure skips with a warning if its input JSON is absent, so a partial run still
produces what it can. Any `--help` and both CPU steps run without colpali/torch-cuda, so an evaluator can
sanity-check the harness on a laptop before spending GPU time.

---

## F. Notes / limitations for the evaluator
- **Cost.** The entire result set was produced for **well under $15** (many $0.5–1 auto-torn-down runs).
- **Determinism.** GPU MaxSim is scoped `warn_only`; headline cells are 1.00/0.00 and reproduce exactly.
  Trust the `"fingerprint"` block (git sha, `git_dirty`, torch/CUDA/GPU) in each JSON over wall-clock time.
- **Honest scope.** The learned defense (C14) is **non-adaptive-only** — C15 breaks it; the constructive
  fix is the information-destroying nullspace defense (P3-B6a/b). Real-doc transfer (C12, P3-A3) is
  **weak on dense small-font forms** and rises with glyph size — that is the paper's honest scope, not a
  bug.
- **File map.** `docs/RESULTS.md` = the numbers; `docs/COMPLETION_PLAN.md` = the item↔experiment
  rationale (A/B/C/D IDs referenced above); `reproduce.sh` = the driver; `experiments/` = one module per
  claim.
