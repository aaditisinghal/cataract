# Your Embeddings Are Photographs

**Research protocol — v1.1 (revised)**

> Multi-vector visual-document embeddings (ColPali-family patch grids) are invertible to field-level PII. The *same spatial locality* that makes them leak is also the primitive for a defense and the reason naive erasure fails. This is the paper.

- **Target venues:** USENIX Security (primary) · IEEE S&P (secondary) · NeurIPS D&B (fallback / if it becomes a measurement+benchmark paper)
- **Status:** planning. Nothing built. First artifact to exist is the *kill-test frontier plot*.
- **What changed from v1.0:** see [§0 Revision notes](#0-revision-notes--what-changed-and-why). The science is the same; the *narrative spine, sequencing, and three risk calls* changed.

---

## 0. Revision notes — what changed and why

These are the edits folded into this version. They do not add scope; they re-order and re-frame so the paper survives review.

1. **Narrative re-centered on Claim 1c + Claim 3 + Claim 4 (the spine).**
   "Multi-vector leaks more than pooled" (Claim 1, headline reading) is nearly information-theoretic — you retain ~1,000× more floats per page, so "of course it leaks more." A reviewer fixates on that trivial reading and misses the point. The **non-obvious, load-bearing** contribution is:
   - **1c** — spatial *arrangement* leaks structured, *localizable* PII beyond raw content;
   - **3** — that same locality is a defense primitive pooled embeddings *cannot* have (patch-scoped noise);
   - **4** — and the VLM's cross-patch contextualization means deleting a field's patches does **not** erase it.
   Claim 1 (1a/1b) is demoted to *scaffolding* that isolates architecture from capacity. It stays in the paper; it does not lead the intro.

2. **A minimal Claim-4a probe is pulled forward to right after the kill test.**
   Erasure-via-contextualization-bleed is the single most citable, most self-contained, most legally-sharp result (Article 17 / EDPB). In v1.0 it lived at month 5 and would be the first thing cut under time pressure — i.e. the best result was the most at-risk. It is cheap once the attack + alignment exist, so a minimal version runs early to de-risk differentiation.

3. **Related-work map is a *week-1 blocking task*, not a pre-submission cleanup.**
   The novelty claim is "the four literatures don't intersect." That claim is only as real as the citations under it. Several arXiv IDs in the source plan are unverified and a few pattern-match to hallucinated / misremembered references. **Verify every ID against arXiv before trusting the gap** — if a "closest precursor" doesn't exist, the framing changes; if a real neighbor was missed, the novelty claim is wrong. See [§10](#10-reading-order--citation-verification-week-1-blocking).

4. **Determinism claim scoped down.** `torch.use_deterministic_algorithms(True)` *raises* on several ops in the diffusion attack; bitwise-reproducible diffusion training is not realistic on GPU. The reproducibility claim is **"stable across seeds within reported CI,"** not "bitwise identical." See [§7](#7-infrastructure).

5. **IDNet reframed as a validity axis, not just "primary corpus."** Synthetic, templated IDs may be *unnaturally easy* to invert (clean fonts, fixed layouts), inflating the headline. Real documents (FUNSD / CORD / DocLayNet-finance) carry the "does it hold on real docs" story, which is stated loudly.

6. **Headline attack number = cross-corpus transfer.** The realistic adversary has the stolen index but no matched training corpus. The *trained-decoder-on-matched-data* number is an upper bound; the **decoder trained on public DocLayNet that then attacks IDNet** is the headline robustness claim, because it answers "where does the attacker get training data?"

7. **Patch↔field alignment elevated to the #1 silent risk** (above BiPali availability). An off-by-one in grid mapping corrupts *every* defense number invisibly. Its visual validator gates all downstream work.

8. **Compute budget padded** from ~$500 to a planning figure of **~$1.5–2k** so a slip doesn't force a scientific compromise. $500 remains the optimistic floor.

9. **S0 premise gate PASSED (2026-07-22) → GO.** All four literatures verified real against arXiv; the intersection is empty; the closest precursor (Ghost Vectors) is real but text-only/single-vector, so we are not scooped. See [`RELATED_WORK_VERIFIED.md`](RELATED_WORK_VERIFIED.md) for the full table and a **7-item fix-before-writing list** (LeakyCLIP SSIM 258% not 358%; Vec2Text "RQ4" belongs to the reproducibility study; EntroGuard is "entropy-driven" + drop the ε≈0.036 figure; PRESS has no arXiv ID; **EDPB 05/2019 mischaracterized — re-ground Claim 4's legal hook in Article 17 itself**; IDNet is 10 US states + 10 EU countries via SD2.0 inpainting of real IDs, not DMV templates; TrustCLIP's patch-token claim needs an in-body check).

---

## 1. Contribution, in one paragraph

We show that the per-patch multi-vector embeddings stored by ColPali-family visual document retrievers are invertible to **field-level PII** — not merely perceptually similar images — under a storage-only threat model (read access to the vector index; no source documents, no retriever weights, no query logs). We isolate that the leak is driven by **spatial arrangement**, not just retained capacity (1c); demonstrate that existing embedding-privacy defenses fail because they have **no locality primitive** (Claim 2); introduce **patch-scoped perturbation** that dominates the privacy–utility frontier by spending noise only where PII lives (Claim 3); and prove that naive patch deletion **does not erase**, because the VLM contextualizes each patch within the whole document — information bleeds outward (Claim 4). We release a reproducible framework, a new **PII Field Recovery Rate (PFRR)** metric, and a deployable Qdrant adapter.

---

## 2. Threat model

- **Capability:** read access to the stored patch-embedding index. **Not** source documents, **not** retriever weights (test both white- and black-box), **not** query logs.
- **Instantiations:** curious cloud vector-DB provider; misconfigured / publicly-exposed index; insider with storage-layer access only; residual data after soft delete; backup leak.
- **Goal:** field-level PII extraction (name, ID number, account number, DOB, address), **not** perceptual similarity.
- **Headline adversary:** has the index + a *public* document corpus to train a decoder (DocLayNet), and attacks a *different* domain (IDNet). Cross-corpus transfer is the realistic story.
- **Upper-bound adversary:** matched training corpus (same-domain labelled images). Reported as a ceiling.
- **Training-free adversary:** no corpus at all (implicit-inversion / optimization variant). Reported as the realistic floor.
- **Explicitly out of scope** (name it to pre-empt "why didn't you also"): prompt injection, poisoning, membership inference, and any attack needing live query access.

---

## 3. Claims and experiments

### The spine: 1c → 3 → 4. Claim 1(a/b) is scaffolding; Claim 2 is necessary scaffolding.

### Claim 1 — architecture vs. capacity (scaffolding, do not lead with it)
- **1a — ColPali vs. BiPali.** Same base model, same training data. Any delta is *architecture* (late-interaction multi-vector vs. pooled bi-encoder).
- **1b — matched bytes-per-page.** Sweep BiPali dimensionality up / ColPali patches down to equalize stored bytes. Kills "you just gave it more capacity." *Known caveat:* extreme-dim BiPali is out-of-distribution and its retrieval may degrade — report that honestly rather than pretending the control is clean.
- **1c — joint vs. independent patch inversion (the real result).** Invert patches independently, then jointly (cross-patch attention). The gap quantifies how much *arrangement* leaks beyond content — the mechanism behind the title. Same decoder capacity, same training budget; only cross-patch modeling differs.

### Claim 2 — existing defenses don't transfer
Port EntroGuard, PRESS, Koga (token-selective budget), flat Gaussian, TrustCLIP — **equal automated tuning budget for every method, ours included** (one config-driven sweep, all runs logged to a public W&B project). Predicted failure mechanism (falsifiable): no locality primitive, so they degrade all ~1,024 patches to protect the ~40 that hold PII. Document each visual-adaptation decision in an appendix; if reachable, have a baseline author sanity-check the port.

### Claim 3 — patch-scoped perturbation dominates
- **3a — the frontier**, swept over noise scale.
- **3b — localization ablation:** oracle boxes → trained detector → off-the-shelf OCR+NER. The **oracle→deployable gap is the honest deployment cost**; report it prominently.
- **3c — adaptive adversary who knows the defense, the ε, and which patches were perturbed. Main body, not appendix.** Non-adaptive-only evaluation is the top rejection reason at security venues. Two adaptive attacks: (i) denoise-then-invert (decoder trained on perturbed embeddings); (ii) **inpaint** (treat protected patches as missing, reconstruct from spatial context — the real threat, same mechanism as Claim 4a).

### Claim 4 — erasure (elevated; minimal probe runs early)
- **4a — delete the patches covering a field, reconstruct from the remainder.** Hypothesis: still recoverable, because the VLM contextualizes each patch within the document's overall structure — information bleeds outward. If true: **naive patch deletion does not erase.** Self-contained, novel, legally sharp. *A minimal version of this runs immediately after the kill test* (see [§8](#8-timeline)).
- **4b — sweep mask dilation**, find the erasure radius, report its utility cost. Deliverable operators want: "to erase field X, delete its patches + k neighbors, costing y NDCG points."
- **Storage-layer check:** mirror the Ghost-Vectors setup on Qdrant — confirm the DB actually *retains* multi-vectors through deletion, else the residual-data instantiation weakens (fall back to backup-leak / curious-provider). Validate this *early*, not at month 5.

---

## 4. Statistics plan (a differentiator — most security papers are weak here)

- **5 seeds per config**, fixed and reported. Every number is **mean ± 95% CI**. No single-run results.
- **Power:** two-proportion test for a 15pp difference (0.50→0.65), α=0.05, power=0.80 → **~165 docs/arm**. IDNet supplies 837k; report achieved power and note the analysis was **pre-specified**.
- **Tests:** two-proportion *z* + bootstrap CI (10k resamples) for recovery rate; **Wilcoxon signed-rank** paired by query for NDCG@5, with rank-biserial effect size.
- **Frontier dominance:** don't eyeball. **AUC with bootstrap CIs**, plus dominance at three fixed utility levels (within 1% / 3% / 5% of undefended).
- **Multiple comparisons:** Holm–Bonferroni across the primary family, stated explicitly.
- **Report per-document variance.** A defense that works on average and fails on 5% of documents isn't a defense — finding that is a contribution.
- **Pre-register frontier thresholds** in the repo with a commit timestamp *before* running the adaptive attack.

---

## 5. Metrics

- **Primary — PII Field Recovery Rate (PFRR).** OCR the reconstruction, exact-match against ground truth after normalization, reported per field type. *SSIM measures whether an image looks similar; it doesn't measure whether the attacker got the account number.* Defining this metric well is itself a contribution.
  - Report **both raw exact-match and normalized exact-match**, everywhere. Freeze the OCR-confusion table (`0/O`, `1/l/I`, `5/S`, `8/B`, `rn/m`) in `configs/ocr_normalization.yaml`, versioned. Argue the normalized number (an attacker with a checksum / Luhn / DB lookup resolves ambiguity) but never hide the raw.
  - **Two independent OCR engines** (e.g. Tesseract + PaddleOCR or a small VLM); report both. A result that holds under only one OCR engine isn't a result.
- **Secondary:** field-level edit distance, SSIM/LPIPS (comparability with the vision literature), page CER/WER, signature-presence detection.
- **Utility:** NDCG@5 on ViDoRe, plus Recall@1 and MRR.
- **Efficiency:** index/query latency and storage delta. A defense tripling query time won't deploy.

---

## 6. Data — with the validity axis made explicit

| Dataset | Scale | Role |
|---|---|---|
| **IDNet** | 837,060 synthetic ID images, ~490 GB, 20 types / 10 US states | Primary attack corpus, PII fields pre-identified. **Validity caveat: synthetic + templated → possibly easier to invert; not the "does it hold on real docs" evidence.** |
| **DocLayNet** | 80,863 human-annotated pages, 11 classes, COCO boxes + per-cell text/coords | Decoder training corpus (public) → powers the **cross-corpus headline**; free patch↔field mapping; finance subset. |
| **FUNSD** | 199 annotated real forms, word-level boxes | **Month-1 kill test**; real-document validity. |
| **CORD** | ~1,000 real receipts, multi-level labels | Financial variety; real-document validity. |
| **ViDoRe** | 127,346 pages | Utility benchmark. |

**Real-vs-synthetic is a stated axis:** the headline attack must be shown to hold on *real* documents (FUNSD/CORD/DocLayNet-finance), not only on IDNet.

**Ethics asset (corrected per verification):** IDNet uses **Stable Diffusion 2.0 to inpaint away PII from real ID images** and fills **synthetic** PII (GPT-3.5-generated); coverage is **10 US states + 10 European countries**. Frame as *"final images contain only synthetic PII"* — not "no real person's data is at risk" (templates derive from real docs). Layouts stay faithful. Still a strong ethics anchor. See [`RELATED_WORK_VERIFIED.md`](RELATED_WORK_VERIFIED.md).

---

## 7. Infrastructure

```
patchguard/
├── Makefile                 # single entry: make test | make kill-test | make paper
├── Dockerfile               # pinned CUDA + torch — this IS the reproducibility claim
├── pyproject.toml           # uv, lockfile committed
├── configs/                 # hydra; one experiment = one YAML, no experiment-specific Python
├── patchguard/
│   ├── repro.py             # seed_everything + run_fingerprint — written FIRST, imported everywhere
│   ├── retrievers/base.py   # Retriever protocol: encode_page()->(n_patches,d), encode_query(), score()=MaxSim
│   ├── data/align.py        # boxes_to_patch_mask() — the crux; visual validator gates everything
│   ├── attack/              # decoder.py | optim.py (training-free) | diffusion.py | adaptive.py
│   ├── defense/             # localize | perturb | erase | accountant
│   ├── eval/                # pfrr | utility | frontier
│   └── api.py               # 6-line public interface
├── experiments/             # one script per claim, deterministic, ~30 lines each (reads a config)
├── tests/                   # fast CPU tests on a 4-page fixture, <60s, every commit
└── reproduce.sh             # one command regenerates every table & figure
```

**Non-negotiables**

- Seed torch/numpy/python/CUDA; set `CUBLAS_WORKSPACE_CONFIG=:4096:8`; `torch.backends.cudnn.benchmark=False`.
- **Determinism is scoped, not absolute.** `torch.use_deterministic_algorithms(True)` will *raise* on some ops used by the diffusion attack. Enable it where it works (retrieval, alignment, decoder eval); where it can't, fall back to `warn_only=True` and make the claim **"stable across seeds within reported CI,"** not bitwise-identical. Do **not** promise bitwise-reproducible diffusion training in the artifact.
- **Run fingerprint** (`git_sha`, `git_dirty`, torch/CUDA/GPU) stamped into every checkpoint, metric JSON, and figure. If `git_dirty`, results are **provisional** and cannot enter a paper table — enforced in CI.
- **Results as data, not logs.** Every run writes `results/{run_id}/metrics.json` with the fingerprint; `experiments/make_tables.py` generates every LaTeX table from those files. **Never hand-copy a number into LaTeX.**
- **CI from day one:** lint, `mypy --strict` on `patchguard/`, CPU tests on every PR. Nightly GPU integration test runs the kill test on 20 docs and asserts numbers haven't drifted.
- **Everything to a public W&B project** (failed runs included) — evidence there was no cherry-picking.
- **Storage:** don't download all of IDNet. Streaming loader pulls a stratified sample (balanced across 20 types); cache *encoded patch tensors*, not images.

**Deployment adapter:** ship **Qdrant** (native multi-vector storage + MaxSim; Milvus also works; pgvector needs custom query-time MaxSim).

**Compute (planning figure ~$1.5–2k; $500 is the optimistic floor):**
ColModernVBERT dev on a single consumer GPU (~$0) · diffusion-attack training on A100 (~$60+, likely 2–3× once seeds × HP-search counted) · ColPali/ColQwen3 validation (~$150) · IDNet eval (~$80) · baselines × tuning sweeps × 5 seeds (the real cost sink) · slack.

---

## 8. Timeline (kill-test-gated; erasure probe pulled forward)

- **Week 1 (blocking, before code) — verify the related-work map.** Confirm every arXiv ID exists and says what we think; confirm the four-literature intersection is actually empty. If a precursor is real, re-frame; if a "paper" is phantom, drop it. See [§10](#10-reading-order--citation-verification-week-1-blocking).
- **M1 — KILL TEST.** FUNSD, 200 docs, ColPali vs. BiPali at matched storage, Attack v0 only, patch-scoped vs. flat Gaussian × 6 noise levels, 5 seeds. **Pre-register two thresholds (commit before running):** PFRR delta (ColPali − BiPali) **≥ 15pp** *AND* frontier-AUC difference excludes zero at 95%.
  - **Immediately after (cheap):** minimal **Claim-4a probe** — delete a field's patches, reconstruct from remainder on ~50 docs. De-risks the best result early.
  - **Gate:** both pass → title earned, proceed. 1 passes / 2 fails → measurement paper, demote defense, retitle. 1 fails → architecture story wrong, **stop and reassess.**
- **M2 — attack at scale.** IDNet + three variants (decoder / training-free / diffusion) + cross-domain (train DocLayNet → attack IDNet = headline). If reconstructions are blurry rather than OCR-readable, **retitle now**, not in M6.
- **M3 — baselines.** Feels unproductive; determines acceptance. Equal-budget automated sweep, public W&B.
- **M4 — defense + frontier + adaptive attacker** (3a/3b/3c, adaptive in main body).
- **M5 — erasure (4a full / 4b dilation), storage-layer check, theory attempt, framework packaging.**
- **M6 — write, release, responsible disclosure (ColPali/ColQwen maintainers + Qdrant/Milvus before preprint), artifact-evaluation submission.**

---

## 9. Objections to pre-empt *in the paper*

- **"Reconstruction quality too low."** → PFRR framing + a figure of low-SSIM reconstructions from which OCR still recovers the field.
- **"Unrealistic adversary."** → Ghost Vectors establishes storage-layer access as live; the training-free variant needs no corpus.
- **"Just CLIP inversion on documents."** → No: architecture isolation (1a/1b), the joint-inversion result (1c), and a defense exploiting the same structure (3). Say it in the intro, not the rebuttal.
- **"Of course more vectors leak more."** → 1b (matched bytes) + 1c (arrangement beyond content) separate capacity from architecture.
- **"Does synthetic transfer to real?"** → real-document validity on FUNSD/CORD/DocLayNet-finance, stated as a first-class result.
- **"Where does the attacker get training data?"** → cross-corpus headline (public DocLayNet → IDNet) + training-free floor.
- **"Why not encrypt?"** → HE approaches exist and pay heavy performance costs; report latency.

---

## 10. Reading order + citation verification (week-1 blocking)

**⚠️ Verify every arXiv ID against arXiv directly before citing — several in the source plan came from search results and are unconfirmed; a wrong ID in a submission is an unforced error. Some 2026-dated IDs pattern-match to hallucinated references. This verification is a gate, not a courtesy.**

- **Confident-real (spot-check only):** ColPali / ColBERT · Vec2Text (note RQ4 noise-vs-recovery curve — the experimental template) · CLIP-inversion line · certified removal: Guo et al. (ICML 2020), Ginart et al. (NeurIPS 2019) · DocLayNet · FUNSD · IDNet.
- **Must verify existence and claims before relying on them:** LeakyCLIP · TrustCLIP · "Implicit Inversion" · Ghost Vectors (our *closest precursor* — if it doesn't exist as described, the framing shifts) · EntroGuard · PRESS · Koga et al. · PRAG · P²RAG · NinjaDesc · GEIA.
- **Week 1:** ColPali (read the BiPali ablation twice) → Vec2Text (RQ4) → LeakyCLIP → Ghost Vectors.
- **Week 2:** TrustCLIP → Koga et al. → Implicit Inversion → Vec2Text reproducibility study.
- **Monitor** `github.com/Arstanley/Awesome-Trustworthy-RAG` monthly for scoops.

---

## 11. Where you'll actually lose time (ranked by damage)

1. **Patch↔field alignment bugs** — silent, corrupts every downstream number, discovered late. Mitigation: the visual validator (render page + box overlay + highlighted patch mask on 50 docs/dataset, eyeball all 50) *gates* everything. Traps: aspect-ratio policy (letterbox vs. squash — read the actual `image_processor` config), partial coverage (state the `coverage_threshold` choice + appendix sensitivity at 0.0/0.25/0.5), grid off-by-one / prepended special tokens.
2. **BiPali not cleanly available** — kills Claim 1. Find out in Phase 1; if absent, reconstruct it (same backbone, mean-pool patches, retrain projection head) — budget 3 days.
3. **OCR engine variance** — result flips by engine. Mitigation: two engines from the start.
4. **Impressive-looking but illegible reconstructions** — fix with text-region-weighted loss (upweight loss on patches inside field boxes 5–10×); if it fails, retitle in M2.
5. **Lazily-ported baselines** — invisible to you, obvious to reviewers. Equal-budget sweep + public W&B + documented adaptations.

---

## 12. The one thing that decides this

**Month 1.** Two hundred FUNSD documents, ColPali vs. BiPali, one frontier plot, plus the cheap Claim-4a probe. If reconstructions are OCR-readable and the frontier separates, the title is earned and the rest is execution. **Build nothing else until that plot exists.**
