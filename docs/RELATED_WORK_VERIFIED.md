# Related-Work Verification (S0 premise gate)

> Citation audit for *The Persistence of Vision: State-of-the-Art Privacy for Multi-Vector VLM
> Retrievers*. [Paper (PDF)](../paper/PersistenceOfVision.pdf) · [README](../README.md).

**Date:** 2026-07-22 · **Method:** each arXiv ID fetched directly (raw HTTP + export API, not memory);
a bogus control ID returned 404 to confirm the checker wasn't rubber-stamping.
**Corrections applied:** 2026-07-23 (D1).

## Verdict: **GO.** The premise holds.

- The **closest claimed precursor — Ghost Vectors — is real, text-only, and uses Vec2Text**, exactly
  as positioned. It does *not* touch visual / multi-vector / patch embeddings. **We are not scooped.**
- All four literatures exist and are real; **no single paper occupies the intersection** (visual
  multi-vector document retrieval × field-level PII inversion × patch-scoped defense × erasure).
- Every load-bearing arXiv ID resolves. A short list of **factual errors** was found and has now been
  **corrected in place (D1, 2026-07-23)** — they were the exact "unforced errors" the protocol warned
  about. None change the contribution.

---

## Corrections applied (was: fix-before-writing list — unforced errors resolved, D1)

All seven corrections below have been applied to this document; each item states the corrected fact,
and the full verification table further down reflects them.

1. **LeakyCLIP SSIM figure — corrected.** The improvement on ViT-B-16 is **>258%** (was mistyped as
   358%). Every reference now reads **258%**.
2. **Vec2Text "RQ4" — corrected.** The RQ1–RQ4 structure is **not** in the original Vec2Text paper
   (2310.06816); it belongs to the **2025 reproducibility-and-inversion study (2507.07700)**, which
   re-runs and stress-tests the embedding-inversion attack under that RQ framing. The noise-vs-recovery
   *defense* is genuinely in the original; the "RQ4" label is attributed to the reproducibility paper only.
3. **EntroGuard — corrected.** Real (ID **2503.12896**), framed **entropy-driven** (not "bound-aware").
   The spurious **ε≈0.036 figure has been dropped** — it was unverified and not in the body.
4. **PRESS — corrected.** Real but **ICASSP 2025, venue-only (NO arXiv ID)** — cite the IEEE
   proceedings version, never an arXiv id. Mechanism = "Embedding Space Shifting."
5. **Claim 4 legal hook — re-grounded on the statute.** The earlier citation to **EDPB Guidelines
   05/2019 has been removed**: that document concerns **search-engine delisting** and does **not** say
   erasure must be "verifiable and irreversible." Claim 4 is now grounded directly in the **text of
   GDPR Article 17** ("Right to erasure / right to be forgotten"): Art. **17(1)** gives the data
   subject the right "to obtain from the controller the erasure of personal data concerning him or her
   without undue delay" and places the corresponding "obligation to erase" on the controller; Art.
   **17(2)** requires the controller, "taking account of available technology and the cost of
   implementation, [to] take reasonable steps, including technical measures," to effect that erasure.
   The impossibility hook is now on the statute itself: deleting the source document does **not**
   discharge the Art. 17(1) erasure obligation while the personal data remain recoverable from the
   retained multi-vector embeddings — and Art. 17(2)'s "available technology / technical measures" is
   exactly where the certified-removal literature (Guo et al. 1911.03030; Ginart et al. 1907.05012)
   attaches. No EDPB-guideline text is relied on.
6. **IDNet composition — corrected.** IDNet spans **10 US + 10 EU document types** (one per 10 US
   states and 10 European countries; 20 types total) — *not* US-only. The images are produced by
   **Stable-Diffusion-2.0 inpainting** of real ID images plus GPT-3.5 synthetic PII — *not* "public DMV
   templates." Ethics wording corrected accordingly: the final images derive from real IDs but
   **contain only synthetic PII**, so we say "final images contain only synthetic PII" rather than
   "no real person's data is at risk."
7. **TrustCLIP patch-token claim — demoted to body prose.** "Patch tokens are more reconstructible
   than CNN/SSL descriptors" is **not** a TrustCLIP headline result/metric and is **not** in the
   abstract; it is a body-level assertion. It is therefore cited as **supporting body prose for
   Claim 1** (a corroborating observation), not as TrustCLIP's thesis or a headline figure.

## arXiv IDs added (were missing — now filled into the verification table below)

| Paper | Correct ID |
|---|---|
| ColBERT (Khattab & Zaharia) | **2004.12832** |
| GEIA (generative embedding inversion, text) | **2305.03010** |
| ModernVBERT / ColModernVBERT | **2510.01149** (ColModernVBERT is the late-interaction variant *inside* this paper) |
| NinjaDesc | **2112.12785** |
| Guo et al., Certified Data Removal (ICML 2020) | **1911.03030** |
| Ginart et al., Making AI Forget You (NeurIPS 2019) | **1907.05012** |
| FUNSD | **1905.13538** |

---

## Full verification table

| Paper | arXiv ID | Status | Actual title / correction |
|---|---|---|---|
| ColPali | 2407.01449 | ✅ VERIFIED | All deep claims confirmed: BiPali ablation, SigLIP-So400m, ~1024 patches, 128-dim, MaxSim |
| ColBERT | 2004.12832 | ✅ VERIFIED (added) | "…Late Interaction over BERT", SIGIR 2020 |
| Vec2Text | 2310.06816 | ✅ VERIFIED | "Text Embeddings Reveal (Almost) As Much As Text" (EMNLP 2023). 92% @32 tokens ✅. **RQ1–RQ4 structure is the reproducibility study's (2507.07700), not this paper's** |
| Vec2Text reproducibility | 2507.07700 | ✅ VERIFIED | July **2025** (not 2026). Reproducibility-and-inversion study; the RQ1–RQ4 structure lives here |
| GEIA | 2305.03010 | ✅ VERIFIED (added) | Li, Xu, Song, 2023 |
| ColModernVBERT | 2510.01149 | ✅ VERIFIED (added) | Inside "ModernVBERT: Towards Smaller Visual Document Retrievers". 250M + within 0.6 NDCG@5 ✅ |
| ColQwen3 | — (no paper) | ✅ (model release) | Model releases; cite ColPali paper + illuin-tech/colpali repo |
| LeakyCLIP | 2508.00756 | ✅ VERIFIED | "LeakyCLIP: Extracting Training Data from CLIP". **SSIM = 258%** (corrected from a 358% typo) |
| TrustCLIP | 2607.04484 | ✅ VERIFIED (real 2026 paper) | Meta/NUS. "Directly optimizes against a generative reconstruction attacker" ✅. Patch-token point is **body prose (not a headline metric)** → supporting evidence for Claim 1 |
| Implicit Inversion | 2505.23161 | ✅ VERIFIED | Exact title: "Implicit Inversion turns CLIP into a Decoder". All components confirmed |
| NinjaDesc | 2112.12785 | ✅ VERIFIED (added) | "Content-Concealing Visual Descriptors via Adversarial Learning", CVPR 2022 |
| **Ghost Vectors** | 2606.18497 | ✅ **VERIFIED (closest precursor, real)** | Soft-deleted embeddings recoverable from HNSW index files via Vec2Text. 25.5% names / 46.4% locations / up to 100% structured medical. **Text-only — our visual angle is open** |
| Koga et al. | 2412.04697 | ✅ VERIFIED | "Privacy-Preserving RAG with Differential Privacy", Dec 2024 |
| EntroGuard | 2503.12896 | ✅ VERIFIED (added) | "…Entropy-Driven Perturbation". **Entropy-driven** (not "bound-aware"); spurious **ε≈0.036 figure dropped** (unverified) |
| PRESS | — (venue-only) | ⚠️ ICASSP 2025, **NO arXiv ID** | "Defending Privacy in RAG via Embedding Space Shifting" — cite the IEEE proceedings version |
| PRAG | 2604.26525 | ✅ VERIFIED | "End-to-End Privacy-Preserving RAG", Apr 2026 |
| P²RAG | 2603.14778 | ✅ VERIFIED | "Efficient Privacy-Preserving RAG… Arbitrary Top-k", Mar 2026 |
| Guo et al. | 1911.03030 | ✅ VERIFIED (added) | Certified Data Removal, ICML 2020 |
| Ginart et al. | 1907.05012 | ✅ VERIFIED (added) | Making AI Forget You, NeurIPS 2019 |
| GDPR Art. 17 *(replaces EDPB 05/2019)* | — (statute, no arXiv) | ✅ RE-GROUNDED | Claim 4 hook now on **Art. 17(1)** erasure right/obligation + **17(2)** "available technology … technical measures". EDPB Guidelines 05/2019 dropped (delisting scope; never stated verifiable+irreversible) |
| IDNet | 2408.01690 | ✅ VERIFIED (correct coverage) | 837k / 490GB / 20 types ✅. **10 US + 10 EU document types** (10 US states + 10 European countries); SD-2.0 inpainting of real IDs + GPT-3.5 synthetic PII |
| DocLayNet | 2206.01062 | ✅ VERIFIED | 80,863 pages, 11 classes, COCO ✅ |
| FUNSD | 1905.13538 | ✅ VERIFIED (added) | 199 forms ✅ |
| CORD | — (NeurIPS 2019 wksp) | ✅ VERIFIED | ~1,000 receipts ✅, no arXiv abs |
| ViDoRe | (ColPali) | ✅ VERIFIED | 127,346 pages ✅, introduced in ColPali paper |
| Qdrant / Milvus / pgvector | — | ✅ VERIFIED | Qdrant+Milvus native multi-vector MaxSim; pgvector lacks it ✅ |

---

## What this means for the paper

- **Novelty statement survives** — write it in the intro with confidence: text-embedding inversion
  (Vec2Text, GEIA), vision-encoder inversion (LeakyCLIP, Implicit Inversion), descriptor defenses
  (NinjaDesc, TrustCLIP), and RAG/erasure privacy (Ghost Vectors, Koga, PRAG, EntroGuard) each exist,
  but none address **multi-vector visual document retrieval at the patch level**.
- **Ghost Vectors is the citation to position against most carefully** — same storage-layer threat
  model, but text-only + single-vector. Our 1c/3/4 contributions are orthogonal to it.
- **TrustCLIP doubles as the strongest defense baseline** (Claim 2) and theoretical support (Claim 1).
  Its patch-token point is a **body-prose assertion** (a corroborating detail), not a headline metric —
  cite it as supporting prose, not as TrustCLIP's thesis.
- **Claim 4's erasure-impossibility hook is grounded on GDPR Article 17 itself** (17(1) erasure
  obligation + 17(2) "available technology / technical measures"), backed by the certified-removal
  literature (Guo, Ginart) — not on the mischaracterized EDPB 05/2019 delisting guideline.
- **All 7 corrections above are now applied (D1, 2026-07-23) and the 7 missing arXiv IDs are filled
  into the table.** The document is clean for drafting — these were cheap now, expensive in a submission.
