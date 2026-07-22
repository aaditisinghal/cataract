# Related-Work Verification (S0 premise gate)

**Date:** 2026-07-22 · **Method:** each arXiv ID fetched directly (raw HTTP + export API, not memory);
a bogus control ID returned 404 to confirm the checker wasn't rubber-stamping.

## Verdict: **GO.** The premise holds.

- The **closest claimed precursor — Ghost Vectors — is real, text-only, and uses Vec2Text**, exactly
  as positioned. It does *not* touch visual / multi-vector / patch embeddings. **We are not scooped.**
- All four literatures exist and are real; **no single paper occupies the intersection** (visual
  multi-vector document retrieval × field-level PII inversion × patch-scoped defense × erasure).
- Every load-bearing arXiv ID resolves. A short list of **factual errors** was found and must be fixed
  before writing (they are the exact "unforced errors" the protocol warned about). None change the
  contribution.

---

## Fix-before-writing list (unforced errors surfaced)

1. **LeakyCLIP SSIM figure:** paper says **>258%** improvement on ViT-B-16, not 358%. Correct it.
2. **Vec2Text "RQ4":** the RQ1–RQ4 structure belongs to the **reproducibility study (2507.07700)**,
   not the original (2310.06816). The noise-vs-recovery *defense* is genuinely in the original, but
   attribute the "RQ4" label to the reproducibility paper.
3. **EntroGuard:** real (add ID **2503.12896**), but it's framed **entropy-driven**, not "bound-aware,"
   and the **ε≈0.036 figure is unverified / likely fabricated** — drop it unless found in the body.
4. **PRESS:** real but **ICASSP 2025, no arXiv ID** — cite the IEEE version. Mechanism = "Embedding
   Space Shifting."
5. **EDPB Guidelines 05/2019 — MISCHARACTERIZED (legal hook for Claim 4).** The document is about
   **search-engine delisting**, and does **not** say erasure must be "verifiable and irreversible."
   **Re-ground the legal hook in Article 17 GDPR text itself** (+ certified-removal literature),
   not this document. This is the one correction that touches a claim's framing — handle with care.
6. **IDNet coverage/construction:** it's **10 US states + 10 European countries** (not US-only), and it
   **inpaints real ID images with Stable Diffusion 2.0** + GPT-3.5 synthetic PII — *not* "public DMV
   templates." Soften the ethics sentence: final images are synthetic, but they derive from real IDs,
   so "no real person's data is at risk" needs rewording to "final images contain only synthetic PII."
7. **TrustCLIP "patch tokens more reconstructible than CNN/SSL descriptors":** not in the abstract;
   **spot-check the PDF body** before citing it as the paper's thesis / support for Claim 1.

## Missing arXiv IDs to add

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

| Paper | Claimed ID | Status | Actual title / correction |
|---|---|---|---|
| ColPali | 2407.01449 | ✅ VERIFIED | All deep claims confirmed: BiPali ablation, SigLIP-So400m, ~1024 patches, 128-dim, MaxSim |
| ColBERT | — | ✅ add 2004.12832 | "…Late Interaction over BERT", SIGIR 2020 |
| Vec2Text | 2310.06816 | ✅ VERIFIED | "Text Embeddings Reveal (Almost) As Much As Text" (EMNLP 2023). 92% @32 tokens ✅. RQ4 label → 2507.07700 |
| Vec2Text reproducibility | 2507.07700 | ✅ VERIFIED | July **2025** (not 2026). RQ1–RQ4 structure lives here |
| GEIA | — | ✅ add 2305.03010 | Li, Xu, Song, 2023 |
| ColModernVBERT | — | ✅ 2510.01149 | Inside "ModernVBERT: Towards Smaller Visual Document Retrievers". 250M + within 0.6 NDCG@5 ✅ |
| ColQwen3 | — | ✅ (no paper) | Model releases; cite ColPali paper + illuin-tech/colpali repo |
| LeakyCLIP | 2508.00756 | ✅ VERIFIED | "LeakyCLIP: Extracting Training Data from CLIP". **SSIM = 258% not 358%** |
| TrustCLIP | 2607.04484 | ✅ VERIFIED (real 2026 paper) | Meta/NUS. "Directly optimizes against a generative reconstruction attacker" ✅. Patch-token claim → check body |
| Implicit Inversion | 2505.23161 | ✅ VERIFIED | Exact title: "Implicit Inversion turns CLIP into a Decoder". All components confirmed |
| NinjaDesc | — | ✅ add 2112.12785 | "Content-Concealing Visual Descriptors via Adversarial Learning", CVPR 2022 |
| **Ghost Vectors** | 2606.18497 | ✅ **VERIFIED (closest precursor, real)** | Soft-deleted embeddings recoverable from HNSW index files via Vec2Text. 25.5% names / 46.4% locations / up to 100% structured medical. **Text-only — our visual angle is open** |
| Koga et al. | 2412.04697 | ✅ VERIFIED | "Privacy-Preserving RAG with Differential Privacy", Dec 2024 |
| EntroGuard | — | ✅ add 2503.12896 | "…Entropy-Driven Perturbation". "entropy-driven" not "bound-aware"; ε≈0.036 unverified |
| PRESS | — | ⚠️ ICASSP 2025, no arXiv | "Defending Privacy in RAG via Embedding Space Shifting" |
| PRAG | 2604.26525 | ✅ VERIFIED | "End-to-End Privacy-Preserving RAG", Apr 2026 |
| P²RAG | 2603.14778 | ✅ VERIFIED | "Efficient Privacy-Preserving RAG… Arbitrary Top-k", Mar 2026 |
| Guo et al. | — | ✅ add 1911.03030 | Certified Data Removal, ICML 2020 |
| Ginart et al. | — | ✅ add 1907.05012 | Making AI Forget You, NeurIPS 2019 |
| EDPB Guidelines 05/2019 | — | ⚠️ EXISTS but MISCHARACTERIZED | Search-engine delisting; does NOT state verifiable+irreversible. Re-ground in Art. 17 |
| IDNet | 2408.01690 | ✅ VERIFIED (correct coverage) | 837k / 490GB / 20 types ✅. **10 US states + 10 EU countries**; SD2.0 inpainting of real IDs |
| DocLayNet | 2206.01062 | ✅ VERIFIED | 80,863 pages, 11 classes, COCO ✅ |
| FUNSD | — | ✅ add 1905.13538 | 199 forms ✅ |
| CORD | — | ✅ (NeurIPS 2019 wksp) | ~1,000 receipts ✅, no arXiv abs |
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
- **TrustCLIP doubles as the strongest defense baseline** (Claim 2) and theoretical support (Claim 1) —
  verify the patch-token sentence in-body first.
- **Fix the 7 errors above before any drafting.** They're cheap now, expensive in a submission.
