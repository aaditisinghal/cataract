# patchguard-paper

Working folder for **"Your Embeddings Are Photographs"** — embedding inversion of ColPali-family
multi-vector visual-document retrievers to field-level PII, plus a patch-scoped defense and an
erasure finding.

- **Target venues:** USENIX Security (primary) · IEEE S&P (secondary) · NeurIPS D&B (fallback)
- **START HERE — master plan (0→100):** [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) — the execution roadmap + gates.
- **Full protocol (the science):** [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md) — v1.1, revised.
- **Compute plan (the infra):** [`docs/GCP_COMPUTE_PLAN.md`](docs/GCP_COMPUTE_PLAN.md) — $25k GCP credits, Vertex ephemeral jobs.

## The spine
Not "more vectors leak more" (trivial). The paper is: **spatial arrangement leaks localizable PII
(1c) → the same locality is a defense primitive pooled embeddings can't have (3) → and it means
naive patch deletion doesn't erase (4).**

## Before any code
1. **Verify the related-work map** (week-1 blocking) — confirm every arXiv ID is real and the
   four-literature intersection is actually empty. See protocol §10.
2. Then **only** the kill test: 200 FUNSD docs, ColPali vs. BiPali, one frontier plot + a cheap
   Claim-4a probe. Build nothing else until that plot exists.

## Build status
- [x] **S1 skeleton** — `repro.py` (determinism + fingerprint), CI, Makefile, pyproject. Repro loop closed (`paper_ready` gate works).
- [x] **S2 interface** — `retrievers/base.py` (PageEncoding + Retriever protocol + MaxSim) + deterministic mock backend. *Real backends (ColPali/BiPali) pending — need GPU.*
- [x] **S3 alignment core** — `data/align.py` (`boxes_to_patch_mask`), visually validated on synthetic page (green patches land on red fields). *Real-dataset eyeballing pending data.*
- [x] **S4 eval math** — `eval/frontier.py` (bootstrap CI, two-proportion z, AUC, dominance) + `eval/pfrr.py`. All pure-logic, tested.
- [x] **35 CPU tests passing**, `experiments/kill_test.py` thresholds pre-registered (not yet runnable).

## Next (needs the GPU stack / data — see GCP plan)
- [ ] Citation verification pass (§10) — still the true premise gate
- [ ] S2 real backends: `colpali.py`, `bipali.py` (+ confirm BiPali availability)
- [ ] S3: run the validator on 50 real docs/dataset once loaders exist
- [ ] S5: `attack/decoder.py` (v0) with text-region-weighted loss → then the ★ kill test
