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

## Next steps (not yet started)
- [ ] Citation verification pass (§10)
- [ ] Phase 0 — `repro.py`, Dockerfile, CI skeleton
- [ ] Phase 1 — retriever protocol + one real backend (+ confirm BiPali availability)
- [ ] Phase 2 — `align.py` + visual validator (the #1 silent risk)
