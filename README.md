# patchguard-paper

Working folder for **"The Persistence of Vision: PII Reconstruction and Patch-Scoped Privacy in
Multi-Vector Vision-Language Retrieval"** — embedding inversion of ColPali-family multi-vector
visual-document retrievers to field-level PII, plus a patch-scoped defense and an erasure finding.

- **Target venues:** IEEE S&P (primary) · USENIX Security (secondary) · NeurIPS D&B (fallback)
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
- [x] **S0 premise** — GO. `docs/RELATED_WORK_VERIFIED.md`: intersection empty, not scooped; 7 errors + 7 missing IDs logged.
- [x] **S1 skeleton + infra** — `repro.py` (determinism + fingerprint), CI, Makefile. **GCP `patchguard-reakon` live; Vertex L4 smoke job SUCCEEDED** (keyless ADC + teardown proven).
- [x] **S2 backends** — `retrievers/base.py` protocol + MaxSim; `colpali.py` (real, guarded), `bipali.py` (pooled control = 1×1 no-locality grid), mock.
- [x] **S3 alignment** — `data/align.py` visually validated (green patches land on red fields). *Real-dataset eyeballing pending data.*
- [x] **S4 eval** — `frontier.py` (bootstrap CI, z-test, AUC, dominance), `pfrr.py`, `killgate.py` (pre-registered gate + `assemble_and_gate`).
- [x] **S5 attack v0** — `attack/decoder.py` (PatchGridDecoder + text-region-weighted loss), CPU shape/backprop tested.
- [x] **S8 defense core** — `defense/perturb.py` (flat vs patch-scoped Gaussian, Claim 3) + `localize.py` (oracle).
- [x] **★ kill test wired** — `experiments/kill_test.py --mock` runs the full pipeline end-to-end → `GO`, writes fingerprinted result. **65 CPU tests passing.**

## Next (needs GPU + data — everything else is done)
- [ ] Decoder **training loop** (`attack/train.py`, S6) — write on CPU, run on Vertex
- [ ] **OCR wiring** in `eval/pfrr` (Tesseract + PaddleOCR) — the real PFRR measurement
- [ ] **FUNSD download** + run `validate_alignment.py` on 50 real docs (S3 gate)
- [ ] Swap real ColPali/BiPali into `kill_test.py` real path → **fire the ★ kill test for real**
