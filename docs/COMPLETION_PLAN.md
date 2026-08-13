# Completion Plan — Attack Phase + Solution Phase

> Planning doc toward *The Persistence of Vision: State-of-the-Art Privacy for Multi-Vector VLM
> Retrievers*. [Paper (PDF)](../paper/PersistenceOfVision.pdf) · [README](../README.md).

**Deep, actionable plan for everything missing** to take the paper from "science locked" to
"submittable." Organized as **Phase 1 (Attack/Findings)**, **Phase 2 (Solution/Defense)**,
**Cross-cutting rigor**, and **Writeup & packaging**. Each item: *what · why · how (concrete design) ·
expected · effort · risk · dependency*. Effort assumes the **warm-VM harness** (~3.5 min/experiment).

**Priority legend:** 🔴 decisive (gates a core claim) · 🟠 should-have (strengthens) · 🟢 completeness.

---

## PHASE 1 — ATTACK / FINDINGS (what's missing to finalize the threat side)

### A1 🟠 FUNSD answer-fields-only (the fair real-doc number)
- **What.** Rerun `funsd_transfer` filtering to `field_type == "answer"` only (real values), excluding
  question/header/"other" boilerplate labels ("FAX NO.", "DATE:").
- **Why.** Current top-1 **0.19** is deflated by non-PII boilerplate; the by-label split already showed
  answer=0.26. The honest real-doc number should be answer-only, broken out by glyph height.
- **How.** Add `--labels answer` to `funsd_transfer.py`; rerun. Report top-1/top-5 + by-glyph buckets + CIs.
- **Expected.** Answer-only top-1 > 0.19, likely ~0.25–0.35, still rising with glyph size.
- **Effort.** ~10 min (1 flag + warm-VM run). **Risk.** low. **Dep.** none.

### A2 🔴 Multi-seed + statistical rigor (Holm–Bonferroni)
- **What.** Rerun each headline experiment (`claim1`, `control_wrongpage`, `erasure`, `cross_model`,
  `retrieval_attack`, `learned_defense`) at **5 seeds**; report mean ± 95% CI; **Holm–Bonferroni** across
  the primary claim family; **per-document variance** for the attack.
- **Why.** The stats plan (protocol §4) and any top venue require it. Single-run headline numbers are a
  rejection risk. We already have bootstrap CIs per run; need seed-level variance too.
- **How.** Add a `--seed` sweep wrapper; aggregate the 5 seeds' JSONs into mean±CI; a `stats/aggregate.py`
  that applies Holm–Bonferroni to the family of primary p-values (McNemar for Claim 1, two-proportion z
  for recovery deltas). Pre-register the family in git before running.
- **Expected.** Headline numbers stable; CIs tight (most are 1.00/0.00). Defense dominance stays positive.
- **Effort.** ~1–2 hr (6 experiments × 5 seeds on warm VM + aggregation script). **Risk.** low. **Dep.** none.

### A3 🟠 Real-corpus breadth (CORD, DocLayNet-finance)
- **What.** Run the retrieval-discrimination attack on **CORD** (~1k receipts, financial PII) and the
  **DocLayNet finance** subset — not just FUNSD.
- **Why.** Multiple real document *types* turn "transfers to FUNSD" into "transfers to real documents,"
  and receipts/finance have the account-number-style PII that matters.
- **How.** Stage CORD + DocLayNet-finance to the data bucket; write loaders (CORD has word-level boxes +
  labels; DocLayNet COCO + per-cell text). Reuse the `funsd_transfer` discrimination harness. Place all
  corpora on the shared **glyph-height axis** (§4.10 method).
- **Expected.** Same pattern — above chance, scales with glyph size/distinctiveness; financial fields
  (larger, distinctive) recover better than FUNSD's dense small text.
- **Effort.** ~half day (staging + 2 loaders). **Risk.** medium (loader/format work). **Dep.** none.

### A4 🟠 Close the id/dob transposition gap (positional rerank)
- **What.** Two-stage attack for numeric fields: MaxSim top-K candidates → **rerank by positional
  consistency** so exact id/dob recovery rises from 0.725/0.40 toward name's 1.00.
- **Why.** The main honest weakness of the attack: MaxSim is bag-of-tokens, so it nails the digit
  *multiset* but transposes order (~20% for id, ~60% for dob). A positional reranker recovers the string.
- **How.** Options, cheapest first: (a) **windowed MaxSim** — score candidates by matching *bigrams/
  positional n-grams* of the value against localized patch windows (needs the box→patch alignment we have);
  (b) a small **trained reranker** on the patch grid + candidate. Report exact-recovery lift.
- **Expected.** id_no top-1 0.725 → ~0.9+, dob 0.40 → ~0.7+. Strengthens the "recovered PII" claim.
- **Effort.** ~half day (design + run). **Risk.** medium (rerank may not fully close it — honest either way).

### A5 🟢 Claim 1c — does spatial *arrangement* leak beyond content?
- **What.** Shuffle patch order before the attack; compare recovery to ordered. (Original protocol Claim 1c.)
- **Why.** Tests whether the leak needs spatial arrangement or is pure bag-of-content. Our finding
  (MaxSim is order-independent) *predicts a NULL* — arrangement doesn't add for the retrieval attack.
- **How.** Permute the patch sequence in the attack; re-measure.
- **Expected.** **NULL** (recovery unchanged) → honest finding: the leak is content, not arrangement.
  Report as a controlled negative (turns a hypothesized claim into a clean fact).
- **Effort.** ~15 min. **Risk.** low. **Dep.** none.

### A6 🟠 Storage-layer threat validation (Qdrant / Ghost-Vectors mirror)
- **What.** Store the multi-vector index in **Qdrant**, insert docs, **soft-delete**, dump the raw index
  files, and run the retrieval attack on the recovered vectors.
- **Why.** Grounds the threat model ("attacker has read access to the stored index") in a real deployed
  vector DB, and connects to Ghost Vectors (soft-deleted vectors remain recoverable). Makes the
  "unrealistic adversary" objection dead.
- **How.** Spin Qdrant (native multi-vector + MaxSim), insert ColPali page embeddings, soft-delete a doc,
  read the raw HNSW/segment files, confirm the vectors persist, run the attack on them.
- **Expected.** Soft-deleted multi-vectors remain physically present + attackable → confirms the model.
- **Effort.** ~half day (Qdrant setup). **Risk.** medium. **Dep.** none.

### A7 🟢 Threat-model refinement — what does the attacker need?
- **What.** Characterize attacker requirements: our attacker uses `encode_query` (needs the model). Test a
  **transfer attack** — attack a ColPali index using a *different* public encoder's queries (or ColQwen2's).
- **Why.** Reviewers ask "black-box or white-box?" Clarify: white-box (has the public retriever) is the
  realistic default (these are open models); test whether a proxy encoder suffices (semi-black-box).
- **How.** Run the attack with mismatched query/index encoders; measure degradation.
- **Expected.** Same-encoder near-perfect; cross-encoder degrades but may transfer partially.
- **Effort.** ~1 hr. **Risk.** low. **Dep.** cross-model backend (have it).

---

## PHASE 2 — SOLUTION / DEFENSE (what's missing to earn the defense claim)

### B1 🔴🔴 ADAPTIVE attacker vs RedactionProjection — THE decisive test
- **What.** Attack the P-defended index with an attacker who **knows P**. Four independent strategies
  (fan out in parallel — ultracode-appropriate):
  - **B1a — Distillation / linear probe on P(patches).** Train an MLP/linear classifier `P(patches) → PII`.
    *This is the most dangerous:* if PII is still linearly decodable from the defended embedding, P only
    broke MaxSim-matching, not information removal. (Mirror §4.3's probe on the *defended* index.)
  - **B1b — Inverse learning.** Train `P⁻¹` (net mapping `P(patch) → patch` or `→ PII`) from
    (P(patch), patch) pairs an attacker could generate by running the public P on their own documents.
  - **B1c — Adaptive query optimization.** With P known, optimize query embeddings to maximize
    `MaxSim(q, P(patches))` for the true value (gradient/search on the query).
  - **B1d — Correlation exploit.** Query *content correlated with PII* (co-occurring fields, topic) to
    infer PII indirectly, sidestepping the suppressed direction.
- **Why.** **Non-adaptive-only evaluation is the #1 security-paper rejection.** Our privacy=1.00 is vs a
  non-adaptive attacker. This experiment *decides whether the defense is real.*
- **How.** `experiments/adaptive_attack.py` running all 4 vs the trained P; report residual recovery per
  strategy. Run at several λ (privacy weights).
- **Expected (honest, uncertain).** B1a is the likely breaker — the information is probably still *present*
  in P(patches), just not MaxSim-accessible; a trained probe may recover it. If so, the honest claim is
  "P defends against the *dictionary/retrieval* attack but not a *trained-probe* attacker" → a scoped,
  still-useful result, and it motivates a stronger (info-removal / certified) P.
- **Effort.** ~1 day (4 strategies). **Risk.** high (may partially break P — that's the point).
  **Dep.** learned defense (have it). **This is the top priority of the whole plan.**

### B2 🔴 Real-retrieval utility — ViDoRe NDCG@5
- **What.** Build a **ColPali + P** index over ViDoRe; measure **NDCG@5 / Recall@1 / MRR** vs vanilla
  ColPali (and vs flat-noise index).
- **Why.** Synthetic topic-Recall@1 (0.95) isn't real utility. ViDoRe is the standard multi-vector
  retrieval benchmark; the defense's utility claim is only credible there.
- **How.** Apply P to ViDoRe page embeddings, run the ViDoRe eval harness (queries × corpus, MaxSim,
  NDCG@5). Sweep λ → real utility axis of the frontier. Compare to flat-noise index at matched privacy.
- **Expected.** P retains most retrieval (small NDCG@5 drop) where PII is incidental; the gap vs flat
  noise persists on real retrieval. *Key risk:* real ViDoRe queries may be *about* PII → smaller margin.
- **Effort.** ~1 day (ViDoRe staging + eval). **Risk.** medium. **Dep.** ViDoRe data.

### B3 🔴 The fundamental floor (where PII IS the retrieval target)
- **What.** Characterize the achievable privacy/utility frontier as a function of **PII↔retrieval
  entanglement**: sweep from "retrieval is about the topic" (PII incidental) to "retrieval is about the
  name" (find-by-name).
- **Why.** The honest scope of the defense. We predicted P works in the incidental case and hits a floor
  in the find-by-name case. Quantifying the floor is a *contribution* (an impossibility boundary).
- **How.** Vary the utility query from topic → name; retrain/eval P; plot best-achievable frontier vs
  entanglement. Show the crossover where no P beats the leak.
- **Expected.** Clean incidental case: P near-perfect. Find-by-name: frontier collapses to the leak (can't
  hide what you retrieve by). Gives operators a decision rule.
- **Effort.** ~half day. **Risk.** low. **Dep.** learned defense.

### B4 🟠 Cross-model defense (P on ColQwen2)
- **What.** Train + evaluate RedactionProjection on **ColQwen2** embeddings.
- **Why.** "The defense generalizes across backbones" mirrors the attack's cross-model result.
- **How.** Reuse `learned_defense.py` with the ColQwen2 backend.
- **Expected.** P works on ColQwen2 too (same subspace-separability mechanism).
- **Effort.** ~2 hr. **Risk.** low. **Dep.** ColQwen2 backend (have it).

### B5 🟠 Baselines (Claim 2) — the field's defenses vs the attack and vs P
- **What.** Port **EntroGuard** (2503.12896), **PRESS** (ICASSP'25), **Koga** (2412.04697); measure their
  privacy/utility vs the retrieval attack, and vs RedactionProjection on the same frontier.
- **Why.** Shows the proposed defenses either fail (if they assume locality) or reduce to global noise,
  and that P dominates them — the "defenses don't transfer" claim, done fairly (equal tuning budget,
  public W&B).
- **How.** Implement each as an index-time transform; one config-driven sweep; overlay all frontiers.
- **Expected.** Most reduce to global-noise-like frontiers; P dominates.
- **Effort.** ~1–2 days (porting). **Risk.** medium (fair-port scrutiny). **Dep.** none.

### B6 🟢 Certified / provable P
- **What.** Add calibrated noise after P with a **DP-style ε bound** (or a certified-removal guarantee à
  la Guo ICML'20 / Ginart NeurIPS'19) → an empirical→provable upgrade.
- **Why.** Turns "attack failed empirically" into "attack *provably* bounded." Strong for a top venue.
- **How.** Post-P Gaussian mechanism with a similarity-bound ε; report the certified privacy at each ε and
  its utility cost.
- **Expected.** A provable floor at some utility cost, dominating uncertified flat noise.
- **Effort.** ~1–2 days (theory + calibration). **Risk.** high (theory). **Dep.** B1 (know what to certify).

### B7 🟢 P ablations
- **What.** Architecture of P: linear vs residual-MLP vs attention; depth; the gate; λ sweep (have λ).
- **Why.** Justify the design; show the minimal P that works (linear may suffice → interpretable + faster).
- **How.** Sweep in `learned_defense.py`.
- **Expected.** Linear may nearly match MLP (subspaces are ~linear) → a clean, interpretable defense.
- **Effort.** ~2 hr. **Risk.** low.

### B8 🟠 Real-doc defense transfer
- **What.** Train P on synthetic cards, evaluate protection on **real FUNSD** field text.
- **Why.** The defense must protect real documents, not just its training distribution.
- **How.** Train P (synthetic), run the FUNSD retrieval attack on P(real FUNSD embeddings).
- **Expected.** Partial transfer; may need training on real docs. Honest either way.
- **Effort.** ~2 hr. **Risk.** medium. **Dep.** B1/B2.

---

## CROSS-CUTTING RIGOR (C)

- **C1 🟠 Efficiency benchmark** — P index-time latency, per-patch FLOPs, storage delta (should be ~0),
  query-time overhead (0). A deploy-cost table (protocol §5). Effort ~1 hr.
- **C2 🟢 Determinism/repro audit** — confirm every paper-table run is `git_dirty=False` fingerprinted;
  `reproduce.sh` regenerates each. Effort ~1 hr.
- **C3 🟢 Ethics section data** — IDNet framing correction, synthetic-only PII statement, disclosure plan.

---

## WRITEUP & PACKAGING (D)

- **D1 🔴 Fix the 7 citation errors** (`RELATED_WORK_VERIFIED.md`): LeakyCLIP SSIM 258% not 358%;
  Vec2Text "RQ4" → reproducibility study; EntroGuard "entropy-driven" + drop ε≈0.036; PRESS no arXiv;
  **EDPB 05/2019 → re-ground Claim 4 in Article 17 text itself**; IDNet 10 US + 10 EU / SD2.0-inpaint;
  TrustCLIP patch-token claim in-body. + add 7 missing IDs (ColBERT 2004.12832, GEIA 2305.03010, etc.).
- **D2 🔴 Paper draft** — sections: intro · threat model · the retrieval attack · the holographic
  mechanism (wrong-page + erasure) · Claim 1/1b · cross-model · defense (impossibility of local + the
  learned P + adaptive results + floor) · eval · ethics · related work. **The narrative is now crisp**
  (RESULTS.md is the skeleton).
- **D3 🟠 Figures** (auto-generate from bucketed JSONs via `experiments/make_figures.py`):
  (1) recovered-PII table; (2) Claim-1 bar (ColPali vs BiPali) + matched-bytes curve; (3) wrong-page 2×2;
  (4) erasure dilation sweep; (5) cross-model panel; (6) glyph-height curve (all corpora on one axis);
  (7) **the defense frontier: learned P vs flat noise vs baselines vs adaptive**.
- **D4 🟠 `reproduce.sh`** — one command regenerating every table/figure from `metrics.json`.
- **D5 🟠 Artifact-evaluation package** (USENIX template) + Docker image + checkpoints on HF/Zenodo.
- **D6 🔴 Responsible disclosure** — ColPali/ColQwen maintainers + Qdrant/Milvus, BEFORE preprint.

---

## CRITICAL PATH & SEQUENCING

**The two experiments that gate the paper's two halves (do these FIRST):**
1. **B1 — adaptive attack** (decides if the defense claim survives). If P holds → strong dual paper.
   If P partially breaks → scope the defense claim honestly (still publishable) and prioritize B6 (certified).
2. **B2 — ViDoRe utility** (makes the defense rigorous, not synthetic).

**Then, in parallel batches on the warm VM (each ~minutes):**
- Attack completeness: A1 (answer-only), A5 (arrangement null), A2 (seeds), A7 (transfer).
- Defense completeness: B3 (floor), B4 (cross-model P), B7 (ablations).
- Heavier data work (can overlap with writing): A3 (CORD/DocLayNet), A6 (Qdrant), B5 (baselines), A4 (rerank).

**Then writeup (D), with D1 (citations) and D6 (disclosure) started early.**

**Minimum for a strong submission:** B1 + B2 + A2 (rigor) + A1 + D1 + D2 + D6. Everything else strengthens.

---

## RISK REGISTER (the honest "what could go wrong")

| Risk | Where | Mitigation / honest fallback |
|---|---|---|
| Adaptive probe (B1a) recovers PII from P(patches) → P doesn't remove info | B1 | scope claim to "defends dictionary attack"; add certified P (B6); or accept impossibility as the result |
| ViDoRe queries are PII-centric → small utility margin | B2 | report per-query-type; the floor (B3) explains it |
| id/dob rerank doesn't fully close transposition | A4 | report multiset-recovery honestly (it's already a finding) |
| Baselines hard to port fairly | B5 | equal-budget sweep + public W&B; document each adaptation |
| Real-corpus transfer stays weak (small-font) | A3 | it's the honest scope: strong on large-font, weak on dense — already the story |
