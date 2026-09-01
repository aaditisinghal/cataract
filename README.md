# Cataract — State-of-the-Art Privacy for Multi-Vector VLM Retrievers

**Paper:** *The Persistence of Vision: State-of-the-Art Privacy for Multi-Vector VLM Retrievers*
**Author:** Krishna Bhatnagar · Reakon Labs Private Limited
**Target venue:** IEEE S&P (primary)

📄 **[Read the paper — PDF](paper/PersistenceOfVision.pdf)** · **[DOCX (exact IEEE formatting)](paper/PersistenceOfVision.docx)**
📊 **[Full results & every metric](docs/RESULTS.md)** — the source of truth behind every number below.

---

## TL;DR

Multi-vector vision-language retrievers (ColPali, ColQwen2) are the **state of the art** in visual
document retrieval — they beat pooled single-vector encoders precisely *because* they store a
fine-grained per-patch embedding for every page. This repo shows that the same fine-grained storage
is a **field-level personal-information oracle**: an attacker with only read access to the stored
index and the public encoder can recover names, dates of birth, and ID numbers with no image
reconstruction at all. We show the leak is **holographic** (deleting a third of the page still
recovers everything), that the obvious learned defense is **adaptively invertible** (a false sense
of privacy), and we introduce **Cataract** — an index-time, information-*destroying* projection that
is the first adaptively-evaluated, tunable, real-privacy defense for this class of retriever.

| | |
|---|---|
| **Attack** | Field-level closed-set PII linkage against multi-vector VLM indices — name/id/dob recovered at **1.00 / 0.85 / 0.82** top-1 (K=200 lineup, chance 0.005) |
| **Mechanism** | Holographic leakage — deleting a field's own patches, or a 33%-of-page dilation, still recovers **1.00**. Reproduces cross-model on ColQwen2. |
| **Naive defense** | A learned residual transform reaches 1.00 non-adaptive privacy — then an **adaptive attacker breaks it completely** (reconstruction cosine 0.998, recovery back to 1.00) |
| **Cataract (this work)** | Index-time orthogonal projection off the PII-discriminative subspace. Certified against linear inversion; holds the strongest adaptive attacker we evaluate to **0.10 recovery @ 0.875 utility** (k=96) |
| **Deploy cost** | **0.64 ms/page**, **+0 bytes** storage, **0** query-path overhead |

---

## Why this is the state-of-the-art privacy result for this class of retriever

Every prior privacy option for multi-vector VLM retrievers is dominated on the measured
privacy–retrieval frontier:

- **Do nothing** → SOTA retrieval, but the index is a PII oracle (this paper's attack).
- **Pool to a single vector** (BiPali-style) → real privacy, but you give up the fine-grained
  retrieval quality that makes multi-vector retrieval state of the art in the first place.
- **Learn a reshaping defense** (the obvious fix) → looks like privacy under a non-adaptive
  attacker, then **collapses to 1.00 recovery** the moment the attacker is adaptive and knows the
  defense — a false sense of privacy, not a real one.
- **Cataract** → destroys the PII-discriminative subspace rather than reshaping it, so there is
  nothing for an adaptive inverse learner to invert. It is the only option in this comparison that
  is *both* tunable (a k-sweep dial from 0.60 → 0.90+ privacy) *and* holds under an adaptive
  attacker, at index-time cost measured in milliseconds.

See `Fig. 2` in the paper (or `paper/build_r5/figs/fig2.png`) for the privacy–retrieval landscape
plot this table summarizes.

---

## Repo structure

```
patchguard/               core library: retriever protocol, Cataract defense, repro/determinism
  retrievers/              ColPali, ColQwen2, BiPali (pooled control), mock backend
  defense/nullspace.py     Cataract — NullspaceRedaction (the paper's constructive defense)
  defense/redact.py        RedactionProjection — the naive learned defense the paper breaks
  data/                    synthetic ID-card generator, FUNSD alignment
experiments/               40+ experiments — attack, holographic mechanism, adaptive break,
                           certified defense, baselines, ablations (all reproducible; see below)
paper/                     PersistenceOfVision.{pdf,docx} + paper/build_r5/ (build source)
docs/RESULTS.md            every metric from every experiment — the source of truth
docs/RELATED_WORK_VERIFIED.md   every citation independently verified against arXiv, not memory
scripts/                   GCP repro-image build + warm-VM + experiment-exec tooling
tests/                     unit + CPU-only integration tests (mock backend, no GPU required)
reproduce.sh               end-to-end reproduction driver
```

---

## Reproducing the results

All decisive experiments were run on **NVIDIA A100 (40GB)** GPUs on GCP, inside a pinned,
git-SHA-tagged reproducible container (`scripts/10_build_image.sh` → `colpali-engine==0.3.5`, torch
2.3.1, CUDA 12.1). A subset of follow-up experiments (defense generalization, corrected lineup
scaling) reran on NVIDIA L4 with the same pinned image. Every result in `docs/RESULTS.md` records the
git SHA and hardware it was produced on.

```bash
# CPU-only: shape/logic tests + mock-backend pipeline, no GPU needed
pytest tests/

# Full GPU reproduction (needs a CUDA GPU + the repro image)
./reproduce.sh
```

See `scripts/20_warm_vm.sh` / `scripts/21_exec.sh` for the exact GCE warm-VM workflow used to run
experiments against a real A100/L4 without repeated cold starts.

---

## Threat model

The attacker has **read access to the stored multi-vector index** (database breach, malicious
insider, leaked backup, or soft-deleted segments) and the **public, open-source encoder** — nothing
else. No original page images, no model secrets. The attack is white-box only on weights anyone can
download, and it does **closed-set linkage** (confirming which candidate from a supplied lineup is
present), not open-world reconstruction. Full details in the paper, §II.

---

## Responsible disclosure

This work characterizes a property of the **multi-vector VLM retriever architecture** (ColPali,
ColQwen2, and by extension any late-interaction visual retriever storing per-patch embeddings), not a
vulnerability in a single deployed product. It extends the storage-layer soft-delete finding of
Chakraborttii et al. (Ghost Vectors) to the multi-vector setting. If you operate a production
multi-vector retrieval system and want to discuss Cataract as a mitigation, open an issue or reach
the author directly.

---

## Citation

```bibtex
@article{bhatnagar2026persistence,
  title   = {The Persistence of Vision: State-of-the-Art Privacy for Multi-Vector VLM Retrievers},
  author  = {Bhatnagar, Krishna},
  year    = {2026},
  note    = {Reakon Labs Private Limited},
}
```

## License

MIT — see [`LICENSE`](LICENSE).
