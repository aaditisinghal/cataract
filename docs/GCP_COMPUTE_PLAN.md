# GCP Compute Plan — PatchGuard

> Compute infrastructure behind *The Persistence of Vision: State-of-the-Art Privacy for
> Multi-Vector VLM Retrievers* — all decisive experiments ran on **NVIDIA A100 (40GB)** GPUs via
> this plan; see the [paper](../paper/PersistenceOfVision.pdf).

**v1.0** · budget: **$25,000 GCP credits** · workload from [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md)

---

## 0. TL;DR and the mental model

**Credits are not the constraint.** The entire project — diffusion-attack training, ColPali/ColQwen3
validation, IDNet-scale eval, 5 baselines × tuning sweeps × 5 seeds — costs **~$1–3k even on
on-demand GPUs**. You have ~10× that. So the plan does **not** optimize for cheapness. It optimizes
for two things:

1. **Ephemeral "run train and close"** — a job provisions a GPU, runs, writes results to a bucket,
   and the machine is *automatically released*. No VM left running overnight burning $88/day.
2. **Not fighting your org's key policy.** Your org (`profitwise-main-app`) blocks API-key and
   service-account-key creation; keyless ADC works for Vertex. That makes **Vertex AI Custom Jobs**
   the path of least resistance — it needs no exported keys, and it *is* the "run and close" model.

**Consequence worth internalizing:** because you're 10× under budget, **run short/critical jobs
on-demand and skip Spot preemption-handling entirely.** Spot saves ~65% but adds resume complexity.
Reserve Spot for one thing only: the long, embarrassingly-parallel IDNet encoding sweep, where a
preemption costs seconds to recover. Simplicity beats a discount you don't need.

---

## 1. Blocking prerequisites (do these before any run — like citation verification, they gate everything)

1. **A dedicated project.** Create `patchguard` (or similar) — *separate* from `profitwise-main-app`.
   Isolates the credits, cost reporting, GPU quota, and IAM; and lets you nuke everything with one
   `gcloud projects delete` at the end. Attach the $25k billing account to it.
2. **GPU quota.** New projects start at **0 GPU quota**. Request increases *now* (approval takes
   hours–1 day):
   - `NVIDIA_L4_GPUS` → 4 (dev, encoding, most eval)
   - `NVIDIA_A100_GPUS` → 2 (diffusion training, ColQwen3-4B validation)
   - Request in `us-central1` (best A100/L4 availability + cheapest) with a `us-east4` fallback.
   - Also raise **`CustomModelTrainingA100GPUs`** / L4 quota under *Vertex AI* specifically — Vertex
     has its own quota bucket separate from Compute Engine.
3. **Confirm the org key policy doesn't block Vertex.** Your memory says keyless ADC works for
   Vertex — verify on the new project with `gcloud auth application-default login`, then a 1-vCPU
   Vertex "hello GPU" job. If the org policy (`iam.disableServiceAccountKeyCreation`,
   `constraints/gcp.resourceLocations`) is inherited and bites, you may need a policy exception on
   this project — sort it before M1, not during.
4. **Budget alerts.** Set a billing budget on the project with alerts at **$500 / $1k / $2.5k / $5k**
   (email + Pub/Sub). You will never hit $5k, which is exactly why an alert there means something
   is wrong (an idle VM, a runaway sweep) — treat it as a tripwire, not a limit.

---

## 2. Architecture

```
project: patchguard  (billing = $25k credit account)
│
├── GCS buckets  (us-central1, same region as GPUs → free egress to them)
│   ├── patchguard-data        # IDNet ~490GB, DocLayNet, FUNSD/CORD/ViDoRe, encoded-patch cache
│   └── patchguard-artifacts   # checkpoints, results/{run_id}/metrics.json, figures, W&B mirror
│
├── Artifact Registry
│   └── patchguard/repro:<git_sha>   # ONE pinned Docker image = the reproduce.sh environment
│
├── Vertex AI Custom Jobs      # the workhorse: submit → provision → run → auto-teardown
│   └── (retriever encode | attack train | defense sweep | eval | erasure)
│
└── (optional) 1× GCE L4 Spot VM   # only if you want an interactive dev box; auto-shutdown on idle
```

**Why this shape:** the Docker image is the same artifact your `reproduce.sh` uses locally, so a
Vertex job and a laptop run are bit-for-bit the same environment (ties directly to the
determinism/reproducibility plan). Results always land in GCS as `metrics.json` + fingerprint;
`make_tables.py` reads them — no number is ever hand-copied.

---

## 3. GPU selection matrix

Prices are **approximate, us-central1, on-demand** — verify against the live pricing page; they
drift. The point is the *ratios*, not the digits.

| Task | Machine | GPU | ~On-demand $/hr | ~Spot $/hr | Use which |
|---|---|---|---|---|---|
| Dev, alignment validator, ColModernVBERT | `g2-standard-8` | 1× L4 24GB | ~$0.85 | ~$0.28 | On-demand (short) |
| **IDNet encoding at scale** (embarrassingly parallel) | `g2-standard-8` ×N | L4 24GB | ~$0.85 | ~$0.28 | **Spot** (only place it's worth it) |
| Diffusion attack training (~40–120 GPU-hr w/ seeds) | `a2-highgpu-1g` | 1× A100 40GB | ~$3.67 | ~$1.3 | On-demand + checkpoint |
| ColPali / ColQwen3-4B validation | `a2-highgpu-1g` | 1× A100 40GB | ~$3.67 | — | On-demand |
| Decoder / baseline sweeps × 5 seeds | `g2-standard-8` or A100 | L4 / A100 | — | — | On-demand, fan out |
| Kill test (M1, 200 docs) | `g2-standard-8` | 1× L4 | ~$0.85 | — | On-demand |

Notes:
- **You do not need H100 / A3.** Nothing here is training a foundation model; ColModernVBERT is 250M
  params and runs on a single consumer-class L4. A100 is only for the diffusion attack and 4B-model
  inference.
- **A100 40GB is enough.** Reach for 80GB (`a2-ultragpu-1g`, ~$5/hr) only if diffusion + a 4B
  retriever must co-reside; usually they don't.
- **Cache encoded patches, not images.** Encoding IDNet once and caching the `(n_patches, d)` tensors
  in `patchguard-data` means every subsequent attack/defense/eval run reads cheap tensors, not 490GB
  of images through a GPU. This is the single biggest cost *and* wall-clock lever.

---

## 4. The "run train and close" workflow

### Path A — Vertex Custom Job (default; no keys, auto-teardown)

Build/push the image once per code change, then submit jobs. The machine exists only for the job.

```bash
# one-time per git_sha: build the pinned env and push
gcloud builds submit --tag \
  us-central1-docker.pkg.dev/patchguard/patchguard/repro:$(git rev-parse --short HEAD)

# submit a training run — provisions A100, runs, writes to GCS, tears down. No idle burn.
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=diffusion-attack-seed0 \
  --worker-pool-spec=\
machine-type=a2-highgpu-1g,\
accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,\
replica-count=1,\
container-image-uri=us-central1-docker.pkg.dev/patchguard/patchguard/repro:$(git rev-parse --short HEAD) \
  --args="experiment=attack/diffusion,seed=0,out=gs://patchguard-artifacts/runs/"
```

The container's entrypoint is just `python -m experiments.<claim> <hydra-overrides>`; it reads config,
runs, writes `gs://patchguard-artifacts/runs/{run_id}/metrics.json` with the run fingerprint, exits.
**Job done → GPU released automatically.** That is the whole "run and close" story — you don't manage
a VM lifecycle at all.

For the 5-seed / sweep pattern, submit 5 jobs (or a Vertex **hyperparameter-tuning job**); they fan
out in parallel up to your quota and each closes independently.

### Path B — GCE Spot VM (only for interactive dev, with a kill-switch)

If you want a box to poke at:

```bash
gcloud compute instances create pg-dev \
  --zone=us-central1-a --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --metadata=startup-script='#! /bin/bash
    # auto-shutdown if GPU idle >30min — the anti-"left it running overnight" guard
    cat >/etc/cron.d/idle <<EOF
*/10 * * * * root [ $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) -lt 5 ] && \
  echo idle >> /var/idle && [ $(wc -l </var/idle) -ge 3 ] && poweroff || rm -f /var/idle
EOF'
```

`--instance-termination-action=DELETE` + the idle-poweroff cron means the box can't quietly bleed
credits. **Always prefer Path A** — a VM you have to remember to stop is a VM you'll forget to stop.

---

## 5. Cost guardrails (the part that actually matters with credits)

- **Label everything** `--labels=project=patchguard,phase=m2` → per-phase cost breakdown in Billing.
- **No committed-use discounts.** CUDs lock spend for 1–3 years; credits favor pure on-demand/Spot.
- **Idle is the only real enemy.** A forgotten `a2` VM is ~$88/day. Path A can't idle; for Path B the
  `DELETE`-on-termination + idle cron is mandatory.
- **Storage is negligible:** 490GB standard ≈ **$10/month**. Don't waste time optimizing it. Do set a
  90-day lifecycle rule to auto-delete stale `runs/` intermediates.
- **Egress:** keep GPUs and buckets in the **same region** → GPU↔GCS traffic is free. You only pay
  egress if you download results to your laptop, which is tiny.
- **Quota is your hard cap.** Keeping A100 quota at 2 means even a runaway sweep can't spin up 50
  A100s. Quota is a better guardrail than willpower.

---

## 6. Cost estimate mapped to the timeline

Generous, on-demand, no Spot discount assumed (so real spend will be lower):

| Phase | What runs | Rough GPU-hr | ~Cost |
|---|---|---|---|
| M1 kill test | L4 encode + decoder, 200 FUNSD × 5 seeds + 4a probe | ~30 L4-hr | ~$30 |
| M2 attack at scale | diffusion train (3 variants × 5 seeds) + IDNet encode | ~120 A100-hr + ~60 L4-hr (Spot) | ~$450 + ~$20 |
| M3 baselines | 5 methods × tuning sweep × 5 seeds, mostly L4 | ~200 L4-hr | ~$170 |
| M4 defense + adaptive | frontier sweep + adaptive attacks | ~100 A100-hr | ~$370 |
| M5 erasure + packaging | dilation sweep, storage-layer test | ~40 A100-hr | ~$150 |
| Storage (6 months) | 490GB + artifacts | — | ~$70 |
| **Total** | | | **~$1.3k on-demand / ~$0.7k with Spot on the sweeps** |

**You will spend under $2k of your $25k.** The remaining ~$23k is slack for: re-runs after bugs,
extra seeds if a CI is wide, an 80GB A100 if memory bites, ViDoRe-scale utility eval, and reviewer
"can you also run X" requests during rebuttal. You are not compute-constrained on this paper at all —
which means the discipline to import is *ephemeral runs and teardown*, not saving money.

---

## 7. Gotchas specific to your setup

1. **Org key policy.** Vertex/GCE use attached service accounts via workload identity — **no exported
   keys needed**, so the `disableServiceAccountKeyCreation` policy shouldn't bite the *runtime*. It
   can bite *local auth*; use `gcloud auth application-default login` (keyless ADC, already confirmed
   working for Vertex in your notes). If `constraints/gcp.resourceLocations` is inherited and pins
   regions, confirm `us-central1` is allowed before requesting quota there.
2. **Zero default GPU quota + Vertex's separate quota bucket.** Request *both* Compute Engine GPU
   quota and Vertex `CustomModelTraining*GPUs` quota, or the job queues forever with a cryptic
   quota error.
3. **A100 Spot availability is spiky.** If you do use Spot for A100, jobs may sit pending. Given your
   budget, just don't — run A100 on-demand. Use Spot only for L4 encoding.
4. **`torch.use_deterministic_algorithms(True)` in a container.** Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`
   as an env var in the Dockerfile, or determinism silently degrades — matches the repro plan.
5. **Deterministic base image.** Pin the CUDA/torch base image by digest, not tag, so a Vertex job in
   M5 uses the exact environment of M2. This is your reproducibility claim living in infra.
6. **Teardown at the end.** When the paper's submitted and artifact-eval mirror is pushed to HF/Zenodo,
   `gcloud projects delete patchguard` stops all residual charges in one command — the reason for the
   dedicated project.

---

## 8. One-screen runbook

1. Create project `patchguard`, attach $25k billing, set budget alerts.
2. Request L4 (×4) + A100 (×2) quota — Compute **and** Vertex buckets — in `us-central1`.
3. `gcloud auth application-default login`; run a tiny Vertex GPU smoke job to prove keyless ADC + quota.
4. Create `patchguard-data` / `patchguard-artifacts` buckets; upload a stratified IDNet sample + FUNSD.
5. Build & push `repro:<sha>` to Artifact Registry.
6. **Encode once, cache patches** to `patchguard-data`.
7. Submit Vertex Custom Jobs per experiment/seed → each writes `metrics.json` + fingerprint → auto-teardown.
8. `make_tables.py` reads GCS results → LaTeX. Never hand-copy a number.
9. On submit: `gcloud projects delete patchguard`.
