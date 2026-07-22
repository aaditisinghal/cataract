# Step 1 — request GPU quota (the one blocking wait; ~hours to 1 day)

New projects start at **0 GPU quota**, and Vertex has a **separate** quota bucket from Compute Engine.
Request both, or jobs queue forever with a cryptic quota error.

## Console path (most reliable)
1. Console → **IAM & Admin → Quotas**. Project = `patchguard-…`.
2. Filter by **Service: Vertex AI API** and request:
   - `Custom model training Nvidia A100 GPUs per region` → **2** (region `us-central1`)
   - `Custom model training Nvidia L4 GPUs per region` → **4**
3. Filter by **Service: Compute Engine API** and request (needed if you also use GCE Spot dev VMs):
   - `NVIDIA_A100_GPUS` (region `us-central1`) → **2**
   - `NVIDIA_L4_GPUS` (region `us-central1`) → **4**
4. Submit with a one-line justification ("academic security research, ephemeral training jobs").
   Approval is usually hours; sometimes a day.

## Region fallback
If `us-central1` A100 is constrained, also request in `us-east4`. Keep the region consistent with
your buckets so GPU↔GCS egress stays free.

## Why keep the numbers small
Quota is your hard cost cap: A100=2 means even a runaway sweep can't spin up 50 A100s. Better than
willpower. Raise later if a sweep genuinely needs more parallelism.
