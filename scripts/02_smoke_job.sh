#!/usr/bin/env bash
# Step 2 — Vertex "hello GPU" smoke job. Proves quota + keyless ADC + ephemeral run+teardown all work
# BEFORE you write real code against them. Uses a prebuilt PyTorch-GPU container (no image build yet).
# Provisions 1 L4, asserts CUDA, writes a file to the bucket, then the machine is released automatically.
set -euo pipefail

# ---- ACTUAL VALUES (match scripts/00_bootstrap.sh) ----
PROJECT_ID="patchguard-reakon"
REGION="us-central1"
# -------------------------------------------------------
BUCKET="gs://${PROJECT_ID}-artifacts"
gcloud config set project "${PROJECT_ID}"

# The smoke payload: assert GPU, print its name, write a proof file to GCS. Runs INSIDE the container.
read -r -d '' PYCODE <<'PY' || true
import torch, datetime, os
from google.cloud import storage
assert torch.cuda.is_available(), "NO CUDA — quota/accelerator misconfigured"
name = torch.cuda.get_device_name(0)
msg = f"OK {name} torch={torch.__version__} at {datetime.datetime.utcnow().isoformat()}Z"
print(msg)
bucket = os.environ["SMOKE_BUCKET"].replace("gs://", "").split("/")[0]
storage.Client().bucket(bucket).blob("smoke/hello_gpu.txt").upload_from_string(msg)
print("wrote gs proof file")
PY

# Prebuilt Vertex training image with torch + google-cloud-storage already inside.
IMG="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest"

gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="patchguard-smoke-gpu" \
  --worker-pool-spec="machine-type=g2-standard-8,accelerator-type=NVIDIA_L4,accelerator-count=1,replica-count=1,container-image-uri=${IMG}" \
  --args="python,-c,${PYCODE}" \
  --update-labels="project=patchguard,phase=s1-smoke" \
  --env-vars="SMOKE_BUCKET=${BUCKET}"

echo ""
echo "Submitted. Watch it:"
echo "  gcloud ai custom-jobs list --region=${REGION} --filter='displayName:patchguard-smoke-gpu'"
echo "Then confirm the proof file (machine already torn down by now):"
echo "  gcloud storage cat ${BUCKET}/smoke/hello_gpu.txt"
echo ""
echo "If that prints 'OK NVIDIA L4 ...', S1's GCP gate is PASSED — keyless ADC + quota + teardown all work."
