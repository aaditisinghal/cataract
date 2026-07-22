#!/usr/bin/env bash
# Step 2 — Vertex "hello GPU" smoke job. Proves quota + keyless ADC + ephemeral run+teardown work
# BEFORE writing real code against them. Uses a prebuilt PyTorch-GPU container (no image build yet).
# Provisions 1 L4, asserts CUDA, writes a proof file to the bucket, then the machine is released.
#
# IMPORTANT: the prebuilt Vertex containers use `run_module.py` as their ENTRYPOINT, which expects a
# Python *module* (`-m pkg`), NOT a raw `python -c` command. Passing a raw command via --args fails
# with "Invalid arguments specified to startup script". The fix is a config YAML that sets
# containerSpec.command, which OVERRIDES the entrypoint. That is what this script does.
set -euo pipefail

# ---- ACTUAL VALUES (match scripts/00_bootstrap.sh) ----
PROJECT_ID="patchguard-reakon"
REGION="us-central1"
# -------------------------------------------------------
BUCKET="${PROJECT_ID}-artifacts"
IMG="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest"
"$HOME/google-cloud-sdk/bin/gcloud" config set project "${PROJECT_ID}"

# Smoke payload (base64'd so no shell/comma escaping issues inside the YAML command string).
PYCODE='import torch
assert torch.cuda.is_available(), "NO CUDA AVAILABLE"
n = torch.cuda.get_device_name(0)
print("SMOKE OK:", n, "| torch", torch.__version__)
try:
    from google.cloud import storage
    storage.Client().bucket("'"${BUCKET}"'").blob("smoke/hello_gpu.txt").upload_from_string("OK "+n)
    print("wrote gs://'"${BUCKET}"'/smoke/hello_gpu.txt")
except Exception as e:
    print("gs write skipped:", repr(e))'
B64=$(printf '%s' "$PYCODE" | base64 | tr -d '\n')

CFG="$(mktemp -t smoke_job).yaml"
cat > "$CFG" <<YAML
workerPoolSpecs:
- machineSpec:
    machineType: g2-standard-8
    acceleratorType: NVIDIA_L4
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    imageUri: ${IMG}
    command:
    - python
    - -c
    - "import base64;exec(base64.b64decode('${B64}').decode())"
YAML

"$HOME/google-cloud-sdk/bin/gcloud" ai custom-jobs create \
  --region="${REGION}" \
  --display-name="patchguard-smoke-gpu" \
  --config="${CFG}" \
  --labels="project=patchguard,phase=s1-smoke"

echo ""
echo "Submitted. When it finishes (machine auto-torn-down), confirm the proof file:"
echo "  gcloud storage cat gs://${BUCKET}/smoke/hello_gpu.txt"
echo "If it prints 'OK NVIDIA L4 ...', S1's GCP gate is PASSED — keyless ADC + quota + teardown all work."
