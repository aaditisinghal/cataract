#!/usr/bin/env bash
# Launch decoder training as an ephemeral Vertex job (1x L4). Reads FUNSD from the data bucket,
# writes checkpoint + metrics to the artifacts bucket, then the machine tears down.
set -euo pipefail
GC="$HOME/google-cloud-sdk/bin/gcloud"
REGION="${1:-us-central1}"   # pass a region as $1 to switch pools, e.g. us-east4 for L4 availability
PROJECT="patchguard-reakon"
SHA="$(git rev-parse --short HEAD)"
IMG="$(cat .last_image 2>/dev/null || echo "${REGION}-docker.pkg.dev/${PROJECT}/patchguard/repro:${SHA}")"
OUT="gs://patchguard-reakon-artifacts/runs/decoder-${SHA}"

CFG="$(mktemp -t train_job).yaml"
cat > "$CFG" <<YAML
workerPoolSpecs:
- machineSpec:
    machineType: g2-standard-8
    acceleratorType: NVIDIA_L4
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    imageUri: ${IMG}
    args:
    - "--data"
    - "gs://patchguard-reakon-data/funsd"
    - "--split"
    - "training_data"
    - "--out"
    - "${OUT}"
    - "--epochs"
    - "40"
    - "--limit"
    - "200"
    - "--granularity"
    - "word"
YAML

echo "==> training job -> ${OUT}"
"$GC" ai custom-jobs create --region="${REGION}" --project="${PROJECT}" \
  --display-name="patchguard-decoder-${SHA}" --config="${CFG}" \
  --labels="project=patchguard,phase=s6-attack"
echo "==> when done: gcloud storage cat ${OUT}/metrics.json"
