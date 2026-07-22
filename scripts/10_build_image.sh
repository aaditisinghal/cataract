#!/usr/bin/env bash
# Build + push the repro image to Artifact Registry via Cloud Build (no local Docker needed).
# Tag = short git sha, so every job pins the exact code that produced it (reproducibility).
set -euo pipefail
GC="$HOME/google-cloud-sdk/bin/gcloud"
REGION="us-central1"
PROJECT="patchguard-reakon"
SHA="$(git rev-parse --short HEAD)"
IMG="${REGION}-docker.pkg.dev/${PROJECT}/patchguard/repro:${SHA}"

echo "==> building ${IMG}"
"$GC" builds submit --project="${PROJECT}" --tag="${IMG}" --timeout=1800s .
echo "==> pushed ${IMG}"
echo "${IMG}" > .last_image
echo "(wrote .last_image for scripts/11_train_job.sh)"
