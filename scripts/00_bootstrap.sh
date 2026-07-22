#!/usr/bin/env bash
# PatchGuard GCP bootstrap — project, billing, APIs, buckets, registry, keyless ADC.
# Run ONCE. Idempotent-ish (safe to re-run; existing resources will just warn).
# Prereqs: gcloud CLI installed (`brew install --cask google-cloud-sdk`), and you are an
# owner/editor on the billing account. Your org blocks SA-key creation — we never create keys;
# Vertex/GCE use attached service accounts + your keyless ADC, which is enough.
set -euo pipefail

# ---- ACTUAL VALUES (provisioned 2026-07-22) --------------------------------
PROJECT_ID="patchguard-reakon"             # created under org profitwise.app (483346253498)
BILLING_ACCT="014F80-4CDA1C-D6AEC2"        # "My Billing Account" (holds the $25k credits)
REGION="us-central1"                        # best A100/L4 availability + price
# ---------------------------------------------------------------------------
# NOTE: this project is ALREADY provisioned (project+billing+APIs+buckets+registry+ADC all done).
# This script is kept as the reproducible record / for rebuilding from scratch after teardown.

echo "==> Authenticate (browser)"
gcloud auth login

echo "==> Create project ${PROJECT_ID}"
gcloud projects create "${PROJECT_ID}" --name="PatchGuard" || echo "(project may already exist)"
gcloud config set project "${PROJECT_ID}"

echo "==> Link billing (needs billing-admin; may require org admin if inherited policy blocks)"
gcloud billing projects link "${PROJECT_ID}" --billing-account="${BILLING_ACCT}"

echo "==> Enable required APIs"
gcloud services enable \
  aiplatform.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

echo "==> Create GCS buckets (same region as GPUs => free egress to jobs)"
gcloud storage buckets create "gs://${PROJECT_ID}-data"      --location="${REGION}" || true
gcloud storage buckets create "gs://${PROJECT_ID}-artifacts" --location="${REGION}" || true

echo "==> Create Artifact Registry for the repro image"
gcloud artifacts repositories create patchguard \
  --repository-format=docker --location="${REGION}" \
  --description="PatchGuard repro images" || true

echo "==> Cloud Build IAM: newer Cloud Build runs as the Compute Engine default SA, which on a fresh"
echo "    project can't read its own source bucket / push to Artifact Registry. Grant the roles."
PROJECT_NUM="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
CB_SA="${PROJECT_NUM}-compute@developer.gserviceaccount.com"
for ROLE in roles/storage.admin roles/artifactregistry.writer roles/logging.logWriter roles/cloudbuild.builds.builder; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${CB_SA}" --role="${ROLE}" --condition=None >/dev/null && echo "  granted ${ROLE}"
done

echo "==> Keyless Application Default Credentials (this is the auth your notes confirm works)"
gcloud auth application-default login

echo "==> Budget-alert reminder (console step): Billing > Budgets & alerts >"
echo "    create budget on ${PROJECT_ID} with thresholds \$500 / \$1k / \$2.5k / \$5k."
echo ""
echo "DONE. Next: request GPU quota (scripts/01_quota.md), then run scripts/02_smoke_job.sh"
echo "PROJECT_ID=${PROJECT_ID}  REGION=${REGION}"
