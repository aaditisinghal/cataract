#!/usr/bin/env bash
# PatchGuard GCP bootstrap — project, billing, APIs, buckets, registry, keyless ADC.
# Run ONCE. Idempotent-ish (safe to re-run; existing resources will just warn).
# Prereqs: gcloud CLI installed (`brew install --cask google-cloud-sdk`), and you are an
# owner/editor on the billing account. Your org blocks SA-key creation — we never create keys;
# Vertex/GCE use attached service accounts + your keyless ADC, which is enough.
set -euo pipefail

# ---- EDIT THESE ------------------------------------------------------------
PROJECT_ID="patchguard-$(whoami)"          # must be globally unique; edit if taken
BILLING_ACCT="XXXXXX-XXXXXX-XXXXXX"         # gcloud billing accounts list  -> copy the ID
REGION="us-central1"                        # best A100/L4 availability + price
# ---------------------------------------------------------------------------

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

echo "==> Keyless Application Default Credentials (this is the auth your notes confirm works)"
gcloud auth application-default login

echo "==> Budget-alert reminder (console step): Billing > Budgets & alerts >"
echo "    create budget on ${PROJECT_ID} with thresholds \$500 / \$1k / \$2.5k / \$5k."
echo ""
echo "DONE. Next: request GPU quota (scripts/01_quota.md), then run scripts/02_smoke_job.sh"
echo "PROJECT_ID=${PROJECT_ID}  REGION=${REGION}"
