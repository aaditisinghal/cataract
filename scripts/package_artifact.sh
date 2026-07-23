#!/usr/bin/env bash
# Assemble the artifact-evaluation release bundle: pull every result JSON from the artifacts bucket,
# gather the rendered figures + reproduce.sh + ARTIFACT.md (+ the numbers/plan docs), tar it, and print
# the checksum. The Zenodo / HuggingFace upload is documented as COMMENTED commands at the tail — this
# script never uploads and never touches the GPU.
#
# Usage: scripts/package_artifact.sh [RUNS_PREFIX]
#   RUNS_PREFIX defaults to gs://patchguard-reakon-artifacts/runs
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (path may contain spaces — everything below is quoted)

GC="${GCLOUD:-$HOME/google-cloud-sdk/bin/gcloud}"
RUNS="${1:-gs://patchguard-reakon-artifacts/runs}"
DIST_ROOT="dist"

# Provenance: tag the bundle with the exact code + image that produced the results.
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
IMG="$(cat .last_image 2>/dev/null || echo 'us-central1-docker.pkg.dev/patchguard-reakon/patchguard/repro:'"${SHA}")"
STAGE="${DIST_ROOT}/patchguard-artifact-${SHA}"
TARBALL="${DIST_ROOT}/patchguard-artifact-${SHA}.tar.gz"

echo "==> assembling artifact bundle for sha=${SHA}"
echo "    image  = ${IMG}"
echo "    runs   = ${RUNS}"
echo "    stage  = ${STAGE}"

rm -rf "${STAGE}"
mkdir -p "${STAGE}/results" "${STAGE}/figures" "${STAGE}/docs"

# 1) result JSONs from the artifacts bucket (every per-experiment / per-seed run).
echo "==> pulling result JSONs from ${RUNS}"
if [ "${SKIP_GCS:-0}" != "1" ]; then
  "${GC}" storage cp -r "${RUNS}" "${STAGE}/results/runs" \
    || echo "!! WARN: could not pull ${RUNS} (auth? empty bucket?) — bundling local results/ only"
fi
# also fold in any locally-generated JSON (efficiency_bench, make_figures inputs) if present.
if compgen -G "results/**/*.json" >/dev/null 2>&1 || find results -name '*.json' -print -quit 2>/dev/null | grep -q .; then
  mkdir -p "${STAGE}/results/local"
  find results -name '*.json' -exec cp {} "${STAGE}/results/local/" \; 2>/dev/null || true
fi

# 2) rendered figures (paper-ready PDF + PNG).
echo "==> gathering figures from paper/figures"
if find paper/figures -type f \( -name '*.pdf' -o -name '*.png' \) -print -quit 2>/dev/null | grep -q .; then
  find paper/figures -type f \( -name '*.pdf' -o -name '*.png' \) -exec cp {} "${STAGE}/figures/" \;
else
  echo "!! WARN: no figures under paper/figures — run 'python3 -m experiments.make_figures' first"
fi

# 3) the driver + the appendix + the numbers-of-record.
echo "==> gathering reproduce.sh + docs"
cp reproduce.sh                 "${STAGE}/reproduce.sh"
cp docs/ARTIFACT.md             "${STAGE}/docs/ARTIFACT.md"
cp docs/RESULTS.md              "${STAGE}/docs/RESULTS.md"       2>/dev/null || true
cp docs/COMPLETION_PLAN.md      "${STAGE}/docs/COMPLETION_PLAN.md" 2>/dev/null || true
cp README.md                    "${STAGE}/README.md"            2>/dev/null || true

# 4) a manifest so the bundle is self-describing (image pin + contents + build time).
{
  echo "PatchGuard artifact bundle"
  echo "git_sha:    ${SHA}"
  echo "repro_image: ${IMG}"
  echo "runs_prefix: ${RUNS}"
  echo "built_at:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "contents:"
  ( cd "${STAGE}" && find . -type f | sort | sed 's/^/  /' )
} > "${STAGE}/MANIFEST.txt"

# 5) tar it up.
echo "==> writing ${TARBALL}"
tar -C "${DIST_ROOT}" -czf "${TARBALL}" "patchguard-artifact-${SHA}"

# 6) checksum (portable: prefer sha256sum, fall back to shasum -a 256).
echo "==> checksum"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${TARBALL}" | tee "${TARBALL}.sha256"
else
  shasum -a 256 "${TARBALL}" | tee "${TARBALL}.sha256"
fi

echo "==> done. bundle: ${TARBALL}"
echo "    unpack: tar xzf ${TARBALL}"

# ==================================================================================================
# UPLOAD STEP — documented only; do NOT run automatically (needs a human + credentials + a DOI reserve).
# ==================================================================================================
# Zenodo (reserve a DOI, create a deposition, attach the tarball, then publish in the web UI):
#   ZENODO_TOKEN=...   # personal access token with 'deposit:write'
#   DEP=$(curl -s -X POST "https://zenodo.org/api/deposit/depositions?access_token=${ZENODO_TOKEN}" \
#            -H "Content-Type: application/json" -d '{}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
#   BUCKET=$(curl -s "https://zenodo.org/api/deposit/depositions/${DEP}?access_token=${ZENODO_TOKEN}" \
#            | python3 -c 'import sys,json;print(json.load(sys.stdin)["links"]["bucket"])')
#   curl -s -X PUT "${BUCKET}/$(basename "${TARBALL}")?access_token=${ZENODO_TOKEN}" --upload-file "${TARBALL}"
#   # then set metadata (title/authors/license) and hit Publish in the Zenodo UI to mint the DOI.
#
# HuggingFace (dataset or model repo; large-file friendly):
#   huggingface-cli login
#   huggingface-cli repo create patchguard-artifact --type dataset
#   huggingface-cli upload patchguard-artifact "${TARBALL}" "$(basename "${TARBALL}")" --repo-type dataset
#
# (Optionally archive the repro image itself for offline evaluators — requires docker, not run here:)
#   docker pull "${IMG}" && docker save "${IMG}" | gzip > "${DIST_ROOT}/repro-image-${SHA}.tar.gz"
