#!/usr/bin/env bash
# Stage FUNSD (199 real forms, ~50MB) into the data bucket for the kill test.
# Layout after upload: gs://<bucket>/funsd/{training_data,testing_data}/{annotations,images}
set -euo pipefail
GC="$HOME/google-cloud-sdk/bin/gcloud"
BUCKET="patchguard-reakon-data"

TMP="$(mktemp -d)"
echo "==> downloading FUNSD"
curl -fsSL -o "$TMP/funsd.zip" "https://guillaumejaume.github.io/FUNSD/dataset.zip"
echo "==> extracting"
unzip -q "$TMP/funsd.zip" -d "$TMP/x"
# the zip nests the data under a top dir; find the one that actually contains training_data
SRC="$(dirname "$(find "$TMP/x" -type d -name training_data | head -1)")"
test -n "$SRC" || { echo "FUNSD layout unexpected"; exit 1; }
echo "==> uploading to gs://${BUCKET}/funsd/"
"$GC" storage cp -r "$SRC"/training_data "$SRC"/testing_data "gs://${BUCKET}/funsd/"
echo "==> done. verify:"
"$GC" storage ls "gs://${BUCKET}/funsd/training_data/" | head
rm -rf "$TMP"
