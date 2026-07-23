#!/usr/bin/env bash
# Run an experiment on the warm VM via SSH (image cached, ColPali cached). ~2 min instead of ~20.
# Usage: scripts/21_exec.sh <experiment_module> <out_subdir> [extra args...]
#   scripts/21_exec.sh experiments.claim1 claim1 --n 60 --distractors 200 --font-size 24
set -euo pipefail
GC="$HOME/google-cloud-sdk/bin/gcloud"
VM="$(cat .warm_vm)"; ZONE="$(cat .warm_zone)"; IMG="$(cat .last_image)"
MOD="$1"; SUB="$2"; shift 2
TAG="$(basename "$IMG" | tr ':' '-')"
OUT="gs://patchguard-reakon-artifacts/runs/${SUB}-${TAG}"

# wait until the warm-up finished (ColPali cached)
"$GC" compute ssh "$VM" --zone="$ZONE" --quiet --command="until [ -f /opt/warm_ready ]; do sleep 10; done; echo READY" 2>/dev/null

echo "==> exec ${MOD} on ${VM} -> ${OUT}"
"$GC" compute ssh "$VM" --zone="$ZONE" --quiet --command="\
  sudo docker run --gpus all -v /opt/hfcache:/root/.cache/huggingface --entrypoint python ${IMG} \
  -m ${MOD} --out ${OUT} $*" 2>&1 | tail -40
echo "==> done. result: gcloud storage cat ${OUT}/*.json"
