#!/usr/bin/env bash
# Option (c): run decoder training on a directly-provisioned GCE L4 VM. Bypasses Vertex's framework
# provisioning (which stalled). Fails FAST if no capacity, instead of queueing.
#
# Teardown is on us, so THREE guards ensure it can't run away:
#   1. startup script self-deletes the VM the moment the container exits (fast path)
#   2. --max-run-duration + --instance-termination-action=DELETE  (platform-enforced 45-min hard cap)
#   3. on-demand (not spot) so it isn't preempted mid-run; job is minutes so cost ~$0.40, cap ~$0.65
set -euo pipefail
GC="$HOME/google-cloud-sdk/bin/gcloud"
PROJECT="patchguard-reakon"
ZONE="${1:-us-central1-a}"          # same region as image+buckets => no cross-region egress
PROFILE="${2:-a100}"                # l4 | a100  (both run our bf16 image; no rebuild)
VM="patchguard-train-$(date +%s | tail -c 5)"

# GPU profile -> machine + accelerator flags. a2 machines have the A100 built in (no --accelerator).
case "$PROFILE" in
  l4)   MACHINE="g2-standard-8"; ACCEL=(--accelerator=type=nvidia-l4,count=1); IMG_FAM="common-cu129-ubuntu-2204-nvidia-580";;
  a100) MACHINE="a2-highgpu-1g"; ACCEL=();                                    IMG_FAM="common-cu129-ubuntu-2204-nvidia-580";;
  *) echo "unknown profile $PROFILE"; exit 2;;
esac
IMG="$(cat .last_image 2>/dev/null || echo us-central1-docker.pkg.dev/${PROJECT}/patchguard/repro:d89f705)"
OUT="gs://patchguard-reakon-artifacts/runs/decoder-gce-$(basename "$IMG" | tr ':' '-')"

STARTUP="$(cat <<STARTUP_EOF
#!/bin/bash
exec > /var/log/patchguard-startup.log 2>&1
set -x
IMG="${IMG}"
OUT="${OUT}"
NAME=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print \$NF}')
self_delete() { gsutil cp /var/log/patchguard-startup.log "\$OUT/vm_startup.log" || true; gcloud compute instances delete "\$NAME" --zone="\$ZONE" --quiet; }
trap self_delete EXIT
for i in \$(seq 1 60); do nvidia-smi >/dev/null 2>&1 && docker info >/dev/null 2>&1 && break; sleep 5; done
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker pull "\$IMG"
docker run --gpus all "\$IMG" \
  --data gs://patchguard-reakon-data/funsd --split training_data \
  --out "\$OUT" --epochs 40 --limit 200 --granularity word
echo "TRAIN_EXIT=\$?"
STARTUP_EOF
)"

echo "==> launching ${VM} (${PROFILE}/${MACHINE}) in ${ZONE}  (image ${IMG})"
echo "==> results -> ${OUT}"
"$GC" compute instances create "${VM}" \
  --project="${PROJECT}" --zone="${ZONE}" \
  --machine-type="${MACHINE}" \
  ${ACCEL[@]+"${ACCEL[@]}"} \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --image-family="${IMG_FAM}" \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=120GB \
  --scopes=cloud-platform \
  --max-run-duration=2700s \
  --instance-termination-action=DELETE \
  --metadata-from-file=startup-script=<(printf '%s' "$STARTUP") \
  --labels=project=patchguard,phase=s6-attack-gce

echo "${VM}" > .last_vm
echo "==> launched. self-deletes on completion; hard-capped at 45 min."
echo "==> when done: gcloud storage cat ${OUT}/metrics.json"
