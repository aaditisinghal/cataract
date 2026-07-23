#!/usr/bin/env bash
# Option (c): run a GPU job on a directly-provisioned GCE VM. Bypasses Vertex's framework provisioning
# (which stalled) and FAILS FAST on stockout instead of queueing.
#
# Usage: scripts/12_train_vm.sh [ZONE] [PROFILE l4|a100] [JOB train|killtest]
#
# Three teardown guards: startup self-delete on container exit; --max-run-duration + DELETE (45-min
# hard cap); on-demand (no preemption). Worst-case cost ~$0.65 (L4) / ~$1 (A100).
set -euo pipefail
GC="$HOME/google-cloud-sdk/bin/gcloud"
PROJECT="patchguard-reakon"
ZONE="${1:-us-central1-a}"
PROFILE="${2:-a100}"     # l4 | a100
JOB="${3:-train}"        # train | killtest
VM="patchguard-${JOB}-$(date +%s | tail -c 5)"
IMG="$(cat .last_image 2>/dev/null || echo us-central1-docker.pkg.dev/${PROJECT}/patchguard/repro:latest)"

case "$PROFILE" in
  l4)   MACHINE="g2-standard-8"; ACCEL=(--accelerator=type=nvidia-l4,count=1);;
  a100) MACHINE="a2-highgpu-1g"; ACCEL=();;
  *) echo "unknown profile $PROFILE"; exit 2;;
esac
IMG_FAM="common-cu129-ubuntu-2204-nvidia-580"

# container command per job — IMG/OUT fully resolved here (no runtime vars needed).
TAG="$(basename "$IMG" | tr ':' '-')"
case "$JOB" in
  killtest)
    OUT="gs://patchguard-reakon-artifacts/runs/killtest-${TAG}"
    CMD="docker run --gpus all --entrypoint python ${IMG} -m experiments.run_killtest --data gs://patchguard-reakon-data/funsd --out ${OUT} --epochs 120 --train-limit 149 --test-limit 20 --noise-levels 5"
    ;;
  diffusion)
    OUT="gs://patchguard-reakon-artifacts/runs/diffusion-${TAG}"
    CMD="docker run --gpus all --entrypoint python ${IMG} -m experiments.train_diffusion --data gs://patchguard-reakon-data/funsd --out ${OUT} --epochs 80 --limit 149 --dump 6"
    ;;
  overfit)
    OUT="gs://patchguard-reakon-artifacts/runs/overfit-${TAG}"
    CMD="docker run --gpus all --entrypoint python ${IMG} -m experiments.overfit_probe --data gs://patchguard-reakon-data/funsd --out ${OUT} --n 8 --epochs 800 --resolution 768 --channels 256"
    ;;
  *)
    OUT="gs://patchguard-reakon-artifacts/runs/decoder-gce-${TAG}"
    CMD="docker run --gpus all ${IMG} --data gs://patchguard-reakon-data/funsd --split training_data --out ${OUT} --epochs 40 --limit 200 --granularity word"
    ;;
esac

STARTUP="$(cat <<STARTUP_EOF
#!/bin/bash
exec > /var/log/patchguard-startup.log 2>&1
set -x
OUT="${OUT}"
NAME=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=\$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print \$NF}')
self_delete() { gsutil cp /var/log/patchguard-startup.log "\$OUT/vm_startup.log" || true; gcloud compute instances delete "\$NAME" --zone="\$ZONE" --quiet; }
trap self_delete EXIT
export DEBIAN_FRONTEND=noninteractive
for i in \$(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 5; done
apt-get update -y && apt-get install -y docker.io curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker && sleep 3
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker pull "${IMG}"
${CMD}
echo "JOB_EXIT=\$?"
STARTUP_EOF
)"

echo "==> launching ${VM} (${PROFILE}/${MACHINE}, job=${JOB}) in ${ZONE}"
echo "==> image ${IMG}"
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
  --max-run-duration=3000s \
  --instance-termination-action=DELETE \
  --metadata-from-file=startup-script=<(printf '%s' "$STARTUP") \
  --labels=project=patchguard,phase=s6-${JOB}-gce

echo "${VM}" > .last_vm
echo "==> launched. self-deletes on completion; hard-capped at 50 min."
echo "==> when done: gcloud storage cat ${OUT}/$([ "$JOB" = killtest ] && echo killtest.json || echo metrics.json)"
