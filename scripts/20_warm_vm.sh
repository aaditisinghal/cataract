#!/usr/bin/env bash
# Warm GPU VM: pay the cold start ONCE (docker install + image pull + ColPali download), then run
# many experiments fast via scripts/21_exec.sh. NO self-delete on container exit (stays warm); a
# max-run-duration backstop + DELETE guards against runaway cost. Delete it yourself when done:
#   gcloud compute instances delete $(cat .warm_vm) --zone=$(cat .warm_zone) --quiet
set -euo pipefail
GC="$HOME/google-cloud-sdk/bin/gcloud"
PROJECT="patchguard-reakon"
ZONE="${1:-us-central1-a}"
PROFILE="${2:-a100}"
VM="patchguard-warm-$(date +%s | tail -c 5)"
IMG="$(cat .last_image)"
case "$PROFILE" in
  l4)   MACHINE="g2-standard-8"; ACCEL=(--accelerator=type=nvidia-l4,count=1);;
  a100) MACHINE="a2-highgpu-1g"; ACCEL=();;
esac

STARTUP="$(cat <<EOF
#!/bin/bash
exec > /var/log/warm.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive
for i in \$(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 5; done
apt-get update -y && apt-get install -y docker.io curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker && sleep 3
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker pull ${IMG}
mkdir -p /opt/hfcache && chmod 777 /opt/hfcache
# warm ColPali into the persistent cache so experiments skip the download
docker run --gpus all -v /opt/hfcache:/root/.cache/huggingface --entrypoint python ${IMG} -c "from colpali_engine.models import ColPali, ColPaliProcessor; ColPali.from_pretrained('vidore/colpali-v1.3'); ColPaliProcessor.from_pretrained('vidore/colpali-v1.3'); print('COLPALI WARM')"
touch /opt/warm_ready
echo WARM_VM_READY
EOF
)"

echo "==> creating warm VM ${VM} (${PROFILE}) in ${ZONE} with image ${IMG}"
"$GC" compute instances create "${VM}" \
  --project="${PROJECT}" --zone="${ZONE}" --machine-type="${MACHINE}" \
  ${ACCEL[@]+"${ACCEL[@]}"} \
  --maintenance-policy=TERMINATE --provisioning-model=STANDARD \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB --scopes=cloud-platform \
  --max-run-duration=28800s --instance-termination-action=DELETE \
  --metadata-from-file=startup-script=<(printf '%s' "$STARTUP") \
  --labels=project=patchguard,phase=warm
echo "${VM}" > .warm_vm; echo "${ZONE}" > .warm_zone
echo "==> warming (docker + image + ColPali). Poll: scripts/21_exec.sh waits for /opt/warm_ready."
