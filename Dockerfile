# PatchGuard repro image — the reproducibility claim lives here.
# Torch + CUDA come from the base (matches the smoke job's torch 2.3.x / cu121); we add only our
# package + retriever/attack deps so we do NOT reinstall torch (which would break the CUDA match).
#
# Pin the base by DIGEST once the first build succeeds (replace :tag with @sha256:...). Build+iterate:
# colpali-engine may pin a torch range — if the build complains, constrain it here, don't upgrade torch.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HToken=0

WORKDIR /app

# System libs for PIL image IO.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# Our package (core dep = numpy) + the model/attack/runtime stack, WITHOUT torch (base provides it).
COPY pyproject.toml ./
COPY patchguard ./patchguard
COPY experiments ./experiments
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir \
        pillow \
        transformers>=4.44 \
        colpali-engine \
        google-cloud-storage \
        && pip install --no-cache-dir lpips || true

# Default: show the entrypoint help.
ENTRYPOINT ["python", "-m", "experiments.train_funsd"]
CMD ["--help"]
