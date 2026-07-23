# PatchGuard repro image — the reproducibility claim lives here.
# Torch + CUDA come from the base (matches the smoke job's torch 2.3.x / cu121); we add only our
# package + retriever/attack deps so we do NOT reinstall torch (which would break the CUDA match).
#
# Pin the base by DIGEST once the first build succeeds (replace :tag with @sha256:...). Build+iterate:
# colpali-engine may pin a torch range — if the build complains, constrain it here, don't upgrade torch.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# System libs for PIL image IO + Tesseract OCR (PFRR measurement).
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git tesseract-ocr fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Our package (core dep = numpy) + the model/attack/runtime stack, WITHOUT torch (base provides it).
COPY pyproject.toml ./
COPY patchguard ./patchguard
COPY experiments ./experiments
# Pin colpali-engine to the release compatible with the base image's torch 2.3.1:
#   0.3.5 -> torch>=2.2 (base 2.3.1 OK) + transformers 4.46.x (avoids the transformers-5.x break).
# Do NOT pin transformers separately — let colpali-engine resolve its own compatible transformers/peft.
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir \
        "pillow" \
        "colpali-engine==0.3.5" \
        "google-cloud-storage" \
        "pytesseract" \
        "diffusers==0.31.0" \
        "accelerate" \
        "qwen-vl-utils" \
    && (pip install --no-cache-dir lpips || true)

# Default: show the entrypoint help.
ENTRYPOINT ["python", "-m", "experiments.train_funsd"]
CMD ["--help"]
