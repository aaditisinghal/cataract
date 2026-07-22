"""Real decoder-training entrypoint (MASTER_PLAN S5/S6). Runs as the Vertex job's container command.

Pipeline: load FUNSD (image + field boxes) -> encode with ColPali -> train PatchGridDecoder with the
text-region-weighted loss -> checkpoint + metrics. GCS-aware: gs:// paths for --data/--out are staged
locally and results uploaded back, so the container is self-contained.

Local dry run (no GPU/model — proves argument wiring on the mock backend):
    python -m experiments.train_funsd --selftest --out results/decoder_selftest
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from patchguard.attack.train import Sample, TrainConfig, build_dataset, train_decoder


def load_funsd_samples(root: str, split: str, limit: int | None, granularity: str) -> list[Sample]:
    from PIL import Image

    from patchguard.data.funsd import iter_funsd

    samples: list[Sample] = []
    for ps in iter_funsd(root, split=split, granularity=granularity):
        img = np.array(Image.open(ps.image_path).convert("RGB"))
        samples.append(Sample(image=img, field_boxes=ps.boxes(), orig_size=ps.size))
        if limit and len(samples) >= limit:
            break
    if not samples:
        raise RuntimeError(f"no FUNSD samples under {root}/{split}")
    return samples


def _selftest_samples(n: int = 6, seed: int = 0) -> list[Sample]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        img[8:20, 8:48] = 30
        out.append(Sample(img, [(8, 8, 48, 20)], (64, 64)))
    return out


def _gcs_download(uri: str, dest: Path) -> Path:
    from google.cloud import storage

    bucket_name, _, prefix = uri.replace("gs://", "").partition("/")
    client = storage.Client()
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        rel = blob.name[len(prefix) :].lstrip("/")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target)
    return dest


def _gcs_upload(src: Path, uri: str) -> None:
    from google.cloud import storage

    bucket_name, _, prefix = uri.replace("gs://", "").partition("/")
    bucket = storage.Client().bucket(bucket_name)
    for p in src.rglob("*"):
        if p.is_file():
            bucket.blob(f"{prefix.rstrip('/')}/{p.relative_to(src)}").upload_from_filename(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="FUNSD root (local dir or gs:// prefix)")
    ap.add_argument("--split", default="training_data")
    ap.add_argument("--out", default="results/decoder", help="output dir (local or gs://)")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--granularity", default="word", choices=["entity", "word"])
    ap.add_argument("--selftest", action="store_true", help="mock backend, tiny synthetic pages")
    args = ap.parse_args()

    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)

    if args.selftest:
        from patchguard.retrievers.mock import MockRetriever

        retriever = MockRetriever(grid=(8, 8), input_size=(64, 64), dim=16, n_prefix_tokens=1)
        samples = _selftest_samples()
        cfg = TrainConfig(
            out_size=(64, 64), epochs=3, batch_size=3, base_channels=16,
            use_lpips=False, w_lpips=0.0, ckpt_dir=str(local_out),
        )
    else:
        from patchguard.retrievers.colpali import ColPaliRetriever

        data_root = args.data
        if str(data_root).startswith("gs://"):
            data_root = str(_gcs_download(data_root, Path(tempfile.mkdtemp())))
        retriever = ColPaliRetriever(model_name=args.model)
        samples = load_funsd_samples(data_root, args.split, args.limit, args.granularity)
        cfg = TrainConfig(epochs=args.epochs, ckpt_dir=str(local_out))

    result = build_dataset(samples, retriever, cfg)
    out = train_decoder(result, cfg)
    print(f"trained decoder on {len(samples)} pages -> {out['checkpoint']}")
    print(f"final: {out['history'][-1]}")

    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
        print(f"uploaded results to {args.out}")
        shutil.rmtree(local_out, ignore_errors=True)


if __name__ == "__main__":
    main()
