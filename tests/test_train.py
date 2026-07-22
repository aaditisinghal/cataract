"""CPU training-loop test on the mock retriever + tiny synthetic pages."""

import json
from pathlib import Path

import numpy as np

from patchguard.attack.train import Sample, TrainConfig, build_dataset, train_decoder
from patchguard.retrievers.mock import MockRetriever


def _samples(n=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        img = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        img[4:12, 4:24] = 30  # a dark "field" region
        out.append(Sample(image=img, field_boxes=[(4, 4, 24, 12)], orig_size=(32, 32)))
    return out


def _cfg(tmp):
    return TrainConfig(
        out_size=(32, 32), epochs=4, batch_size=3, lr=1e-3, base_channels=16,
        w_lpips=0.0, use_lpips=False, seed=0, ckpt_dir=str(tmp),
    )


def test_build_dataset_shapes():
    r = MockRetriever(grid=(4, 4), input_size=(32, 32), dim=8, n_prefix_tokens=1)
    ds = build_dataset(_samples(4), r, _cfg("results/_tmp"))
    assert ds.grid == (4, 4) and ds.dim == 8
    patches, target, weight = ds[0]
    assert patches.shape == (16, 8)  # image patches only (prefix stripped)
    assert target.shape == (3, 32, 32)
    assert weight.shape == (1, 32, 32)


def test_training_reduces_loss(tmp_path):
    r = MockRetriever(grid=(4, 4), input_size=(32, 32), dim=8, n_prefix_tokens=1)
    cfg = _cfg(tmp_path)
    ds = build_dataset(_samples(6), r, cfg)
    out = train_decoder(ds, cfg)
    hist = out["history"]
    assert len(hist) == cfg.epochs
    # loss should not increase over training on this trivial fit
    assert hist[-1]["total"] <= hist[0]["total"] + 1e-6
    assert np.isfinite(hist[-1]["total"])


def test_checkpoint_and_metrics_written(tmp_path):
    r = MockRetriever(grid=(4, 4), input_size=(32, 32), dim=8, n_prefix_tokens=1)
    cfg = _cfg(tmp_path)
    ds = build_dataset(_samples(4), r, cfg)
    out = train_decoder(ds, cfg)
    assert Path(out["checkpoint"]).exists()
    metrics = json.loads((Path(cfg.ckpt_dir) / "metrics.json").read_text())
    assert "fingerprint" in metrics and "history" in metrics
    assert metrics["n_samples"] == 4
