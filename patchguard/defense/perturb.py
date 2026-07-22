"""Embedding perturbation defenses (MASTER_PLAN S3a/S8 — the heart of Claim 3).

Two mechanisms the kill test contrasts:
  * ``flat_gaussian``       — noise on EVERY image patch (the naive floor; no locality).
  * ``patch_scoped_gaussian`` — noise ONLY on the patches a localizer flagged as PII.

The thesis (Claim 3): patch-scoped spends its utility budget on ~40 patches instead of ~1024, so it
destroys invertibility of PII while barely moving retrieval — dominating the privacy/utility frontier.
A pooled bi-encoder (BiPali, a 1x1 grid) has no locality primitive, so patch-scoping is impossible
there; that asymmetry is the whole point.

Calibration (protocol S8): noise is scaled to each patch's own norm by default (``calibrate=
"local_norm"``), not a single global scale, so sparse and dense regions are protected evenly. ColPali
patch embeddings live ~on the unit sphere, so perturbed patches are renormalized by default.
"""

from __future__ import annotations

import numpy as np

from patchguard.retrievers.base import PageEncoding


def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _perturb_rows(
    patches: np.ndarray,
    rows: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    calibrate: str,
    renormalize: bool,
) -> np.ndarray:
    """Return a copy of ``patches`` with Gaussian noise added to the given row indices."""
    out = patches.copy()
    if rows.size == 0 or sigma <= 0:
        return out
    d = out.shape[1]
    noise = rng.standard_normal((rows.size, d)).astype(out.dtype)
    if calibrate == "local_norm":
        scale = np.linalg.norm(out[rows], axis=1, keepdims=True)
    elif calibrate == "global":
        scale = 1.0
    else:
        raise ValueError("calibrate must be 'local_norm' or 'global'")
    out[rows] = out[rows] + sigma * scale * noise
    if renormalize:
        out[rows] = _l2(out[rows])
    return out


def _image_rows(enc: PageEncoding) -> np.ndarray:
    gh, gw = enc.grid
    start = enc.n_prefix_tokens
    return np.arange(start, start + gh * gw)


def _with_patches(enc: PageEncoding, patches: np.ndarray) -> PageEncoding:
    return PageEncoding(
        patches=patches,
        grid=enc.grid,
        input_size=enc.input_size,
        model_id=enc.model_id,
        resize_policy=enc.resize_policy,
        n_prefix_tokens=enc.n_prefix_tokens,
    )


def flat_gaussian(
    enc: PageEncoding,
    sigma: float,
    seed: int = 0,
    calibrate: str = "local_norm",
    renormalize: bool = True,
) -> PageEncoding:
    """Add noise to ALL image patches. The naive baseline in the kill test's frontier."""
    rng = np.random.default_rng(seed)
    rows = _image_rows(enc)
    return _with_patches(enc, _perturb_rows(enc.patches, rows, sigma, rng, calibrate, renormalize))


def patch_scoped_gaussian(
    enc: PageEncoding,
    mask: np.ndarray,
    sigma: float,
    seed: int = 0,
    calibrate: str = "local_norm",
    renormalize: bool = True,
) -> PageEncoding:
    """Add noise ONLY to patches selected by ``mask`` (a full-sequence boolean from a localizer).

    ``mask`` length must equal ``enc.n_patches``; only True positions are perturbed. Prefix/trailing
    tokens are left untouched unless the mask marks them (localizers never do).
    """
    if mask.shape[0] != enc.n_patches:
        raise ValueError(f"mask length {mask.shape[0]} != n_patches {enc.n_patches}")
    rng = np.random.default_rng(seed)
    rows = np.nonzero(mask)[0]
    return _with_patches(enc, _perturb_rows(enc.patches, rows, sigma, rng, calibrate, renormalize))
