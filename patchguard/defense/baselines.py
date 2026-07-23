"""Field baselines for embedding-privacy — the comparison set for Claim 2 (MASTER_PLAN S8).

Our RedactionProjection (``redact.py``) is a *learned, anisotropic* index-time transform. To claim it
is not just "a defense" but "the defense that dominates the frontier", we must beat the published
alternatives on the SAME privacy/utility frontier and the SAME threat model: the transform touches only
the STORED patches; queries run through the vanilla frozen encoder. This module ports three field
defenses, faithful-in-spirit, as index-time patch transforms with a common ``(patches, strength)``
signature so a single runner can sweep each one:

  * ``entroguard`` — arXiv:2503.12896 (EntroGuard). Entropy-/variance-driven perturbation: inject noise
    per EMBEDDING DIMENSION scaled by that dimension's spread across the stored patches, so
    high-information directions are noised more. Adaptation: their text-embedding entropy signal becomes
    the per-dim standard deviation of the multi-vector patch matrix (a cheap entropy proxy on the sphere).
  * ``press``     — ICASSP'25 PRESS (Privacy-pREserving Subspace removal). Estimate the top-k "PII"
    principal directions and project them OUT of every stored patch. Adaptation: if a PII-vs-content
    contrast is supplied (rows spanning the PII subspace, e.g. name-query minus topic-query directions),
    remove that subspace's top-k right singular vectors; otherwise fall back to the top-k principal
    components of the stored patches. The strength knob maps to k = round(strength * dim).
  * ``koga``      — arXiv:2412.04697. Adversarial embedding perturbation: an isotropic random component
    plus a "learned-ish" adversarial component. Adaptation: with no attacker to backprop through, the
    adversarial direction is approximated by a centroid-collapse push (drag each patch toward the corpus
    centroid, eroding separability) mixed with isotropic per-patch noise.

All three are UNINFORMED about which directions carry PII versus content (PRESS gets a contrast only when
one is handed in), so they are expected to trade utility for privacy roughly isotropically — the null
hypothesis our anisotropic learned P must dominate. A positive Claim-2 result is: at matched privacy,
learned P retains more retrieval utility than every baseline's best frontier point.

Dependency-light (numpy core; torch tensors are accepted and returned in kind). Every transform returns
UNIT-NORMALIZED patches of the SAME shape (ColPali patches live on the unit sphere).
"""

from __future__ import annotations

import numpy as np

BASELINE_NAMES = ("entroguard", "press", "koga")


# --------------------------------------------------------------------------------------------------
# numpy/torch plumbing — operate in numpy, hand back whatever type came in.
# --------------------------------------------------------------------------------------------------
def _is_torch(x: object) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _to_numpy(x):
    if _is_torch(x):
        return x.detach().cpu().numpy().astype(np.float32, copy=False), x
    return np.asarray(x, dtype=np.float32), None


def _restore(y: np.ndarray, ref):
    if ref is None:
        return y
    import torch  # local: never import torch at module top-level (see repo contract)

    return torch.as_tensor(y, dtype=ref.dtype, device=ref.device)


def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _as_rng(rng) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng(0)
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


# --------------------------------------------------------------------------------------------------
# Baselines. Each: (patches (..., d), strength float) -> unit-normalized patches, same shape/type.
# --------------------------------------------------------------------------------------------------
def entroguard(patches, strength: float, *, rng=None):
    """EntroGuard (arXiv:2503.12896), adapted: per-DIMENSION noise scaled by that dim's spread.

    strength scales a Gaussian whose per-dim standard deviation matches the patch matrix's own per-dim
    std (the entropy proxy). High-variance / high-information dimensions therefore absorb more noise.
    """
    x, ref = _to_numpy(patches)
    rng = _as_rng(rng)
    flat = x.reshape(-1, x.shape[-1])
    per_dim = flat.std(axis=0, keepdims=True) + 1e-8  # (1, d) entropy/variance proxy
    noise = rng.standard_normal(x.shape).astype(np.float32) * per_dim.reshape(
        (1,) * (x.ndim - 1) + (x.shape[-1],)
    )
    y = x + float(strength) * noise
    return _restore(_l2(y), ref)


def press(patches, strength: float, *, contrast=None):
    """PRESS (ICASSP'25), adapted: project out the top-k PII/principal subspace of the stored patches.

    k = max(1, round(strength * dim)), clamped to the available rank. If ``contrast`` (m, d) is given —
    rows spanning the estimated PII subspace — its top-k right singular vectors are removed; otherwise
    the top-k principal components of the (mean-centred) stored patches are removed.
    """
    x, ref = _to_numpy(patches)
    d = x.shape[-1]
    flat = x.reshape(-1, d)
    k = max(1, int(round(float(strength) * d)))
    if contrast is not None and len(contrast) > 0:
        C = np.asarray(contrast, dtype=np.float32).reshape(-1, d)
        C = C - C.mean(axis=0, keepdims=True)
        max_rank = min(C.shape[0], d)
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
    else:
        Xc = flat - flat.mean(axis=0, keepdims=True)
        max_rank = min(flat.shape[0], d)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, max(1, max_rank))
    U = Vt[:k]  # (k, d) orthonormal directions to remove
    # remove the component of every patch that lies in span(U): y = x - (x U^T) U
    proj = (flat @ U.T) @ U  # (N, d)
    y = (flat - proj).reshape(x.shape)
    return _restore(_l2(y), ref)


def koga(patches, strength: float, *, rng=None):
    """KOGA-style adversarial perturbation (arXiv:2412.04697), adapted: isotropic + centroid-collapse.

    Mixes a unit-normalized isotropic per-patch noise (the isotropic term) with a push of each patch
    toward the corpus centroid (the "learned-ish" adversarial term that erodes separability). ``strength``
    scales the combined perturbation.
    """
    x, ref = _to_numpy(patches)
    rng = _as_rng(rng)
    d = x.shape[-1]
    flat = x.reshape(-1, d)
    centroid = flat.mean(axis=0, keepdims=True)  # (1, d)
    iso = rng.standard_normal(x.shape).astype(np.float32)
    iso = iso / (np.linalg.norm(iso, axis=-1, keepdims=True) + 1e-8)  # unit per-patch isotropic
    pull = (centroid.reshape((1,) * (x.ndim - 1) + (d,)) - x)  # adversarial: collapse toward centroid
    y = x + float(strength) * (0.5 * iso + 0.5 * pull)
    return _restore(_l2(y), ref)


# --------------------------------------------------------------------------------------------------
# Registry / common dispatch — the frontier runner sweeps `strength` through this.
# --------------------------------------------------------------------------------------------------
def get_baseline(name: str):
    try:
        return {"entroguard": entroguard, "press": press, "koga": koga}[name]
    except KeyError:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}") from None


def apply_baseline(name: str, patches, strength: float, *, rng=None, contrast=None):
    """Uniform entry point. ``rng`` is used by entroguard/koga; ``contrast`` by press. Extra kwargs are
    ignored per-baseline so a single call site can drive all three with one signature."""
    fn = get_baseline(name)
    if name == "press":
        return fn(patches, strength, contrast=contrast)
    return fn(patches, strength, rng=rng)
