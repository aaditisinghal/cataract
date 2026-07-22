"""One interface over the whole ColPali family (MASTER_PLAN S2).

Everything downstream (attack, defense, alignment, eval) depends only on ``PageEncoding`` +
``Retriever``, never on a concrete backend. That isolation is what lets Claim 1 (ColPali vs BiPali)
be a config swap rather than a rewrite.

Conventions fixed here to avoid the off-by-one / transpose trap called out in RESEARCH_PROTOCOL S11:
  * ``grid`` is ``(rows, cols)`` == ``(gh, gw)``.
  * ``input_size`` is ``(width, height)`` == ``(iw, ih)`` — pixel size the model resized the page to.
  * patch sequence order is ROW-MAJOR: patch (row i, col j) lives at index ``n_prefix + i*gw + j``.
  * ``n_prefix_tokens`` counts non-image tokens the model prepends to the patch sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# Resize policy the model's image processor uses. Read it from the actual processor config;
# do NOT assume (ColPali-family models differ — some squash, some letterbox).
ResizePolicy = str  # Literal-ish: "squash" | "letterbox"
VALID_RESIZE_POLICIES = ("squash", "letterbox")


@dataclass(frozen=True)
class PageEncoding:
    """What an attacker reads from the index: the per-patch multi-vector grid for one page."""

    patches: np.ndarray  # (n_patches, d) float — includes prefix tokens if the model emits them
    grid: tuple[int, int]  # (rows, cols) of image patches, e.g. (32, 32)
    input_size: tuple[int, int]  # (width, height) the page was resized to, e.g. (448, 448)
    model_id: str
    resize_policy: ResizePolicy = "squash"
    n_prefix_tokens: int = 0  # non-image tokens at the FRONT of `patches`

    def __post_init__(self) -> None:
        if self.patches.ndim != 2:
            raise ValueError(f"patches must be (n_patches, d); got shape {self.patches.shape}")
        gh, gw = self.grid
        expected = self.n_prefix_tokens + gh * gw
        if self.patches.shape[0] != expected:
            raise ValueError(
                f"patch count {self.patches.shape[0]} != n_prefix({self.n_prefix_tokens}) "
                f"+ grid({gh}x{gw}={gh * gw}) = {expected}"
            )
        if self.resize_policy not in VALID_RESIZE_POLICIES:
            raise ValueError(f"resize_policy must be one of {VALID_RESIZE_POLICIES}")

    @property
    def n_patches(self) -> int:
        return int(self.patches.shape[0])

    @property
    def dim(self) -> int:
        return int(self.patches.shape[1])

    def image_patches(self) -> np.ndarray:
        """Just the spatial patches, prefix tokens stripped — the thing align.py maps boxes onto."""
        return self.patches[self.n_prefix_tokens :]


@runtime_checkable
class Retriever(Protocol):
    """Minimal surface every backend implements. Keep it this small."""

    model_id: str

    def encode_page(self, image: np.ndarray) -> PageEncoding:
        """image: (H, W, 3) uint8/float -> per-patch multi-vector encoding."""
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """text -> (n_query_tokens, d) multi-vector query encoding."""
        ...

    def score(self, query: np.ndarray, page: PageEncoding) -> float:
        """Late-interaction (MaxSim) relevance score."""
        ...


def maxsim(query_tokens: np.ndarray, patch_tokens: np.ndarray) -> float:
    """ColBERT/ColPali late interaction: sum over query tokens of the max dot with any patch.

    query_tokens: (nq, d), patch_tokens: (np, d). Returns a scalar. This is the default ``score``
    for multi-vector backends; bi-encoders (BiPali) override with a single-vector dot product.
    """
    if query_tokens.ndim != 2 or patch_tokens.ndim != 2:
        raise ValueError("maxsim expects 2-D (tokens, d) arrays")
    if query_tokens.shape[1] != patch_tokens.shape[1]:
        raise ValueError(
            f"dim mismatch: query d={query_tokens.shape[1]} vs patch d={patch_tokens.shape[1]}"
        )
    # (nq, np) similarity, max over patches, sum over query tokens.
    sim = query_tokens @ patch_tokens.T
    return float(sim.max(axis=1).sum())
