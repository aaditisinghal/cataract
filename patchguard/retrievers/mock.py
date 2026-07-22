"""Deterministic fake retriever for tests and CI (no model download, no GPU).

Turns an image into a patch grid by average-pooling pixel blocks and projecting through a fixed
seeded matrix. Not meaningful for retrieval — its only job is to exercise the ``Retriever`` protocol,
the ``PageEncoding`` invariants, and the MaxSim path on the 4-page CPU fixture.
"""

from __future__ import annotations

import numpy as np

from patchguard.retrievers.base import PageEncoding, maxsim


class MockRetriever:
    model_id = "mock-colpali"

    def __init__(
        self,
        grid: tuple[int, int] = (4, 4),
        input_size: tuple[int, int] = (32, 32),
        dim: int = 8,
        n_prefix_tokens: int = 1,
        resize_policy: str = "squash",
        seed: int = 0,
    ) -> None:
        self.grid = grid
        self.input_size = input_size
        self.dim = dim
        self.n_prefix_tokens = n_prefix_tokens
        self.resize_policy = resize_policy
        rng = np.random.default_rng(seed)
        gh, gw = grid
        # Fixed projection: pooled 3-channel block stats (mean per channel) -> dim.
        self._proj = rng.standard_normal((3, dim)).astype(np.float32)
        self._prefix = rng.standard_normal((n_prefix_tokens, dim)).astype(np.float32)
        self._vocab_proj = rng.standard_normal((256, dim)).astype(np.float32)

    def encode_page(self, image: np.ndarray) -> PageEncoding:
        img = np.asarray(image, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) image; got {img.shape}")
        iw, ih = self.input_size
        gh, gw = self.grid
        # Resize by nearest-block pooling to (ih, iw), then pool into the grid.
        h, w, _ = img.shape
        ys = (np.linspace(0, h, gh + 1)).astype(int)
        xs = (np.linspace(0, w, gw + 1)).astype(int)
        feats = np.zeros((gh * gw, self.dim), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                block = img[ys[i] : max(ys[i] + 1, ys[i + 1]), xs[j] : max(xs[j] + 1, xs[j + 1])]
                mean_rgb = block.reshape(-1, 3).mean(axis=0) / 255.0
                feats[i * gw + j] = mean_rgb @ self._proj
        patches = np.concatenate([self._prefix, feats], axis=0)
        patches = _l2_normalize(patches)
        return PageEncoding(
            patches=patches,
            grid=self.grid,
            input_size=self.input_size,
            model_id=self.model_id,
            resize_policy=self.resize_policy,
            n_prefix_tokens=self.n_prefix_tokens,
        )

    def encode_query(self, text: str) -> np.ndarray:
        tokens = [ord(c) % 256 for c in text] or [0]
        q = self._vocab_proj[tokens]
        return _l2_normalize(q)

    def score(self, query: np.ndarray, page: PageEncoding) -> float:
        return maxsim(query, page.patches)


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
