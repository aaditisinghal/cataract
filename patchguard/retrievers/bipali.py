"""BiPali control — the pooled bi-encoder (MASTER_PLAN S2, Claim 1).

BiPali is the confound-free control: same backbone and training data as ColPali, but a single pooled
page vector instead of a multi-vector patch grid. Any inversion delta is therefore *architecture*,
not capacity.

Two ways to get it:
  1. A trained BiPali checkpoint (strongest; retrains the projection head). Preferred — swap it in when
     available / confirmed.
  2. **Mean-pool an existing ColPali** (this class, `PooledRetriever`). A faithful-enough ablation for
     the kill test and the honest fallback the plan allows. Documented as such so we don't overclaim.

Representing a pooled embedding as a (1,1) grid is not a hack — it is the point: a bi-encoder has **no
locality primitive**, so align.py maps *every* field box onto the single vector. That is exactly the
asymmetry Claim 3's patch-scoped defense exploits and the baselines (Claim 2) cannot.
"""

from __future__ import annotations

import numpy as np

from patchguard.retrievers.base import PageEncoding, Retriever, maxsim


def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


class PooledRetriever:
    """Wrap any multi-vector Retriever and mean-pool it to a single page/query vector."""

    def __init__(self, base: Retriever, tag: str = "bipali-pooled") -> None:
        self.base = base
        self.model_id = f"{tag}({getattr(base, 'model_id', 'base')})"

    def encode_page(self, image: np.ndarray) -> PageEncoding:
        enc = self.base.encode_page(image)
        pooled = _l2(enc.image_patches().mean(axis=0, keepdims=True))  # (1, d)
        return PageEncoding(
            patches=pooled,
            grid=(1, 1),  # no locality: the whole page is one "patch"
            input_size=enc.input_size,
            model_id=self.model_id,
            resize_policy=enc.resize_policy,
            n_prefix_tokens=0,
        )

    def encode_query(self, text: str) -> np.ndarray:
        q = self.base.encode_query(text)
        return _l2(q.mean(axis=0, keepdims=True))  # (1, d)

    def score(self, query: np.ndarray, page: PageEncoding) -> float:
        # Both are single vectors -> maxsim reduces to a dot product.
        return maxsim(query, page.patches)
