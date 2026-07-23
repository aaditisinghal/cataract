"""Information-DESTROYING redaction: project stored patches off the PII-discriminative subspace.

Motivated by the adaptive-attack result (RESULTS.md §4.13): the residual ``RedactionProjection`` is
(near-)bijective, so a white-box attacker learns ``P^-1``, reconstructs the patches at cosine ~0.998,
and the dictionary attack returns to 1.00. The lesson: a real defense must REMOVE the PII information,
not merely reshape it.

``NullspaceRedaction`` is RANK-DEFICIENT by construction. It estimates the ``k`` directions in patch
space that name-VALUE queries use to match (after removing the topic/content directions, so legitimate
retrieval is spared) and projects every stored patch onto their orthogonal complement. The PII component
is then information-theoretically gone — no inverse map can recover a dimension that was annihilated —
so the very attack that broke the residual ``P`` (B1b inverse learning) cannot recover it here. The
price is utility only if the removed directions overlap the content directions; the sweep over ``k``
traces exactly that privacy/utility trade under the ADAPTIVE (inverse) attacker.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _top_dirs(M: torch.Tensor, r: int) -> torch.Tensor:
    """Top-``r`` orthonormal right-singular directions of ``M`` (T,d) as columns of a (d,r) matrix."""
    if r <= 0 or M.numel() == 0:
        return torch.zeros(M.shape[1], 0, dtype=M.dtype, device=M.device)
    # M = U S Vh ; the d-space directions are the rows of Vh (columns of Vh.T).
    _, _, Vh = torch.linalg.svd(M, full_matrices=False)  # Vh: (min(T,d), d)
    r = int(min(r, Vh.shape[0]))
    return Vh[:r].T.contiguous()  # (d, r)


def pii_directions(
    name_tokens: np.ndarray, topic_tokens: np.ndarray, k: int, r_topic: int = 8
) -> torch.Tensor:
    """Orthonormal (d,k) basis of the name-DISCRIMINATIVE subspace.

    Directions the name-value queries span AFTER the top-``r_topic`` topic-query directions are removed,
    so the subspace we annihilate carries PII matching but not the legitimate content signal.
    """
    N = torch.as_tensor(np.asarray(name_tokens, dtype=np.float32))
    Tm = torch.as_tensor(np.asarray(topic_tokens, dtype=np.float32))
    Vt = _top_dirs(Tm, r_topic)  # (d, r_topic) content directions to SPARE
    N_res = N - N @ Vt @ Vt.T if Vt.shape[1] > 0 else N  # name signal orthogonal to content
    return _top_dirs(N_res, k)  # (d, k)


class NullspaceRedaction(nn.Module):
    """Fixed, rank-deficient index-time transform: drop the span(D) component, then L2-normalize.

    Forward contract matches ``RedactionProjection`` ((...,d) -> normalized (...,d)) so it is a drop-in
    for the same attack/eval harness — but unlike the residual transform it is NOT invertible on span(D).
    """

    def __init__(self, D: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("D", D)  # (d, k) orthonormal, k directions to remove

    @property
    def k(self) -> int:
        return int(self.D.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.D.shape[1] > 0:
            D = self.D.to(x.dtype)
            x = x - (x @ D) @ D.T  # annihilate the PII subspace (lossy, non-invertible)
        return F.normalize(x, dim=-1)
