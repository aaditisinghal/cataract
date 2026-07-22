"""Real ColPali backend (MASTER_PLAN S2).

Wraps `colpali_engine`'s ColPali + processor behind the Retriever protocol. Heavy imports (torch,
PIL, colpali_engine) are deferred to construction/first-use so importing this module never forces the
GPU stack — the tested logic core stays clean.

⚠️ VERIFY-ON-REAL-DATA (S3 gate): the grid, prefix-token count, and resize policy below are the
documented PaliGemma/ColPali defaults (448x448 -> 32x32 = 1024 image tokens, placed FIRST in the
sequence, bilinear squash resize). Confirm them against the live processor with
experiments/validate_alignment.py before trusting any defense number. If the processor letterboxes or
prepends tokens, fix `resize_policy` / `n_prefix_tokens` here.
"""

from __future__ import annotations

import numpy as np

from patchguard.retrievers.base import PageEncoding, maxsim

# ColPali (PaliGemma/SigLIP-So400m) defaults — see warning above.
_COLPALI_GRID = (32, 32)
_COLPALI_INPUT = (448, 448)
_COLPALI_PREFIX_TOKENS = 0  # image tokens are first; instruction tokens TRAIL the 1024-patch block
_COLPALI_RESIZE = "squash"


class ColPaliRetriever:
    model_id: str

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        device: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.model_id = model_name
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype)
        self.model = (
            ColPali.from_pretrained(model_name, torch_dtype=torch_dtype).eval().to(self.device)
        )
        self.processor = ColPaliProcessor.from_pretrained(model_name)

    def encode_page(self, image: np.ndarray) -> PageEncoding:
        from PIL import Image

        pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        batch = self.processor.process_images([pil]).to(self.device)
        with self._torch.no_grad():
            emb = self.model(**batch)  # (1, n_tokens, 128)
        patches = emb[0].float().cpu().numpy()
        return PageEncoding(
            patches=patches,
            grid=_COLPALI_GRID,
            input_size=_COLPALI_INPUT,
            model_id=self.model_id,
            resize_policy=_COLPALI_RESIZE,
            n_prefix_tokens=_COLPALI_PREFIX_TOKENS,
        )

    def encode_query(self, text: str) -> np.ndarray:
        batch = self.processor.process_queries([text]).to(self.device)
        with self._torch.no_grad():
            emb = self.model(**batch)
        return emb[0].float().cpu().numpy()

    def score(self, query: np.ndarray, page: PageEncoding) -> float:
        # Late interaction over the FULL page token set (image + instruction tokens), as in ColPali.
        return maxsim(query, page.patches)
