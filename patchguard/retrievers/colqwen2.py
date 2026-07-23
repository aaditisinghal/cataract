"""ColQwen2 backend (MASTER_PLAN S2 cross-model) — a DIFFERENT multi-vector visual retriever.

Qwen2-VL backbone + late interaction, distinct from ColPali's SigLIP/PaliGemma. Used to test whether
the leak + holographic bleed are a general multi-vector-VLM property or a ColPali quirk. Qwen2-VL uses
dynamic resolution (variable patch count), so there's no fixed 32x32 grid — but the model-agnostic
tests (retrieval attack, wrong-page, random-patch-removal bleed) need only the raw patch array, not a
box->patch alignment. encode_page therefore returns a light object exposing `.patches`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from patchguard.retrievers.base import maxsim


class ColQwen2Retriever:
    model_id: str

    def __init__(self, model_name: str = "vidore/colqwen2-v1.0", device: str | None = None,
                 dtype: str = "bfloat16") -> None:
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self.model_id = model_name
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            ColQwen2.from_pretrained(model_name, torch_dtype=getattr(torch, dtype)).eval().to(self.device)
        )
        self.processor = ColQwen2Processor.from_pretrained(model_name)

    def encode_page(self, image: np.ndarray):
        from PIL import Image

        pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        batch = self.processor.process_images([pil]).to(self.device)
        with self._torch.no_grad():
            emb = self.model(**batch)
        return SimpleNamespace(patches=emb[0].float().cpu().numpy())

    def encode_query(self, text: str) -> np.ndarray:
        batch = self.processor.process_queries([text]).to(self.device)
        with self._torch.no_grad():
            emb = self.model(**batch)
        return emb[0].float().cpu().numpy()

    def score(self, query: np.ndarray, page) -> float:
        return maxsim(query, page.patches)
