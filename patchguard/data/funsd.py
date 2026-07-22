"""FUNSD loader (MASTER_PLAN S5/★ — the kill-test corpus).

FUNSD ships as `<split>/annotations/*.json` + `<split>/images/*.png`. Each annotation is:

    {"form": [
        {"id": 0, "text": "REGISTRATION NO.", "box": [x0,y0,x1,y1],
         "label": "question", "words": [{"text": "...", "box": [x0,y0,x1,y1]}, ...],
         "linking": [[0, 1]]},
        ...]}

Boxes are already in original-image pixel coords (top-left origin), which is exactly what align.py
expects. The parser is pure-json (no image decode) so it is testable without PIL; image size is read
lazily only when needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from patchguard.data.fields import AnnotatedField, Box, PageSample

# FUNSD's four entity labels. Kept verbatim as field_type; the kill test reports PFRR per label.
FUNSD_LABELS = ("question", "answer", "header", "other")


def parse_funsd_form(
    doc: dict[str, Any], granularity: str = "entity"
) -> list[AnnotatedField]:
    """Turn one FUNSD annotation dict into AnnotatedFields.

    granularity="entity" -> one field per form element (its full text + bounding box).
    granularity="word"   -> one field per word (finer boxes; better for tight patch alignment).
    """
    if granularity not in ("entity", "word"):
        raise ValueError("granularity must be 'entity' or 'word'")
    fields: list[AnnotatedField] = []
    for elem in doc.get("form", []):
        label = elem.get("label", "other")
        if granularity == "entity":
            box = _as_box(elem["box"])
            text = elem.get("text", "") or ""
            if text.strip():
                fields.append(AnnotatedField(field_type=label, text=text, box=box))
        else:
            for w in elem.get("words", []):
                text = w.get("text", "") or ""
                if text.strip():
                    fields.append(
                        AnnotatedField(field_type=label, text=text, box=_as_box(w["box"]))
                    )
    return fields


def load_funsd_annotation(
    json_path: str | Path, image_path: str | Path, size: tuple[int, int], granularity: str = "entity"
) -> PageSample:
    p = Path(json_path)
    doc = json.loads(p.read_text())
    return PageSample(
        image_path=str(image_path),
        size=size,
        fields=parse_funsd_form(doc, granularity),
        doc_id=p.stem,
    )


def iter_funsd(root: str | Path, split: str = "testing_data", granularity: str = "entity") -> Iterator[PageSample]:
    """Yield PageSamples from a FUNSD split dir. Requires PIL to read image sizes."""
    base = Path(root) / split
    ann_dir = base / "annotations"
    img_dir = base / "images"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"no FUNSD annotations at {ann_dir}")
    for jp in sorted(ann_dir.glob("*.json")):
        img = img_dir / f"{jp.stem}.png"
        size = _image_size(img)
        yield load_funsd_annotation(jp, img, size, granularity)


def _as_box(raw: list[float]) -> Box:
    x0, y0, x1, y1 = raw
    # FUNSD boxes are already [x0,y0,x1,y1] top-left; normalize ordering defensively.
    return (float(min(x0, x1)), float(min(y0, y1)), float(max(x0, x1)), float(max(y0, y1)))


def _image_size(img_path: Path) -> tuple[int, int]:
    from PIL import Image  # lazy: only needed when actually iterating real images

    with Image.open(img_path) as im:
        return (im.width, im.height)
