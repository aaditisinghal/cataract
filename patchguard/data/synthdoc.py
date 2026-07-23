"""Synthetic large-font ID documents (MASTER_PLAN S6 diagnostic — the big-font PII test).

The controlled version of the IDNet experiment: render simple ID cards with PII (name, DOB, ID number)
in LARGE, clearly-legible fonts at 448px, with exact ground-truth text + boxes. This isolates the
resolution hypothesis — if legible big-font PII still doesn't reconstruct from ColPali embeddings, the
info is fundamentally lost (not a resolution artifact). Font size is a knob, so legibility-vs-size is
directly testable. No 490GB download; ground truth is exact.
"""

from __future__ import annotations

import os

import numpy as np

from patchguard.data.fields import AnnotatedField, Box

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # container (apt fonts-dejavu-core)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS local
    "/Library/Fonts/Arial.ttf",
]

_FIRST = ["JAMES", "MARIA", "ROBERT", "LINDA", "MICHAEL", "SARAH", "DAVID", "EMILY",
          "JOHN", "ANNA", "WILLIAM", "OLIVIA", "RICHARD", "SOPHIA", "THOMAS", "GRACE"]
_LAST = ["SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER", "DAVIS",
         "RODRIGUEZ", "MARTINEZ", "HERNANDEZ", "LOPEZ", "GONZALEZ", "WILSON", "ANDERSON"]


def _font(size: int):
    from PIL import ImageFont

    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _pii(rng: np.random.Generator) -> dict[str, str]:
    return {
        "name": f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
        "dob": f"{int(rng.integers(1, 13)):02d}/{int(rng.integers(1, 29)):02d}/{int(rng.integers(1950, 2005))}",
        "id_no": f"{int(rng.integers(10_000_000, 99_999_999))}",
    }


def generate_id_card(
    seed: int, size: tuple[int, int] = (448, 448), value_font_size: int = 34
) -> tuple[np.ndarray, list[AnnotatedField]]:
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, (238, 240, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size[0], 56], fill=(35, 70, 130))
    d.text((16, 14), "IDENTITY CARD", fill=(255, 255, 255), font=_font(26))

    pii = _pii(rng)
    label_font = _font(16)
    value_font = _font(value_font_size)
    fields: list[AnnotatedField] = []
    y = 84
    for label, key in (("NAME", "name"), ("DATE OF BIRTH", "dob"), ("ID NUMBER", "id_no")):
        d.text((16, y), label, fill=(95, 95, 95), font=label_font)
        vy = y + 22
        val = pii[key]
        d.text((16, vy), val, fill=(15, 15, 15), font=value_font)
        x0, y0, x1, y1 = d.textbbox((16, vy), val, font=value_font)
        fields.append(
            AnnotatedField(field_type=key, text=val, box=(float(x0), float(y0), float(x1), float(y1)))
        )
        y = vy + value_font_size + 26
    return np.array(img), fields


def generate_id_cards(
    n: int, seed: int = 0, size: tuple[int, int] = (448, 448), value_font_size: int = 34
) -> list[tuple[np.ndarray, list[AnnotatedField]]]:
    return [generate_id_card(seed * 10_000 + i, size, value_font_size) for i in range(n)]
