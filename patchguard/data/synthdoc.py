"""Synthetic ID documents with TEMPLATE VARIATION (MASTER_PLAN S6 diagnostic).

The big-font PII test — but with layout variance baked in, because a FIXED template confounds any
probe (it can key on position/geometry, and deterministic rendering makes held-out cards duplicates of
train). With ``vary=True`` each card randomizes: font family, value font size, left margin, vertical
start, field spacing, field ORDER (so field-type != position), header height, background tint, and
light pixel noise. Same name therefore renders differently across cards, and field-type is decoupled
from position — the two controls the confound demands.

``generate_same_name_cards`` fixes the name and varies everything else (the same-name-different-template
control): if a probe/MaxSim still recovers that name across templates, the signal is glyph content, not
template geometry.
"""

from __future__ import annotations

import os

import numpy as np

from patchguard.data.fields import AnnotatedField, Box

# font families (fonts-dejavu-core provides these in the container; macOS fallbacks for local)
_FAMILIES = [
    ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"],
    ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/System/Library/Fonts/Supplemental/Courier New.ttf"],
    ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", "/System/Library/Fonts/Supplemental/Arial Italic.ttf"],
]

_FIRST = ["JAMES", "MARIA", "ROBERT", "LINDA", "MICHAEL", "SARAH", "DAVID", "EMILY",
          "JOHN", "ANNA", "WILLIAM", "OLIVIA", "RICHARD", "SOPHIA", "THOMAS", "GRACE"]
_LAST = ["SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER", "DAVIS",
         "RODRIGUEZ", "MARTINEZ", "HERNANDEZ", "LOPEZ", "GONZALEZ", "WILSON", "ANDERSON"]


def _font(size: int, family: int = 0):
    from PIL import ImageFont

    for p in _FAMILIES[family % len(_FAMILIES)]:
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
    seed: int, size: tuple[int, int] = (448, 448), value_font_size: int = 34,
    vary: bool = True, fixed_name: str | None = None, fixed_pii: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[AnnotatedField]]:
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    if vary:
        fam = int(rng.integers(0, len(_FAMILIES)))
        vfs = int(np.clip(value_font_size + rng.integers(-6, 7), 12, 60))
        margin = int(rng.integers(10, 44))
        y = int(rng.integers(66, 104))
        spacing = int(rng.integers(18, 36))
        header_h = int(rng.integers(40, 64))
        bg = tuple(int(x) for x in rng.integers(224, 250, 3))
        order = list(rng.permutation(3))  # shuffle field order -> field-type decoupled from position
    else:
        fam, vfs, margin, y, spacing, header_h, bg, order = 1, value_font_size, 16, 84, 26, 56, (238, 240, 245), [0, 1, 2]

    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size[0], header_h], fill=(35, 70, 130))
    d.text((margin, header_h // 2 - 12), "IDENTITY CARD", fill=(255, 255, 255), font=_font(24, fam))

    pii = dict(fixed_pii) if fixed_pii is not None else _pii(rng)
    if fixed_name is not None:
        pii["name"] = fixed_name
    label_font = _font(15, fam)
    value_font = _font(vfs, fam)
    spec = [("NAME", "name"), ("DATE OF BIRTH", "dob"), ("ID NUMBER", "id_no")]

    fields: list[AnnotatedField] = []
    for idx in order:
        label, key = spec[idx]
        d.text((margin, y), label, fill=(95, 95, 95), font=label_font)
        vy = y + 20
        val = pii[key]
        d.text((margin, vy), val, fill=(15, 15, 15), font=value_font)
        x0, y0, x1, y1 = d.textbbox((margin, vy), val, font=value_font)
        fields.append(AnnotatedField(field_type=key, text=val, box=(float(x0), float(y0), float(x1), float(y1))))
        y = vy + vfs + spacing

    arr = np.array(img).astype(np.int16)
    if vary:
        arr = arr + rng.integers(-6, 7, arr.shape)  # light pixel noise so same content != identical
    return np.clip(arr, 0, 255).astype(np.uint8), fields


def generate_id_cards(
    n: int, seed: int = 0, size: tuple[int, int] = (448, 448), value_font_size: int = 34,
    vary: bool = True,
) -> list[tuple[np.ndarray, list[AnnotatedField]]]:
    return [generate_id_card(seed * 10_000 + i, size, value_font_size, vary=vary) for i in range(n)]


def generate_same_name_cards(
    name: str, n: int, seed: int = 0, value_font_size: int = 34
) -> list[tuple[np.ndarray, list[AnnotatedField]]]:
    """Same name, fully varied template — the same-name-different-template control."""
    return [generate_id_card(seed * 10_000 + i, value_font_size=value_font_size, vary=True, fixed_name=name)
            for i in range(n)]
