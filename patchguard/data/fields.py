"""Annotated field types — the bridge between a dataset, align.py, and PFRR.

An ``AnnotatedField`` is one labelled region on a page: its box (original-image pixels), the ground-
truth text (for PFRR exact-match), and a ``field_type`` string. Datasets decide the field_type
vocabulary (FUNSD: question/answer/header/other; IDNet: name/id_no/dob/address/...).
"""

from __future__ import annotations

from dataclasses import dataclass

# (x0, y0, x1, y1) in original-image pixels.
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class AnnotatedField:
    field_type: str
    text: str
    box: Box

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.box
        if x1 < x0 or y1 < y0:
            raise ValueError(f"degenerate box {self.box}")


@dataclass(frozen=True)
class PageSample:
    """One document page: where the image lives, its size, and its annotated fields."""

    image_path: str
    size: tuple[int, int]  # (width, height) in original pixels
    fields: list[AnnotatedField]
    doc_id: str

    def boxes(self) -> list[Box]:
        return [f.box for f in self.fields]

    def boxes_of_type(self, field_type: str) -> list[Box]:
        return [f.box for f in self.fields if f.field_type == field_type]
