"""Composition helpers for charts built from blocks and labels.

A chart is a mid-grey frame with opaque rectangular blocks placed on it
and white labels drawn over the result. Each block renders over its own
bounding box, so its arithmetic costs the box rather than the frame,
and lands in the frame through one gather and one ``where``
(:func:`place`). Labels render in one pass for the whole chart through a
built-in bitmap font (:func:`draw_labels`), so a chart needs no imaging
library and stays in the core catalog (§spec:package-shape).

Both are functional and branch on no array contents
(§spec:backend-portability). Their lookup tables are built host-side
from parameters: one integer per frame row or column, never per pixel.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from display_patterns.patterns import _backend

# Upper bound on either frame dimension: 16384 clears 8K with headroom,
# and nothing past it is a real video frame.
MAX_DIMENSION = 16384

# Chart values in the [0, 1] float convention.
BLACK = 0.0
GREY = 0.5
WHITE = 1.0

# The font: 5x7 glyphs on a 6-column advance, so adjacent glyphs keep one
# blank column between them. Only what chart labels need: digits, capital
# letters, a little punctuation, and the "x" of a raster size.
GLYPH_WIDTH = 5
GLYPH_HEIGHT = 7
ADVANCE = GLYPH_WIDTH + 1
_FONT = {
    " ": ("00000",) * GLYPH_HEIGHT,
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "x": ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    "/": ("00000", "00001", "00010", "00100", "01000", "10000", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "10001", "11001", "10101", "10011", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "10001", "01010", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}
_CODES = {char: code for code, char in enumerate(_FONT)}
_BLANK = _CODES[" "]
# Each glyph padded to its advance, so the spacing column is a lookup of
# blank like any other.
_GLYPHS = np.array(
    [[[bit == "1" for bit in row + "0"] for row in glyph] for glyph in _FONT.values()],
    dtype=bool,
)


@dataclass(frozen=True)
class Label:
    """``text`` drawn white with its top-left corner at ``(x, y)``, each
    font pixel ``scale`` frame pixels square."""

    x: int
    y: int
    text: str
    scale: int

    @property
    def width(self) -> int:
        """Frame columns from the first glyph's left edge to the last's
        right edge."""
        return (len(self.text) * ADVANCE - 1) * self.scale

    @property
    def height(self) -> int:
        return GLYPH_HEIGHT * self.scale


@dataclass(frozen=True)
class Block:
    """An opaque block's top-left corner in the frame."""

    x: int
    y: int
    width: int
    height: int

    def overlaps(self, other: "Block") -> bool:
        return (
            self.x < other.x + other.width
            and other.x < self.x + self.width
            and self.y < other.y + other.height
            and other.y < self.y + self.height
        )


def label_block(label: Label) -> Block:
    """The frame rectangle a label inks."""
    return Block(label.x, label.y, label.width, label.height)


def check_layout(
    width: int, height: int, blocks: tuple[Block, ...], labels: tuple[Label, ...]
) -> None:
    """Refuse a layout whose blocks or labels leave the frame or collide.

    Blocks may not overlap each other, nor labels each other; a label
    may sit over a block. Raises ``ValueError`` naming the offender.
    """
    for kind, rects in (
        ("block", blocks),
        ("label", tuple(label_block(label) for label in labels)),
    ):
        for rect in rects:
            if (
                rect.x < 0
                or rect.y < 0
                or rect.x + rect.width > width
                or rect.y + rect.height > height
            ):
                raise ValueError(
                    f"{kind} {rect} leaves the {width}x{height} frame: the frame "
                    f"is too small for this layout."
                )
        for index, rect in enumerate(rects):
            for other in rects[index + 1 :]:
                if rect.overlaps(other):
                    raise ValueError(
                        f"{kind}s {rect} and {other} overlap: the frame is too "
                        f"small for this layout."
                    )
    for label in labels:
        unknown = set(label.text) - set(_FONT)
        if unknown:
            raise ValueError(
                f"label {label.text!r} uses {sorted(unknown)}, which the chart "
                f"font does not carry."
            )


def place(frame: Any, block: Any, origin: Block, rows: Any, cols: Any, xp: Any) -> Any:
    """``frame`` with ``block`` over it at ``origin``.

    One gather of the block at clamped coordinates and one ``where``:
    a frame pixel outside the block keeps its value. ``rows`` and
    ``cols`` are the frame's coordinate vectors.
    """
    block_rows = rows - origin.y
    block_cols = cols - origin.x
    in_rows = (block_rows >= 0) & (block_rows < origin.height)
    in_cols = (block_cols >= 0) & (block_cols < origin.width)
    block_rows = xp.where(in_rows, block_rows, 0)
    block_cols = xp.where(in_cols, block_cols, 0)
    picked = block[block_rows[:, None], block_cols[None, :]]
    return xp.where(in_rows[:, None] & in_cols[None, :], picked, frame)


@dataclass(frozen=True, eq=False)
class LabelTables:
    """Host-side lookup tables locating every label in a frame.

    Rows covered by the same set of labels form a band, and each band
    holds one frame-width row naming the label at each column, so finding
    a pixel's label is two gathers. Label 0 is the empty label. Built
    once with the geometry that owns the labels, so a compiled render
    reads them as constants rather than tracing their construction.
    """

    band_of_row: np.ndarray
    label_at: np.ndarray
    x: np.ndarray
    y: np.ndarray
    scale: np.ndarray
    chars: np.ndarray

    @classmethod
    def build(cls, labels: tuple[Label, ...], width: int, height: int) -> "LabelTables":
        rows = np.arange(height)[:, None]
        tops = np.array([label.y for label in labels], dtype=np.int64)
        bottoms = tops + np.array([label.height for label in labels], dtype=np.int64)
        covering = (rows >= tops) & (rows < bottoms)
        # One band per distinct set of covering labels, the empty set first.
        bands, band_of_row = np.unique(
            np.vstack([np.zeros((1, len(labels)), dtype=bool), covering]),
            axis=0,
            return_inverse=True,
        )
        label_at = np.zeros((len(bands), width), dtype=np.int32)
        for band, active in enumerate(bands):
            for index in np.flatnonzero(active):
                label = labels[index]
                label_at[band, label.x : label.x + label.width] = index + 1
        # The empty set sorts first among boolean rows, so band 0 is empty.
        band_of_row = band_of_row.reshape(-1)[1:].astype(np.int32)

        longest = max([len(label.text) for label in labels] + [1])
        chars = np.full((len(labels) + 1, longest), _BLANK, dtype=np.int32)
        for index, label in enumerate(labels, start=1):
            chars[index, : len(label.text)] = [_CODES[char] for char in label.text]
        return cls(
            band_of_row=band_of_row,
            label_at=label_at,
            x=np.array([0, *(label.x for label in labels)], dtype=np.int32),
            y=np.array([0, *(label.y for label in labels)], dtype=np.int32),
            scale=np.array([1, *(label.scale for label in labels)], dtype=np.int32),
            chars=chars,
        )


def draw_labels(
    frame: Any, tables: LabelTables, rows: Any, cols: Any, xp: Any, device: Any
) -> Any:
    """``frame`` with every label in ``tables`` inked white over it, in
    one pass."""

    def table(array: np.ndarray) -> Any:
        return _backend.asarray(xp, array, device)

    glyphs = table(_GLYPHS)
    label_at = table(tables.label_at)
    chars = table(tables.chars)

    label = label_at[table(tables.band_of_row)[rows][:, None], cols[None, :]]
    scale = table(tables.scale)[label]
    # Every index is reduced into its table's range, so the empty label's
    # pixels look up a blank glyph wherever they fall.
    glyph_row = ((rows[:, None] - table(tables.y)[label]) // scale) % GLYPH_HEIGHT
    offset = (cols[None, :] - table(tables.x)[label]) // scale
    char = (offset // ADVANCE) % tables.chars.shape[1]
    glyph_col = offset % ADVANCE
    ink = glyphs[chars[label, char], glyph_row, glyph_col]
    return xp.where(ink, WHITE, frame)


def as_rgb(frame: Any, xp: Any) -> Any:
    """A ``(height, width)`` float frame as an achromatic HWC float32 image."""
    frame = _backend.astype(frame, xp.float32)
    return xp.stack([frame, frame, frame], axis=-1)
