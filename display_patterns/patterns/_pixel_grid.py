"""Pixel-level reference grid: geometry and render.

A matrix of high-frequency patches for reading a chain's resampling by
eye (§spec:catalog). Each row is one kind of detail, each column one
cell size in pixels:

- ``checker`` — square cells, the finest detail on both axes at once.
- ``vertical`` / ``horizontal`` — stripes that vary across columns or
  down rows, so each axis reads on its own.
- ``diagonal_down`` / ``diagonal_up`` — stripes at 45 degrees, which a
  scaler that works one axis at a time handles worst.
- ``lines`` — one line each way on black, so ringing shows as halos and
  a sub-pixel shift as a line that splits across two pixels.

Every patch is split into quadrants that sample the pattern one pixel
apart on each axis. A scaler whose output depends on where the pattern
falls against its sample grid (a two-to-one drop, say) renders the
quadrants differently, where an even-handed one renders them alike. The
gutter between patches is the patches' mean grey, so a patch a scaler
averaged away matches its surround.

Under an upscale by a non-integer factor, one-pixel detail beats: its
contrast comes and goes every ``1 / |s - round(s)|`` source pixels. The
scale chart's beat rulers turn that period into a reading.

Value and layout conventions (§spec:render-model): a float pattern,
``(height, width, 3)`` float32, achromatic, holding exactly 0, 0.5 and
1.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from display_patterns.patterns import _backend, _compose

SIZES = (1, 2, 3, 4, 6, 8)
KINDS = ("checker", "vertical", "horizontal", "diagonal_down", "diagonal_up", "lines")

# A quadrant's side is a whole number of periods at every cell size up
# to 6 (periods 2 to 12 pixels), so those stripes and checkers average to
# exactly the gutter grey: the sizes a scaler can average away. Eight-pixel
# cells come within a few percent. Patches step in twice this, one
# quadrant per half.
_QUADRANT_STEP = 24
_PATCH_STEP = 2 * _QUADRANT_STEP

# Fraction of each frame dimension left as a margin around a
# full-frame grid.
_MARGIN_FRACTION = 0.04

# The narrowest gutter, in label-scale units: wide enough to read as
# grey between patches.
_MIN_GUTTER = 2


def text_scale_for(height: int) -> int:
    """Font pixels per frame pixel for a chart ``height`` rows tall: one
    at 1080 lines and below, growing with the frame above it."""
    return max(1, round(height / 1080))


@dataclass(frozen=True)
class PixelGridGeometry:
    """Where the grid lands: its top-left corner, cell pitch, and patch
    side in frame pixels, with a header row of size labels above."""

    width: int
    height: int
    x: int
    y: int
    cell: int
    patch: int
    header: int
    text_scale: int
    label_tables: _compose.LabelTables = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _compose.check_layout(self.width, self.height, (self.block,), self.labels)
        object.__setattr__(
            self,
            "label_tables",
            _compose.LabelTables.build(self.labels, self.width, self.height),
        )

    @classmethod
    def for_frame(cls, width: int, height: int) -> "PixelGridGeometry":
        """Centre the grid in a ``width`` x ``height`` frame.

        Raises
        ------
        ValueError
            If the frame cannot hold a patch of at least two 24-pixel
            quadrants.
        """
        margin_x = round(width * _MARGIN_FRACTION)
        margin_y = round(height * _MARGIN_FRACTION)
        return cls.for_region(
            width,
            height,
            _compose.Block(
                margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y
            ),
            text_scale_for(height),
        )

    @classmethod
    def for_region(
        cls, width: int, height: int, region: _compose.Block, text_scale: int
    ) -> "PixelGridGeometry":
        """Centre the grid in ``region`` of a ``width`` x ``height``
        frame, labelled at ``text_scale``."""
        header = (_compose.GLYPH_HEIGHT + 4) * text_scale
        cell = min(region.width // len(SIZES), (region.height - header) // len(KINDS))
        patch = (cell - 2 * _MIN_GUTTER * text_scale) // _PATCH_STEP * _PATCH_STEP
        if patch < _PATCH_STEP:
            raise ValueError(
                f"a {region.width}x{region.height} region is too small for the "
                f"pixel grid: each patch needs at least {_PATCH_STEP}px plus its "
                f"gutter."
            )
        x = region.x + (region.width - len(SIZES) * cell) // 2
        y = region.y + (region.height - header - len(KINDS) * cell) // 2
        return cls(width, height, x, y, cell, patch, header, text_scale)

    @property
    def sizes(self) -> tuple[int, ...]:
        return SIZES

    @property
    def kinds(self) -> tuple[str, ...]:
        return KINDS

    @property
    def gutter(self) -> int:
        """Grey between a patch and its cell's edge, on each side."""
        return (self.cell - self.patch) // 2

    @property
    def block(self) -> _compose.Block:
        """The frame rectangle the grid covers, header included."""
        return _compose.Block(
            self.x,
            self.y,
            len(SIZES) * self.cell,
            self.header + len(KINDS) * self.cell,
        )

    @property
    def labels(self) -> tuple[_compose.Label, ...]:
        """Each column's cell size, centred over it."""
        labels = []
        for column, size in enumerate(SIZES):
            text = str(size)
            width = (len(text) * _compose.ADVANCE - 1) * self.text_scale
            labels.append(
                _compose.Label(
                    self.x + column * self.cell + (self.cell - width) // 2,
                    self.y + 2 * self.text_scale,
                    text,
                    self.text_scale,
                )
            )
        return tuple(labels)

    def patch_origin(self, kind: str, size: int) -> tuple[int, int]:
        """The ``(x, y)`` frame pixel at the top-left of a patch."""
        column = SIZES.index(size)
        row = KINDS.index(kind)
        return (
            self.x + column * self.cell + self.gutter,
            self.y + self.header + row * self.cell + self.gutter,
        )


def render_block(geometry: PixelGridGeometry, xp: Any, device: Any) -> Any:
    """The grid over its own bounding box, ``(height, width)`` float32."""
    block = geometry.block
    cell, patch, gutter = geometry.cell, geometry.patch, geometry.gutter
    half = patch // 2
    quarter = half // 2

    rows = _backend.arange(xp, block.height, xp.int32, device) - geometry.header
    cols = _backend.arange(xp, block.width, xp.int32, device)
    row_cell = rows // cell
    col_cell = cols // cell
    local_y = rows - row_cell * cell - gutter
    local_x = cols - col_cell * cell - gutter
    in_patch = ((rows >= 0) & (local_y >= 0) & (local_y < patch))[:, None] & (
        (local_x >= 0) & (local_x < patch)
    )[None, :]

    sizes = _backend.asarray(xp, np.array(SIZES, dtype=np.int32), device)
    size = sizes[col_cell % len(SIZES)][None, :]
    kind = (row_cell % len(KINDS))[:, None]

    # One pixel's phase step in the right and lower quadrants.
    step_x = xp.where(local_x >= half, 1, 0)
    step_y = xp.where(local_y >= half, 1, 0)
    x = (local_x + step_x)[None, :]
    y = (local_y + step_y)[:, None]

    # Each quadrant of a lines patch holds one cross, centred in it and
    # moved by the quadrant's phase step.
    across = (local_x % half - (quarter + step_x))[None, :] + size // 2
    down = (local_y % half - (quarter + step_y))[:, None] + size // 2
    lines = ((across >= 0) & (across < size)) | ((down >= 0) & (down < size))

    pattern = xp.where(
        kind == 0,
        (x // size + y // size) % 2,
        xp.where(
            kind == 1,
            (x // size) % 2,
            xp.where(
                kind == 2,
                (y // size) % 2,
                xp.where(
                    kind == 3,
                    ((x + y) // size) % 2,
                    xp.where(kind == 4, ((x - y) // size) % 2, xp.where(lines, 1, 0)),
                ),
            ),
        ),
    )
    value = _backend.astype(pattern, xp.float32)
    return xp.where(in_patch, value, _compose.GREY)


def pixel_grid(
    geometry: PixelGridGeometry,
    *,
    frame: int | Any = 0,  # noqa: ARG001 — a still ignores its frame index
    xp: Any = np,
    device: Any = None,
) -> Any:
    """Render the pixel-level reference grid.

    A still (§spec:render-model): the frame index is accepted for the
    shared calling convention and ignored.

    Parameters
    ----------
    geometry : PixelGridGeometry
        Frame size and layout; build with
        :meth:`PixelGridGeometry.for_frame`.
    frame : int or array, optional
        Ignored.
    xp : namespace, optional
        Array namespace to render through (numpy default; torch on GPU
        hosts).
    device : optional
        Device placement for ``xp`` backends that take one.

    Returns
    -------
    array
        ``(height, width, 3)`` float32: the grid on mid-grey, its
        patches 0 and 1, its labels 1.
    """
    rows = _backend.arange(xp, geometry.height, xp.int32, device)
    cols = _backend.arange(xp, geometry.width, xp.int32, device)
    frame_value = _backend.full(
        xp, (geometry.height, geometry.width), _compose.GREY, xp.float32, device
    )
    frame_value = _compose.place(
        frame_value, render_block(geometry, xp, device), geometry.block, rows, cols, xp
    )
    frame_value = _compose.draw_labels(
        frame_value, geometry.label_tables, rows, cols, xp, device
    )
    return _compose.as_rgb(frame_value, xp)
