"""Scale chart: a full frame for reading a chain's scaling by eye.

Rendered into one end of a chain and inspected at the other, the chart
answers how the chain resized the picture, on each axis (§spec:catalog):

- **Beat rulers** read an upscale. A one-pixel stripe strip, upscaled
  by ``s``, beats every ``1 / |s - round(s)|`` source pixels; beside it,
  combs tick every N pixels, each labelled ``1 + 1/N``. The comb whose
  ticks fall one per band names the scale. The strip and the combs pass
  through the same scaler, so the reading holds whatever raster the
  output is.
- **Zoom ladders** read a crop-and-enlarge. Rungs sit where a centred
  zoom by z puts the frame's edge, each labelled z, so the outermost
  rung left in the picture is the zoom: the top ladder for rows, the
  left one for columns.
- **Stripe rulers** read a downscale. A sweep runs from coarse stripes
  to stripes at 0.45 cycles per pixel, ticked with the scale at which a
  chain can no longer carry them. Past the tick for its scale, a chain
  turns the sweep to grey or moire. Each tick also names the raster
  that scale implies.
- The **pixel grid** (:mod:`display_patterns.patterns._pixel_grid`)
  shows what the scaler does to fine detail: banding, softening, phase,
  ringing.
- The **raster label** names the frame size the chart was rendered at,
  so a chain whose output format differs reads its factor from the two.

Value and layout conventions (§spec:render-model): a float pattern,
``(height, width, 3)`` float32 in [0, 1], achromatic.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from display_patterns.patterns import _backend, _compose, _pixel_grid

# The upscale combs: a comb ticking every N pixels reads 1 + 1/N.
BEAT_PERIODS = (50, 25, 20, 10, 5, 4)

# The zoom ladder's rungs.
ZOOMS = (1.00, 1.02, 1.05, 1.10, 1.15, 1.20, 1.25, 1.33, 1.50)

# The stripe rulers' ticks, and the sweep's top frequency in cycles per
# pixel. Stopping short of Nyquist keeps the sweep's far end from
# beating against the pixel grid at 1:1, which would read as a scaler's
# moire; a tick at s marks where the sweep reaches s / 2 cycles per pixel.
RULER_SCALES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_SWEEP_TOP = 0.45

# The layout, as fractions of the frame's width (X) and height (Y).
_LADDER_TOP_X = 0.20
_LADDER_LEFT_Y = 0.20
_LADDER_LENGTH_X = 0.03
_LADDER_LENGTH_Y = 0.03
_BEAT_X = (0.31, 0.97)
_BEAT_Y = (0.03, 0.155)
_BEAT_STRIP_Y = 0.025
_VERTICAL_BEAT_X = (0.02, 0.17)
_VERTICAL_BEAT_Y = (0.30, 0.97)
_VERTICAL_BEAT_STRIP_X = 0.02
_GRID = (0.20, 0.24, 0.62, 0.60)
_RULER_X = (0.20, 0.95)
_RULER_Y = 0.865
_RULER_TICK_Y = 0.012
_RULER_BAND_Y = 0.04
_VERTICAL_RULER_X = 0.86
_VERTICAL_RULER_Y = (0.24, 0.84)
_VERTICAL_RULER_TICK_X = 0.01
_VERTICAL_RULER_BAND_X = 0.025

# The raster label's font scale, in multiples of the chart's.
_RASTER_LABEL_SCALE = 3

# The smallest frame the layout fits.
_MIN_WIDTH = 1280
_MIN_HEIGHT = 720


def _px(fraction: float, extent: int) -> int:
    return round(fraction * extent)


def ruler_position(scale: float, length: int) -> int:
    """The pixel along a stripe ruler ``length`` long at which the sweep
    reaches ``scale / 2`` cycles per pixel, the most a raster ``scale``
    times the chart's can carry."""
    centre = scale / (2 * _SWEEP_TOP) * length
    return min(length - 1, round(centre - 0.5))


def zoom_edge(zoom: float, extent: int) -> int:
    """Where a centred zoom by ``zoom`` puts the frame's leading edge,
    in pixels from it along an axis ``extent`` long."""
    return round(extent * (1 - 1 / zoom) / 2)


@dataclass(frozen=True, eq=False)
class _Layout:
    """Every block and label of the chart, derived from the frame size."""

    text_scale: int
    rung_width: int
    tick_width: int
    top_ladder: _compose.Block
    left_ladder: _compose.Block
    beat: _compose.Block
    strip: _compose.Block
    comb_rows: tuple[int, ...]
    comb_pitch: int
    vertical_beat: _compose.Block
    vertical_strip: _compose.Block
    vertical_comb_pitch: int
    grid: _pixel_grid.PixelGridGeometry
    ruler: _compose.Block
    ruler_tick: int
    vertical_ruler: _compose.Block
    vertical_ruler_tick: int
    labels: tuple[_compose.Label, ...]
    top_rungs: np.ndarray
    left_rungs: np.ndarray
    combs: tuple[np.ndarray, np.ndarray]
    vertical_combs: tuple[np.ndarray, np.ndarray]
    ruler_marks: np.ndarray
    vertical_ruler_marks: np.ndarray

    @classmethod
    def for_frame(cls, width: int, height: int) -> "_Layout":
        g = _pixel_grid.text_scale_for(height)
        pad = 2 * g
        label_height = _compose.GLYPH_HEIGHT * g
        rung_width = max(2, g)
        tick_width = min(g, 2)
        labels: list[_compose.Label] = []

        def text_width(text: str, scale: int = g) -> int:
            return (len(text) * _compose.ADVANCE - 1) * scale

        # Zoom ladders.
        top_ladder = _compose.Block(
            _px(_LADDER_TOP_X, width),
            0,
            _px(_LADDER_LENGTH_X, width),
            zoom_edge(ZOOMS[-1], height) + rung_width,
        )
        for zoom in ZOOMS:
            labels.append(
                _compose.Label(
                    top_ladder.x + top_ladder.width + pad,
                    zoom_edge(zoom, height) + rung_width + 1,
                    f"{zoom:.2f}",
                    g,
                )
            )
        left_ladder = _compose.Block(
            0,
            _px(_LADDER_LEFT_Y, height),
            zoom_edge(ZOOMS[-1], width) + rung_width,
            _px(_LADDER_LENGTH_Y, height),
        )
        for index, zoom in enumerate(ZOOMS):
            labels.append(
                _compose.Label(
                    zoom_edge(zoom, width) + rung_width + 1,
                    left_ladder.y
                    + left_ladder.height
                    + pad
                    + (index % 2) * (label_height + pad),
                    f"{zoom:.2f}",
                    g,
                )
            )

        # Horizontal beat ruler: comb labels in a column on its left.
        comb_label = text_width("1.00") + 2 * pad
        beat_left, beat_right = (_px(x, width) for x in _BEAT_X)
        beat_top, beat_bottom = (_px(y, height) for y in _BEAT_Y)
        strip_height = _px(_BEAT_STRIP_Y, height)
        comb_pitch = (beat_bottom - beat_top - strip_height - g) // len(BEAT_PERIODS)
        beat = _compose.Block(
            beat_left,
            beat_top,
            beat_right - beat_left,
            strip_height + g + len(BEAT_PERIODS) * comb_pitch,
        )
        strip = _compose.Block(
            beat_left + comb_label, beat_top, beat.width - comb_label, strip_height
        )
        comb_tops = [
            beat_top + strip_height + g + k * comb_pitch
            for k in range(len(BEAT_PERIODS))
        ]
        for period, top in zip(BEAT_PERIODS, comb_tops, strict=True):
            labels.append(
                _compose.Label(
                    beat_left + pad,
                    top + (comb_pitch - g - label_height) // 2,
                    f"{1 + 1 / period:.2f}",
                    g,
                )
            )
        labels.append(
            _compose.Label(
                beat_left,
                beat_bottom + pad,
                f"{width}x{height}",
                _RASTER_LABEL_SCALE * g,
            )
        )

        # Vertical beat ruler: comb labels in a header above.
        header = label_height + 2 * pad
        vb_left, vb_right = (_px(x, width) for x in _VERTICAL_BEAT_X)
        vb_top, vb_bottom = (_px(y, height) for y in _VERTICAL_BEAT_Y)
        strip_width = _px(_VERTICAL_BEAT_STRIP_X, width)
        vertical_pitch = (vb_right - vb_left - strip_width - g) // len(BEAT_PERIODS)
        vertical_beat = _compose.Block(
            vb_left,
            vb_top,
            strip_width + g + len(BEAT_PERIODS) * vertical_pitch,
            vb_bottom - vb_top,
        )
        vertical_strip = _compose.Block(
            vb_left, vb_top + header, strip_width, vertical_beat.height - header
        )
        for k, period in enumerate(BEAT_PERIODS):
            labels.append(
                _compose.Label(
                    vb_left + strip_width + g + k * vertical_pitch,
                    vb_top + pad,
                    f"{1 + 1 / period:.2f}",
                    g,
                )
            )

        grid = _pixel_grid.PixelGridGeometry.for_region(
            width,
            height,
            _compose.Block(
                _px(_GRID[0], width),
                _px(_GRID[1], height),
                _px(_GRID[2], width),
                _px(_GRID[3], height),
            ),
            g,
        )
        labels.extend(grid.labels)

        # Horizontal stripe ruler: ticks above the sweep, two label rows
        # below it.
        ruler_left, ruler_right = (_px(x, width) for x in _RULER_X)
        ruler_tick = _px(_RULER_TICK_Y, height)
        ruler = _compose.Block(
            ruler_left,
            _px(_RULER_Y, height),
            ruler_right - ruler_left,
            ruler_tick + _px(_RULER_BAND_Y, height),
        )
        for scale in RULER_SCALES:
            centre = ruler.x + ruler_position(scale, ruler.width)
            for row, text in enumerate((f"{scale:.1f}", str(round(scale * width)))):
                labels.append(
                    _compose.Label(
                        centre - text_width(text) // 2,
                        ruler.y + ruler.height + pad + row * (label_height + pad),
                        text,
                        g,
                    )
                )

        # Vertical stripe ruler: ticks left of the sweep, labels right.
        vr_top, vr_bottom = (_px(y, height) for y in _VERTICAL_RULER_Y)
        vertical_ruler_tick = _px(_VERTICAL_RULER_TICK_X, width)
        vertical_ruler = _compose.Block(
            _px(_VERTICAL_RULER_X, width),
            vr_top,
            vertical_ruler_tick + _px(_VERTICAL_RULER_BAND_X, width),
            vr_bottom - vr_top,
        )
        for scale in RULER_SCALES:
            centre = vertical_ruler.y + ruler_position(scale, vertical_ruler.height)
            x = vertical_ruler.x + vertical_ruler.width + pad
            for text in (f"{scale:.1f}", str(round(scale * height))):
                labels.append(_compose.Label(x, centre - label_height // 2, text, g))
                x += text_width(text) + 3 * pad

        return cls(
            text_scale=g,
            rung_width=rung_width,
            tick_width=tick_width,
            top_ladder=top_ladder,
            left_ladder=left_ladder,
            beat=beat,
            strip=strip,
            comb_rows=tuple(top + (comb_pitch - g) // 2 for top in comb_tops),
            comb_pitch=comb_pitch,
            vertical_beat=vertical_beat,
            vertical_strip=vertical_strip,
            vertical_comb_pitch=vertical_pitch,
            grid=grid,
            ruler=ruler,
            ruler_tick=ruler_tick,
            vertical_ruler=vertical_ruler,
            vertical_ruler_tick=vertical_ruler_tick,
            labels=tuple(labels),
            top_rungs=_rung_table(
                [zoom_edge(zoom, height) for zoom in ZOOMS],
                top_ladder.height,
                rung_width,
            ),
            left_rungs=_rung_table(
                [zoom_edge(zoom, width) for zoom in ZOOMS],
                left_ladder.width,
                rung_width,
            ),
            combs=_comb_tables(strip_height, g, comb_pitch),
            vertical_combs=_comb_tables(strip_width, g, vertical_pitch),
            ruler_marks=_mark_table(ruler.width, rung_width),
            vertical_ruler_marks=_mark_table(vertical_ruler.height, rung_width),
        )

    @property
    def blocks(self) -> tuple[_compose.Block, ...]:
        return (
            self.top_ladder,
            self.left_ladder,
            self.beat,
            self.vertical_beat,
            self.grid.block,
            self.ruler,
            self.vertical_ruler,
        )


@dataclass(frozen=True)
class ScaleChartGeometry:
    """The scale chart's layout for one frame size. Build with
    :meth:`for_frame`; every position derives from ``width`` and
    ``height``, so the render and a reader agree on where each mark is."""

    width: int
    height: int
    layout: _Layout = field(init=False, repr=False, compare=False)
    label_tables: _compose.LabelTables = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for label, value, least in (
            ("width", self.width, _MIN_WIDTH),
            ("height", self.height, _MIN_HEIGHT),
        ):
            if not least <= value <= _compose.MAX_DIMENSION:
                raise ValueError(
                    f"{label} {value} is out of range: the scale chart's layout "
                    f"needs a frame between {_MIN_WIDTH}x{_MIN_HEIGHT} and "
                    f"{_compose.MAX_DIMENSION} pixels a side; this one is too "
                    f"small or too large."
                )
        layout = _Layout.for_frame(self.width, self.height)
        _compose.check_layout(self.width, self.height, layout.blocks, layout.labels)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(
            self,
            "label_tables",
            _compose.LabelTables.build(layout.labels, self.width, self.height),
        )

    @classmethod
    def for_frame(cls, width: int, height: int) -> "ScaleChartGeometry":
        """Lay the chart out on a ``width`` x ``height`` frame.

        Raises
        ------
        ValueError
            If the frame is smaller than 1280x720 or larger than 16384
            pixels a side.
        """
        return cls(width, height)

    @property
    def labels(self) -> tuple[_compose.Label, ...]:
        return self.layout.labels

    @property
    def beat_periods(self) -> tuple[int, ...]:
        return BEAT_PERIODS

    @property
    def strip(self) -> _compose.Block:
        """The horizontal beat ruler's one-pixel stripe strip."""
        return self.layout.strip

    @property
    def comb_rows(self) -> tuple[int, ...]:
        """A frame row through each horizontal comb, in
        :attr:`beat_periods` order."""
        return self.layout.comb_rows

    @property
    def vertical_strip(self) -> _compose.Block:
        """The vertical beat ruler's one-pixel stripe strip."""
        return self.layout.vertical_strip

    @property
    def top_ladder(self) -> _compose.Block:
        return self.layout.top_ladder

    @property
    def left_ladder(self) -> _compose.Block:
        return self.layout.left_ladder

    @property
    def rung_width(self) -> int:
        return self.layout.rung_width

    @property
    def zoom_rows(self) -> dict[float, int]:
        """Each top-ladder rung's first row, keyed by the zoom it reads."""
        return {zoom: zoom_edge(zoom, self.height) for zoom in ZOOMS}

    @property
    def zoom_cols(self) -> dict[float, int]:
        """Each left-ladder rung's first column, keyed by the zoom it
        reads."""
        return {zoom: zoom_edge(zoom, self.width) for zoom in ZOOMS}

    @property
    def horizontal_ruler(self) -> _compose.Block:
        """The horizontal stripe ruler's sweep, without its ticks."""
        ruler = self.layout.ruler
        tick = self.layout.ruler_tick
        return _compose.Block(ruler.x, ruler.y + tick, ruler.width, ruler.height - tick)

    @staticmethod
    def ruler_position(scale: float, length: int) -> int:
        return ruler_position(scale, length)


def _rung_table(edges: list[int], extent: int, rung: int) -> np.ndarray:
    """Which of ``extent`` rows a ladder lights: ``rung`` rows from each
    of ``edges``."""
    lit = np.zeros(extent, dtype=bool)
    for edge in edges:
        lit[edge : edge + rung] = True
    return lit


def _comb_tables(strip: int, gap: int, pitch: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row of a beat ruler: what it holds (0 grey, 1 stripe strip,
    2 comb) and the comb's period (1 off a comb)."""
    depth = strip + gap + len(BEAT_PERIODS) * pitch
    kind = np.zeros(depth, dtype=np.int32)
    period = np.ones(depth, dtype=np.int32)
    kind[:strip] = 1
    for index, beat in enumerate(BEAT_PERIODS):
        top = strip + gap + index * pitch
        kind[top : top + pitch - gap] = 2
        period[top : top + pitch - gap] = beat
    return kind, period


def _mark_table(length: int, mark: int) -> np.ndarray:
    """Which of a stripe ruler's ``length`` columns carry a tick,
    ``mark`` wide about each ruler scale's position."""
    marked = np.zeros(length, dtype=bool)
    for scale in RULER_SCALES:
        start = ruler_position(scale, length) - mark // 2
        marked[max(0, start) : start + mark] = True
    return marked


def _ladder_block(lit: np.ndarray, length: int, xp: Any, device: Any) -> Any:
    """Rungs across a ``(len(lit), length)`` block: white rows where
    ``lit``, grey between."""
    rows = _backend.asarray(xp, lit, device)
    line = _backend.full(xp, (1, length), _compose.WHITE, xp.float32, device)
    return xp.where(rows[:, None], line, _compose.GREY)


def _beat_block(
    label_band: int,
    length: int,
    tables: tuple[np.ndarray, np.ndarray],
    tick: int,
    xp: Any,
    device: Any,
) -> Any:
    """A beat ruler laid along columns: a grey band for labels, then
    ``length`` columns of the one-pixel stripe strip and, below it, one
    comb per beat period, as ``tables`` lays out the rows."""
    kind = _backend.asarray(xp, tables[0], device)[:, None]
    period = _backend.asarray(xp, tables[1], device)[:, None]
    cols = _backend.arange(xp, label_band + length, xp.int32, device)
    along = (cols - label_band)[None, :]
    stripes = _backend.astype(along % 2, xp.float32)
    ticks = xp.where(along % period < tick, _compose.WHITE, _compose.BLACK)
    value = xp.where(kind == 1, stripes, xp.where(kind == 2, ticks, _compose.GREY))
    return xp.where(along >= 0, value, _compose.GREY)


def _ruler_block(marked: np.ndarray, tick: int, band: int, xp: Any, device: Any) -> Any:
    """A stripe ruler laid along columns: ``tick`` rows of tick marks
    where ``marked``, over ``band`` rows of the sweep."""
    length = marked.size
    marks = _backend.asarray(xp, marked, device)[None, :]

    # The sweep's phase in cycles is _SWEEP_TOP * t**2 / (2 * length) at
    # pixel centre t. Doubling t keeps it an integer u, which puts the
    # phase at 9 * u**2 / (160 * length) for a top of 0.45: an exact
    # integer reduced modulo one cycle before any float arithmetic.
    steps = 160 * length
    u = 2 * _backend.arange(xp, length, xp.int32, device) + 1
    within = ((u * u) % steps * 9) % steps
    sweep = _compose.GREY + 0.5 * xp.cos(
        _backend.astype(within, xp.float32) * (2 * math.pi / steps)
    )
    sweep = _backend.astype(sweep, xp.float32)[None, :]

    rows = _backend.arange(xp, tick + band, xp.int32, device)[:, None]
    tick_row = xp.where(marks, _compose.WHITE, _compose.GREY)
    return xp.where(rows < tick, tick_row, sweep)


def scale_chart(
    geometry: ScaleChartGeometry,
    *,
    frame: int | Any = 0,  # noqa: ARG001 — a still ignores its frame index
    xp: Any = np,
    device: Any = None,
) -> Any:
    """Render the scale chart.

    A still (§spec:render-model): the frame index is accepted for the
    shared calling convention and ignored.

    Parameters
    ----------
    geometry : ScaleChartGeometry
        Frame size and layout; build with
        :meth:`ScaleChartGeometry.for_frame`.
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
        ``(height, width, 3)`` float32 in [0, 1] on mid-grey.
    """
    layout = geometry.layout
    rows = _backend.arange(xp, geometry.height, xp.int32, device)
    cols = _backend.arange(xp, geometry.width, xp.int32, device)
    value = _backend.full(
        xp, (geometry.height, geometry.width), _compose.GREY, xp.float32, device
    )

    def put(block: Any, origin: _compose.Block) -> None:
        nonlocal value
        value = _compose.place(value, block, origin, rows, cols, xp)

    top, left = layout.top_ladder, layout.left_ladder
    put(_ladder_block(layout.top_rungs, top.width, xp, device), top)
    put(_ladder_block(layout.left_rungs, left.height, xp, device).T, left)

    beat, strip = layout.beat, layout.strip
    put(
        _beat_block(
            strip.x - beat.x, strip.width, layout.combs, layout.tick_width, xp, device
        ),
        beat,
    )
    vertical_beat, vertical_strip = layout.vertical_beat, layout.vertical_strip
    put(
        _beat_block(
            vertical_strip.y - vertical_beat.y,
            vertical_strip.height,
            layout.vertical_combs,
            layout.tick_width,
            xp,
            device,
        ).T,
        vertical_beat,
    )

    put(_pixel_grid.render_block(layout.grid, xp, device), layout.grid.block)

    ruler = layout.ruler
    put(
        _ruler_block(
            layout.ruler_marks,
            layout.ruler_tick,
            ruler.height - layout.ruler_tick,
            xp,
            device,
        ),
        ruler,
    )
    vertical_ruler = layout.vertical_ruler
    put(
        _ruler_block(
            layout.vertical_ruler_marks,
            layout.vertical_ruler_tick,
            vertical_ruler.width - layout.vertical_ruler_tick,
            xp,
            device,
        ).T,
        vertical_ruler,
    )

    value = _compose.draw_labels(value, geometry.label_tables, rows, cols, xp, device)
    return _compose.as_rgb(value, xp)
