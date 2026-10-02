# display-patterns

**Deterministic display test patterns, device-free and exact.**

Test-pattern math tends to live trapped inside device tools. This
library is the importable version: solid fields, checkerboards, zone
plates, scale charts, charts, and machine-readable temporal-alignment patterns, rendered as
pure functions of their parameters and a frame index. Integer
patterns deliver exact code values at a stated bit depth — the
property measurement work depends on — and the render path never
touches a device, a clock, or a file.

The core depends on numpy alone and renders into a caller-supplied
array namespace (numpy or torch, CPU or GPU), so it serves CLI signal
generators and GPU-resident render graphs alike. Render bodies are
functional and free of Python-level branching on frame data, so a
consumer can compile them into its own graph. Chart authoring and TIFF
export install as extras.

Used by
[bmd-signal-gen](https://github.com/OpenDisplayEval/bmd-signal-gen),
which sends the rendered patterns to a display over SDI or HDMI.
Related:
[display-measure](https://github.com/OpenDisplayEval/display-measure),
[display-report](https://github.com/OpenDisplayEval/display-report), and
[methodology](https://github.com/OpenDisplayEval/methodology), which
measure, report on, and document the characterization of a display
surface.

## Installation

```sh
pip install display-patterns              # core: numpy only
pip install "display-patterns[charts,io]" # chart authoring + TIFF export
```

Until the first PyPI release, install from a checkout:
`pip install .` (or `uv pip install .`).

## Usage

Render a 12-bit checkerboard at exact code values. Every catalog entry
is a pure function of its parameters and a frame index (stills ignore
the index), rendered through a caller-supplied array namespace —
numpy by default, torch on GPU hosts:

```python
from display_patterns import checkerboard

frame = checkerboard(
    [[4095, 2048, 0], [0, 0, 0]], width=1920, height=1080, bit_depth=12
)  # uint16 (1080, 1920, 3)

# GPU-resident render, no host round trip:
# frame = checkerboard(..., xp=torch, device="cuda")
```

`dtype` states the output type, `uint16` by default. Any type that
carries the stated bit depth exactly is accepted; one that cannot is
refused rather than silently wrapping. See [Backends](#backends) for
when to state it.

Encode a frame counter as machine-readable bit-cells and decode it back
— render one end of a video chain, measure latency and frame skew at
the other:

```python
from display_patterns import PanelGeometry, decode_counter, render_counter_panel

geometry = PanelGeometry.for_frame(width=1920, height=1080, bits=16)
overlay, mask = render_counter_panel(1234, geometry)  # float32 [0, 1], HWC
assert decode_counter(overlay, geometry) == 1234
```

The counter carries 1 to 31 bits and wraps at `2**bits` — 31 bits runs
over a year at 60 Hz. Pass the frame index as a zero-dimensional array
rather than a Python integer when compiling (see below).

Render a zone plate to find out whether a chain resamples the picture,
and by how much. The plate's rings tighten from the centre to one pixel
per half-cycle at the edge of the largest circle the frame holds. A
scaler cannot carry the outer rings, so it folds them into moiré, and
`zone_plate_scale` reads the fold back from a capture as a scale per
axis:

```python
from display_patterns import ZonePlateGeometry, zone_plate, zone_plate_scale

geometry = ZonePlateGeometry.for_frame(width=3840, height=2160)
plate = zone_plate(geometry)  # float32 [0, 1], HWC; mid-grey outside the circle
zone_plate_scale(capture, geometry)  # ZonePlateScale(horizontal=0.5, vertical=0.5)
```

An untouched chain reads 1.0 on both axes, and so does one that only
enlarges the picture. Near half scale, a scaler with no antialiasing can
read as `1 - s`, because there the fold lands on the plate itself.

To read a chain's scaling by eye instead, render the scale chart into
it and look at the output:

```python
from display_patterns import ScaleChartGeometry, scale_chart

chart = scale_chart(ScaleChartGeometry.for_frame(width=3840, height=2160))
```

Each element is captioned, and a legend in the top-left corner says how
to read it:

- **Upscale:** the one-pixel strip beside the combs breaks into bands.
  The comb whose ticks fall one per band reads the scale (1.02 to
  1.25). The bands give only the distance to a whole number, so 1.1x
  and 1.9x look alike; a whole-number upscale shows no bands at all.
  The raster label settles both, read against the output format.
- **Zoom:** the outermost ladder rung left in the picture reads the
  zoom: the top ladder for rows, the left one for columns.
- **Downscale:** the stripe ruler's sweep goes to grey or moiré near
  the tick for the chain's scale; a filtering scaler fades it over the
  stretch before the tick. Each tick also names the raster that scale
  implies.
- **Detail:** the pixel grid shows what the scaler does to 1- to 8-pixel
  checkers, stripes, diagonals and lines. `pixel_grid` renders the grid
  on its own.

View the output at 1:1. A monitor that scales the picture to fit adds
its own scaling to the reading.

## Backends

Every core pattern renders through the `xp` namespace with an optional
`device`, and returns the same values on each. Two things are worth
knowing before putting one inside a compiled graph.

**Pass the frame index as array data.** A Python integer is a
compile-time constant, so stepping one recompiles the pattern on every
frame — and past the recompile limit `torch.compile` gives up on the
call site and falls back to eager for good:

```python
counter = torch.tensor(frame_number, dtype=torch.int32, device=device)
overlay, mask = render_counter_panel(counter, geometry, xp=torch, device=device)
```

**State a dtype your backend can compile.** `uint16` renders correctly
on MPS in eager mode, but Inductor's Metal code generator has no
mapping for it, so a `uint16` output cannot be compiled there at all.
An `int32` or `float32` output can, and carries 12-bit code values just
as exactly:

```python
frame = checkerboard(colors, ..., xp=torch, device=device, dtype=torch.int32)
```

Which types a backend supports is the backend's business and moves
between its releases, which is why the library takes the dtype from the
caller rather than fixing one.

Verified on numpy and on torch's CPU build in CI; on CUDA and MPS by
`pytest -m cuda` and `pytest -m mps`, which are deselected by default
and run where the hardware is.

With the `charts` and `io` extras, author a chart in YAML, render it,
and write a 16-bit TIFF:

```python
from display_patterns.charts import render_chart, write_chart_tiff
from display_patterns.charts.loaders import load_chart

layout = load_chart("my_chart.yaml")
image = render_chart(layout, bit_depth=12)
write_chart_tiff("my_chart.tiff", image, layout)
```

## API

- `display_patterns` / `display_patterns.patterns` — the core catalog:
  `checkerboard` (one color renders a solid), `ROI`, `ColorRangeError`,
  the temporal-alignment counter panel (`PanelGeometry`,
  `render_counter_panel`, `decode_counter`), and the zone plate
  (`ZonePlateGeometry`, `zone_plate`, `zone_plate_scale`,
  `ZonePlateScale`), the pixel grid (`PixelGridGeometry`, `pixel_grid`),
  and the scale chart (`ScaleChartGeometry`, `scale_chart`). The root
  is the canonical
  import surface; no dependency beyond numpy — torch renders through
  the `xp` parameter without ever being required.
- `display_patterns.image_generators` — legacy class surface
  (`PatternGenerator`), kept for existing consumers; delegates to the
  catalog.
- `display_patterns.charts` — chart types, colorimetric conversion, and
  rendering (`charts` extra); TIFF read/write (`io` extra).
- `display_patterns.charts.loaders` — YAML chart definitions
  (`charts` extra).

All public functions and classes carry NumPy-style docstrings; the
package ships `py.typed` for static type checking.

## Governance

- [REQUIREMENTS.md](REQUIREMENTS.md) — the problem space
- [SPEC.md](SPEC.md) — design and rationale
- [ROADMAP.md](ROADMAP.md) — work remaining

## License

BSD 3-Clause.
