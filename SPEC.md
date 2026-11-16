# display-patterns — Specification

Declarative description of the system. Each section states what the
system does and why. See ROADMAP.md for work remaining.

## Problem §spec:problem

*Status: complete*

Display test-pattern math lives trapped inside device tools
(§req:problem-statement). bmd-signal-gen's pattern and chart modules
import nothing from its DeckLink layer yet ship only inside the device
tool; a renderer re-derived frame-counter panel math for want of an
importable source. General imaging libraries carry color-management
opinions and cannot make a measurement claim: driving
*exact* integer code values at a stated bit depth is the point, and a
library that rescales or quantizes behind the caller's back defeats
it.

display-patterns is that importable source: deterministic pattern
math, device-free, exact by construction. bmd-signal-gen's SPEC
records the extraction decision on its side (`§spec:pattern-library`
there).

## Package shape §spec:package-shape

*Status: in progress*

The distribution is `display-patterns` on PyPI; the import package is
`display_patterns`. The core installs with numpy as its only
dependency and renders every core catalog pattern
(§req:success-criteria). Optional extras price heavier dependencies
separately (§req:quality-attributes footprint):

- `charts` — chart definitions, colorimetric conversion, and chart
  rendering (colour-science, Pillow, PyYAML).
- `io` — file export (tifffile).

**Why extras and not packages:** one repository, one version, one
release cycle keeps the extraction and its consumers simple; the
extras keep a numpy-only consumer from paying for an imaging stack it
never imports. Two packages would add a version seam with no consumer
on the far side of it.

The package runs on CPython 3.12+ on macOS, Linux, and Windows
(§req:constraints); nothing in it opens a device, a socket, or a
display.

## Rendering model §spec:render-model

*Status: complete*

A pattern is a pure function: parameters (geometry, values) and a
frame index in, an array out. The same inputs produce identical
arrays on every run, platform, and backend (§req:success-criteria).
Stills ignore the frame index; temporal patterns use it as their only
time source. The render path performs no I/O and holds no state.

**Why frame-indexed and not stateful:** consumers render inside their
own frame loops on their own clocks (§req:user-stories) — a DeckLink
frame loop, a GPU graph's per-frame tick. A pure function of frame
index is seekable, parallelizable, and testable in isolation; an
iterator or clock abstraction would drag playback concerns into pure
math. Playback, pacing, and scheduling stay with consumers
(§req:constraints).

Pattern math renders through a caller-supplied array namespace (numpy
on CPU hosts, torch on GPU hosts) with an optional device placement,
so a GPU-resident consumer pays no copy through host memory
(§req:constraints). **Why a namespace parameter and not a backend
registry:** the caller already knows its backend; a parameter is the
whole mechanism, and numpy remains the default for the common case.

Values are the caller's contract (§req:quality-attributes exactness).
Integer patterns deliver requested code values unmodified and validate
them against the stated bit depth; float patterns state their range
convention. **Why no canonical value space:** measurement drives exact
drive-space integers, render graphs author scene-referred floats —
privileging either would force the other through a conversion that
manufactures error. The library owns geometry; meaning stays with the
caller.

Catalog entries share one calling convention: pattern parameters, a
`frame` index, and the `xp`/`device` keywords. Rasters are HWC
(height, width, channels); integer patterns return the dtype the
caller states, `uint16` by default (§spec:backend-portability); a
pattern whose output is more than one plane (the counter panel's
overlay + mask) returns a documented tuple. **Why HWC and not the
source runtime's NCHW:** these are images consumers composite and
encode, not batched graph payloads; the batch and channel-first axes
belong to the runtime that needs them.

## Catalog §spec:catalog

*Status: complete*

The core catalog, renderable with numpy alone (§req:success-criteria):

- **Checkerboard** fills (one entry point; a single color renders a
  solid) with the established color expansion (two alternating, four
  in a 2×2 tile), bounded by an optional **region of interest**,
  validated against the stated bit depth with a specific error for
  out-of-range values.
- **Temporal-alignment counter panel** — a frame counter encoded as
  binary bit-cells in a title-safe row, with the matching decoder.
  Encode and decode ship together so any consumer can render the
  panel into one end of a video chain and read frame skew and latency
  back out of the other (§req:user-stories). The geometry is
  deterministic, so encoder and decoder agree on every cell from
  parameters alone, and decode thresholds at the cell midpoint to
  survive a lossy chain (§req:success-criteria). Ported from a
  renderer's alignment-probe math, which retires its bespoke node in
  favor of generic primitives.
- **Zone plate** — a circular chirp whose frequency rises from zero at
  the centre to Nyquist on the largest inscribed circle, mid-grey
  outside it, with a measurement that reads from a capture the raster
  scale a chain resampled the frame through, per axis. A scaler folds
  the band it cannot carry, and demodulating the capture by the plate's
  own phase turns every fold into a tone at the scaler's sampling rate,
  so the scale is a spectral peak rather than a threshold on blur
  (§req:user-stories). A capture sampled once per pixel cannot tell a
  scale `s` from `1 - s`; whether the plate survived between the two
  decides, which is least certain near half scale for a scaler that
  does not filter.
- **Pixel grid** — patches of checkers, stripes on each axis, both
  diagonals, and isolated lines at cell sizes from 1 to 8 pixels, each
  split into quadrants one pixel apart in phase, on a gutter at the
  patches' mean grey. Read by eye: a scaler bands, softens, decimates,
  or phase-shifts the patches, and the finest size that survives
  intact bounds its scale (§req:user-stories).
- **Scale chart** — a full frame for reading a chain's scaling by eye,
  per axis, either side of 1:1. Beat rulers read an upscale: a
  one-pixel stripe strip upscaled by `s` beats every
  `1 / |s - round(s)|` source pixels, beside combs labelled with the
  scale each period implies. Zoom ladders read a crop-and-enlarge as
  the outermost labelled rung left in frame. Stripe rulers read a
  downscale as the tick past which a sweep has gone to grey or moiré.
  The chart also carries the pixel grid and names the raster it was
  rendered at. **Why beats, not a blur threshold, for upscale:** a
  clean upscale destroys no detail, so nothing fades; the one
  output-space reference is the pixel grid, and one-pixel detail
  beating against it is visible at normal viewing distance. The beat
  gives only the distance from `s` to a whole number, so a whole-number
  upscale shows no bands and reads from the raster label or the pixel
  grid instead.

The `charts` extra adds chart production (§req:user-stories): a chart
is authored as a YAML patch list carrying colorimetric values,
converted to display RGB, and rendered with patch labels. The
conversion states its colorimetric conventions; chart math makes no
claim about the display beyond what the author provides.

**Why the counter panel is core, not an extra:** it is pure geometry
with no dependencies beyond the array namespace, and its decoder is
the library's distinguishing measurement feature — a machine-readable
pattern, not just a visible one.

Catalog growth (motion material, PLUGE, ramps) enters as
consumers need it (§req:priorities); each entry follows the rendering
model and carries a decode side when one is meaningful.

## Backend portability §spec:backend-portability

*Status: complete*

The core catalog renders the same values on every backend a consumer
brings: numpy on a CPU host, torch on CUDA, torch on Apple's MPS
(§req:quality-attributes portability). A counter panel rendered on any
of them decodes to the frame index it encodes
(§req:success-criteria).

Render bodies are functional. An entry point computes its result from
broadcast arithmetic over coordinate and bit-position arrays and
returns it, rather than allocating a buffer and writing strided slices
into it. No render body branches on array contents, and none reads a
value back to the host; parameter validation stays host-side, where
the parameters already are.

**Why functional and not in-place:** a strided scatter is a
materialized intermediate no compiler fuses away, so an in-place body
spends a pass over the frame per write where the whole pattern is one
kernel — measured at 2160p on MPS, 3.5 ms for the strided body against
0.7 ms for the functional one, before any compilation. In-place bodies
also exclude any backend whose arrays are immutable. The library never
calls `torch.compile` itself — compilation belongs to the consumer's
graph (§spec:non-goals) — so its obligation is to stay *compilable*,
and staying compilable is what vectorized means here
(§req:quality-attributes performance).

A temporal pattern's frame index is array data, not a Python integer:
`frame` accepts a scalar or a zero-dimensional array, and the
counter's bits are extracted arithmetically. **Why the frame index is
data:** to a tracing compiler a Python integer is a compile-time
constant, so a consumer compiling its frame loop would recompile the
pattern on every frame and the fused path would cost more than the
eager one it replaced. The counter is bounded at 31 bits so its
arithmetic fits the signed 32-bit integers every backend supports; at
60 Hz that counts for over a year before wrapping.

Output dtype is the caller's, defaulting to the exact-integer type the
pattern has always returned. An integer pattern renders into any dtype
that represents its stated bit depth without loss and rejects one that
cannot, so exactness is preserved by the check rather than by a fixed
type (§req:quality-attributes exactness). **Why the caller states the
dtype:** a fixed return type forces every consumer whose value space
differs to pay a conversion pass, and it collides with what a backend
can compile. `uint16` renders on MPS in eager mode but has no Metal
code-generation mapping, so a `uint16` output cannot be compiled there
at all — and which types a backend supports is the backend's business,
moving between its releases. The exactness claim is about the values
that arrive; any type wide enough carries them just as exactly.

Verification is proportional to where the hardware is
(§req:priorities). The torch leg runs on every change — torch's CPU
build is a development dependency, so backend equivalence is a gate
rather than a test that skips silently. The CUDA and MPS legs are
opt-in markers run on the hardware that has them. CI asserts the
backend contract; hardware asserts the device.

The `charts` extra is numpy-only and stays so: chart rendering draws
text through an imaging library and converts through a colorimetry
library, neither of which has a device backend. Backend portability is
a core-catalog property.

## Extraction and compatibility §spec:extraction

*Status: complete*

The initial code was bmd-signal-gen's `bmd_sg/image_generators/` and
`bmd_sg/charts/` with their tests, moved verbatim before any reshaping
(§req:constraints). bmd-signal-gen adopted the package at its v0.2.1
behind one release cycle of deprecation shims, with bit-identical CLI
output verified array-for-array against its last pre-split commit —
`0640015fc8b851d8db987eb5d81f1bc003ddeffa`, the pin any re-check needs,
because every later bmd-signal-gen commit ships shims of this package
and compares the library to itself (§req:success-criteria); the shims
live on its side (`§road:pattern-library` there).

**Why verbatim first:** bit-identical adoption is checkable only
against an unchanged implementation. With that claim banked, the
equivalence scaffolding (test module and CI job) is retired and the
rendering-model reshape lands as separate, visible changes. The
extraction-era `image_generators` class surface stays importable and
delegates to the catalog, so adopters upgrade without code changes.

## File export §spec:file-export

*Status: in progress*

The `io` extra writes rendered arrays to 16-bit TIFF, serving the
chart review workflow (§req:user-stories). Export is one-way — the
library reads nothing back but its own counter panels.

**Why TIFF only:** the one consumer workflow that needs a file gets
one. Consumers with artifact contracts of their own — byte-exact,
deliberately untagged probe imagery — write those files themselves so
no library default can reinterpret them (ocio-display-gen records
that rationale in its `§spec:verification`).

## Scope boundaries §spec:non-goals

*Status: complete*

Out of scope, with their owners: device output and signaling
(bmd-signal-gen, pydecklink); playback, clocks, and frame pacing
(consumers' runtimes); color management and display characterization
(ocio-display-gen); instrument I/O and measurement sessions
(colour-specio, and the surface-characterization umbrella that drives
it); runtime graph integration (consumers' render runtimes). The
library defines the mapping from parameters to image and nothing on
either side of it.
