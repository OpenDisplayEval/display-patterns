# Requirements

Problem-space document for display-patterns: a device-free library of
display test patterns. Extracted from
[bmd-signal-gen](https://github.com/OpenDisplayEval/bmd-signal-gen), whose
SPEC records the split decision (`§spec:pattern-library` there).

## Problem statement §req:problem-statement

Target users are display and video engineers evaluating, calibrating,
and aligning displays — LED walls foremost — and the tools they build:
signal generators driving SDI output, chart authoring scripts,
real-time render graphs, and measurement pipelines.

Test-pattern math lives trapped inside device tools. bmd-signal-gen's
checkerboard and chart generation imports nothing from its DeckLink
layer, yet a consumer who wants only the patterns installs the whole
device tool. A renderer re-derived its own frame-counter panel math
because no importable source existed. Each new consumer rewrites
geometry that is pure, deterministic, and identical across delivery
paths.

General imaging libraries fall short in the other direction: they
carry color-management and encoding opinions and offer no measurement
semantics. Measurement work drives *exact* integer code values at a
stated bit depth; a library that scales, tags, or quantizes behind the
caller's back cannot make a checkable radiometric claim.

## Success criteria §req:success-criteria

- bmd-signal-gen consumes this library in place of its in-tree
  `image_generators` and `charts` modules, and its CLI output is
  bit-identical before and after the switch.
- A consumer with only numpy installed imports the package and renders
  every core catalog pattern — no device SDK, no imaging stack.
- A requested code value arrives in the rendered array unmodified: a
  12-bit checkerboard authored at (4095, 2048, 0) contains exactly
  those integers.
- A frame-counter panel rendered at frame N decodes back to N — from
  the rendered array, and from a captured frame after a lossy trip
  through a real video chain.
- The same pattern renders under numpy and under torch and decodes to
  the same result, without a copy through host memory on the torch
  path.
- Rendering a still is a pure function of its parameters; rendering
  frame N of a motion pattern is a pure function of its parameters and
  N. Two calls with the same inputs produce identical arrays on any
  platform.

## User stories §req:user-stories

- As a display engineer, I drive a checkerboard at exact 12-bit code
  values through bmd-signal-gen to compare an LED wall's response
  under partial load against full-field drive, so power-supply droop
  is measurable rather than anecdotal.
- As a color scientist, I author a chart as a YAML patch list with
  colorimetric values, render it, and export a TIFF for review, so
  chart production is scriptable and reproducible.
- As a video-systems engineer, I display a frame-counter pattern into
  one end of a chain and decode captured frames at the other, so I can
  measure end-to-end latency and integer frame skew of a chain I do
  not control.
- As a video-systems engineer, I display a zone plate into one end of a
  chain and measure the capture at the other, so I can tell whether the
  chain resamples the picture, and by how much on each axis.
- As a real-time-pipeline developer, I render patterns frame by frame
  inside my own loop, on my own clock, into my own GPU tensors, so
  motion patterns cost me no playback machinery and no host round
  trip.
- As the bmd-signal-gen maintainer, I delete the in-tree pattern
  modules and depend on this library, so the math is maintained once
  and shared by every consumer.

## Quality attributes §req:quality-attributes

- **Determinism.** Same inputs, same array — across runs, platforms,
  and array backends. No hidden state, no wall-clock dependence.
- **Exactness.** Integer code values are delivered exactly; float
  patterns state their range convention and honor it. The library
  never rescales or quantizes on the caller's behalf.
- **Performance.** Rendering is vectorized and allocation-conscious;
  a 4K frame renders comfortably inside a 60 Hz consumer's frame
  budget on commodity hardware. The render path performs no I/O.
- **Footprint.** The core depends on numpy alone. Chart rendering,
  colorimetric conversion, and file export are optional extras with
  their own dependencies.
- **Portability.** Pure Python on CPython 3.12+, running wherever its
  consumers run (macOS, Linux, Windows; CPU or GPU hosts).

## Constraints §req:constraints

- Public repository under OpenDisplayEval; BSD-3-Clause; published to
  PyPI as `display-patterns`.
- Seeded by extraction from bmd-signal-gen (`bmd_sg/image_generators/`,
  `bmd_sg/charts/` and their tests, moved verbatim before any
  reshaping); bmd-signal-gen adoption is non-breaking behind one
  release cycle of deprecation shims.
- No device I/O and no playback: the library defines the mapping from
  parameters (and frame index) to image; output devices, frame pacing,
  clocks, and scheduling belong to consumers.
- No color-management decisions in the core: value meaning (code
  values, scene-linear, display-referred) is the caller's contract.
  Colorimetric chart math lives in an extra and states its
  conventions.
- Consumers include GPU-resident torch pipelines that cannot afford a
  numpy round trip: pattern math renders into a caller-supplied array
  namespace.
- Motion is in scope from the first release: the rendering signature
  takes a frame index so temporal patterns (frame counters, motion
  test material) are ordinary catalog entries, not a second API.

## Priorities §req:priorities

Essential, in adoption order:

1. Core catalog with exact code values: solid, checkerboard, region
   of interest, bit-depth validation.
2. bmd-signal-gen adoption (requires the charts and TIFF-export
   extras) with bit-identical output.
3. Frame-indexed rendering signature on every pattern.
4. Temporal-alignment counter panel, encode and decode (ported from
   a renderer's probe math).

Nice-to-have, after adoption:

- Additional motion patterns (scrolling bars, flicker and judder
  material).
- Torch-backend verification in CI.
- Catalog growth as consumers contribute (PLUGE, ramps, zone plates).
