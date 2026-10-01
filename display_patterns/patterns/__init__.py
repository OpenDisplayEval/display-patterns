"""Core pattern catalog on the frame-indexed rendering model.

Every entry point is a pure function (§spec:render-model): parameters
and a frame index in, an array out, rendered through a caller-supplied
array namespace ``xp`` (numpy default) with optional ``device``
placement. Stills ignore the frame index; temporal patterns use it as
their only time source. The render path performs no I/O and holds no
state.

Render bodies are functional — built from broadcast arithmetic rather
than written into an allocation — and branch on no array contents, so a
consumer may compile one into its own graph (§spec:backend-portability).
Two caller choices decide whether that compiles well:

- Pass a temporal pattern's frame index as array data. A Python integer
  is a compile-time constant, so stepping one recompiles the pattern
  every frame, and past the recompile limit the compiler abandons the
  call site entirely.
- State an output ``dtype`` the backend can compile. ``uint16`` is the
  default and renders on every backend in eager mode, but has no Metal
  code-generation mapping, so a ``uint16`` output cannot be compiled on
  MPS; ``int32`` and ``float32`` can, and carry the same code values
  exactly.

The library never compiles anything itself — that belongs to the
consumer's graph (§spec:non-goals).
"""

from display_patterns.patterns._counter_panel import (
    PanelGeometry,
    decode_counter,
    render_counter_panel,
)
from display_patterns.patterns._fills import (
    ROI,
    ColorRangeError,
    checkerboard,
)
from display_patterns.patterns._zone_plate import (
    ZonePlateGeometry,
    ZonePlateScale,
    zone_plate,
    zone_plate_scale,
)

__all__ = [
    "ROI",
    "ColorRangeError",
    "PanelGeometry",
    "ZonePlateGeometry",
    "ZonePlateScale",
    "checkerboard",
    "decode_counter",
    "render_counter_panel",
    "zone_plate",
    "zone_plate_scale",
]
