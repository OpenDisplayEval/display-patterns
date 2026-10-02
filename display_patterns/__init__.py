"""
display-patterns: deterministic display test patterns, device-free and exact.

The package root is the canonical import surface; the core catalog is
implemented in ``display_patterns.patterns`` and resolves here lazily,
so importing the package costs nothing until a pattern is used. Chart
authoring and TIFF export live in ``display_patterns.charts`` behind
the ``charts`` and ``io`` extras.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from display_patterns.patterns import (
        ROI,
        ColorRangeError,
        PanelGeometry,
        PixelGridGeometry,
        ScaleChartGeometry,
        ZonePlateGeometry,
        ZonePlateScale,
        checkerboard,
        decode_counter,
        pixel_grid,
        render_counter_panel,
        scale_chart,
        zone_plate,
        zone_plate_scale,
    )

__all__ = [
    "ROI",
    "ColorRangeError",
    "PanelGeometry",
    "PixelGridGeometry",
    "ScaleChartGeometry",
    "ZonePlateGeometry",
    "ZonePlateScale",
    "checkerboard",
    "decode_counter",
    "pixel_grid",
    "render_counter_panel",
    "scale_chart",
    "zone_plate",
    "zone_plate_scale",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        return getattr(importlib.import_module("display_patterns.patterns"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
