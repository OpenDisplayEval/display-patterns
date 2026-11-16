"""Circular zone plate and its scale measurement (§spec:catalog).

The plate sweeps spatial frequency from zero at the centre to Nyquist at
the edge of the largest circle the frame holds, so a chain that resamples
the frame folds the band where its smallest raster runs out. The
measurement reads that fold back from a capture, per axis: a chain that
passes the frame untouched measures 1.0, and one that scaled the frame
down and back measures the scale it went through. The resampling chains
here run through torch's ``interpolate``, which stands in for a
downstream scaler.
"""

import numpy as np
import pytest

from display_patterns import (
    ZonePlateGeometry,
    ZonePlateScale,
    zone_plate,
    zone_plate_scale,
)
from tests.conftest import device_or_skip, to_host, torch_or_skip

# A frame small enough to keep the suite fast and large enough that the
# plate carries enough cycles to place a fold (64 at a 512px diameter).
WIDTH = 896
HEIGHT = 512

# Cross-backend agreement: cosine is not correctly rounded on every
# backend, so float32 renders may differ by an ulp or two.
BACKEND_ATOL = 1e-6

# How closely a measured scale has to land on the chain's. The spectral
# peak resolves to one bin, 1/896 of a cycle per pixel here.
SCALE_TOLERANCE = 0.02

UNTOUCHED = ZonePlateScale(horizontal=1.0, vertical=1.0)


def _geometry() -> ZonePlateGeometry:
    return ZonePlateGeometry.for_frame(width=WIDTH, height=HEIGHT)


def _resample(
    frame: np.ndarray,
    scale: tuple[float, float],
    mode: str,
    antialias: bool,
) -> np.ndarray:
    """``frame`` scaled by ``scale`` (vertical, horizontal) and back to
    its own size: a chain that carries the picture through a smaller
    raster."""
    torch = torch_or_skip()
    import torch.nn.functional as functional

    height, width = frame.shape[:2]
    nchw = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None]
    small_size = (round(height * scale[0]), round(width * scale[1]))
    options = {} if mode in ("nearest", "area") else {"align_corners": False}
    small = functional.interpolate(
        nchw, size=small_size, mode=mode, antialias=antialias, **options
    )
    back = functional.interpolate(small, size=(height, width), mode=mode, **options)
    return back[0].permute(1, 2, 0).numpy()


class TestGeometry:
    def test_reaches_nyquist_at_the_largest_inscribed_circle(self) -> None:
        """At UHD the circle spans the frame's height."""
        geom = ZonePlateGeometry.for_frame(width=3840, height=2160)

        assert geom.diameter == 2160
        assert geom.nyquist_radius == 1080

    def test_rejects_an_out_of_range_dimension(self) -> None:
        with pytest.raises(ValueError, match="between"):
            ZonePlateGeometry.for_frame(width=0, height=64)
        with pytest.raises(ValueError, match="between"):
            ZonePlateGeometry.for_frame(width=64, height=16385)

    def test_rejects_a_frame_too_small_to_carry_a_cycle(self) -> None:
        """A plate under one full cycle has no band to measure."""
        with pytest.raises(ValueError, match="cycle"):
            ZonePlateGeometry.for_frame(width=4, height=4)


class TestRender:
    def test_is_an_hwc_float_image_in_the_unit_range(self) -> None:
        plate = zone_plate(_geometry())

        assert plate.shape == (HEIGHT, WIDTH, 3)
        assert plate.dtype == np.float32
        assert plate.min() >= 0.0
        assert plate.max() <= 1.0

    def test_is_achromatic(self) -> None:
        plate = zone_plate(_geometry())

        np.testing.assert_array_equal(plate[..., 0], plate[..., 1])
        np.testing.assert_array_equal(plate[..., 0], plate[..., 2])

    def test_is_mid_grey_outside_the_circle(self) -> None:
        """The corners and the margins beside the circle carry no
        pattern, so rings there can only come from downstream."""
        plate = zone_plate(_geometry())

        np.testing.assert_array_equal(plate[0, 0], [0.5, 0.5, 0.5])
        np.testing.assert_array_equal(plate[HEIGHT // 2, 0], [0.5, 0.5, 0.5])
        np.testing.assert_array_equal(plate[HEIGHT // 2, -1], [0.5, 0.5, 0.5])

    def test_centre_is_bright(self) -> None:
        """Zero frequency at the centre: the cosine starts at its peak."""
        plate = zone_plate(_geometry())

        assert plate[HEIGHT // 2, WIDTH // 2, 0] > 0.99

    def test_frequency_rises_to_nyquist_at_the_circle(self) -> None:
        """Along a radius the plate completes ``diameter / 8`` cycles, and
        its last cycles alternate pixel by pixel."""
        geom = _geometry()
        plate = zone_plate(geom)
        row = plate[HEIGHT // 2, WIDTH // 2 :, 0] - 0.5
        radius = row[: int(geom.nyquist_radius)]

        crossings = int(np.count_nonzero(np.diff(np.sign(radius)) != 0))
        assert abs(crossings - 2 * geom.diameter // 8) <= 2
        # Near Nyquist neighbouring pixels sit on opposite sides of grey.
        edge = radius[-8:]
        assert np.all(np.sign(edge[1:]) != np.sign(edge[:-1]))

    def test_still_ignores_the_frame_index(self) -> None:
        geom = _geometry()

        np.testing.assert_array_equal(
            zone_plate(geom, frame=0), zone_plate(geom, frame=99)
        )

    def test_renders_alike_under_torch(self) -> None:
        torch = torch_or_skip()
        geom = _geometry()

        actual = zone_plate(geom, xp=torch)
        np.testing.assert_allclose(
            to_host(actual), zone_plate(geom), rtol=0, atol=BACKEND_ATOL
        )

    def test_compiles_whole(self) -> None:
        """The render body is functional, so dynamo traces it as one
        graph with no break (§spec:backend-portability)."""
        torch = torch_or_skip()
        import torch._dynamo as dynamo

        dynamo.reset()
        compiled = torch.compile(zone_plate, backend="eager", fullgraph=True)
        np.testing.assert_allclose(
            to_host(compiled(_geometry(), xp=torch)),
            zone_plate(_geometry()),
            rtol=0,
            atol=BACKEND_ATOL,
        )


class TestScale:
    def test_an_untouched_chain_reads_whole(self) -> None:
        geom = _geometry()

        assert zone_plate_scale(zone_plate(geom), geom) == UNTOUCHED

    def test_gain_and_lift_read_whole(self) -> None:
        """A chain that changes contrast without resampling folds
        nothing."""
        geom = _geometry()
        captured = 0.8 * zone_plate(geom) + 0.05

        assert zone_plate_scale(captured, geom) == UNTOUCHED

    def test_quantization_reads_whole(self) -> None:
        """A 10-bit wire folds nothing."""
        geom = _geometry()
        captured = np.round(zone_plate(geom) * 1023) / 1023

        assert zone_plate_scale(captured, geom) == UNTOUCHED

    @pytest.mark.parametrize("scale", [0.25, 1 / 3, 0.5, 2 / 3, 0.75])
    @pytest.mark.parametrize(
        ("mode", "antialias"),
        [("bilinear", True), ("bilinear", False), ("bicubic", True)],
    )
    def test_a_round_trip_through_a_smaller_raster_reads_its_scale(
        self, scale: float, mode: str, antialias: bool
    ) -> None:
        """Scaled down and back, the chain reads the scale it went
        through, on both axes."""
        geom = _geometry()
        captured = _resample(zone_plate(geom), (scale, scale), mode, antialias)

        measured = zone_plate_scale(captured, geom)
        assert measured.horizontal == pytest.approx(scale, abs=SCALE_TOLERANCE)
        assert measured.vertical == pytest.approx(scale, abs=SCALE_TOLERANCE)

    @pytest.mark.parametrize("mode", ["nearest", "area"])
    def test_an_unfiltered_halving_reads_half(self, mode: str) -> None:
        """Dropping or averaging every other pixel, the commonest
        downstream monitor path, reads half."""
        geom = _geometry()
        captured = _resample(zone_plate(geom), (0.5, 0.5), mode, antialias=False)

        measured = zone_plate_scale(captured, geom)
        assert measured.horizontal == pytest.approx(0.5, abs=SCALE_TOLERANCE)
        assert measured.vertical == pytest.approx(0.5, abs=SCALE_TOLERANCE)

    def test_each_axis_reads_its_own_scale(self) -> None:
        """A scaler that halves only the width leaves the height whole."""
        geom = _geometry()
        captured = _resample(zone_plate(geom), (1.0, 0.5), "bilinear", True)

        measured = zone_plate_scale(captured, geom)
        assert measured.horizontal == pytest.approx(0.5, abs=SCALE_TOLERANCE)
        assert measured.vertical == 1.0

    def test_near_half_an_unfiltered_scale_reads_one_side_of_the_mirror(
        self,
    ) -> None:
        """Within 0.05 of half, a scaler with no antialiasing folds the
        plate onto itself, so the measurement finds the fold but may put
        it at ``1 - s``. Pinned so the documented limit stays true."""
        geom = _geometry()
        scale = 7 / 15
        captured = _resample(zone_plate(geom), (scale, scale), "bilinear", False)

        measured = zone_plate_scale(captured, geom)
        for axis in measured:
            assert min(axis, 1 - axis) == pytest.approx(scale, abs=SCALE_TOLERANCE)

    def test_reads_a_device_resident_capture(self) -> None:
        torch = torch_or_skip()
        geom = _geometry()

        assert zone_plate_scale(zone_plate(geom, xp=torch), geom) == UNTOUCHED

    def test_rejects_a_capture_of_another_size(self) -> None:
        geom = _geometry()
        with pytest.raises(ValueError, match="shape"):
            zone_plate_scale(np.zeros((HEIGHT, WIDTH // 2, 3)), geom)


@pytest.mark.parametrize(
    "device_name",
    [
        pytest.param("cuda", marks=pytest.mark.cuda),
        pytest.param("mps", marks=pytest.mark.mps),
    ],
)
def test_renders_on_a_device(device_name: str) -> None:
    torch = torch_or_skip()
    device = device_or_skip(device_name)
    geom = _geometry()

    plate = zone_plate(geom, xp=torch, device=device)
    assert plate.device.type == device_name
    np.testing.assert_allclose(
        to_host(plate), zone_plate(geom), rtol=0, atol=BACKEND_ATOL
    )
