"""Pixel-level reference grid (§spec:catalog).

A matrix of high-frequency patches (checkers, stripes on each axis,
both diagonals, and isolated lines) at cell sizes from 1 to 8 pixels,
each split into four quadrants whose phases differ by one pixel, on a
gutter at the patches' mean grey. Read by eye on a chain's output: a
scaler that resamples the frame bands, softens, decimates, or
phase-shifts the patches, and the gutter shows where a patch has gone
to its mean.
"""

import numpy as np
import pytest

from display_patterns import PixelGridGeometry, pixel_grid
from tests.conftest import device_or_skip, to_host, torch_or_skip

WIDTH = 1920
HEIGHT = 1080


def _geometry() -> PixelGridGeometry:
    return PixelGridGeometry.for_frame(width=WIDTH, height=HEIGHT)


def _patch(frame: np.ndarray, geom: PixelGridGeometry, kind: str, size: int):
    x, y = geom.patch_origin(kind, size)
    return frame[y : y + geom.patch, x : x + geom.patch, 0]


def _contrast_period(row: np.ndarray) -> int:
    """The period, in samples, of the rise and fall of neighbour-to-
    neighbour contrast along ``row``: the beat a resampler leaves."""
    contrast = np.abs(np.diff(row))
    contrast = (contrast[:-1] + contrast[1:]) / 2
    centred = contrast - contrast.mean()
    correlation = np.correlate(centred, centred, "full")[centred.size - 1 :]
    correlation = correlation / correlation[0]
    for lag in range(2, correlation.size - 1):
        if (
            correlation[lag] > correlation[lag - 1]
            and correlation[lag] >= correlation[lag + 1]
            and correlation[lag] > 0.3
        ):
            return lag
    raise AssertionError("no beat in the contrast")


@pytest.fixture(scope="module")
def grid() -> np.ndarray:
    return pixel_grid(_geometry())


class TestGeometry:
    def test_lays_every_kind_at_every_size(self) -> None:
        geom = _geometry()

        assert geom.sizes == (1, 2, 3, 4, 6, 8)
        assert set(geom.kinds) == {
            "checker",
            "vertical",
            "horizontal",
            "diagonal_down",
            "diagonal_up",
            "lines",
        }

    def test_patches_are_large_enough_to_show_a_slow_beat(self) -> None:
        """A 1.02x upscale beats every 50 source pixels; a patch shows
        at least two of them."""
        assert _geometry().patch >= 100

    def test_rejects_a_frame_too_small(self) -> None:
        with pytest.raises(ValueError, match="small"):
            PixelGridGeometry.for_frame(width=320, height=180)


class TestRender:
    def test_is_an_achromatic_hwc_float_image(self, grid: np.ndarray) -> None:
        assert grid.shape == (HEIGHT, WIDTH, 3)
        assert grid.dtype == np.float32
        np.testing.assert_array_equal(grid[..., 0], grid[..., 2])
        assert set(np.unique(grid)) <= {0.0, 0.5, 1.0}

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 6, 8])
    def test_checker_cells_are_the_stated_size(
        self, grid: np.ndarray, size: int
    ) -> None:
        patch = _patch(grid, _geometry(), "checker", size)
        row = patch[0, : 4 * size]

        expected = np.repeat([0.0, 1.0, 0.0, 1.0], size)
        np.testing.assert_array_equal(row, expected)

    @pytest.mark.parametrize(
        ("kind", "along_rows"), [("vertical", False), ("horizontal", True)]
    )
    def test_stripes_run_along_their_axis(
        self, grid: np.ndarray, kind: str, along_rows: bool
    ) -> None:
        patch = _patch(grid, _geometry(), kind, 2)
        quadrant = patch[: patch.shape[0] // 2, : patch.shape[1] // 2]
        if along_rows:
            quadrant = quadrant.T
        # Constant down each column, alternating across columns.
        assert np.all(quadrant == quadrant[:1])
        np.testing.assert_array_equal(quadrant[0, :4], [0.0, 0.0, 1.0, 1.0])

    def test_diagonals_run_both_ways(self, grid: np.ndarray) -> None:
        geom = _geometry()
        down = _patch(grid, geom, "diagonal_down", 1)
        up = _patch(grid, geom, "diagonal_up", 2)

        assert down[0, 1] == down[1, 0]
        assert up[0, 0] == up[1, 1] == up[2, 2]

    @pytest.mark.parametrize("kind", ["checker", "vertical", "diagonal_down"])
    def test_quadrants_differ_by_a_one_pixel_phase(
        self, grid: np.ndarray, kind: str
    ) -> None:
        """At one-pixel cells the right half samples the pattern one
        pixel along, which inverts it."""
        patch = _patch(grid, _geometry(), kind, 1)
        half = patch.shape[1] // 2

        np.testing.assert_array_equal(patch[:4, half : half + 4], 1 - patch[:4, :4])

    def test_lines_are_single_lines_on_black(self, grid: np.ndarray) -> None:
        patch = _patch(grid, _geometry(), "lines", 1)
        quadrant = patch[: patch.shape[0] // 2, : patch.shape[1] // 2]

        # One lit column and one lit row: a cross, one pixel wide.
        assert np.count_nonzero(quadrant.all(axis=0)) == 1
        assert np.count_nonzero(quadrant.all(axis=1)) == 1
        assert quadrant.mean() < 0.05

    @pytest.mark.parametrize("kind", ["checker", "vertical", "diagonal_up"])
    def test_patches_average_to_the_gutter_grey(
        self, grid: np.ndarray, kind: str
    ) -> None:
        """A patch a scaler decimates to its mean matches the gutter, at
        every size whose period divides a quadrant (all but eight)."""
        geom = _geometry()
        for size in (1, 2, 3, 4, 6):
            assert _patch(grid, geom, kind, size).mean() == pytest.approx(0.5, abs=0.02)
        x, y = geom.patch_origin(kind, 1)
        assert grid[y - 1, x, 0] == 0.5

    def test_still_ignores_the_frame_index(self) -> None:
        geom = _geometry()
        np.testing.assert_array_equal(
            pixel_grid(geom, frame=0), pixel_grid(geom, frame=7)
        )


class TestReadByEye:
    """What a downstream scaler does to the grid, simulated with torch's
    ``interpolate``."""

    def _scaled(self, grid: np.ndarray, size: tuple[int, int], mode: str):
        torch = torch_or_skip()
        import torch.nn.functional as functional

        nchw = torch.from_numpy(np.ascontiguousarray(grid)).permute(2, 0, 1)[None]
        options = {} if mode == "nearest" else {"align_corners": False}
        out = functional.interpolate(nchw, size=size, mode=mode, **options)
        return out[0].permute(1, 2, 0).numpy()

    def test_a_two_to_one_drop_exposes_its_phase(self, grid: np.ndarray) -> None:
        """Dropping every other pixel turns a one-pixel checker solid, and
        which solid depends on phase: neighbouring quadrants disagree."""
        geom = _geometry()
        decimated = grid[::2, ::2]
        x, y = geom.patch_origin("checker", 1)
        half = geom.patch // 2
        left = decimated[y // 2 + 2 : y // 2 + 6, x // 2 + 2 : x // 2 + 6, 0]
        right = decimated[
            y // 2 + 2 : y // 2 + 6, (x + half) // 2 + 2 : (x + half) // 2 + 6, 0
        ]

        assert np.unique(left).size == 1
        assert np.unique(right).size == 1
        assert left[0, 0] != right[0, 0]

    @pytest.mark.parametrize(("scale", "period"), [(1.1, 10), (1.25, 4)])
    def test_an_upscale_bands_the_finest_stripes(
        self, grid: np.ndarray, scale: float, period: int
    ) -> None:
        """Upscaled by 1 + 1/N, one-pixel stripes beat every N source
        pixels, while two-pixel and coarser stripes keep their contrast."""
        geom = _geometry()
        out = self._scaled(
            grid, (round(HEIGHT * scale), round(WIDTH * scale)), "bilinear"
        )
        x, y = geom.patch_origin("vertical", 1)
        row = out[round((y + 4) * scale), :, 0]
        start = round(x * scale)
        # Trim the edges against the gutter, whose step is not a beat.
        quadrant = row[start + 3 : start + round(geom.patch // 2 * scale) - 3]

        assert _contrast_period(quadrant) == pytest.approx(period * scale, abs=1)


def test_renders_alike_under_torch(grid: np.ndarray) -> None:
    torch = torch_or_skip()
    np.testing.assert_array_equal(to_host(pixel_grid(_geometry(), xp=torch)), grid)


def test_compiles_whole() -> None:
    torch = torch_or_skip()
    import torch._dynamo as dynamo

    dynamo.reset()
    geom = PixelGridGeometry.for_frame(width=640, height=360)
    compiled = torch.compile(pixel_grid, backend="eager", fullgraph=True)
    np.testing.assert_array_equal(to_host(compiled(geom, xp=torch)), pixel_grid(geom))


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

    rendered = pixel_grid(geom, xp=torch, device=device)
    assert rendered.device.type == device_name
    np.testing.assert_array_equal(to_host(rendered), pixel_grid(geom))
