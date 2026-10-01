"""Scale chart: a full frame for reading a chain's scaling by eye
(§spec:catalog).

Each test runs the chart through a simulated chain (torch's
``interpolate``, or a crop) and checks that the feature the chart offers
for that case reads the scale the chain applied:

- an upscale by 1 + 1/N bands the beat ruler's one-pixel strip every N
  source pixels, the spacing of the comb labelled 1 + 1/N;
- a centred zoom by z pushes the ladder rung labelled z to the frame's
  edge;
- a downscale by s turns the stripe ruler's sweep to mush past the tick
  labelled s;
- the raster label names the frame size the chart was rendered at.
"""

import numpy as np
import pytest

from display_patterns import ScaleChartGeometry, scale_chart
from tests.conftest import device_or_skip, to_host, torch_or_skip

WIDTH = 1920
HEIGHT = 1080


@pytest.fixture(scope="module")
def geom() -> ScaleChartGeometry:
    return ScaleChartGeometry.for_frame(width=WIDTH, height=HEIGHT)


@pytest.fixture(scope="module")
def chart(geom: ScaleChartGeometry) -> np.ndarray:
    return scale_chart(geom)


def _resize(frame: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    torch = torch_or_skip()
    import torch.nn.functional as functional

    nchw = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None]
    options = {} if mode == "nearest" else {"align_corners": False}
    antialias = mode in ("bilinear", "bicubic")
    out = functional.interpolate(
        nchw, size=size, mode=mode, antialias=antialias, **options
    )
    return out[0].permute(1, 2, 0).numpy()


def _contrast_period(samples: np.ndarray) -> int:
    """The period of the rise and fall of neighbour-to-neighbour
    contrast along ``samples``."""
    contrast = np.abs(np.diff(samples))
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


@pytest.mark.parametrize(
    ("width", "height"),
    [(1280, 720), (1920, 1080), (2048, 1080), (3840, 2160), (4096, 2160), (7680, 4320)],
)
def test_lays_out_at_every_broadcast_and_cinema_raster(width: int, height: int) -> None:
    """Geometry validates its own layout: nothing leaves the frame and
    nothing collides."""
    ScaleChartGeometry.for_frame(width=width, height=height)


def test_rejects_a_frame_too_small() -> None:
    with pytest.raises(ValueError, match="small"):
        ScaleChartGeometry.for_frame(width=640, height=360)


class TestRender:
    def test_is_an_achromatic_hwc_float_image(self, chart: np.ndarray) -> None:
        assert chart.shape == (HEIGHT, WIDTH, 3)
        assert chart.dtype == np.float32
        np.testing.assert_array_equal(chart[..., 0], chart[..., 1])
        assert chart.min() >= 0.0
        assert chart.max() <= 1.0

    def test_names_its_own_raster(self, geom: ScaleChartGeometry) -> None:
        assert "1920x1080" in {label.text for label in geom.labels}

    def test_labels_every_comb_rung_and_tick(self, geom: ScaleChartGeometry) -> None:
        texts = {label.text for label in geom.labels}

        assert {"1.02", "1.10", "1.25"} <= texts
        assert {"0.1", "0.5", "0.9"} <= texts
        assert {"960", "540"} <= texts

    def test_captions_every_element(self, geom: ScaleChartGeometry) -> None:
        texts = {label.text for label in geom.labels}

        assert {
            "ZOOM V",
            "ZOOM H",
            "UPSCALE H",
            "UPSCALE V",
            "DOWNSCALE H",
            "DOWNSCALE V",
            "PIXEL GRID",
        } <= texts

    def test_a_legend_says_how_to_read_each_element(
        self, geom: ScaleChartGeometry
    ) -> None:
        legend = [label.text for label in geom.labels if ":" in label.text]

        for element in ("ZOOM", "UPSCALE", "DOWNSCALE", "GRID"):
            assert any(line.startswith(f"{element}:") for line in legend)
        assert any("1:1" in line for line in legend)

    def test_ladder_rungs_hang_from_a_spine(
        self, chart: np.ndarray, geom: ScaleChartGeometry
    ) -> None:
        """Each ladder's rungs join one lit spine, so the ladder reads as
        one object, and its rungs are long enough to see."""
        top = geom.top_ladder
        assert (chart[: top.height, top.x, 0] == 1.0).all()
        left = geom.left_ladder
        assert (chart[left.y, : left.width, 0] == 1.0).all()
        assert top.width >= WIDTH // 16
        assert left.height >= HEIGHT // 16
        assert geom.rung_width >= 3

    def test_labels_are_inked(
        self, chart: np.ndarray, geom: ScaleChartGeometry
    ) -> None:
        for label in geom.labels:
            box = chart[
                label.y : label.y + label.height, label.x : label.x + label.width, 0
            ]
            assert (box == 1.0).any(), label

    def test_combs_tick_at_their_period(
        self, chart: np.ndarray, geom: ScaleChartGeometry
    ) -> None:
        for period, row in zip(geom.beat_periods, geom.comb_rows, strict=True):
            line = chart[row, geom.strip.x : geom.strip.x + geom.strip.width, 0]
            starts = np.flatnonzero(np.diff(line) > 0) + 1
            assert set(np.diff(starts)) == {period}

    def test_still_ignores_the_frame_index(self, geom: ScaleChartGeometry) -> None:
        np.testing.assert_array_equal(
            scale_chart(geom, frame=0), scale_chart(geom, frame=5)
        )


class TestReadByEye:
    @pytest.mark.parametrize("period", [10, 5, 4])
    def test_an_upscale_bands_the_strip_at_its_comb(
        self, chart: np.ndarray, geom: ScaleChartGeometry, period: int
    ) -> None:
        """Upscaled by 1 + 1/N, the one-pixel strip beats every N source
        pixels: one band per tick of the comb labelled 1 + 1/N."""
        scale = 1 + 1 / period
        out = _resize(chart, (round(HEIGHT * scale), round(WIDTH * scale)), "bilinear")
        strip = geom.strip
        row = out[round((strip.y + strip.height // 2) * scale), :, 0]
        samples = row[round(strip.x * scale) + 3 : round((strip.x + 600) * scale)]

        assert _contrast_period(samples) == pytest.approx(period * scale, abs=1)

    def test_the_vertical_strip_reads_the_vertical_scale(
        self, chart: np.ndarray, geom: ScaleChartGeometry
    ) -> None:
        """Stretched on rows alone, the vertical strip bands and the
        horizontal one does not."""
        scale = 1.1
        out = _resize(chart, (round(HEIGHT * scale), WIDTH), "bilinear")
        strip = geom.vertical_strip
        column = out[:, strip.x + strip.width // 2, 0]
        samples = column[round(strip.y * scale) + 3 : round((strip.y + 400) * scale)]

        assert _contrast_period(samples) == pytest.approx(10 * scale, abs=1)

    @pytest.mark.parametrize("zoom", [1.05, 1.1, 1.25])
    def test_a_centred_zoom_pushes_its_rung_to_the_edge(
        self, chart: np.ndarray, geom: ScaleChartGeometry, zoom: float
    ) -> None:
        """Cropped about the centre and enlarged by z, the rung labelled
        z is the outermost rung left in the frame, at its edge."""
        crop_h, crop_w = round(HEIGHT / zoom), round(WIDTH / zoom)
        top, left = (HEIGHT - crop_h) // 2, (WIDTH - crop_w) // 2
        cropped = chart[top : top + crop_h, left : left + crop_w]
        out = _resize(cropped, (HEIGHT, WIDTH), "nearest")

        # The rung labelled z sits where the crop begins, so it is the
        # first lit row (and column) of the ladder in the output.
        edge = geom.rung_width * zoom + 1
        assert abs(geom.zoom_rows[zoom] - top) * zoom <= edge
        assert abs(geom.zoom_cols[zoom] - left) * zoom <= edge

        ladder = geom.top_ladder
        column = out[:, round((ladder.x + ladder.width // 2 - left) * zoom), 0]
        assert np.flatnonzero(column == 1.0)[0] <= edge
        ladder = geom.left_ladder
        row = out[round((ladder.y + ladder.height // 2 - top) * zoom), :, 0]
        assert np.flatnonzero(row == 1.0)[0] <= edge

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("scale", [0.3, 0.5, 0.7])
    def test_a_downscale_fades_the_sweep_out_by_its_tick(
        self, chart: np.ndarray, geom: ScaleChartGeometry, scale: float, mode: str
    ) -> None:
        """Scaled down and back through a filtering scaler, the sweep
        fades over the stretch before the tick labelled s: clear at half
        the tick's distance, gone a little past it. The reading is where
        the stripes are gone."""
        small = _resize(chart, (round(HEIGHT * scale), round(WIDTH * scale)), mode)
        out = _resize(small, (HEIGHT, WIDTH), mode)
        ruler = geom.horizontal_ruler
        rows = slice(ruler.y, ruler.y + ruler.height)
        columns = slice(ruler.x, ruler.x + ruler.width)
        band, original = out[rows, columns, 0], chart[rows, columns, 0]
        tick = geom.ruler_position(scale, ruler.width)
        window = ruler.width // 40

        def kept(at: float) -> float:
            centre = round(tick * at)
            span = slice(centre - window, centre + window)
            return float(band[:, span].std() / original[:, span].std())

        assert kept(0.5) > 0.5
        assert kept(1.3) < 0.2


def test_renders_alike_under_torch(chart: np.ndarray, geom: ScaleChartGeometry) -> None:
    torch = torch_or_skip()
    np.testing.assert_allclose(
        to_host(scale_chart(geom, xp=torch)), chart, rtol=0, atol=1e-6
    )


def test_compiles_whole() -> None:
    torch = torch_or_skip()
    import torch._dynamo as dynamo

    dynamo.reset()
    geom = ScaleChartGeometry.for_frame(width=1280, height=720)
    compiled = torch.compile(scale_chart, backend="eager", fullgraph=True)
    np.testing.assert_allclose(
        to_host(compiled(geom, xp=torch)), scale_chart(geom), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize(
    "device_name",
    [
        pytest.param("cuda", marks=pytest.mark.cuda),
        pytest.param("mps", marks=pytest.mark.mps),
    ],
)
def test_renders_on_a_device(device_name: str, geom: ScaleChartGeometry) -> None:
    torch = torch_or_skip()
    device = device_or_skip(device_name)

    rendered = scale_chart(geom, xp=torch, device=device)
    assert rendered.device.type == device_name
    np.testing.assert_allclose(to_host(rendered), scale_chart(geom), rtol=0, atol=1e-6)
