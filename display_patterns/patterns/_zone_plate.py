"""Circular zone plate: geometry, render, and scale measurement.

The plate is a radial chirp, ``cos`` of a phase that grows with the
square of the radius, so its local spatial frequency rises linearly from
zero at the centre to Nyquist (half a cycle per pixel) at the edge of the
largest circle the frame holds (§spec:catalog). Outside that circle the
frame is mid-grey. A chain that resamples the picture can carry only the
frequencies its smallest raster holds, so past that radius the rings
blur out or alias into moiré; on the grey margin, any ring at all is the
chain's.

The measurement reads the scale back. Multiplying a capture by the
plate's own phasor collapses the plate to a constant and turns every
alias and image a scaler left into a pure tone at the scaler's sampling
rate, so one spectral peak per axis names the scale — up to the mirror
``s`` against ``1 - s`` that any once-per-pixel capture carries, which
whether the plate survived between the two decides.

Value and layout conventions (§spec:render-model): this is a float
pattern; the plate is ``(height, width, 3)`` float32 in [0, 1],
achromatic. A consumer driving integer code values scales [0, 1] into
its own value space and normalizes a capture back before measuring it.
"""

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from display_patterns.patterns import _backend

# Upper bound on either frame dimension, shared with the counter panel's
# reasoning: 16384 clears 8K with headroom, and nothing past it is a
# real video frame.
_MAX_DIMENSION = 16384

# The plate's phase advances one cycle each time the squared doubled
# radius (see ``ZonePlateGeometry``) advances by this many diameters.
# Eight puts Nyquist exactly on the inscribed circle: phase in cycles is
# r**2 / (2 * diameter), whose radial derivative r / diameter reaches
# one half at r = diameter / 2.
_DOUBLED_RADIUS_SQ_PER_CYCLE_PER_DIAMETER = 8

# Pattern values: mid-grey ground, full-range swing about it.
_GREY = 0.5
_AMPLITUDE = 0.5

# The smallest scale the measurement reports. Its spectral search skips
# frequencies this near zero, where the plate itself collapses to under
# demodulation; a scale this near one folds too little of the band to
# find, and reads as 1.0.
_MIN_SCALE = 0.05

# A fold counts when its tone carries this fraction of the plate's own
# demodulated energy. An untouched 10-bit capture sits near 0.01; every
# filtering or sampling scaler measured sits above 0.06.
_FOLD_PROMINENCE = 0.03

# Half-width of the sector about each axis that measures the plate's
# survival along it.
_SECTOR_HALF_ANGLE_DEGREES = 15

# The plate survived between the two candidate scales when its gain
# there is at least this fraction of its low-frequency gain.
_SURVIVAL = 0.2

# The low-frequency gain is the median over annuli up to this fraction
# of Nyquist, inside the passband of any scale worth measuring.
_REFERENCE_BAND = 0.2


@dataclass(frozen=True)
class ZonePlateGeometry:
    """Where the plate lands in the frame: centred, reaching Nyquist on
    the largest inscribed circle. Deterministic, so :func:`zone_plate` and
    :func:`zone_plate_passband` agree on every pixel's phase."""

    width: int
    height: int
    diameter: int

    @classmethod
    def for_frame(cls, width: int, height: int) -> "ZonePlateGeometry":
        """Inscribe the plate in a ``width`` x ``height`` frame.

        Raises
        ------
        ValueError
            If a dimension falls outside [1, 16384], or the inscribed
            circle is too small to hold one full cycle of the plate.
        """
        for label, value in (("width", width), ("height", height)):
            if not 1 <= value <= _MAX_DIMENSION:
                raise ValueError(
                    f"{label} {value} is out of range: a frame dimension must be "
                    f"between 1 and {_MAX_DIMENSION} pixels."
                )
        diameter = min(width, height)
        if diameter < _DOUBLED_RADIUS_SQ_PER_CYCLE_PER_DIAMETER:
            raise ValueError(
                f"a {width}x{height} frame inscribes a {diameter}px circle, under "
                f"one full cycle of the plate: the diameter must be at least "
                f"{_DOUBLED_RADIUS_SQ_PER_CYCLE_PER_DIAMETER}px."
            )
        return cls(width, height, diameter)

    @property
    def nyquist_radius(self) -> float:
        """The radius in pixels at which the plate reaches Nyquist."""
        return self.diameter / 2

    @property
    def shape(self) -> tuple[int, int, int]:
        """The ``(height, width, 3)`` shape of a rendered plate."""
        return (self.height, self.width, 3)


def _doubled_offsets(xp: Any, size: int, limit: int, device: Any) -> Any:
    """Each pixel centre's offset from the frame centre along one axis,
    doubled so it is an integer whatever the parity of ``size``, and held
    to ``limit`` so its square cannot overflow int32. Clamping changes
    no pixel inside the circle, whose doubled offsets never exceed the
    diameter."""
    doubled = 2 * _backend.arange(xp, size, xp.int32, device) + (1 - size)
    return xp.where(doubled > limit, limit, xp.where(doubled < -limit, -limit, doubled))


def _phase_steps(geometry: ZonePlateGeometry, xp: Any, device: Any) -> tuple[Any, Any]:
    """Every pixel's squared doubled radius ``n``, and whether it falls
    inside the circle.

    The plate's phase in cycles is ``n / (8 * diameter)``. Keeping ``n``
    an exact integer lets the render reduce the phase modulo one cycle
    before any float arithmetic, so the cosine sees an argument under
    2*pi on every backend rather than thousands of radians.
    """
    d = geometry.diameter
    u = _doubled_offsets(xp, geometry.width, d, device)
    v = _doubled_offsets(xp, geometry.height, d, device)
    n = v[:, None] * v[:, None] + u[None, :] * u[None, :]
    return n, n <= d * d


def zone_plate(
    geometry: ZonePlateGeometry,
    *,
    frame: int | Any = 0,  # noqa: ARG001 — a still ignores its frame index
    xp: Any = np,
    device: Any = None,
) -> Any:
    """Render a circular zone plate.

    A still (§spec:render-model): the frame index is accepted for the
    shared calling convention and ignored.

    Parameters
    ----------
    geometry : ZonePlateGeometry
        Frame size and circle; build with :meth:`ZonePlateGeometry.for_frame`.
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
        ``(height, width, 3)`` float32 in [0, 1]: ``0.5 + 0.5 * cos`` of
        the plate's phase inside the inscribed circle, bright at the
        centre and alternating pixel by pixel at its edge, and 0.5
        outside it.
    """
    steps_per_cycle = _DOUBLED_RADIUS_SQ_PER_CYCLE_PER_DIAMETER * geometry.diameter
    n, inside = _phase_steps(geometry, xp, device)
    within_cycle = _backend.astype(n % steps_per_cycle, xp.float32)
    swing = xp.cos(within_cycle * (2 * math.pi / steps_per_cycle))
    plate = xp.where(inside, _GREY + _AMPLITUDE * swing, _GREY)
    plate = _backend.astype(plate, xp.float32)
    return xp.stack([plate, plate, plate], axis=-1)


class ZonePlateScale(NamedTuple):
    """The raster scale a chain carried the plate through, per axis.

    1.0 on an axis where nothing folded the plate's band, either because
    the chain kept every pixel or because it only enlarged the frame.
    """

    horizontal: float
    vertical: float


def _annulus_gains(
    luma: np.ndarray, n: np.ndarray, steps_per_cycle: int, diameter: int
) -> tuple[np.ndarray, np.ndarray]:
    """The chain's gain at each one-cycle annulus of the plate, and that
    annulus's frequency as a fraction of Nyquist.

    Each annulus holds one whole period, so the plate is zero-mean within
    it and projecting the capture onto it is well conditioned at every
    radius. Energy the chain moved to another frequency projects to
    nothing.
    """
    annulus = n // steps_per_cycle
    reference = np.cos(2 * np.pi * (n % steps_per_cycle) / steps_per_cycle)
    count = np.bincount(annulus)
    divisor = np.maximum(count, 1)

    def per_annulus(weights: np.ndarray) -> np.ndarray:
        return np.bincount(annulus, weights=weights, minlength=count.size)

    luma = luma - (per_annulus(luma) / divisor)[annulus]
    reference = reference - (per_annulus(reference) / divisor)[annulus]
    energy = per_annulus(reference * reference)
    gain = np.divide(
        per_annulus(luma * reference),
        energy,
        out=np.zeros_like(energy),
        where=energy > 0,
    )
    frequency = per_annulus(np.sqrt(n) / diameter) / divisor
    present = count > 0
    return frequency[present], gain[present]


def _axis_scale(
    demodulated: np.ndarray,
    summed_axis: int,
    survival: tuple[np.ndarray, np.ndarray],
) -> float:
    """One axis's raster scale: the fold's tone, and which side of half
    scale the plate's survival puts it.

    ``demodulated`` is the capture times the plate's own phasor. The
    plate itself collapses to a constant under it, while a scaler that
    sampled the frame at ``s`` samples per pixel along this axis leaves
    a pure tone at ``s`` cycles per pixel — its aliases and its images
    alike. Summing across the other axis keeps that tone and averages
    the rest away, so the profile's spectral peak is the scale.

    A capture sampled once per pixel cannot tell a tone at ``s`` from
    one at ``1 - s``, so the peak names two candidates either side of
    half scale. The scaler at the smaller one has nothing left of the
    plate between the two radii; the one at the larger still carries
    it. ``survival`` is this axis's sector of annulus gains, which
    decides between them.
    """
    profile = demodulated.sum(axis=summed_axis)
    spectrum = np.abs(np.fft.fft(profile))
    frequency = np.fft.fftfreq(profile.size)
    searched = np.where(np.abs(frequency) >= _MIN_SCALE, spectrum, 0.0)
    peak = int(np.argmax(searched))
    if searched[peak] < _FOLD_PROMINENCE * abs(profile.sum()):
        return 1.0

    tone = abs(float(frequency[peak]))
    low, high = min(tone, 1.0 - tone), max(tone, 1.0 - tone)
    annulus_frequency, gain = survival
    between = (annulus_frequency >= low) & (annulus_frequency < high)
    if not between.any():
        return low
    passband = annulus_frequency <= min(low, _REFERENCE_BAND)
    if not passband.any():
        passband = annulus_frequency == annulus_frequency.min()
    reference_gain = float(np.median(gain[passband]))
    survived = float(np.mean(gain[between])) >= _SURVIVAL * reference_gain
    return high if survived else low


def zone_plate_scale(captured: Any, geometry: ZonePlateGeometry) -> ZonePlateScale:
    """Measure the raster scale a chain carried a rendered plate through.

    Reads a capture of :func:`zone_plate` back into a number, as
    :func:`decode_counter` reads a counter panel. A scaler that took the
    frame through a raster ``s`` times its size, down and back, folds
    the plate's band at ``s`` times Nyquist; this finds that fold on each
    axis.

    ``captured`` is any ``(height, width, channels)`` array in the
    [0, 1] range convention, on any backend; its channels are averaged.
    It is read to the host in one transfer and measured there.

    The fold is found reliably, but which side of half scale it lies on
    is decided by whether the plate survived between ``s`` and
    ``1 - s``, so it is least certain near half scale. A scaler that
    samples without filtering (nearest neighbour) above half scale, or
    one with no antialiasing within 0.05 of half, can read as ``1 - s``.
    Scales within :data:`_MIN_SCALE` of zero or one read as 1.0.

    Returns
    -------
    ZonePlateScale
        The scale along each axis, 1.0 where the band was not folded.

    Raises
    ------
    ValueError
        If ``captured`` is not the geometry's frame size.
    """
    image = _backend.to_host(captured)
    if image.ndim != 3 or image.shape[:2] != (geometry.height, geometry.width):
        raise ValueError(
            f"captured shape {image.shape} does not match the plate's "
            f"{geometry.shape}: measure a capture at the rendered frame size."
        )
    luma = image.astype(np.float64).mean(axis=-1)

    d = geometry.diameter
    steps_per_cycle = _DOUBLED_RADIUS_SQ_PER_CYCLE_PER_DIAMETER * d
    u = np.abs(_doubled_offsets(np, geometry.width, d, None)).astype(np.int64)
    v = np.abs(_doubled_offsets(np, geometry.height, d, None)).astype(np.int64)
    n = v[:, None] * v[:, None] + u[None, :] * u[None, :]
    inside = n <= d * d
    phase = (2 * np.pi / steps_per_cycle) * (n % steps_per_cycle)
    demodulated = np.where(inside, luma * np.exp(1j * phase), 0)

    # A sector about each axis: a separable scaler folds each axis at
    # its own scale, and a whole ring would mix in the diagonal, which
    # folds at a scale root-two higher.
    spread = math.tan(math.radians(_SECTOR_HALF_ANGLE_DEGREES))
    horizontal = inside & (v[:, None] <= spread * u[None, :])
    vertical = inside & (u[None, :] <= spread * v[:, None])

    def survival(sector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _annulus_gains(luma[sector], n[sector], steps_per_cycle, d)

    return ZonePlateScale(
        horizontal=_axis_scale(demodulated, 0, survival(horizontal)),
        vertical=_axis_scale(demodulated, 1, survival(vertical)),
    )
