"""Internal helpers for projection-matrix construction.

The public projection API still lives on :class:`multi_pinhole.world.World`.
This module contains geometry-independent bookkeeping that is shared by the
ordinary sparse and future factorized projection builders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pair(value, name: str) -> np.ndarray:
    pair = np.asarray(value, dtype=float)
    if pair.ndim == 0:
        pair = np.repeat(pair, 2)
    if pair.shape != (2,) or not np.all(np.isfinite(pair)) or np.any(pair <= 0.0):
        raise ValueError(f"{name} must be a positive scalar or length-2 sequence")
    return pair


def projected_axis_spans(camera, eye_index: int, centers: np.ndarray,
                         edge_lengths: np.ndarray) -> np.ndarray:
    """Project each source-cell axis chord onto the detector plane.

    For every cell center and each world axis, this function projects the two
    endpoints ``center +/- edge_length[axis] / 2`` through one Eye.  The
    absolute endpoint displacement is returned in physical screen ``(u, v)``
    coordinates.  Evaluating the actual endpoints, instead of using a
    far-field scalar approximation, includes camera rotation, off-axis
    perspective, nonuniform cell sizes, and depth-axis perspective exactly.

    Parameters
    ----------
    camera : Camera
        Camera defining the world-to-camera transform and detector plane.
    eye_index : int
        Eye used for the point projection.
    centers : ndarray, shape (n, 3)
        Cell centers in world coordinates.
    edge_lengths : ndarray, shape (n, 3)
        Positive cell lengths along the world x, y, and z axes.

    Returns
    -------
    ndarray, shape (n, 3, 2)
        Absolute projected chord lengths. Axis 1 selects the source world axis
        and axis 2 selects detector ``(u, v)``. Cells with an endpoint behind
        the Eye receive ``nan`` for the affected source axis.
    """
    centers = np.asarray(centers, dtype=float)
    edge_lengths = np.asarray(edge_lengths, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (n, 3)")
    if edge_lengths.shape != centers.shape:
        raise ValueError("edge_lengths must have the same shape as centers")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite values")
    if not np.all(np.isfinite(edge_lengths)) or np.any(edge_lengths <= 0.0):
        raise ValueError("edge_lengths must contain only positive finite values")
    if not 0 <= eye_index < len(camera.eyes):
        raise IndexError("eye_index is out of range")

    n_cells = centers.shape[0]
    offsets = np.zeros((n_cells, 3, 3), dtype=float)
    diagonal = np.arange(3)
    offsets[:, diagonal, diagonal] = 0.5 * edge_lengths
    endpoints = np.stack((centers[:, None, :] - offsets,
                          centers[:, None, :] + offsets), axis=2)

    points_camera = camera.world2camera(endpoints.reshape(-1, 3))
    rays = camera.eyes[eye_index].calc_rays(points_camera)
    projected_uv = camera.screen.xy2uv(rays.XY).reshape(n_cells, 3, 2, 2)
    return np.abs(projected_uv[:, :, 1] - projected_uv[:, :, 0])


@dataclass(frozen=True)
class SourceResolutionEstimate:
    """Axis-wise source quadrature recommendation and diagnostics.

    Attributes
    ----------
    resolution : ndarray
        Selected integer axis resolutions, shape ``(n, 3)`` in ``x, y, z`` order.
    projected_span_cells : ndarray
        Estimated axis spans in detector-cell units, shape ``(n, 3)``.
    uncapped_resolution : ndarray
        Heuristic recommendation before the configured ceiling, shape ``(n, 3)``.
    capped : ndarray
        Boolean mask, shape ``(n, 3)``; true axes did not reach that recommendation.
    """

    resolution: np.ndarray
    projected_span_cells: np.ndarray
    uncapped_resolution: np.ndarray
    capped: np.ndarray


@dataclass(frozen=True)
class PointSourceResolutionEstimate:
    """Local-perspective source quadrature recommendation.

    Attributes
    ----------
    resolution : ndarray
        Selected integer resolution, shape ``(n, 3)`` in ``x, y, z`` order.
    ratio : ndarray
        Projected-diameter/reference-scale ratio, shape ``(n,)``, dimensionless.
    projected_diameter, reference_size : ndarray
        Local projected size and comparison scale, shape ``(n,)``, in detector
        length units.
    point_source : ndarray
        Boolean sampling-policy decision, shape ``(n,)``; not an error test.
    ideal_resolution : ndarray
        Uncapped heuristic resolution, shape ``(n, 3)``; ``ideal`` does not
        imply a numerical error guarantee.
    capped : ndarray
        Boolean axis mask, shape ``(n, 3)``.
    valid : ndarray
        Boolean geometry-valid mask, shape ``(n,)``; invalid rows use fallback.
    """

    resolution: np.ndarray
    ratio: np.ndarray
    projected_diameter: np.ndarray
    reference_size: np.ndarray
    point_source: np.ndarray
    ideal_resolution: np.ndarray
    capped: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class EyeProjectionWorkEstimate:
    """Preflight source-sample counts for one camera Eye.

    Attributes
    ----------
    camera_key : object
        Camera mapping key.
    eye_index : int
        Zero-based Eye index.
    full_voxels, partial_voxels : int
        Exact counts from the cached corner-visibility classification.
    full_samples : int
        Exact scheduled sample count for fully visible voxels.
    partial_samples_upper_bound : int
        Count before sample-center inside/visibility masking.
    full_resolution_buckets : tuple
        ``((rx, ry, rz), count)`` groups in ``x, y, z`` order.
    partial_resolution : tuple of int
        Fixed partial-voxel resolution in ``x, y, z`` order.
    ideal_p50, ideal_p95, ideal_max : tuple or None
        Percentiles of uncapped heuristic axis resolutions.
    point_source_voxels, capped_axes, invalid_voxels : int
        Diagnostic counts; capped or invalid entries need not meet the
        heuristic recommendation.
    """

    camera_key: object
    eye_index: int
    full_voxels: int
    partial_voxels: int
    full_samples: int
    partial_samples_upper_bound: int
    full_resolution_buckets: tuple[tuple[tuple[int, int, int], int], ...]
    partial_resolution: tuple[int, int, int]
    ideal_p50: tuple[float, float, float] | None
    ideal_p95: tuple[float, float, float] | None
    ideal_max: tuple[float, float, float] | None
    point_source_voxels: int
    capped_axes: int
    invalid_voxels: int

    @property
    def visible_voxels(self) -> int:
        return self.full_voxels + self.partial_voxels

    @property
    def total_samples_upper_bound(self) -> int:
        return self.full_samples + self.partial_samples_upper_bound


@dataclass(frozen=True)
class ProjectionWorkEstimate:
    """Projection preflight report across all cameras and Eyes.

    Attributes
    ----------
    eyes : tuple of EyeProjectionWorkEstimate
        Per-Eye reports in camera iteration order.
    """

    eyes: tuple[EyeProjectionWorkEstimate, ...]

    @property
    def total_visible_voxels(self) -> int:
        return sum(eye.visible_voxels for eye in self.eyes)

    @property
    def total_samples_upper_bound(self) -> int:
        return sum(eye.total_samples_upper_bound for eye in self.eyes)

    def summary(self, max_buckets: int = 8) -> str:
        """Return a compact human-readable work estimate."""
        lines = [
            "Projection preflight: "
            f"{self.total_samples_upper_bound:,} source samples (upper bound), "
            f"{self.total_visible_voxels:,} visible camera-voxels",
        ]
        for eye in self.eyes:
            lines.append(
                f"  camera={eye.camera_key!r} eye={eye.eye_index}: "
                f"full={eye.full_voxels:,}, partial={eye.partial_voxels:,}, "
                f"samples<={eye.total_samples_upper_bound:,}",
            )
            shown = eye.full_resolution_buckets[:max_buckets]
            bucket_text = ", ".join(
                f"{resolution} x {count:,}"
                for resolution, count in shown
            ) or "none"
            if len(eye.full_resolution_buckets) > max_buckets:
                bucket_text += f", ... (+{len(eye.full_resolution_buckets) - max_buckets})"
            lines.append(f"    full buckets: {bucket_text}")
            if eye.ideal_p50 is not None:
                lines.append(
                    f"    ideal p50={eye.ideal_p50}, p95={eye.ideal_p95}, "
                    f"max={eye.ideal_max}; point-source={eye.point_source_voxels:,}, "
                    f"capped axes={eye.capped_axes:,}, invalid={eye.invalid_voxels:,}",
                )
            if eye.partial_voxels:
                lines.append(
                    f"    partial res={eye.partial_resolution}, "
                    f"samples<={eye.partial_samples_upper_bound:,} before masking",
                )
        return "\n".join(lines)


def select_circumsphere_resolution(
        points_in_eye: np.ndarray, edge_lengths: np.ndarray,
        focal_length: float, reference_size,
        fallback_resolution=4,
        point_source_threshold: float = 1.0 / 8.0,
        ) -> PointSourceResolutionEstimate:
    """Recommend axis-wise source resolution from a local circumsphere scale.

    Parameters
    ----------
    points_in_eye : ndarray
        Voxel centers in Eye coordinates, shape ``(n, 3)``, in a consistent
        length unit (mm in project examples).
    edge_lengths : ndarray
        Positive voxel edge lengths, shape ``(n, 3)`` in ``x, y, z`` order,
        in the same length unit.
    focal_length : float
        Nonzero Eye focal length in the same length unit.
    reference_size : float or ndarray
        Positive detector/PSF comparison scale, scalar or shape ``(n,)``, in
        detector length units.
    fallback_resolution : int or tuple of int or None, default=4
        Axis-wise ``x, y, z`` ceiling and invalid-geometry fallback. ``None``
        requests the uncapped heuristic and rejects invalid geometry.
    point_source_threshold : float, default=1/8
        Positive dimensionless sampling-policy threshold. It is not an image
        error tolerance.

    Returns
    -------
    PointSourceResolutionEstimate
        Per-voxel resolutions and diagnostics in input order.

    Notes
    -----
    The local perspective Jacobian has magnification
    ``abs(f) / (Z*cos(theta))``. Multiplying it by the voxel circumsphere
    diameter gives a rotation-independent, worst-direction projected size.
    The same scale determines a near-cubic ideal axis-wise resolution. The
    caller's ``fallback_resolution`` is used as an axis-wise ceiling. This is
    a voxel-center heuristic, not a rigorous projected-size upper bound over a
    finite voxel; it can underestimate a large voxel close to the Eye. Voxels
    behind the Eye or whose circumsphere reaches the Eye plane are invalid and
    use the fallback. A capped axis may not reach the heuristic recommendation,
    and the field named ``ideal_resolution`` is uncapped—not an error guarantee.
    """
    points_in_eye = np.asarray(points_in_eye, dtype=float)
    edge_lengths = np.asarray(edge_lengths, dtype=float)
    if points_in_eye.ndim != 2 or points_in_eye.shape[1] != 3:
        raise ValueError("points_in_eye must have shape (n, 3)")
    if edge_lengths.shape != points_in_eye.shape:
        raise ValueError("edge_lengths must have the same shape as points_in_eye")
    if not np.all(np.isfinite(edge_lengths)) or np.any(edge_lengths <= 0.0):
        raise ValueError("edge_lengths must contain only positive finite values")
    if not np.isfinite(focal_length) or focal_length == 0.0:
        raise ValueError("focal_length must be finite and nonzero")
    if (not np.isfinite(point_source_threshold) or
            point_source_threshold <= 0.0):
        raise ValueError("point_source_threshold must be positive and finite")

    fallback = None
    if fallback_resolution is not None:
        try:
            fallback = np.asarray(
                np.broadcast_to(fallback_resolution, 3), dtype=float,
            )
        except ValueError as exc:
            raise ValueError(
                "fallback_resolution must be None, an integer, or a length-3 sequence",
            ) from exc
        if (not np.all(np.isfinite(fallback)) or np.any(fallback < 1) or
                np.any(fallback != np.floor(fallback))):
            raise ValueError("fallback_resolution must contain positive integers")
        fallback = fallback.astype(np.int64)

    reference_size = np.asarray(reference_size, dtype=float)
    if reference_size.ndim == 0:
        reference_size = np.full(points_in_eye.shape[0], reference_size)
    elif reference_size.shape != (points_in_eye.shape[0],):
        raise ValueError("reference_size must be scalar or shape (n,)")
    if not np.all(np.isfinite(reference_size)) or np.any(reference_size <= 0.0):
        raise ValueError("reference_size must contain only positive finite values")

    diameter = np.linalg.norm(edge_lengths, axis=1)
    radius = 0.5 * diameter
    distance = np.linalg.norm(points_in_eye, axis=1)
    axial_distance = points_in_eye[:, 2]
    valid = (
        np.all(np.isfinite(points_in_eye), axis=1) &
        (distance > 0.0) &
        (axial_distance > radius)
    )

    projected_diameter = np.full(points_in_eye.shape[0], np.inf)
    # Z*cos(theta) = Z**2 / distance. This form avoids a separate angle and
    # makes the off-axis 1/cos(theta) safety factor explicit algebraically.
    projected_diameter[valid] = (
        abs(float(focal_length)) * diameter[valid] * distance[valid] /
        axial_distance[valid] ** 2
    )
    ratio = projected_diameter / reference_size
    point_source = valid & (ratio <= point_source_threshold)
    magnification = np.full(points_in_eye.shape[0], np.inf)
    magnification[valid] = (
        abs(float(focal_length)) * distance[valid] /
        axial_distance[valid] ** 2
    )
    # For a cube, a subcell edge h has circumsphere diameter sqrt(3)*h.
    # Choosing h from the allowed projected diameter therefore reproduces
    # ceil(ratio/threshold) on cubic voxels while keeping anisotropic cells
    # close to cubic after subdivision.
    target_edge = np.zeros(points_in_eye.shape[0], dtype=float)
    target_edge[valid] = (
        point_source_threshold * reference_size[valid] /
        (np.sqrt(3.0) * magnification[valid])
    )
    ideal_float = np.full((points_in_eye.shape[0], 3), np.inf)
    refinable = valid & ~point_source
    ideal_float[point_source] = 1.0
    ideal_float[refinable] = np.maximum(
        1.0,
        np.ceil(np.nextafter(
            edge_lengths[refinable] / target_edge[refinable, None],
            -np.inf,
        )),
    )
    if fallback is None:
        if not np.all(np.isfinite(ideal_float)):
            raise ValueError(
                "uncapped ideal resolution is undefined for invalid geometry; "
                "specify fallback_resolution",
            )
        if np.any(ideal_float > np.iinfo(np.int64).max):
            raise OverflowError("ideal source resolution exceeds int64")
        capped = np.zeros(ideal_float.shape, dtype=bool)
        resolution = ideal_float.astype(np.int64)
    else:
        capped = (~np.isfinite(ideal_float)) | (ideal_float > fallback[None, :])
        resolution = np.where(
            np.isfinite(ideal_float),
            np.minimum(ideal_float, fallback[None, :]),
            fallback[None, :],
        ).astype(np.int64)
    return PointSourceResolutionEstimate(
        resolution=resolution,
        ratio=ratio,
        projected_diameter=projected_diameter,
        reference_size=reference_size,
        point_source=point_source,
        ideal_resolution=ideal_float,
        capped=capped,
        valid=valid,
    )


def select_source_resolution(projected_spans: np.ndarray, detector_pitch,
                             max_resolution=4,
                             max_projected_step: float = 1.0
                             ) -> SourceResolutionEstimate:
    """Choose axis-wise source resolution from projected cell-axis spans.

    The projected displacement of one source subcell is approximated by the
    full cell-axis displacement divided by the number of source samples on
    that axis.  The smallest integer resolution is selected such that this
    displacement is no larger than ``max_projected_step`` detector cells in
    either detector direction.

    ``detector_pitch`` may be a common physical pixel/subpixel pitch or a
    per-cell optical scale such as the local finite-Eye PSF footprint.

    Parameters
    ----------
    projected_spans : ndarray, shape (n, 3, 2)
        Output of :func:`projected_axis_spans` in physical screen units.
    detector_pitch : float, (float, float), or ndarray shape (n, 2)
        Reference scale in screen ``(u, v)`` coordinates. A per-cell array
        supports depth-dependent finite-Eye PSF footprints.
    max_resolution : int or (int, int, int), optional
        Per-source-axis resolution ceiling. Defaults to four.
    max_projected_step : float, optional
        Maximum projected source-subcell displacement in detector-cell units.
        ``1`` permits one reference detector cell per source subcell; smaller
        values request finer source quadrature.

    Returns
    -------
    SourceResolutionEstimate
        ``resolution`` has shape ``(n, 3)``. ``projected_span_cells`` stores
        the unrefined displacement of each source axis, and ``capped`` marks
        axes whose requested resolution exceeded the ceiling or was invalid.
    """
    projected_spans = np.asarray(projected_spans, dtype=float)
    if projected_spans.ndim != 3 or projected_spans.shape[1:] != (3, 2):
        raise ValueError("projected_spans must have shape (n, 3, 2)")
    if np.any(projected_spans < 0.0):
        raise ValueError("projected_spans must be nonnegative")
    detector_pitch = np.asarray(detector_pitch, dtype=float)
    if detector_pitch.ndim == 0:
        detector_pitch = np.full((projected_spans.shape[0], 2), detector_pitch)
    elif detector_pitch.shape == (2,):
        detector_pitch = np.broadcast_to(
            detector_pitch, (projected_spans.shape[0], 2),
        )
    elif detector_pitch.shape != (projected_spans.shape[0], 2):
        raise ValueError("detector_pitch must be scalar, length 2, or shape (n, 2)")
    if not np.all(np.isfinite(detector_pitch)) or np.any(detector_pitch <= 0.0):
        raise ValueError("detector_pitch must contain only positive finite values")
    if not np.isfinite(max_projected_step) or max_projected_step <= 0.0:
        raise ValueError("max_projected_step must be positive and finite")

    try:
        maximum = np.asarray(np.broadcast_to(max_resolution, 3), dtype=float)
    except ValueError as exc:
        raise ValueError("max_resolution must be an integer or length-3 sequence") from exc
    if not np.all(np.isfinite(maximum)) or np.any(maximum < 1) or \
            np.any(maximum != np.floor(maximum)):
        raise ValueError("max_resolution must contain positive integers")
    maximum = maximum.astype(np.int64)

    span_cells = np.max(projected_spans / detector_pitch[:, None, :], axis=2)
    requested_float = span_cells / max_projected_step
    # Move exact floating-point integers infinitesimally downward so a value
    # such as 1.0000000000000002 from projection arithmetic does not
    # spuriously request the next resolution.
    finite = np.isfinite(requested_float)
    uncapped = np.full(requested_float.shape, np.inf)
    uncapped[finite] = np.maximum(
        1.0, np.ceil(np.nextafter(requested_float[finite], -np.inf)),
    )
    capped = (~finite) | (uncapped > maximum[None, :])
    resolution = np.where(
        finite,
        np.minimum(uncapped, maximum[None, :]),
        maximum[None, :],
    ).astype(np.int64)
    return SourceResolutionEstimate(
        resolution=resolution,
        projected_span_cells=span_cells,
        uncapped_resolution=uncapped,
        capped=capped,
    )


@dataclass(frozen=True)
class OpticalBinning:
    """Packed visible-sample ordering for independent optical bins.

    Attributes
    ----------
    order : ndarray
        Input sample indices in packed order, shape ``(n_samples,)``.
    scope_offsets : ndarray
        Integer offsets, shape ``(n_scopes + 1,)``.
    scope_keys : ndarray
        Integer optical-bin keys in detector ``(u, v)`` order, shape
        ``(n_scopes, 2)``.
    scope_costs : ndarray
        Expanded sample cost per scope, shape ``(n_scopes,)``.
    bin_width_uv : ndarray
        Physical bin width in detector ``(u, v)`` order, shape ``(2,)``.

    Notes
    -----
    ``order`` contains indices into the input point array.  Each consecutive
    interval ``scope_offsets[i]:scope_offsets[i + 1]`` is one independent
    compression scope.  A very large optical bin may be represented by
    several adjacent scopes with the same ``scope_keys`` entry; scopes are
    never joined during compression.
    """

    order: np.ndarray
    scope_offsets: np.ndarray
    scope_keys: np.ndarray
    scope_costs: np.ndarray
    bin_width_uv: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.order.size)

    @property
    def n_scopes(self) -> int:
        return int(self.scope_offsets.size - 1)

    def scopes(self) -> list[np.ndarray]:
        """Return sample-index views, one per independent optical scope."""
        return [self.order[start:stop]
                for start, stop in zip(self.scope_offsets[:-1], self.scope_offsets[1:])]

    def work_offsets(self, max_samples: int) -> np.ndarray:
        """Pack complete optical scopes into memory work chunks.

        The limit is soft when one scope itself is larger than ``max_samples``;
        callers that require a hard memory bound should set ``max_scope_samples``
        while constructing the binning.
        """
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        if self.n_scopes == 0:
            return np.array([0], dtype=np.int64)
        offsets = [0]
        accumulated_cost = 0
        for scope_number, scope_cost in enumerate(self.scope_costs):
            scope_start = int(self.scope_offsets[scope_number])
            scope_stop = int(self.scope_offsets[scope_number + 1])
            if accumulated_cost and accumulated_cost + scope_cost > max_samples:
                offsets.append(scope_start)
                accumulated_cost = 0
            accumulated_cost += int(scope_cost)
            if accumulated_cost >= max_samples:
                offsets.append(scope_stop)
                accumulated_cost = 0
        if offsets[-1] != self.n_samples:
            offsets.append(self.n_samples)
        return np.asarray(offsets, dtype=np.int64)

    def work_chunks(self, max_samples: int) -> list[np.ndarray]:
        """Return sample-index views for memory-bounded work chunks."""
        offsets = self.work_offsets(max_samples)
        return [self.order[start:stop]
                for start, stop in zip(offsets[:-1], offsets[1:])]


def make_optical_binning(camera, eye_index: int, points: np.ndarray,
                         bin_width_pixels=1.0,
                         max_scope_samples: int | None = None,
                         sample_costs: np.ndarray | None = None) -> OpticalBinning:
    """Order already-visible source samples by Eye projection direction.

    Parameters
    ----------
    camera : Camera
        Camera used to transform world points and define detector pitch.
    eye_index : int
        Eye whose optical coordinates define the bins.
    points : ndarray, shape (n, 3)
        Source points that have already passed visibility filtering.
    bin_width_pixels : float or (float, float), optional
        Optical-bin width in detector-pixel pitches.  ``1`` means one detector
        pixel, ``0.5`` half a pixel, and ``2`` two pixels.
    max_scope_samples : int, optional
        Split an oversized bin in increasing ``f / Z_e`` order when its
        cumulative expanded-sample cost exceeds this value.
    sample_costs : ndarray of int, shape (n,), optional
        Expanded sub-voxel count represented by each input point.  Defaults to
        one, as used when the input points are already sub-voxel samples.

    Returns
    -------
    OpticalBinning
        Packed sample order and independent optical scopes. Empty input returns
        zero scopes with a single zero offset.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    width_pixels = _pair(bin_width_pixels, "bin_width_pixels")
    if max_scope_samples is not None:
        if isinstance(max_scope_samples, (bool, np.bool_)) or \
                int(max_scope_samples) != max_scope_samples or max_scope_samples < 1:
            raise ValueError("max_scope_samples must be a positive integer")
        max_scope_samples = int(max_scope_samples)
    if sample_costs is None:
        sample_costs = np.ones(points.shape[0], dtype=np.int64)
    else:
        sample_costs = np.asarray(sample_costs)
        if sample_costs.shape != (points.shape[0],) or \
                not np.issubdtype(sample_costs.dtype, np.integer) or \
                np.any(sample_costs <= 0):
            raise ValueError("sample_costs must be positive integers with shape (n,)")
        sample_costs = sample_costs.astype(np.int64, copy=False)
    if points.shape[0] == 0:
        return OpticalBinning(
            order=np.empty(0, dtype=np.int64),
            scope_offsets=np.array([0], dtype=np.int64),
            scope_keys=np.empty((0, 2), dtype=np.int64),
            scope_costs=np.empty(0, dtype=np.int64),
            bin_width_uv=camera.screen.pixel_size * width_pixels,
        )

    points_camera = camera.world2camera(points)
    eye = camera.eyes[eye_index]
    points_eye = eye.camera2eye(points_camera)
    if np.any(points_eye[:, 2] <= 0.0):
        raise ValueError("all optical-binning points must lie in front of the Eye")

    xi_eta = points_eye[:, :2] / points_eye[:, 2, None]
    projected_xy = -eye.focal_length * xi_eta + eye.principal_point[None, :2]
    projected_uv = camera.screen.xy2uv(projected_xy)
    bin_width_uv = camera.screen.pixel_size * width_pixels
    bin_keys = np.floor(projected_uv / bin_width_uv[None, :]).astype(np.int64)
    zoom_rate = 1.0 + eye.focal_length / points_eye[:, 2]
    order = np.lexsort((zoom_rate, bin_keys[:, 1], bin_keys[:, 0])).astype(np.int64)

    ordered_keys = bin_keys[order]
    ordered_costs = sample_costs[order]
    boundaries = np.flatnonzero(np.any(np.diff(ordered_keys, axis=0), axis=1)) + 1
    bin_offsets = np.concatenate(([0], boundaries, [order.size])).astype(np.int64)

    scope_offsets = [0]
    scope_keys = []
    scope_costs = []
    for start, stop in zip(bin_offsets[:-1], bin_offsets[1:]):
        scope_start = int(start)
        accumulated_cost = 0
        for position in range(int(start), int(stop)):
            cost = int(ordered_costs[position])
            if max_scope_samples is not None and accumulated_cost and \
                    accumulated_cost + cost > max_scope_samples:
                scope_offsets.append(position)
                scope_keys.append(ordered_keys[start])
                scope_costs.append(accumulated_cost)
                scope_start = position
                accumulated_cost = 0
            accumulated_cost += cost
        if scope_start < stop:
            scope_offsets.append(int(stop))
            scope_keys.append(ordered_keys[start])
            scope_costs.append(accumulated_cost)

    return OpticalBinning(
        order=order,
        scope_offsets=np.asarray(scope_offsets, dtype=np.int64),
        scope_keys=np.asarray(scope_keys, dtype=np.int64).reshape(-1, 2),
        scope_costs=np.asarray(scope_costs, dtype=np.int64),
        bin_width_uv=bin_width_uv,
    )
