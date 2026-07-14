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
    """Axis-wise source quadrature recommendation and its diagnostics."""

    resolution: np.ndarray
    projected_span_cells: np.ndarray
    uncapped_resolution: np.ndarray
    capped: np.ndarray


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
