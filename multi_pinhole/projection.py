"""Internal helpers for projection-matrix construction.

The public projection API still lives on :class:`multi_pinhole.world.World`.
This module contains geometry-independent bookkeeping that is shared by the
ordinary sparse and future factorized projection builders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


def _pair(value, name: str) -> np.ndarray:
    pair = np.asarray(value, dtype=float)
    if pair.ndim == 0:
        pair = np.repeat(pair, 2)
    if pair.shape != (2,) or not np.all(np.isfinite(pair)) or np.any(pair <= 0.0):
        raise ValueError(f"{name} must be a positive scalar or length-2 sequence")
    return pair


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


@dataclass(frozen=True)
class PSFFactorization:
    """Sensitivity-preserving factorization ``I ~= Q R``."""

    Q: sparse.csr_matrix
    R: sparse.csr_matrix
    group_index: np.ndarray
    group_max_l1: np.ndarray
    group_max_relative_l2: np.ndarray
    sensitivity: np.ndarray


def _psf_distances(columns: np.ndarray, representative: np.ndarray):
    difference = columns - representative[:, None]
    l1 = np.abs(difference).sum(axis=0)
    norm = np.linalg.norm(columns, axis=0)
    relative_l2 = np.linalg.norm(difference, axis=0) / np.maximum(norm, 1e-30)
    return l1, relative_l2


def _psf_metric(columns: np.ndarray, representative: np.ndarray, metric: str):
    l1, relative_l2 = _psf_distances(columns, representative)
    if metric == "relative_l2":
        return relative_l2
    if metric == "l1":
        return l1
    raise ValueError("metric must be 'relative_l2' or 'l1'")


def _recursive_psf_groups(columns: np.ndarray, sensitivity: np.ndarray,
                          tolerance: float, metric: str) -> list[np.ndarray]:
    groups: list[np.ndarray] = []

    def split(indices):
        indices = np.asarray(indices, dtype=np.int64)
        block = columns[:, indices]
        weights = sensitivity[indices]
        representative = (block * weights[None, :]).sum(axis=1) / weights.sum()
        errors = _psf_metric(block, representative, metric)
        if indices.size == 1 or errors.max(initial=0.0) <= tolerance:
            groups.append(indices)
            return

        first = block[:, int(np.argmax(errors))]
        second = block[:, int(np.argmax(_psf_metric(block, first, metric)))]
        first_distance = _psf_metric(block, first, metric)
        second_distance = _psf_metric(block, second, metric)
        left = first_distance <= second_distance
        if np.all(left) or not np.any(left):
            order = np.argsort(errors, kind="stable")
            left = np.zeros(indices.size, dtype=bool)
            left[order[:indices.size // 2]] = True
        split(indices[left])
        split(indices[~left])

    split(np.arange(columns.shape[1], dtype=np.int64))
    return groups


def factorize_psf_columns(I, scope_offsets, tolerance: float,
                          metric: str = "relative_l2") -> PSFFactorization | None:
    """Factorize PSF columns independently within consecutive scopes.

    A return value of ``None`` is the explicit direct-block signal.  In
    particular, ``tolerance == 0`` returns before converting, normalizing, or
    densifying ``I``; callers should compute the exact ``I @ S`` block.
    """
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if tolerance == 0.0:
        return None
    if metric not in {"relative_l2", "l1"}:
        raise ValueError("metric must be 'relative_l2' or 'l1'")

    I_csc = sparse.csc_matrix(I)
    offsets = np.asarray(scope_offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size < 1 or offsets[0] != 0 or \
            offsets[-1] != I_csc.shape[1] or np.any(np.diff(offsets) <= 0):
        raise ValueError("scope_offsets must strictly partition all I columns")

    sensitivity = np.asarray(I_csc.sum(axis=0)).ravel()
    positive = sensitivity > 0.0
    group_index = np.full(I_csc.shape[1], -1, dtype=np.int64)
    group_members: list[np.ndarray] = []
    q_columns: list[sparse.csc_matrix] = []
    max_l1: list[float] = []
    max_relative_l2: list[float] = []

    for start, stop in zip(offsets[:-1], offsets[1:]):
        members = np.arange(start, stop, dtype=np.int64)
        members = members[positive[members]]
        if members.size == 0:
            continue
        block = I_csc[:, members].toarray()
        normalized = block / sensitivity[members][None, :]
        for local_group in _recursive_psf_groups(
                normalized, sensitivity[members], tolerance, metric):
            global_members = members[local_group]
            total_sensitivity = sensitivity[global_members].sum()
            representative = np.asarray(
                I_csc[:, global_members].sum(axis=1)
            ).ravel() / total_sensitivity
            l1, relative_l2 = _psf_distances(
                normalized[:, local_group], representative,
            )
            group_number = len(group_members)
            group_index[global_members] = group_number
            group_members.append(global_members)
            q_columns.append(sparse.csc_matrix(representative[:, None]))
            max_l1.append(float(l1.max(initial=0.0)))
            max_relative_l2.append(float(relative_l2.max(initial=0.0)))

    if q_columns:
        Q = sparse.hstack(q_columns, format="csr")
        columns = np.concatenate(group_members)
        rows = np.repeat(
            np.arange(len(group_members)), [members.size for members in group_members],
        )
        R = sparse.csr_matrix(
            (sensitivity[columns], (rows, columns)),
            shape=(len(group_members), I_csc.shape[1]),
        )
    else:
        Q = sparse.csr_matrix((I_csc.shape[0], 0))
        R = sparse.csr_matrix((0, I_csc.shape[1]))

    return PSFFactorization(
        Q=Q, R=R, group_index=group_index,
        group_max_l1=np.asarray(max_l1),
        group_max_relative_l2=np.asarray(max_relative_l2),
        sensitivity=sensitivity,
    )


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
