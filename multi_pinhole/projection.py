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
        chunk_start = 0
        for scope_stop in self.scope_offsets[1:]:
            if scope_stop - chunk_start > max_samples and offsets[-1] != scope_stop:
                previous_stop = int(self.scope_offsets[
                    np.searchsorted(self.scope_offsets, scope_stop) - 1
                ])
                if previous_stop > chunk_start:
                    offsets.append(previous_stop)
                    chunk_start = previous_stop
            if scope_stop - chunk_start >= max_samples:
                offsets.append(int(scope_stop))
                chunk_start = int(scope_stop)
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
                         max_scope_samples: int | None = None) -> OpticalBinning:
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
        Split an oversized bin in increasing ``f / Z_e`` order.  The split
        pieces remain separate compression scopes even though their bin keys
        are identical.
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
    if points.shape[0] == 0:
        return OpticalBinning(
            order=np.empty(0, dtype=np.int64),
            scope_offsets=np.array([0], dtype=np.int64),
            scope_keys=np.empty((0, 2), dtype=np.int64),
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
    boundaries = np.flatnonzero(np.any(np.diff(ordered_keys, axis=0), axis=1)) + 1
    bin_offsets = np.concatenate(([0], boundaries, [order.size])).astype(np.int64)

    scope_offsets = [0]
    scope_keys = []
    for start, stop in zip(bin_offsets[:-1], bin_offsets[1:]):
        step = stop - start if max_scope_samples is None else max_scope_samples
        for scope_start in range(int(start), int(stop), int(step)):
            scope_offsets.append(min(scope_start + int(step), int(stop)))
            scope_keys.append(ordered_keys[start])

    return OpticalBinning(
        order=order,
        scope_offsets=np.asarray(scope_offsets, dtype=np.int64),
        scope_keys=np.asarray(scope_keys, dtype=np.int64).reshape(-1, 2),
        bin_width_uv=bin_width_uv,
    )
