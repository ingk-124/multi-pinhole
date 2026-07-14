"""Internal helpers for projection-matrix construction.

The public projection API still lives on :class:`multi_pinhole.world.World`.
This module contains geometry-independent bookkeeping that is shared by the
ordinary sparse and future factorized projection builders.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

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
    """Sensitivity-preserving grouping data for ``I ~= Q R``.

    ``R`` is intentionally not materialized.  Its column ``sample`` has one
    logical nonzero at ``(group_index[sample], sample)`` with value
    ``sensitivity[sample]``.  Keeping those two one-dimensional arrays avoids
    CSR bookkeeping for a matrix that is only a weighted group assignment.
    """

    Q: sparse.csr_matrix
    group_index: np.ndarray
    group_max_l1: np.ndarray
    group_max_relative_l2: np.ndarray
    sensitivity: np.ndarray

    @property
    def n_groups(self) -> int:
        """Number of representative PSF columns."""
        return int(self.Q.shape[1])

    @property
    def n_active_samples(self) -> int:
        """Number of source samples with nonzero detector sensitivity."""
        return int(np.count_nonzero(self.group_index >= 0))


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


def _leader_psf_groups(columns: np.ndarray, sensitivity: np.ndarray,
                       tolerance: float, metric: str) -> list[np.ndarray]:
    """Single-pass leader grouping with exact checks after mean updates."""
    groups: list[list[int]] = []
    representatives: list[np.ndarray] = []
    for column_number in range(columns.shape[1]):
        if not groups:
            groups.append([column_number])
            representatives.append(columns[:, column_number].copy())
            continue

        candidate = columns[:, column_number:column_number + 1]
        candidate_distance = np.asarray([
            _psf_metric(candidate, representative, metric)[0]
            for representative in representatives
        ])
        accepted = False
        for group_number in np.argsort(candidate_distance, kind="stable"):
            proposed = np.asarray(
                groups[group_number] + [column_number], dtype=np.int64,
            )
            weights = sensitivity[proposed]
            representative = (
                columns[:, proposed] * weights[None, :]
            ).sum(axis=1) / weights.sum()
            # Rechecking every existing member prevents the moving mean from
            # drifting outside the requested tolerance.
            if _psf_metric(
                    columns[:, proposed], representative, metric,
            ).max(initial=0.0) <= tolerance:
                groups[group_number].append(column_number)
                representatives[group_number] = representative
                accepted = True
                break
        if not accepted:
            groups.append([column_number])
            representatives.append(columns[:, column_number].copy())
    return [np.asarray(group, dtype=np.int64) for group in groups]


def factorize_psf_columns(I, scope_offsets, tolerance: float,
                          metric: str = "relative_l2",
                          algorithm: str = "recursive",
                          max_scope_dense_bytes: int | None = None,
                          ) -> PSFFactorization | None:
    """Factorize PSF columns independently within consecutive scopes.

    A return value of ``None`` is the explicit direct-block signal.  In
    particular, ``tolerance == 0`` returns before converting, normalizing, or
    densifying ``I``; callers should compute the exact ``I @ S`` block.

    Only detector rows that contain a nonzero in the current scope are made
    dense for distance calculations.  Their indices remain *global detector
    row indices*; group and sample indices always refer to columns of the
    original ``I``.  If ``max_scope_dense_bytes`` is given, an oversized scope
    is split before allocation.  This may miss compression across the split,
    but cannot increase the approximation error.
    """
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if tolerance == 0.0:
        return None
    if metric not in {"relative_l2", "l1"}:
        raise ValueError("metric must be 'relative_l2' or 'l1'")
    if algorithm not in {"recursive", "leader"}:
        raise ValueError("algorithm must be 'recursive' or 'leader'")
    if max_scope_dense_bytes is not None and max_scope_dense_bytes < 1:
        raise ValueError("max_scope_dense_bytes must be positive or None")

    I_csc = sparse.csc_matrix(I)
    offsets = np.asarray(scope_offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size < 1 or offsets[0] != 0 or \
            offsets[-1] != I_csc.shape[1] or np.any(np.diff(offsets) <= 0):
        raise ValueError("scope_offsets must strictly partition all I columns")

    sensitivity = np.asarray(I_csc.sum(axis=0)).ravel()
    positive = sensitivity > 0.0
    group_index = np.full(I_csc.shape[1], -1, dtype=np.int32)
    group_members: list[np.ndarray] = []
    q_columns: list[sparse.csc_matrix] = []
    max_l1: list[float] = []
    max_relative_l2: list[float] = []

    def _factorize_scope(members: np.ndarray):
        """Append groups for one scope, splitting only for dense memory."""
        if members.size == 0:
            return

        I_scope = I_csc[:, members]
        # ``I_scope.indices`` are global detector-row numbers in CSC format.
        # Rows absent from this union are exactly zero for every scope column,
        # so dropping them leaves both L1 and L2 distances unchanged.
        active_pixel_rows = np.unique(I_scope.indices)
        dense_itemsize = max(I_csc.dtype.itemsize, sensitivity.dtype.itemsize)
        # Normalized columns, a difference buffer, and metric temporaries can
        # coexist.  Four dense arrays is a conservative working-set estimate.
        estimated_dense_bytes = (
            4 * active_pixel_rows.size * members.size * dense_itemsize
        )
        if max_scope_dense_bytes is not None and \
                estimated_dense_bytes > max_scope_dense_bytes and members.size > 1:
            middle = members.size // 2
            _factorize_scope(members[:middle])
            _factorize_scope(members[middle:])
            return

        # This is a local detector-row coordinate system used only while
        # comparing PSFs.  Q below is reconstructed from I_csc and therefore
        # remains indexed in the original global detector coordinates.
        block = I_scope[active_pixel_rows, :].toarray()
        normalized = block / sensitivity[members][None, :]
        group_function = (
            _recursive_psf_groups if algorithm == "recursive"
            else _leader_psf_groups
        )
        for local_group in group_function(
                normalized, sensitivity[members], tolerance, metric):
            global_members = members[local_group]
            total_sensitivity = sensitivity[global_members].sum()
            representative = np.asarray(
                I_csc[:, global_members].sum(axis=1)
            ).ravel() / total_sensitivity
            local_representative = representative[active_pixel_rows]
            l1, relative_l2 = _psf_distances(
                normalized[:, local_group], local_representative,
            )
            group_number = len(group_members)
            group_index[global_members] = group_number
            group_members.append(global_members)
            q_columns.append(sparse.csc_matrix(representative[:, None]))
            max_l1.append(float(l1.max(initial=0.0)))
            max_relative_l2.append(float(relative_l2.max(initial=0.0)))

    for start, stop in zip(offsets[:-1], offsets[1:]):
        members = np.arange(start, stop, dtype=np.int64)
        _factorize_scope(members[positive[members]])

    if q_columns:
        Q = sparse.hstack(q_columns, format="csr")
    else:
        Q = sparse.csr_matrix((I_csc.shape[0], 0))

    return PSFFactorization(
        Q=Q, group_index=group_index,
        group_max_l1=np.asarray(max_l1),
        group_max_relative_l2=np.asarray(max_relative_l2),
        sensitivity=sensitivity,
    )


def build_psf_group_matrix(
        factorization: PSFFactorization, S,
) -> sparse.csr_matrix:
    """Build ``A = R S`` without materializing the weighted one-hot ``R``.

    ``S`` rows and ``factorization.group_index`` entries use the same global
    sample-column order as the ``I`` passed to :func:`factorize_psf_columns`.
    ``S`` columns are global voxel indices and are copied unchanged into A.
    """
    S_csr = sparse.csr_matrix(S)
    n_samples = factorization.group_index.size
    if S_csr.shape[0] != n_samples:
        raise ValueError("S rows must match the number of factorized I columns")

    row_nnz = np.diff(S_csr.indptr)
    # Each nonzero S[i, voxel] is accumulated into A[group_index[i], voxel]
    # with the logical R value sensitivity[i].  These repeated arrays are
    # indexed per S nonzero, not per detector pixel.
    group_per_nonzero = np.repeat(factorization.group_index, row_nnz)
    sensitivity_per_nonzero = np.repeat(factorization.sensitivity, row_nnz)
    assigned = group_per_nonzero >= 0
    A = sparse.coo_matrix(
        (
            S_csr.data[assigned] * sensitivity_per_nonzero[assigned],
            (group_per_nonzero[assigned], S_csr.indices[assigned]),
        ),
        shape=(factorization.n_groups, S_csr.shape[1]),
    ).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


def _sparse_nbytes(matrix: sparse.spmatrix) -> int:
    """Return actual CSR payload bytes, including indices and row pointers."""
    matrix = sparse.csr_matrix(matrix)
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


class HybridProjectionOperator:
    """Projection represented as an exact direct block plus ``Q @ A``.

    The named :meth:`project` and :meth:`backproject` methods are the primary
    API.  ``@`` and ``.T @`` are convenience aliases for code that already
    treats a projection as a matrix.
    """

    def __init__(self, direct, Q, A, compression_stats=()):
        self.direct = sparse.csr_matrix(direct)
        self.Q = sparse.csr_matrix(Q)
        self.A = sparse.csr_matrix(A)
        self.compression_stats = tuple(compression_stats)
        if self.Q.shape[1] != self.A.shape[0]:
            raise ValueError("Q columns must match A rows")
        if self.direct.shape != (self.Q.shape[0], self.A.shape[1]):
            raise ValueError("direct, Q, and A have incompatible shapes")

    @classmethod
    def empty(cls, shape, dtype=float):
        """Construct an all-zero operator with no factorized groups."""
        n_rows, n_columns = shape
        return cls(
            sparse.csr_matrix(shape, dtype=dtype),
            sparse.csr_matrix((n_rows, 0), dtype=dtype),
            sparse.csr_matrix((0, n_columns), dtype=dtype),
        )

    @property
    def shape(self):
        return self.direct.shape

    @property
    def dtype(self):
        return np.result_type(self.direct.dtype, self.Q.dtype, self.A.dtype)

    @property
    def storage_nbytes(self) -> int:
        """Actual bytes occupied by the three CSR payloads."""
        return sum(_sparse_nbytes(matrix) for matrix in (self.direct, self.Q, self.A))

    def project(self, emission):
        """Apply ``(direct + Q A)`` to voxel emission values."""
        return self.direct @ emission + self.Q @ (self.A @ emission)

    def backproject(self, detector_values):
        """Apply the transpose to detector-space values."""
        return (self.direct.T @ detector_values
                + self.A.T @ (self.Q.T @ detector_values))

    def dot(self, emission):
        """Alias for :meth:`project` for sparse-matrix familiarity."""
        return self.project(emission)

    def __matmul__(self, emission):
        return self.project(emission)

    @property
    def T(self):
        return _TransposedProjectionOperator(self)

    def transpose(self):
        """Return a lightweight transposed operator view."""
        return self.T

    def to_sparse(self) -> sparse.csr_matrix:
        """Materialize the approximation only when legacy CSR is required."""
        return (self.direct + self.Q @ self.A).tocsr()

    def left_multiply(self, matrix):
        """Apply a detector-side sparse transform without expanding ``Q A``."""
        matrix = sparse.csr_matrix(matrix)
        if matrix.shape[1] != self.shape[0]:
            raise ValueError("left transform columns must match detector rows")
        return HybridProjectionOperator(
            matrix @ self.direct, matrix @ self.Q, self.A,
            compression_stats=self.compression_stats,
        )


class _TransposedProjectionOperator:
    """Lightweight transpose view returned by ``HybridProjectionOperator.T``."""

    def __init__(self, parent: HybridProjectionOperator):
        self._parent = parent

    @property
    def shape(self):
        return self._parent.shape[::-1]

    @property
    def dtype(self):
        return self._parent.dtype

    def project(self, detector_values):
        return self._parent.backproject(detector_values)

    def backproject(self, emission):
        return self._parent.project(emission)

    def dot(self, detector_values):
        return self.project(detector_values)

    def __matmul__(self, detector_values):
        return self.project(detector_values)

    @property
    def T(self):
        return self._parent

    def transpose(self):
        return self._parent

    def to_sparse(self) -> sparse.csr_matrix:
        return self._parent.to_sparse().T.tocsr()


@dataclass(frozen=True)
class ProjectionCompressionStats:
    """Compression decision for one independent optical scope."""

    n_samples: int
    n_active_samples: int
    n_groups: int
    group_fraction: float
    used_factorization: bool
    max_l1: float
    max_relative_l2: float
    n_active_pixel_rows: int
    direct_nbytes: int
    q_nbytes: int
    a_nbytes: int
    grouping_seconds: float
    assembly_seconds: float

    @property
    def stored_nbytes(self) -> int:
        """CSR payload bytes selected for this scope."""
        return self.direct_nbytes + self.q_nbytes + self.a_nbytes


def build_projection_block(
        I, S, *, tolerance: float, metric: str = "relative_l2",
        algorithm: str = "recursive", max_group_fraction: float | None = 0.8,
        max_scope_dense_bytes: int | None = None,
) -> tuple[HybridProjectionOperator, ProjectionCompressionStats]:
    """Build one direct or factorized scope without computing both forms.

    The decision uses ``N_group / N_active_sample``.  If the ratio is at or
    above ``max_group_fraction``, ``I @ S`` is calculated and the temporary
    factorization is discarded.  Otherwise A is accumulated directly from the
    weighted group assignment.  ``tolerance == 0`` bypasses all grouping.
    """
    I_sparse = sparse.csr_matrix(I)
    S_sparse = sparse.csr_matrix(S)
    build_start = time.perf_counter()
    if I_sparse.shape[1] != S_sparse.shape[0]:
        raise ValueError("I columns must match S rows")
    if max_group_fraction is not None and not 0.0 <= max_group_fraction <= 1.0:
        raise ValueError("max_group_fraction must be in [0, 1] or None")

    n_detector, n_samples = I_sparse.shape
    n_voxels = S_sparse.shape[1]
    n_active_pixel_rows = int(np.count_nonzero(I_sparse.getnnz(axis=1)))
    if n_samples == 0:
        operator = HybridProjectionOperator.empty(
            (n_detector, n_voxels), dtype=I_sparse.dtype,
        )
        stats = ProjectionCompressionStats(
            n_samples=0, n_active_samples=0, n_groups=0,
            group_fraction=0.0, used_factorization=False,
            max_l1=0.0, max_relative_l2=0.0,
            n_active_pixel_rows=0,
            direct_nbytes=_sparse_nbytes(operator.direct),
            q_nbytes=_sparse_nbytes(operator.Q),
            a_nbytes=_sparse_nbytes(operator.A),
            grouping_seconds=0.0,
            assembly_seconds=time.perf_counter() - build_start,
        )
        operator.compression_stats = (stats,)
        return operator, stats
    # Both switches are explicit direct-mode requests.  Test them before any
    # normalization or grouping work, just as tolerance == 0 is handled by
    # factorize_psf_columns itself.
    if tolerance == 0.0 or max_group_fraction == 0.0:
        assembly_start = time.perf_counter()
        operator = HybridProjectionOperator(
            I_sparse @ S_sparse,
            sparse.csr_matrix((n_detector, 0), dtype=I_sparse.dtype),
            sparse.csr_matrix((0, n_voxels), dtype=I_sparse.dtype),
        )
        assembly_seconds = time.perf_counter() - assembly_start
        n_active = int(np.count_nonzero(
            np.asarray(I_sparse.sum(axis=0)).ravel() > 0.0
        ))
        stats = ProjectionCompressionStats(
            n_samples=n_samples, n_active_samples=n_active, n_groups=0,
            group_fraction=1.0, used_factorization=False,
            max_l1=0.0, max_relative_l2=0.0,
            n_active_pixel_rows=n_active_pixel_rows,
            direct_nbytes=_sparse_nbytes(operator.direct),
            q_nbytes=_sparse_nbytes(operator.Q),
            a_nbytes=_sparse_nbytes(operator.A),
            grouping_seconds=0.0,
            assembly_seconds=assembly_seconds,
        )
        operator.compression_stats = (stats,)
        return operator, stats
    grouping_start = time.perf_counter()
    factorization = factorize_psf_columns(
        I_sparse, np.array([0, n_samples]), tolerance=tolerance,
        metric=metric, algorithm=algorithm,
        max_scope_dense_bytes=max_scope_dense_bytes,
    )
    grouping_seconds = time.perf_counter() - grouping_start

    n_active = factorization.n_active_samples
    group_fraction = factorization.n_groups / max(n_active, 1)
    use_factorization = (
        n_active > 0
        and (max_group_fraction is None or group_fraction < max_group_fraction)
    )
    assembly_start = time.perf_counter()
    if use_factorization:
        A = build_psf_group_matrix(factorization, S_sparse)
        operator = HybridProjectionOperator(
            sparse.csr_matrix((n_detector, n_voxels), dtype=I_sparse.dtype),
            factorization.Q, A,
        )
    else:
        operator = HybridProjectionOperator(
            I_sparse @ S_sparse,
            sparse.csr_matrix((n_detector, 0), dtype=I_sparse.dtype),
            sparse.csr_matrix((0, n_voxels), dtype=I_sparse.dtype),
        )
    assembly_seconds = time.perf_counter() - assembly_start
    stats = ProjectionCompressionStats(
        n_samples=n_samples,
        n_active_samples=n_active,
        n_groups=factorization.n_groups,
        group_fraction=group_fraction,
        used_factorization=use_factorization,
        max_l1=float(factorization.group_max_l1.max(initial=0.0)),
        max_relative_l2=float(
            factorization.group_max_relative_l2.max(initial=0.0)
        ),
        n_active_pixel_rows=n_active_pixel_rows,
        direct_nbytes=_sparse_nbytes(operator.direct),
        q_nbytes=_sparse_nbytes(operator.Q),
        a_nbytes=_sparse_nbytes(operator.A),
        grouping_seconds=grouping_seconds,
        assembly_seconds=assembly_seconds,
    )
    operator.compression_stats = (stats,)
    return operator, stats


def combine_projection_operators(operators) -> HybridProjectionOperator:
    """Sum equal-shaped operators while retaining all factorized blocks."""
    operators = list(operators)
    if not operators:
        raise ValueError("at least one projection operator is required")
    shape = operators[0].shape
    if any(operator.shape != shape for operator in operators):
        raise ValueError("all projection operators must have the same shape")
    direct = sum(
        (operator.direct for operator in operators),
        sparse.csr_matrix(shape, dtype=operators[0].dtype),
    )
    q_blocks = [operator.Q for operator in operators if operator.Q.shape[1]]
    a_blocks = [operator.A for operator in operators if operator.A.shape[0]]
    if q_blocks:
        Q = sparse.hstack(q_blocks, format="csr")
        A = sparse.vstack(a_blocks, format="csr")
    else:
        Q = sparse.csr_matrix((shape[0], 0), dtype=operators[0].dtype)
        A = sparse.csr_matrix((0, shape[1]), dtype=operators[0].dtype)
    compression_stats = tuple(
        stats
        for operator in operators
        for stats in operator.compression_stats
    )
    return HybridProjectionOperator(
        direct, Q, A, compression_stats=compression_stats,
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
