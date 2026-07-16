"""Private numerical builders for voxel-to-pixel projection matrices.

The functions here know about quadrature, optical work ordering, and sparse
matrix assembly. They do not resolve camera keys, own caches, or mutate a
``World`` instance.
"""

import time
from collections.abc import Callable

import numpy as np
from scipy import sparse

from .utils.my_stdio import my_print, my_tqdm


def sparse_nbytes(matrix) -> int:
    """Return the bytes occupied by a matrix in CSR representation."""
    matrix = matrix.tocsr()
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def sum_eye_projections(matrices, shape) -> sparse.csr_matrix:
    """Combine per-eye pixel matrices without changing their sparse format."""
    return sum(matrices, sparse.csr_matrix(shape))


def build_optical_projection_matrix(
        *, voxel, camera, eye_idx: int,
        full_voxels: np.ndarray, partial_voxels: np.ndarray,
        full_subvoxel_res, partial_subvoxel_res,
        max_nnz: int, max_working_memory: int,
        optical_bin_width_pixels, verbose: int,
        point_visibility: Callable[[np.ndarray], np.ndarray],
        inside_points: Callable[[np.ndarray], np.ndarray],
        make_binning: Callable,
) -> sparse.csr_matrix:
    """Build one eye's CSR projection using optical-bin work ordering.

    ``point_visibility`` and ``inside_points`` are explicit geometry queries
    used only for partial voxels. The returned matrix has physical pixels as
    rows and source voxels as columns.
    """
    screen = camera.screen
    n_vox = voxel.N
    index_start = time.perf_counter()
    full_resolution = tuple(int(r) for r in np.broadcast_to(full_subvoxel_res, 3))
    partial_resolution = tuple(int(r) for r in np.broadcast_to(partial_subvoxel_res, 3))
    full_cost = int(np.prod(full_resolution))
    partial_cost = int(np.prod(partial_resolution))
    visible_voxels = np.concatenate((full_voxels, partial_voxels))
    sample_costs = np.concatenate((
        np.full(full_voxels.size, full_cost, dtype=np.int64),
        np.full(partial_voxels.size, partial_cost, dtype=np.int64),
    ))
    voxel_centers = voxel.get_gravity_center(visible_voxels)
    if visible_voxels.size == 0:
        return sparse.csr_matrix((screen.N_pixel, n_vox))

    if full_voxels.size:
        sample_voxels = full_voxels[:min(full_voxels.size, 20)]
        sample_resolution = full_resolution
    else:
        sample_voxels = partial_voxels[:min(partial_voxels.size, 20)]
        sample_resolution = partial_resolution
    sample_points = voxel.get_sub_voxel_centers(
        sample_voxels, res=sample_resolution,
    )
    sample_S = voxel.sub_voxel_interpolator_from_centers(
        sample_voxels, res=sample_resolution, points=sample_points,
    )
    sample_I = camera.calc_image_vec(
        eye_idx, points=sample_points, verbose=0, check_visibility=False,
    )
    sample_result = screen.transform_matrix @ (sample_I @ sample_S)
    sample_count = sample_points.shape[0]
    transient_bytes = (
        sample_points.nbytes + sparse_nbytes(sample_I) + sparse_nbytes(sample_S)
        + sparse_nbytes(sample_result)
    )
    bytes_per_sample = max(1.0, transient_bytes / sample_count)
    memory_samples = max(1, int((max_working_memory // 2) // bytes_per_sample))
    nnz_per_sample = sample_I.nnz / sample_count
    if nnz_per_sample == 0.0:
        nnz_per_sample = screen.N_subpixel * 0.01
    nnz_samples = max(1, int(np.ceil(max_nnz / nnz_per_sample)))
    max_samples = min(memory_samples, nnz_samples, int(sample_costs.sum()))
    del sample_I, sample_S, sample_result

    binning = make_binning(
        camera, eye_idx, voxel_centers,
        bin_width_pixels=optical_bin_width_pixels,
        max_scope_samples=max_samples,
        sample_costs=sample_costs,
    )
    work_chunks = binning.work_chunks(max_samples=max_samples)
    index_elapsed = time.perf_counter() - index_start
    my_print(
        f"Processing {visible_voxels.size} visible voxels "
        f"({sample_costs.sum()} sub-voxel samples before partial masking) in "
        f"{binning.n_scopes} optical scopes and {len(work_chunks)} work chunks "
        f"(max_samples={max_samples})",
        show=verbose > 0,
    )

    projection_start = time.perf_counter()
    result = sparse.csr_matrix((screen.N_pixel, n_vox))
    data_buf, row_buf, col_buf = [], [], []
    buffer_nbytes = 0
    result_buffer_limit = max(1, min(128 * 2 ** 20, max_working_memory // 4))

    def flush():
        nonlocal result, data_buf, row_buf, col_buf, buffer_nbytes
        if not data_buf:
            return
        block = sparse.coo_matrix(
            (np.concatenate(data_buf),
             (np.concatenate(row_buf), np.concatenate(col_buf))),
            shape=result.shape,
        ).tocsr()
        result += block
        data_buf, row_buf, col_buf = [], [], []
        buffer_nbytes = 0

    def project_voxels(owners, resolution, check_point_visibility):
        if owners.size == 0:
            return None
        points = voxel.get_sub_voxel_centers(owners, res=resolution)
        interpolator = voxel.sub_voxel_interpolator_from_centers(
            owners, res=resolution, points=points,
        )
        if check_point_visibility:
            mask = point_visibility(points).reshape(-1)
            mask &= inside_points(points)
            if not np.any(mask):
                return None
            points = points[mask]
            interpolator = interpolator[mask]
        subpixel_image = camera.calc_image_vec(
            eye_idx, points=points, verbose=0, check_visibility=False,
        )
        return (
            screen.transform_matrix @ (subpixel_image @ interpolator)
        ).tocoo()

    for chunk in my_tqdm(
            work_chunks, desc="Processing optical work chunks",
            disable=verbose <= 0):
        ordered_voxels = visible_voxels[chunk]
        is_full = chunk < full_voxels.size
        chunk_results = (
            project_voxels(ordered_voxels[is_full], full_resolution, False),
            project_voxels(ordered_voxels[~is_full], partial_resolution, True),
        )
        for projection_chunk in chunk_results:
            if projection_chunk is None:
                continue
            data_buf.append(projection_chunk.data)
            row_buf.append(projection_chunk.row)
            col_buf.append(projection_chunk.col)
            buffer_nbytes += (
                projection_chunk.data.nbytes + projection_chunk.row.nbytes
                + projection_chunk.col.nbytes
            )
        if buffer_nbytes >= result_buffer_limit:
            flush()
    flush()
    projection_elapsed = time.perf_counter() - projection_start
    my_print(
        f"Optical sparse-P timing: index={index_elapsed:.3f}s, "
        f"projection={projection_elapsed:.3f}s",
        show=verbose > 1,
    )
    return result
