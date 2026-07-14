"""Tests for optical projection bin and work-chunk bookkeeping."""

import numpy as np
from scipy import sparse

from multi_pinhole import Camera, Eye, Screen
from multi_pinhole.projection import (
    build_psf_group_matrix,
    build_projection_block,
    combine_projection_operators,
    factorize_psf_columns,
    make_optical_binning,
)


def _camera():
    return Camera(
        eyes=[Eye(position=(0.0, 0.0), focal_length=20.0,
                  eye_type="pinhole", eye_shape="circle", eye_size=1.0)],
        apertures=[],
        screen=Screen(screen_shape="rectangle", screen_size=(12.0, 12.0),
                      pixel_shape=(12, 12), subpixel_resolution=4),
        camera_position=(0.0, 0.0, 0.0),
    )


def _points():
    x, y = np.meshgrid(np.linspace(-15.0, 15.0, 9),
                       np.linspace(-12.0, 12.0, 7), indexing="ij")
    z = 120.0 + 20.0 * np.sin(x / 15.0)
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)


def test_optical_bin_width_uses_detector_pixel_pitch_not_subpixel_pitch():
    camera = _camera()
    points = _points()

    one_pixel = make_optical_binning(camera, 0, points, bin_width_pixels=1.0)
    one_subpixel = make_optical_binning(camera, 0, points, bin_width_pixels=0.25)
    two_pixels = make_optical_binning(camera, 0, points, bin_width_pixels=2.0)

    np.testing.assert_allclose(one_pixel.bin_width_uv, camera.screen.pixel_size)
    np.testing.assert_allclose(one_subpixel.bin_width_uv,
                               camera.screen.pixel_size / 4.0)
    assert one_subpixel.n_scopes >= one_pixel.n_scopes >= two_pixels.n_scopes


def test_work_chunks_cover_each_sample_once_without_splitting_scopes():
    points = _points()
    binning = make_optical_binning(
        _camera(), 0, points, bin_width_pixels=1.0, max_scope_samples=6,
    )
    chunks = binning.work_chunks(max_samples=15)

    np.testing.assert_array_equal(np.sort(np.concatenate(chunks)),
                                  np.arange(points.shape[0]))
    work_offsets = set(binning.work_offsets(max_samples=15).tolist())
    assert work_offsets.issubset(set(binning.scope_offsets.tolist()))
    assert max(scope.size for scope in binning.scopes()) <= 6


def test_scope_and_work_limits_use_expanded_sample_costs():
    camera = _camera()
    points = np.array([
        [-0.3, 0.0, 120.0], [-0.1, 0.0, 120.0],
        [0.1, 0.0, 120.0], [0.3, 0.0, 120.0],
    ])
    sample_costs = np.array([8, 8, 1, 1])
    binning = make_optical_binning(
        camera, 0, points, bin_width_pixels=2.0,
        max_scope_samples=9, sample_costs=sample_costs,
    )

    assert binning.scope_costs.sum() == sample_costs.sum()
    assert np.all(binning.scope_costs <= 9)
    work_offsets = binning.work_offsets(max_samples=9)
    assert set(work_offsets).issubset(set(binning.scope_offsets))


def test_invalid_optical_binning_parameters_are_rejected():
    camera = _camera()
    points = _points()
    for width in (0.0, -1.0, (1.0, 0.0), (1.0, 2.0, 3.0)):
        try:
            make_optical_binning(camera, 0, points, bin_width_pixels=width)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid bin width accepted: {width!r}")


def test_zero_tolerance_bypasses_all_psf_factorization_work():
    class MustNotBeInspected:
        pass

    assert factorize_psf_columns(
        MustNotBeInspected(), scope_offsets=None, tolerance=0.0,
    ) is None


def test_psf_factorization_preserves_sensitivity_and_scope_boundaries():
    # Columns 0 and 2 are identical but deliberately belong to different
    # optical scopes, so they must not share one representative.
    I = sparse.csr_matrix(np.array([
        [0.8, 0.78, 0.8, 0.1],
        [0.2, 0.22, 0.2, 0.9],
    ]))
    factorization = factorize_psf_columns(
        I, scope_offsets=np.array([0, 2, 4]), tolerance=0.1,
    )

    assert factorization is not None
    assert factorization.group_index.dtype == np.int32
    assert factorization.group_index[0] != factorization.group_index[2]
    assert factorization.group_max_relative_l2.max(initial=0.0) <= 0.1 + 1e-12
    # Identity S exposes the logical R through A without ever constructing R.
    A = build_psf_group_matrix(factorization, sparse.eye(I.shape[1], format="csr"))
    np.testing.assert_allclose(
        np.asarray((factorization.Q @ A).sum(axis=0)),
        np.asarray(I.sum(axis=0)), rtol=0.0, atol=1e-14,
    )


def test_factorization_uses_global_pixel_rows_after_local_dense_grouping():
    # The PSFs occupy non-consecutive global detector rows.  Local dense row
    # numbers must never be mistaken for the original detector indices in Q.
    I = sparse.csc_matrix((
        np.array([0.8, 0.2, 0.78, 0.22]),
        np.array([10, 90, 10, 90]),
        np.array([0, 2, 4]),
    ), shape=(100, 2))
    factorization = factorize_psf_columns(
        I, scope_offsets=np.array([0, 2]), tolerance=0.1,
    )

    assert factorization is not None
    assert factorization.n_groups == 1
    np.testing.assert_array_equal(factorization.Q.nonzero()[0], [10, 90])
    A = build_psf_group_matrix(factorization, sparse.eye(2, format="csr"))
    np.testing.assert_allclose((factorization.Q @ A).toarray().sum(axis=0), [1.0, 1.0])


def test_factorization_algorithm_and_dense_memory_split_keep_error_bound():
    I = sparse.csr_matrix(np.array([
        [0.8, 0.78, 0.2, 0.22],
        [0.2, 0.22, 0.8, 0.78],
    ]))
    for algorithm in ("recursive", "leader"):
        factorization = factorize_psf_columns(
            I, scope_offsets=np.array([0, 4]), tolerance=0.1,
            algorithm=algorithm, max_scope_dense_bytes=64,
        )
        assert factorization is not None
        assert factorization.group_max_relative_l2.max(initial=0.0) <= 0.1 + 1e-12


def test_group_matrix_uses_global_voxel_columns_and_ignores_zero_psfs():
    I = sparse.csr_matrix(np.array([
        [1.0, 0.0, 0.9],
        [0.0, 0.0, 0.1],
    ]))
    S = sparse.csr_matrix((
        np.array([0.25, 0.75, 1.0, 0.4, 0.6]),
        np.array([2, 7, 5, 2, 7]),
        np.array([0, 2, 3, 5]),
    ), shape=(3, 9))
    factorization = factorize_psf_columns(
        I, scope_offsets=np.array([0, 3]), tolerance=0.2,
    )
    assert factorization is not None
    assert factorization.group_index[1] == -1

    A = build_psf_group_matrix(factorization, S)
    approximation = factorization.Q @ A
    reference = I @ S
    np.testing.assert_allclose(
        np.asarray(approximation.sum(axis=0)),
        np.asarray(reference.sum(axis=0)), rtol=0.0, atol=1e-14,
    )
    np.testing.assert_array_equal(np.unique(A.nonzero()[1]), [2, 7])


def test_hybrid_operator_project_backproject_and_transpose_match_materialization():
    I = sparse.csr_matrix(np.array([
        [0.8, 0.78, 0.0],
        [0.2, 0.22, 0.0],
    ]))
    S = sparse.csr_matrix(np.array([
        [0.5, 0.5],
        [0.2, 0.8],
        [1.0, 0.0],
    ]))
    operator, stats = build_projection_block(
        I, S, tolerance=0.1, max_group_fraction=0.8,
    )
    assert stats.used_factorization
    assert stats.n_active_samples == 2
    assert stats.n_groups == 1

    materialized = operator.to_sparse()
    emission = np.array([1.5, -0.25])
    detector = np.array([0.7, -0.3])
    np.testing.assert_allclose(operator.project(emission), materialized @ emission)
    np.testing.assert_allclose(operator @ emission, materialized @ emission)
    np.testing.assert_allclose(operator.backproject(detector), materialized.T @ detector)
    np.testing.assert_allclose(operator.T @ detector, materialized.T @ detector)
    assert operator.T.T is operator


def test_group_fraction_selects_direct_without_building_both_storage_forms():
    I = sparse.eye(3, format="csr")
    S = sparse.csr_matrix(np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
    ]))
    operator, stats = build_projection_block(
        I, S, tolerance=0.01, max_group_fraction=0.8,
    )
    assert not stats.used_factorization
    assert stats.group_fraction == 1.0
    assert operator.Q.shape[1] == operator.A.shape[0] == 0
    np.testing.assert_allclose(operator.to_sparse().toarray(), (I @ S).toarray())


def test_projection_operators_combine_and_left_multiply_without_expanding_qa():
    I = sparse.csr_matrix(np.array([
        [0.8, 0.79],
        [0.2, 0.21],
    ]))
    S = sparse.eye(2, format="csr")
    factorized, _ = build_projection_block(
        I, S, tolerance=0.1, max_group_fraction=None,
    )
    direct, _ = build_projection_block(I, S, tolerance=0.0)
    combined = combine_projection_operators([factorized, direct])
    transform = sparse.csr_matrix([[0.25, 0.75]])
    transformed = combined.left_multiply(transform)

    assert transformed.shape == (1, 2)
    np.testing.assert_allclose(
        transformed.to_sparse().toarray(),
        (transform @ combined.to_sparse()).toarray(),
    )
