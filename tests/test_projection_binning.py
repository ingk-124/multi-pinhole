"""Tests for optical projection bin and work-chunk bookkeeping."""

import numpy as np
from scipy import sparse

from multi_pinhole import Camera, Eye, Screen
from multi_pinhole.projection import factorize_psf_columns, make_optical_binning


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
    assert factorization.group_index[0] != factorization.group_index[2]
    assert factorization.group_max_relative_l2.max(initial=0.0) <= 0.1 + 1e-12
    np.testing.assert_allclose(
        np.asarray((factorization.Q @ factorization.R).sum(axis=0)),
        np.asarray(I.sum(axis=0)), rtol=0.0, atol=1e-14,
    )
