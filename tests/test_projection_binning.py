"""Tests for optical projection bin and work-chunk bookkeeping."""

import numpy as np

from multi_pinhole import Camera, Eye, Screen
from multi_pinhole.projection import make_optical_binning


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
