"""Tests for optical projection bin and work-chunk bookkeeping."""

import numpy as np

from multi_pinhole import Camera, Eye, Screen
from multi_pinhole.projection import (
    make_optical_binning,
    projected_axis_spans,
    select_circumsphere_resolution,
    select_source_resolution,
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


def test_projected_axis_spans_matches_exact_on_axis_pinhole_geometry():
    camera = _camera()
    centers = np.array([[0.0, 0.0, 120.0]])
    edge_lengths = np.array([[12.0, 8.0, 20.0]])

    spans = projected_axis_spans(camera, 0, centers, edge_lengths)

    # The Eye is at camera z=f=20, hence source depth from the Eye is 100.
    # screen.xy2uv swaps x/y, so world x maps to detector v and world y to u.
    np.testing.assert_allclose(spans[0, 0], [0.0, 20.0 * 12.0 / 100.0])
    np.testing.assert_allclose(spans[0, 1], [20.0 * 8.0 / 100.0, 0.0])
    np.testing.assert_allclose(spans[0, 2], [0.0, 0.0])


def test_projected_depth_axis_span_includes_off_axis_perspective():
    camera = _camera()
    centers = np.array([[10.0, 0.0, 120.0]])
    edge_lengths = np.array([[2.0, 2.0, 20.0]])

    spans = projected_axis_spans(camera, 0, centers, edge_lengths)

    near_depth, far_depth = 90.0, 110.0
    expected = 20.0 * 10.0 * abs(1.0 / near_depth - 1.0 / far_depth)
    np.testing.assert_allclose(spans[0, 2], [0.0, expected])


def test_projected_axis_spans_respects_camera_rotation():
    camera = _camera()
    camera.set_rotation_euler("z", 90.0)
    centers = np.array([[0.0, 0.0, 120.0]])
    edge_lengths = np.array([[12.0, 8.0, 20.0]])

    spans = projected_axis_spans(camera, 0, centers, edge_lengths)

    # A 90-degree camera rotation exchanges which detector axis receives the
    # world-x and world-y source chords.
    np.testing.assert_allclose(spans[0, 0], [20.0 * 12.0 / 100.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(spans[0, 1], [0.0, 20.0 * 8.0 / 100.0], atol=1e-15)


def test_projected_axis_spans_rejects_invalid_inputs():
    camera = _camera()
    with np.testing.assert_raises(ValueError):
        projected_axis_spans(camera, 0, np.zeros((2, 2)), np.ones((2, 3)))
    with np.testing.assert_raises(ValueError):
        projected_axis_spans(camera, 0, np.zeros((2, 3)), np.ones((3, 3)))
    with np.testing.assert_raises(ValueError):
        projected_axis_spans(camera, 0, np.zeros((2, 3)), np.zeros((2, 3)))


def test_select_source_resolution_is_axiswise_and_reports_ceiling():
    spans = np.array([[[0.2, 0.4], [1.1, 0.2], [4.1, 0.0]]])

    estimate = select_source_resolution(
        spans, detector_pitch=(1.0, 1.0), max_resolution=(4, 2, 3),
    )

    np.testing.assert_array_equal(estimate.resolution, [[1, 2, 3]])
    np.testing.assert_allclose(estimate.projected_span_cells, [[0.4, 1.1, 4.1]])
    np.testing.assert_allclose(estimate.uncapped_resolution, [[1.0, 2.0, 5.0]])
    np.testing.assert_array_equal(estimate.capped, [[False, False, True]])


def test_select_source_resolution_tracks_detector_subpixel_pitch():
    spans = np.array([[[0.6, 0.0], [0.0, 0.6], [0.0, 0.0]]])

    pixel_estimate = select_source_resolution(spans, detector_pitch=1.0,
                                              max_resolution=8)
    subpixel_estimate = select_source_resolution(spans, detector_pitch=0.25,
                                                 max_resolution=8)

    np.testing.assert_array_equal(pixel_estimate.resolution, [[1, 1, 1]])
    np.testing.assert_array_equal(subpixel_estimate.resolution, [[3, 3, 1]])


def test_select_source_resolution_accepts_per_cell_psf_scales():
    spans = np.array([
        [[0.6, 0.0], [0.0, 0.6], [0.0, 0.0]],
        [[0.6, 0.0], [0.0, 0.6], [0.0, 0.0]],
    ])
    scales = np.array([[0.25, 0.25], [1.0, 1.0]])

    estimate = select_source_resolution(
        spans, detector_pitch=scales, max_resolution=8,
    )

    np.testing.assert_array_equal(estimate.resolution,
                                  [[3, 3, 1], [1, 1, 1]])


def test_select_source_resolution_honors_stricter_step_fraction():
    spans = np.array([[[0.6, 0.0], [0.0, 0.6], [0.0, 0.0]]])

    estimate = select_source_resolution(
        spans, detector_pitch=1.0, max_resolution=8,
        max_projected_step=0.25,
    )

    np.testing.assert_array_equal(estimate.resolution, [[3, 3, 1]])


def test_select_source_resolution_uses_ceiling_for_nonfinite_geometry():
    spans = np.zeros((1, 3, 2))
    spans[0, 1] = np.nan

    estimate = select_source_resolution(spans, detector_pitch=1.0,
                                        max_resolution=(2, 3, 4))

    np.testing.assert_array_equal(estimate.resolution, [[1, 3, 1]])
    np.testing.assert_array_equal(estimate.capped, [[False, True, False]])


def test_circumsphere_resolution_uses_one_eighth_threshold_and_ideal_res():
    points = np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 100.0]])
    edges = np.ones((2, 3))

    estimate = select_circumsphere_resolution(
        points, edges, focal_length=20.0, reference_size=1.0,
        fallback_resolution=8,
    )

    np.testing.assert_allclose(
        estimate.ratio, 20.0 * np.sqrt(3.0) / points[:, 2],
    )
    np.testing.assert_array_equal(estimate.point_source, [True, False])
    np.testing.assert_array_equal(estimate.resolution, [[1, 1, 1], [3, 3, 3]])
    np.testing.assert_array_equal(estimate.ideal_resolution,
                                  [[1, 1, 1], [3, 3, 3]])


def test_circumsphere_resolution_includes_off_axis_factor_and_cubic_subcells():
    points = np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]])
    edges = np.array([[10.0, 10.0, 2.0], [10.0, 10.0, 2.0]])

    estimate = select_circumsphere_resolution(
        points, edges, focal_length=20.0, reference_size=1.0,
        fallback_resolution=32,
    )

    np.testing.assert_allclose(estimate.ratio[1] / estimate.ratio[0], np.sqrt(2.0))
    np.testing.assert_array_equal(estimate.resolution[0], [28, 28, 6])
    subcell_edges = edges[0] / estimate.resolution[0]
    assert np.ptp(subcell_edges) < 0.05


def test_circumsphere_resolution_falls_back_for_invalid_or_capped_geometry():
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 100.0]])
    edges = np.array([[2.0, 2.0, 2.0], [10.0, 10.0, 10.0]])

    estimate = select_circumsphere_resolution(
        points, edges, focal_length=20.0, reference_size=1.0,
        fallback_resolution=(2, 3, 4),
    )

    np.testing.assert_array_equal(estimate.valid, [False, True])
    np.testing.assert_array_equal(estimate.resolution, [[2, 3, 4], [2, 3, 4]])
    assert np.all(estimate.capped)


def test_circumsphere_resolution_can_return_uncapped_ideal_resolution():
    estimate = select_circumsphere_resolution(
        np.array([[0.0, 0.0, 100.0]]), np.ones((1, 3)),
        focal_length=20.0, reference_size=1.0,
        fallback_resolution=None,
    )

    np.testing.assert_array_equal(estimate.resolution, [[3, 3, 3]])
    assert not np.any(estimate.capped)

    with np.testing.assert_raises_regex(ValueError, "undefined for invalid geometry"):
        select_circumsphere_resolution(
            np.array([[0.0, 0.0, 1.0]]), np.full((1, 3), 2.0),
            focal_length=20.0, reference_size=1.0,
            fallback_resolution=None,
        )
