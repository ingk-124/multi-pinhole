import numpy as np
import pytest

from multi_pinhole.utils import stl_utils


def _random_mesh_and_points(seed=124):
    rng = np.random.default_rng(seed)
    vertices = rng.uniform(-3.0, 3.0, size=(60, 3))
    vertices[:, 2] += 5.0
    faces = np.arange(60).reshape(-1, 3)
    model = stl_utils.make_stl(vertices, faces)
    points = rng.uniform(-6.0, 6.0, size=(240, 3))
    points[:, 2] += 8.0
    return model, points.astype(np.float32)


@pytest.mark.parametrize("behind_start_included", [False, True, -2.0])
@pytest.mark.parametrize("batch_points,batch_triangles", [(17, 3), (64, 11), (1000, 1000)])
def test_batched_visibility_matches_reference(behind_start_included, batch_points, batch_triangles):
    model, points = _random_mesh_and_points()
    start = np.array([0.25, -0.5, 0.75], dtype=np.float32)

    expected = stl_utils._check_visible_reference(
        model, start, points, behind_start_included=behind_start_included,
        batch_points=batch_points,
    )
    actual = stl_utils.check_visible(
        model, start, points, behind_start_included=behind_start_included,
        batch_points=batch_points, batch_triangles=batch_triangles,
    )

    np.testing.assert_array_equal(actual, expected)


def test_batched_visibility_skips_points_occluded_by_an_earlier_triangle_batch(monkeypatch):
    vertices = np.array([
        [-5.0, -5.0, 2.0], [5.0, -5.0, 2.0], [0.0, 5.0, 2.0],
        [-5.0, -5.0, 4.0], [5.0, -5.0, 4.0], [0.0, 5.0, 4.0],
    ])
    model = stl_utils.make_stl(vertices, np.array([[0, 1, 2], [3, 4, 5]]))
    points = np.array([[0.0, 0.0, 6.0], [0.5, 0.0, 6.0]], dtype=np.float32)
    candidate_counts = []
    original = stl_utils._check_intersection_pairs

    def recorded(*args, **kwargs):
        candidate_counts.append(len(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(stl_utils, "_check_intersection_pairs", recorded)
    visible = stl_utils.check_visible(
        model, np.zeros(3, dtype=np.float32), points, batch_triangles=1,
    )

    np.testing.assert_array_equal(visible, [False, False])
    assert candidate_counts == [2]


@pytest.mark.parametrize("behind_start_included", [False, True, -2.0])
def test_paired_intersections_match_scalar_at_triangle_boundaries(behind_start_included):
    triangle = np.array([
        [-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0],
    ], dtype=np.float32)
    points = np.array([
        [-2.0, -2.0, 4.0],  # through a vertex
        [2.0, -2.0, 4.0],   # through a vertex
        [0.0, 2.0, 4.0],    # through a vertex
        [0.0, -2.0, 4.0],   # through an edge
        [0.0, 0.0, 4.0],    # through the interior
        [3.0, 0.0, 4.0],    # outside
    ], dtype=np.float32)
    triangles = np.broadcast_to(triangle, (len(points), 3, 3))
    start = np.zeros(3, dtype=np.float32)

    expected = stl_utils.check_intersection(
        triangle, start, points,
        behind_start_included=behind_start_included,
    )
    actual = stl_utils._check_intersection_pairs(
        triangles, start, points,
        behind_start_included=behind_start_included,
    )

    np.testing.assert_array_equal(actual, expected)


def test_wall_visibility_discards_triangles_outside_segment_z_range(monkeypatch):
    vertices = np.array([
        [-2.0, -2.0, -3.0], [2.0, -2.0, -3.0], [0.0, 2.0, -3.0],
        [-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [0.0, 2.0, 2.0],
        [-2.0, -2.0, 9.0], [2.0, -2.0, 9.0], [0.0, 2.0, 9.0],
    ])
    model = stl_utils.make_stl(vertices, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    points = np.array([[0.0, 0.0, 6.0], [3.0, 0.0, 6.0]], dtype=np.float32)
    start = np.zeros(3, dtype=np.float32)
    prepared_triangle_counts = []
    original = stl_utils.delta_cone_prepare

    def recorded(triangles, *args, **kwargs):
        prepared_triangle_counts.append(len(triangles))
        return original(triangles, *args, **kwargs)

    monkeypatch.setattr(stl_utils, "delta_cone_prepare", recorded)
    actual = stl_utils.check_visible(model, start, points)
    expected = stl_utils._check_visible_reference(model, start, points)

    np.testing.assert_array_equal(actual, expected)
    assert prepared_triangle_counts[0] == 1
