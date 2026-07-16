import numpy as np
from scipy import sparse

from multi_pinhole import Aperture, Camera, Eye, Screen, Voxel, World
from multi_pinhole import _projection_matrix
from multi_pinhole import world as world_module


def test_sum_eye_projections_has_fixed_values_and_csr_format():
    first = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    second = sparse.coo_matrix(np.array([[0.0, 3.0], [4.0, 0.0]]))

    actual = _projection_matrix.sum_eye_projections(
        [first, second], shape=(2, 2),
    )

    np.testing.assert_array_equal(actual.toarray(), [[1.0, 3.0], [4.0, 2.0]])
    assert sparse.isspmatrix_csr(actual)
    assert actual.dtype == np.dtype(float)


def test_optical_builder_preserves_world_binning_monkeypatch(monkeypatch):
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=1.0)
    screen = Screen(
        screen_shape="square", screen_size=12.0,
        pixel_shape=(6, 6), subpixel_resolution=1,
    )
    aperture = Aperture(shape="circle", size=20.0, position=(0.0, 0.0, 5.0))
    camera = Camera(
        eyes=[eye], apertures=aperture, screen=screen,
        camera_position=(0.0, 0.0, 0.0),
    )
    voxel = Voxel.uniform_voxel(
        ranges=((-1.0, 1.0), (-1.0, 1.0), (30.0, 32.0)),
        shape=(2, 2, 2),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    original = world_module.make_optical_binning
    calls = []

    def recorded(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(world_module, "make_optical_binning", recorded)
    world.set_projection_matrix(
        res=1, verbose=0, parallel=1, chunk_strategy="optical",
    )

    assert len(calls) == 1
    assert sparse.isspmatrix_csr(world.projection[0][0])
    assert world.projection[0][0].shape == (screen.N_pixel, voxel.N)


def test_projection_builder_api_has_no_world_or_cache_parameter():
    names = _projection_matrix.build_optical_projection_matrix.__annotations__
    assert "world" not in names
    assert "cache" not in names
