"""Run release smoke checks against an explicitly installed wheel directory.

Usage:
    python -I tests/wheel_smoke.py /path/to/isolated/site-packages
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: wheel_smoke.py INSTALLED_PACKAGE_ROOT")
    package_root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(package_root))

    import dill
    import numpy as np

    import multi_pinhole
    from multi_pinhole import Camera, Eye, Screen, World

    config = {
        "schema": "multi-pinhole/world-config",
        "schema_version": 1,
        "units": {"length": "mm", "angle": "rad"},
        "voxel": {
            "type": "uniform",
            "ranges": [[-0.5, 0.5], [-0.5, 0.5], [20.0, 21.0]],
            "shape": [1, 1, 1],
            "coordinate": {
                "type": "cartesian",
                "parameters": {"width": 1.0, "depth": 1.0, "height": 1.0},
            },
            "sub_voxel_resolution": [1, 1, 1],
        },
        "cameras": [],
        "walls": [],
        "inside": {"type": "all", "parameters": {}},
        "verbose": 0,
    }
    world = World.from_config(config)
    camera = Camera(
        eyes=[Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=1.0)],
        apertures=[],
        screen=Screen("square", 100.0, pixel_shape=(1, 1), subpixel_resolution=1),
        camera_position=(0.0, 0.0, 0.0),
    )
    world.add_camera("main", camera)
    world.set_projection_matrix(res=1, parallel=1, verbose=0)
    expected = world.project(np.ones(world.voxel.N), "main")
    assert expected.shape == (1,)
    assert expected[0] > 0.0

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        archive = directory / "world.mpw"
        legacy = directory / "legacy.pkl"
        world.save(archive)
        restored = World.load(archive)
        np.testing.assert_allclose(restored.project(np.ones(1), "main"), expected)
        with legacy.open("wb") as file:
            dill.dump(world, file)
        migrated = World.load(legacy)
        np.testing.assert_allclose(migrated.project(np.ones(1), "main"), expected)

    assert multi_pinhole.__version__ == "1.0.0"
    print("wheel smoke passed")


if __name__ == "__main__":
    main()
