"""Run and plot a tiny voxel projection example.

This script is intentionally small enough to use as a smoke check:

    python examples/small_voxel_projection.py

It writes PNG files instead of opening GUI windows so it can also run under
pytest or on a headless machine.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multi_pinhole_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "multi_pinhole_cache"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from multi_pinhole import Aperture, Camera, Eye, Screen, Voxel, World


def build_small_world():
    """Create a compact world with one eye, one aperture, and 27 voxels."""
    voxel = Voxel.uniform_voxel(
        ranges=[[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]],
        shape=[3, 3, 3],
    )
    camera = Camera(
        eyes=[Eye(eye_type="pinhole", eye_shape="circle", eye_size=1.0, focal_length=12.0, position=[0.0, 0.0])],
        screen=Screen(screen_shape="rectangle", screen_size=[12.0, 12.0], pixel_shape=(8, 8), subpixel_resolution=2),
        apertures=Aperture(shape="circle", size=6.0, position=[0.0, 0.0, 25.0], resolution=24, max_size=24.0),
        camera_position=[0.0, 0.0, -60.0],
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def sample_emission(voxel):
    """Evaluate a smooth source profile on voxel centers."""
    x, y, z = voxel.gravity_center.T
    return np.exp(-((x / 2.2) ** 2 + (y / 1.8) ** 2 + (z / 2.6) ** 2))


def draw_geometry(world, output_path):
    """Draw the optical setup, voxel centers, and representative rays."""
    camera = world.cameras[0]
    voxel = world.voxel

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    camera.draw_optical_system(ax=ax, X_lim=(-8, 8), Y_lim=(-8, 8), Z_lim=(0, 80))

    centers_camera = camera.world2camera(voxel.gravity_center)
    values = sample_emission(voxel)
    ax.scatter(
        centers_camera[:, 2],
        centers_camera[:, 0],
        centers_camera[:, 1],
        c=values,
        cmap="viridis",
        s=35,
        depthshade=False,
        label="Voxel centers",
    )

    eye = camera.eyes[0]
    for point in centers_camera[:: max(1, len(centers_camera) // 8)]:
        ax.plot(
            [eye.position[2], point[2]],
            [eye.position[0], point[0]],
            [eye.position[1], point[1]],
            color="tab:red",
            alpha=0.35,
            linewidth=0.8,
        )

    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_projection_result(world, emission, output_path):
    """Draw per-eye image, aggregated image, and source values."""
    camera = world.cameras[0]
    eye_image = world.projection[0][0] @ emission
    pixel_image = world.P_matrix[0] @ emission

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    camera.screen.show_image(eye_image, ax=axes[0], colorbar=True)
    axes[0].set_title("Eye 0 pixels")
    camera.screen.show_image(pixel_image, ax=axes[1], colorbar=True)
    axes[1].set_title("All-eye pixels")

    centers = world.voxel.gravity_center
    sc = axes[2].scatter(centers[:, 0], centers[:, 1], c=emission, s=80, cmap="viridis")
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("Emission")
    fig.colorbar(sc, ax=axes[2])

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return eye_image, pixel_image


def run(output_dir=None):
    """Run the full tiny projection workflow and save figures."""
    output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "multi_pinhole_small_voxel"
    output_dir.mkdir(parents=True, exist_ok=True)

    world = build_small_world()
    world.set_projection_matrix(res=1, verbose=0, parallel=1)
    emission = sample_emission(world.voxel)

    geometry_path = output_dir / "small_voxel_geometry.png"
    projection_path = output_dir / "small_voxel_projection.png"
    draw_geometry(world, geometry_path)
    eye_image, pixel_image = draw_projection_result(world, emission, projection_path)

    return {
        "world": world,
        "emission": emission,
        "eye_image": eye_image,
        "pixel_image": pixel_image,
        "geometry_path": geometry_path,
        "projection_path": projection_path,
    }


if __name__ == "__main__":
    result = run()
    print(f"geometry: {result['geometry_path']}")
    print(f"projection: {result['projection_path']}")
    print(f"P shape: {result['world'].P_matrix[0].shape}, nnz: {result['world'].P_matrix[0].nnz}")
