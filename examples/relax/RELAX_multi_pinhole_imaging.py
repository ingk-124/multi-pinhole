"""RELAX two-camera multi-pinhole projection example."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stl import mesh

import multi_pinhole
from multi_pinhole import Aperture, Camera, Eye, Screen, Voxel, World


def build_relax_world(voxel_shape=(12, 12, 6)):
    """Build a reduced RELAX scene using the packaged vessel mesh."""
    aperture_1 = Aperture(shape="circle", size=10, position=[0, 0, 80]).set_model(
        resolution=40,
        max_size=200,
    )
    aperture_2 = Aperture(shape="circle", size=10, position=[0, 0, 80]).set_model(
        resolution=40,
        max_size=200,
    )
    eyes_1 = [
        Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.25,
            focal_length=20, position=[0, 0]),
        Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.25,
            focal_length=20, position=[-5, 5]),
    ]
    eyes_2 = [
        Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.25,
            focal_length=20, position=[0, 0]),
        Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.25,
            focal_length=20, position=[-5, 5]),
    ]
    cameras = {
        "upper": Camera(
            eyes=eyes_1,
            screen=Screen(
                screen_shape="rectangle",
                screen_size=[15, 30],
                pixel_shape=(25, 50),
                subpixel_resolution=3,
            ),
            apertures=[aperture_1],
            camera_position=[670, 670, 0],
        ).set_rotation_euler("xyz", (-90, 45, 180), degrees=True),
        "lower": Camera(
            eyes=eyes_2,
            screen=Screen(
                screen_shape="circle",
                screen_size=[17, 17],
                pixel_shape=(32, 32),
                subpixel_resolution=3,
            ),
            apertures=[aperture_2],
            camera_position=[670, -500, 0],
        ).set_rotation_euler("xz", (90, -90), degrees=True),
    }

    voxel = Voxel.uniform_voxel_from_centers(
        ranges=[[-800, 800], [-800, 800], [-300, 300]],
        shape=voxel_shape,
        coordinate_type="torus",
        coordinate_parameters={"major_radius": 500, "minor_radius": 250},
    )
    vessel_path = Path(multi_pinhole.__file__).resolve().parent / "data" / "relax_rotated.stl"
    world = World(
        voxel=voxel,
        cameras=cameras,
        walls=mesh.Mesh.from_file(vessel_path),
        verbose=0,
    )

    def inside_plasma(x, y, z):
        major_radius = 500
        minor_radius = 250
        return (np.sqrt(x ** 2 + y ** 2) - major_radius) ** 2 + z ** 2 <= minor_radius ** 2

    world.set_inside_vertices(inside_plasma)
    return world


def run_projection(world, resolution=1, parallel=1, plot=True):
    """Build projections and display a toroidal test emission."""
    world.set_projection_matrix(
        res=resolution,
        verbose=1,
        parallel=parallel,
    )
    r, _, _ = world.voxel.normalized_coordinates().T
    emission = np.clip(1 - r ** 2, 0, None)

    images = {
        camera_key: world.project(emission, camera_idx=camera_key)
        for camera_key in world.cameras
    }
    if not plot:
        return images

    fig, axes = plt.subplots(1, len(world.cameras), figsize=(9, 4))
    for ax, (camera_key, image) in zip(np.atleast_1d(axes), images.items()):
        world.cameras[camera_key].screen.show_image(image, ax=ax, pixel_image=True)
        ax.set_title(camera_key)
    fig.tight_layout()
    return fig


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=1,
                        help="sub-voxel projection resolution (default: 1)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="projection worker count (default: 1)")
    parser.add_argument("--no-show", action="store_true",
                        help="build the projection without opening figures")
    args = parser.parse_args()

    world = build_relax_world()
    run_projection(
        world,
        resolution=args.resolution,
        parallel=args.parallel,
        plot=not args.no_show,
    )
    if not args.no_show:
        world.draw_camera_orientation(show_fig=True, elev=60, azim=-30, alpha=0.1)
        plt.show()


if __name__ == "__main__":
    main()
