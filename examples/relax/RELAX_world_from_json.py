"""Load and optionally project the RELAX equatorial-camera JSON scene."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle

from multi_pinhole import World


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "relax_world.json"
CAMERA_KEY = "phi=+45deg"


def load_relax_world() -> World:
    """Load the RELAX scene declared in :data:`CONFIG_PATH`."""
    return World.from_config(CONFIG_PATH)


def toroidal_emission(world: World) -> np.ndarray:
    """Return a parabolic emission profile confined to the configured torus."""
    radius, _, _ = world.voxel.normalized_coordinates().T
    return np.where(radius <= 1.0, 1.0 - radius**2, 0.0)


def run_projection(world: World, *, resolution: int = 1, parallel: int = 1):
    """Build and apply the camera projection matrix."""
    world.set_projection_matrix(
        res=resolution,
        res_mode="fixed",
        parallel=parallel,
        verbose=1,
    )
    return world.project(toroidal_emission(world), camera_idx=CAMERA_KEY)


def plot_equatorial_visibility(world: World):
    """Plot camera visibility states for voxel centers on the ``Z=0`` plane."""
    world.find_visible_voxels(verbose=1)
    states = np.max(world.visible_voxels[CAMERA_KEY], axis=0)
    centers = world.voxel.gravity_center
    z_values = np.unique(centers[:, 2])
    equatorial_z = z_values[np.argmin(np.abs(z_values))]
    normalized_radius = world.voxel.normalized_coordinates()[:, 0]
    mask = np.isclose(centers[:, 2], equatorial_z) & (normalized_radius <= 1.0)

    x_values = np.unique(centers[:, 0])
    y_values = np.unique(centers[:, 1])
    image = np.full((y_values.size, x_values.size), np.nan)
    x_index = np.searchsorted(x_values, centers[mask, 0])
    y_index = np.searchsorted(y_values, centers[mask, 1])
    image[y_index, x_index] = states[mask]

    colors = ListedColormap(["#eeeeee", "#f6a340", "#1764ab"])
    colors.set_bad("white")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], colors.N)
    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(
        world.voxel.x_axis,
        world.voxel.y_axis,
        image,
        cmap=colors,
        norm=norm,
        shading="flat",
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            508.0 - 250.0,
            fill=False,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            508.0 + 250.0,
            fill=False,
            color="black",
            linewidth=1.0,
        )
    )
    camera = world.cameras[CAMERA_KEY]
    ax.plot(*camera.camera_position[:2], marker="^", color="crimson", markersize=9)
    ax.quiver(
        *camera.camera_position[:2],
        *camera.camera_z[:2],
        angles="xy",
        scale_units="xy",
        scale=0.004,
        color="crimson",
        width=0.005,
    )
    colorbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2], pad=0.02)
    colorbar.ax.set_yticklabels(["not visible", "partial", "fully visible"])
    ax.set(
        xlabel="x [mm]",
        ylabel="y [mm]",
        title=f"RELAX plasma visible voxels at Z={equatorial_z:g} mm",
        aspect="equal",
    )
    fig.tight_layout()
    return fig, ax


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        action="store_true",
        help="also build the projection matrix and display its image",
    )
    parser.add_argument(
        "--visibility",
        action="store_true",
        help="plot visible voxels on the equatorial plane",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="save the requested plot instead of only displaying it",
    )
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    world = load_relax_world()
    camera = world.cameras[CAMERA_KEY]
    print(f"Loaded {CONFIG_PATH.name}")
    print(f"Voxel coordinate parameters: {world.voxel.coordinate_parameters}")
    print(f"Camera position [mm]: {camera.camera_position}")
    print(f"Camera look direction: {camera.camera_z}")

    figure = None
    if args.visibility:
        figure, _ = plot_equatorial_visibility(world)

    if args.project:
        image = run_projection(
            world,
            resolution=args.resolution,
            parallel=args.parallel,
        )
        if not args.no_show:
            figure, ax = plt.subplots(figsize=(5, 4))
            camera.screen.show_image(image, ax=ax, pixel_image=True)
            ax.set_title(CAMERA_KEY)

    if args.save is not None:
        if figure is None:
            parser.error("--save requires --visibility or --project")
        figure.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"Saved {args.save}")
    if figure is not None and not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
