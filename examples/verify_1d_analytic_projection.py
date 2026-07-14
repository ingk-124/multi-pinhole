"""Compare a wall-free 1-D voxel projection with a pinhole reference model.

The source is a thin slab at a fixed distance from a small circular pinhole.
Only its x coordinate varies, so summing the computed image over pixel rows
gives a one-dimensional image along the screen v axis.

Internally ``Screen`` assigns detector samples the familiar
``dA_screen * cos(theta)**4 / (4*pi)`` weight.  After integrating the projected
pinhole footprint, the equivalent point-pinhole source-side factor is

    A_pinhole * cos(theta)**3 / (4*pi*Z**2),

where Z is the axial source-to-pinhole distance.  The reference below
integrates that expression over the source interval seen by each pixel.  The
finite pinhole and the thin (rather than zero-thickness) slab are the only
geometric approximations in this comparison.

Run from the repository root with, for example::

    python examples/verify_1d_analytic_projection.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multi_pinhole_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "multi_pinhole_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from multi_pinhole import Camera, Eye, Screen, Voxel, World


SOURCE_X_BOUNDS = (-20.0, 20.0)
SOURCE_Y_BOUNDS = (-0.1, 0.1)
SOURCE_Z_BOUNDS = (99.9, 100.1)
FOCAL_LENGTH = 20.0
EYE_DIAMETER = 0.08
SCREEN_SIZE = (0.45, 12.0)
PIXEL_SHAPE = (3, 81)


def profiles(x: np.ndarray | float) -> dict[str, np.ndarray | float]:
    """Return the four center-sampled emission profiles under test."""
    x = np.asarray(x)
    half_width = 0.5 * (SOURCE_X_BOUNDS[1] - SOURCE_X_BOUNDS[0])
    return {
        "constant": np.ones_like(x, dtype=float),
        "linear": 1.0 + 0.6 * x / half_width,
        "square": (x / half_width) ** 2,
        "gaussian": np.exp(-0.5 * (x / 6.0) ** 2),
    }


def build_world(voxel_count: int = 81, subpixel_resolution: int = 24) -> World:
    """Construct the thin 1-D source and wall/aperture-free pinhole camera."""
    voxel = Voxel.uniform_voxel(
        ranges=(SOURCE_X_BOUNDS, SOURCE_Y_BOUNDS, SOURCE_Z_BOUNDS),
        shape=(voxel_count, 1, 1),
    )
    eye = Eye(
        position=(0.0, 0.0),
        focal_length=FOCAL_LENGTH,
        eye_type="pinhole",
        eye_shape="circle",
        eye_size=EYE_DIAMETER,
    )
    screen = Screen(
        screen_shape="rectangle",
        screen_size=SCREEN_SIZE,
        pixel_shape=PIXEL_SHAPE,
        subpixel_resolution=subpixel_resolution,
    )
    camera = Camera(
        eyes=[eye],
        apertures=[],
        screen=screen,
        camera_position=(0.0, 0.0, 0.0),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def analytic_pixel_image(world: World, profile) -> np.ndarray:
    """Integrate the point-pinhole reference over each pixel's source interval."""
    camera = world.cameras[0]
    screen = camera.screen
    eye = camera.eyes[0]
    source_z = 0.5 * sum(SOURCE_Z_BOUNDS)
    axial_distance = source_z - eye.position[2]
    transverse_area = ((SOURCE_Y_BOUNDS[1] - SOURCE_Y_BOUNDS[0])
                       * (SOURCE_Z_BOUNDS[1] - SOURCE_Z_BOUNDS[0]))
    pinhole_area = np.pi * (EYE_DIAMETER / 2.0) ** 2
    v_edges = np.linspace(0.0, screen.screen_size[1], screen.pixel_shape[1] + 1)
    screen_center = screen.screen_size[1] / 2.0

    def integrand(x):
        cosine = axial_distance / np.sqrt(axial_distance ** 2 + x ** 2)
        solid_angle = pinhole_area * cosine ** 3 / (4.0 * np.pi * axial_distance ** 2)
        return float(profile(x)) * transverse_area * solid_angle

    line = np.zeros(screen.pixel_shape[1], dtype=float)
    for j, (v0, v1) in enumerate(zip(v_edges[:-1], v_edges[1:])):
        # Eye.calc_rays uses v = v_center - f*x/Z, hence the reversed limits.
        x0 = -(v1 - screen_center) * axial_distance / FOCAL_LENGTH
        x1 = -(v0 - screen_center) * axial_distance / FOCAL_LENGTH
        x0 = max(x0, SOURCE_X_BOUNDS[0])
        x1 = min(x1, SOURCE_X_BOUNDS[1])
        if x1 > x0:
            line[j] = quad(integrand, x0, x1, epsabs=1e-18, epsrel=1e-11)[0]
    # The source is wholly captured in u, so the row distribution is not part
    # of the 1-D reference.  Return a line image for direct comparison.
    return line


def relative_metrics(computed: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return scale-sensitive image error metrics."""
    reference_norm = np.linalg.norm(reference)
    reference_flux = reference.sum()
    return {
        "relative_l2": float(np.linalg.norm(computed - reference) / reference_norm),
        "relative_flux": float(abs(computed.sum() - reference_flux) / abs(reference_flux)),
        "correlation": float(np.corrcoef(computed, reference)[0, 1]),
    }


def run_validation(output_path: Path, voxel_count: int = 81, voxel_res: int = 8,
                   subpixel_resolution: int = 24, parallel: int = 1):
    """Compute P, compare four profiles, and save a diagnostic pyplot figure."""
    world = build_world(voxel_count=voxel_count,
                        subpixel_resolution=subpixel_resolution)
    world.set_projection_matrix(
        res=(voxel_res, 1, 1),
        partial_res=(voxel_res, 1, 1),
        verbose=0,
        parallel=parallel,
        force=True,
    )
    projection = world.P_matrix[0]
    x_centers = world.voxel.gravity_center[:, 0]
    profile_values = profiles(x_centers)
    screen = world.cameras[0].screen
    v_centers = screen.pixel_position.reshape(*screen.pixel_shape, 2)[0, :, 1]

    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex="col")
    metrics = {}
    for row, (name, emission) in enumerate(profile_values.items()):
        pixel_image = np.asarray(projection @ emission).reshape(screen.pixel_shape)
        computed = pixel_image.sum(axis=0)
        reference = analytic_pixel_image(world, lambda x, key=name: profiles(x)[key])
        metrics[name] = relative_metrics(computed, reference)

        axes[row, 0].plot(v_centers, reference, "k-", lw=2, label="point-pinhole analytic")
        axes[row, 0].plot(v_centers, computed, "o-", ms=3, lw=1, label="P @ f(x_gc)")
        axes[row, 0].set_ylabel(f"{name}\nsignal / pixel")
        axes[row, 0].grid(alpha=0.25)
        if row == 0:
            axes[row, 0].legend()

        scale = max(reference.max(), np.finfo(float).tiny)
        axes[row, 1].plot(v_centers, (computed - reference) / scale, color="tab:red")
        axes[row, 1].axhline(0.0, color="k", lw=0.8)
        axes[row, 1].set_ylabel("difference / max(ref.)")
        axes[row, 1].grid(alpha=0.25)
        axes[row, 1].text(
            0.02, 0.95,
            f"rel. L2 = {metrics[name]['relative_l2']:.3e}\n"
            f"rel. flux = {metrics[name]['relative_flux']:.3e}\n"
            f"corr. = {metrics[name]['correlation']:.6f}",
            transform=axes[row, 1].transAxes, va="top",
        )

    axes[-1, 0].set_xlabel("screen v [mm]")
    axes[-1, 1].set_xlabel("screen v [mm]")
    fig.suptitle(
        "Wall-free 1-D projection: finite-pinhole P versus point-pinhole reference\n"
        f"voxels={voxel_count}, voxel res=({voxel_res},1,1), "
        f"eye diameter={EYE_DIAMETER} mm, subpixel res={subpixel_resolution}"
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {"output_path": output_path, "metrics": metrics, "world": world}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("examples/output/verify_1d_analytic_projection.png"))
    parser.add_argument("--voxels", type=int, default=81)
    parser.add_argument("--voxel-res", type=int, default=8)
    parser.add_argument("--subpixel-res", type=int, default=24)
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    result = run_validation(args.output, args.voxels, args.voxel_res,
                            args.subpixel_res, args.parallel)
    for name, values in result["metrics"].items():
        print(f"{name:8s}: " + ", ".join(f"{key}={value:.6g}" for key, value in values.items()))
    print(f"figure: {result['output_path']}")
