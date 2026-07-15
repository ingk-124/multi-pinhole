"""Measure source-side subvoxel convergence across imaging geometries.

The detector uses analytic spot/cell overlap with detector subpixel resolution
one.  For each combination of voxel size, pixel size, and source distance Z,
the script compares several source-side subvoxel resolutions against a much
finer projection matrix constructed for the same voxel-center emission vector.
This isolates source quadrature error from detector rasterization error.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multi_pinhole_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "multi_pinhole_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multi_pinhole import Camera, Eye, Screen, Voxel, World


X_BOUNDS = (-20.0, 20.0)
FOCAL_LENGTH = 20.0
EYE_DIAMETER = 0.08
SCREEN_SIZE = (0.45, 24.0)


def emission_profiles(x):
    """Profiles evaluated only at the fixed coarse voxel centers."""
    half_width = 0.5 * (X_BOUNDS[1] - X_BOUNDS[0])
    return {
        "constant": np.ones_like(x),
        "linear": 1.0 + 0.6 * x / half_width,
        "square": (x / half_width) ** 2,
        "gaussian": np.exp(-0.5 * (x / 6.0) ** 2),
    }


def build_world(voxel_count: int, pixel_count: int, axial_distance: float) -> World:
    """Build a wall-free thin source slab and a one-dimensional detector."""
    source_z = FOCAL_LENGTH + axial_distance
    voxel = Voxel.uniform_voxel(
        ranges=(X_BOUNDS, (-0.01, 0.01), (source_z - 0.01, source_z + 0.01)),
        shape=(voxel_count, 1, 1),
    )
    eye = Eye(
        position=(0.0, 0.0), focal_length=FOCAL_LENGTH,
        eye_type="pinhole", eye_shape="circle", eye_size=EYE_DIAMETER,
    )
    screen = Screen(
        screen_shape="rectangle", screen_size=SCREEN_SIZE,
        pixel_shape=(3, pixel_count), subpixel_resolution=1,
    )
    camera = Camera(
        eyes=[eye], apertures=[], screen=screen,
        camera_position=(0.0, 0.0, 0.0),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def _images(world, profiles, resolution):
    world.set_projection_matrix(
        res=(resolution, 1, 1), partial_res=(resolution, 1, 1),
        verbose=0, parallel=1, force=True,
    )
    projection = world.P_matrix[0]
    return {name: np.asarray(projection @ values).ravel()
            for name, values in profiles.items()}


def _relative_errors(image, reference):
    return (
        float(np.linalg.norm(image - reference) / np.linalg.norm(reference)),
        float(abs(image.sum() - reference.sum()) / abs(reference.sum())),
    )


def run_sweep(output_dir: Path, voxel_counts=(21, 41, 81),
              pixel_counts=(41, 81, 161), axial_distances=(40.0, 80.0, 160.0),
              resolutions=(1, 2, 4, 8, 16, 32, 64), reference_resolution=256):
    """Run the convergence sweep, save CSV/figure, and return all rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()

    for voxel_count in voxel_counts:
        voxel_size = (X_BOUNDS[1] - X_BOUNDS[0]) / voxel_count
        for pixel_count in pixel_counts:
            pixel_size = SCREEN_SIZE[1] / pixel_count
            for axial_distance in axial_distances:
                world = build_world(voxel_count, pixel_count, axial_distance)
                x = world.voxel.gravity_center[:, 0]
                profiles = emission_profiles(x)
                reference = _images(world, profiles, reference_resolution)
                spot_size = EYE_DIAMETER * (1.0 + FOCAL_LENGTH / axial_distance)
                response_scale = min(pixel_size, spot_size)
                base_sampling_ratio = ((FOCAL_LENGTH / axial_distance) * voxel_size
                                       / response_scale)

                for resolution in resolutions:
                    images = _images(world, profiles, resolution)
                    for name in profiles:
                        relative_l2, relative_flux = _relative_errors(
                            images[name], reference[name],
                        )
                        rows.append({
                            "voxel_count": voxel_count,
                            "voxel_size": voxel_size,
                            "pixel_count": pixel_count,
                            "pixel_size": pixel_size,
                            "axial_distance": axial_distance,
                            "spot_size": spot_size,
                            "subvoxel_resolution": resolution,
                            "projected_subvoxel_step": (
                                FOCAL_LENGTH / axial_distance * voxel_size / resolution
                            ),
                            "sampling_ratio": base_sampling_ratio / resolution,
                            "profile": name,
                            "relative_l2": relative_l2,
                            "relative_flux": relative_flux,
                        })

    csv_path = output_dir / "subvoxel_resolution_sweep.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Collapse the four profiles to a conservative worst-case error for each
    # geometry and resolution.
    grouped = {}
    for row in rows:
        key = (row["voxel_count"], row["pixel_count"],
               row["axial_distance"], row["subvoxel_resolution"])
        grouped[key] = max(grouped.get(key, 0.0), row["relative_l2"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {distance: color for distance, color in zip(
        axial_distances, plt.cm.viridis(np.linspace(0.15, 0.85, len(axial_distances))))}
    for row in rows:
        if row["profile"] != "constant":
            continue
        key = (row["voxel_count"], row["pixel_count"],
               row["axial_distance"], row["subvoxel_resolution"])
        axes[0].scatter(row["sampling_ratio"], grouped[key], s=13,
                        color=colors[row["axial_distance"]], alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"projected subvoxel step / min(pixel, spot)  $\rho$")
    axes[0].set_ylabel("worst profile relative L2")
    axes[0].grid(which="both", alpha=0.25)
    for distance in axial_distances:
        axes[0].scatter([], [], color=colors[distance], label=f"Z={distance:g} mm")
    axes[0].legend()

    targets = (1e-3, 1e-4)
    geometry_keys = sorted({key[:3] for key in grouped})
    has_required_resolution = False
    for target, marker in zip(targets, ("o", "s")):
        difficulties, required = [], []
        for voxel_count, pixel_count, axial_distance in geometry_keys:
            voxel_size = (X_BOUNDS[1] - X_BOUNDS[0]) / voxel_count
            pixel_size = SCREEN_SIZE[1] / pixel_count
            spot_size = EYE_DIAMETER * (1.0 + FOCAL_LENGTH / axial_distance)
            difficulty = ((FOCAL_LENGTH / axial_distance) * voxel_size
                          / min(pixel_size, spot_size))
            passing = [resolution for resolution in resolutions
                       if grouped[(voxel_count, pixel_count, axial_distance, resolution)] <= target]
            if passing:
                difficulties.append(difficulty)
                required.append(min(passing))
        if required:
            has_required_resolution = True
            axes[1].scatter(difficulties, required, marker=marker, label=f"L2 ≤ {target:g}")
    axes[1].set_xscale("log")
    if has_required_resolution:
        axes[1].set_yscale("log", base=2)
    else:
        axes[1].text(0.5, 0.5, "No tested resolution met the targets",
                     ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xlabel(r"difficulty before subdivision  $\rho(r=1)$")
    axes[1].set_ylabel("minimum tested subvoxel resolution")
    axes[1].grid(which="both", alpha=0.25)
    if has_required_resolution:
        axes[1].legend()
    fig.suptitle("Source-side subvoxel convergence with exact detector overlap")
    fig.tight_layout()
    figure_path = output_dir / "subvoxel_resolution_sweep.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    return {
        "rows": rows,
        "csv_path": csv_path,
        "figure_path": figure_path,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _parse_ints(value):
    return tuple(int(item) for item in value.split(","))


def _parse_floats(value):
    return tuple(float(item) for item in value.split(","))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmarks/output/subvoxel_resolution"))
    parser.add_argument("--voxel-counts", type=_parse_ints, default=(21, 41, 81))
    parser.add_argument("--pixel-counts", type=_parse_ints, default=(41, 81, 161))
    parser.add_argument("--distances", type=_parse_floats, default=(40.0, 80.0, 160.0))
    parser.add_argument("--resolutions", type=_parse_ints, default=(1, 2, 4, 8, 16, 32, 64))
    parser.add_argument("--reference-resolution", type=int, default=256)
    args = parser.parse_args()
    result = run_sweep(
        args.output_dir, args.voxel_counts, args.pixel_counts,
        args.distances, args.resolutions, args.reference_resolution,
    )
    print(f"elapsed_seconds: {result['elapsed_seconds']:.3f}")
    print(f"csv: {result['csv_path']}")
    print(f"figure: {result['figure_path']}")
