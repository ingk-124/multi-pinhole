"""Isolate partial-voxel quadrature with controlled boundary cuts.

Each case contains exactly one voxel. A plane or sphere cuts that cell at a
known orientation and offset, so no full-voxel contribution, wall mesh,
aperture edge, or neighbouring-cell interpolation can contaminate the
comparison. Spheres with R/d=5 and 20 bracket the curvature of the MST
minor-radius boundary for d=75 and 25 mm. The high-resolution result is a
numerical reference for the projected image.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multi_pinhole_mpl"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "multi_pinhole_cache"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multi_pinhole import Camera, Eye, Screen, Voxel, World


ORIENTATIONS = {
    "axis": np.array([1.0, 0.0, 0.0]),
    "face_diagonal": np.array([1.0, 1.0, 0.0]),
    "space_diagonal": np.array([1.0, 1.0, 1.0]),
}
BOUNDARIES = {
    "plane": None,
    "sphere_Rd5": 5.0,
    "sphere_Rd20": 20.0,
}


def build_case(
        spacing: float, z_over_f: float, orientation: str, offset: float,
        boundary: str, pixel_shape=(81, 81)) -> World:
    """Build one partially included voxel without walls or apertures."""
    focal_length = 25.0
    normal = ORIENTATIONS[orientation].astype(float)
    normal /= np.linalg.norm(normal)
    radius_over_d = BOUNDARIES[boundary]
    if radius_over_d is None:
        center = np.array([0.0, 0.0, focal_length * (1.0 + z_over_f)])
        camera_position = np.zeros(3)
        coordinate_kwargs = {}
    else:
        radius = radius_over_d * spacing
        center = radius * normal
        camera_position = center - np.array([
            0.0, 0.0, focal_length * (1.0 + z_over_f),
        ])
        coordinate_kwargs = {
            "coordinate_type": "spherical",
            "coordinate_parameters": {"radius": radius},
        }
    half = 0.5 * spacing
    voxel = Voxel(
        x_axis=np.array([center[0] - half, center[0] + half]),
        y_axis=np.array([center[1] - half, center[1] + half]),
        z_axis=np.array([center[2] - half, center[2] + half]),
        **coordinate_kwargs,
    )
    screen_size = max(10.0, 2.5 * spacing / max(z_over_f - 1.0, 1.0))
    camera = Camera(
        eyes=[Eye(position=(0.0, 0.0), focal_length=focal_length,
                  eye_size=0.2)],
        apertures=[],
        screen=Screen(
            screen_shape="square", screen_size=screen_size,
            pixel_shape=pixel_shape, subpixel_resolution=1,
        ),
        camera_position=camera_position,
    )
    threshold = offset * spacing

    def inside(x, y, z):
        points = np.column_stack((x, y, z))
        if radius_over_d is None:
            return ((points - center) @ normal) >= threshold
        return np.linalg.norm(points, axis=1) <= radius + threshold

    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(inside)
    world.find_visible_voxels(verbose=0)
    if world.visible_voxels[0][0, 0] != 1:
        raise RuntimeError(
            f"case is not partial: {boundary=}, {orientation=}, "
            f"{offset=}, {z_over_f=}",
        )
    return world


def _relative_metrics(actual, reference):
    difference = actual - reference
    reference_l1 = np.linalg.norm(reference, ord=1)
    reference_l2 = np.linalg.norm(reference)
    reference_peak = np.max(np.abs(reference))
    return {
        "flux_relative": float(
            abs(actual.sum() - reference.sum()) / abs(reference.sum())
        ),
        "l1_relative": float(np.linalg.norm(difference, ord=1) / reference_l1),
        "l2_relative": float(np.linalg.norm(difference) / reference_l2),
        "max_pixel_relative": float(
            np.max(np.abs(difference)) / reference_peak
        ),
    }


def run(
        output_dir=None, spacing=25.0, reference_res=48,
        resolutions=(1, 2, 3, 4, 5, 6, 8), parallel=4):
    """Evaluate partial quadrature over depth, boundary, angle, and offset."""
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(tempfile.gettempdir()) / "multi_pinhole_partial_resolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    depths = (5.0, 20.0, 80.0)
    offsets = (-0.2, 0.0, 0.2)
    records = []

    for depth in depths:
        for boundary in BOUNDARIES:
            for orientation in ORIENTATIONS:
                for offset in offsets:
                    world = build_case(
                        spacing, depth, orientation, offset, boundary,
                    )
                    images = {}
                    elapsed = {}
                    for resolution in (*resolutions, reference_res):
                        started = time.perf_counter()
                        world.set_projection_matrix(
                            res=1, partial_res=resolution, force=True,
                            verbose=0, parallel=parallel,
                            max_working_memory=500_000_000,
                        )
                        elapsed[resolution] = time.perf_counter() - started
                        images[resolution] = np.asarray(
                            world.P_matrix[0][:, 0].todense(),
                        ).ravel()
                    reference = images[reference_res]
                    for resolution in resolutions:
                        records.append({
                            "boundary": boundary,
                            "z_over_f": depth,
                            "orientation": orientation,
                            "offset_over_d": offset,
                            "partial_res": resolution,
                            "samples": resolution ** 3,
                            "seconds": elapsed[resolution],
                            **_relative_metrics(images[resolution], reference),
                        })

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for boundary_index, boundary in enumerate(BOUNDARIES):
        selected = [record for record in records
                    if record["boundary"] == boundary]
        worst_l2 = [
            max(record["l2_relative"] for record in selected
                if record["partial_res"] == resolution)
            for resolution in resolutions
        ]
        axes[0].plot(
            resolutions, worst_l2, marker="o",
            color=colors[boundary_index], label=boundary,
        )
    for depth_index, depth in enumerate(depths):
        selected = [record for record in records
                    if record["z_over_f"] == depth]
        worst_l2 = [
            max(record["l2_relative"] for record in selected
                if record["partial_res"] == resolution)
            for resolution in resolutions
        ]
        axes[0].plot(
            resolutions, worst_l2,
            color=colors[depth_index + len(BOUNDARIES)],
            linestyle="--", label=f"all boundaries, Z/f={depth:g}",
        )

    metric_names = ("flux_relative", "l2_relative", "max_pixel_relative")
    width = 0.24
    positions = np.arange(len(resolutions))
    for metric_index, metric in enumerate(metric_names):
        worst = [
            max(record[metric] for record in records
                if record["partial_res"] == resolution)
            for resolution in resolutions
        ]
        axes[1].bar(
            positions + (metric_index - 1) * width, worst, width,
            label=metric,
        )

    median_seconds = [
        np.median([
            record["seconds"] for record in records
            if record["partial_res"] == resolution
        ])
        for resolution in resolutions
    ]
    axes[2].plot(
        [resolution ** 3 for resolution in resolutions], median_seconds,
        marker="o",
    )

    axes[0].set(
        xlabel="partial res", ylabel="worst relative L2",
        title="Boundary curvature and depth",
    )
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set(
        xlabel="partial res", ylabel="worst relative error over all cases",
        title=f"Worst case vs res={reference_res} reference",
    )
    axes[1].set_xticks(positions, resolutions)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[2].set(
        xlabel="subvoxel samples per partial cell",
        ylabel="median construction time [s]",
        title="Quadrature cost",
    )
    figure.suptitle(
        f"Controlled partial voxel: d={spacing:g} mm, no wall/aperture",
    )
    figure.tight_layout()
    figure_path = output_dir / "partial_resolution_boundary_cut.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    result = {
        "geometry": {
            "spacing_mm": spacing,
            "z_over_f": depths,
            "boundaries": BOUNDARIES,
            "orientations": list(ORIENTATIONS),
            "offset_over_d": offsets,
            "reference_res": reference_res,
        },
        "records": records,
    }
    json_path = output_dir / "partial_resolution_boundary_cut.json"
    json_path.write_text(json.dumps(result, indent=2))
    return {"result": result, "figure_path": figure_path, "json_path": json_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=25.0)
    parser.add_argument("--reference-res", type=int, default=48)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output = run(
        output_dir=arguments.output_dir, spacing=arguments.spacing,
        reference_res=arguments.reference_res, parallel=arguments.parallel,
    )
    print(output["figure_path"])
    print(output["json_path"])
