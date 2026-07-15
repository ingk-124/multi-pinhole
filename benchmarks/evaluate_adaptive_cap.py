"""Compare a finite auto-resolution cap with explicit uncapped ideal work.

The wall/aperture-free d=10 mm Toy isolates source quadrature from partial
visibility. It spans the range Z/f=16...128 used by the MST discussion and
includes smooth, discontinuous, and single-voxel emission profiles.
"""

import argparse
import json
from pathlib import Path
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multi_pinhole import Camera, Eye, Screen, Voxel, World


def build_case(detector_res=1):
    """Build a fully-visible d=10 mm source column."""
    eye = Eye(position=(0.0, 0.0), focal_length=25.0, eye_size=1.0)
    camera = Camera(
        eyes=[eye], apertures=[],
        screen=Screen(
            screen_shape="square", screen_size=7.5,
            pixel_shape=(61, 61), subpixel_resolution=detector_res,
        ),
        camera_position=(0.0, 0.0, 0.0),
    )
    # The Eye is at camera z=f, giving Eye-relative Z=400...3200 mm.
    voxel = Voxel.uniform_voxel_from_centers(
        ranges=((-20.0, 20.0), (-20.0, 20.0), (425.0, 3225.0)),
        shape=(5, 5, 281),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def emission_profiles(voxel, focal_length=25.0):
    """Return smooth and deliberately difficult center-value profiles."""
    centers = voxel.get_gravity_center()
    z_eye = centers[:, 2] - focal_length
    transverse_squared = centers[:, 0] ** 2 + centers[:, 1] ** 2
    profiles = {
        "constant": np.ones(voxel.N),
        "linear": 0.25 + 0.75 * (z_eye - z_eye.min()) / np.ptp(z_eye),
        "square_near": (
            (z_eye >= 400.0) & (z_eye <= 700.0)
            & (np.abs(centers[:, 0]) <= 10.0)
            & (np.abs(centers[:, 1]) <= 10.0)
        ).astype(float),
        "gaussian_near": (
            np.exp(-0.5 * ((z_eye - 520.0) / 120.0) ** 2)
            * np.exp(-0.5 * transverse_squared / 15.0 ** 2)
        ),
    }
    nearest = np.argmin((z_eye - z_eye.min()) ** 2 + transverse_squared)
    profiles["nearest_impulse"] = np.zeros(voxel.N)
    profiles["nearest_impulse"][nearest] = 1.0
    return profiles


def _sparse_bytes(matrix):
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def _metrics(actual, reference):
    difference = actual - reference
    return {
        "total_relative": float(
            abs(actual.sum() - reference.sum()) / abs(reference.sum())
        ),
        "l1_relative": float(
            np.linalg.norm(difference, ord=1)
            / np.linalg.norm(reference, ord=1)
        ),
        "l2_relative": float(
            np.linalg.norm(difference) / np.linalg.norm(reference)
        ),
        "max_pixel_relative": float(
            np.max(np.abs(difference)) / np.max(np.abs(reference))
        ),
    }


def run(output_dir=None, cap=5, detector_res=1, parallel=4):
    """Run uncapped ideal and capped auto projections and save diagnostics."""
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(tempfile.gettempdir()) / "multi_pinhole_adaptive_cap")
    output_dir.mkdir(parents=True, exist_ok=True)
    world = build_case(detector_res=detector_res)

    settings = {
        "ideal": dict(res=None, res_mode="ideal", partial_res=cap),
        f"auto_cap_{cap}": dict(res=cap, res_mode="auto"),
    }
    preflight = {}
    matrices = {}
    build_seconds = {}
    for name, kwargs in settings.items():
        report = world.preflight_projection(**kwargs)
        row = report.eyes[0]
        preflight[name] = {
            "samples": report.total_samples_upper_bound,
            "buckets": [
                [list(resolution), count]
                for resolution, count in row.full_resolution_buckets
            ],
            "ideal_p50": row.ideal_p50,
            "ideal_p95": row.ideal_p95,
            "ideal_max": row.ideal_max,
            "capped_axes": row.capped_axes,
        }
        started = time.perf_counter()
        world.set_projection_matrix(
            **kwargs, verbose=0, parallel=parallel, force=True,
            max_working_memory=500_000_000,
        )
        build_seconds[name] = time.perf_counter() - started
        matrices[name] = world.P_matrix[0].copy()

    ideal_name = "ideal"
    capped_name = f"auto_cap_{cap}"
    profiles = emission_profiles(world.voxel)
    profile_metrics = {}
    for name, emission in profiles.items():
        reference = np.asarray(matrices[ideal_name] @ emission).reshape(-1)
        actual = np.asarray(matrices[capped_name] @ emission).reshape(-1)
        profile_metrics[name] = _metrics(actual, reference)

    result = {
        "geometry": {
            "voxel_shape": world.voxel.shape,
            "voxel_count": world.voxel.N,
            "d_mm": 10.0,
            "eye_Z_mm": [400.0, 3200.0],
            "focal_length_mm": 25.0,
            "eye_diameter_mm": 1.0,
            "detector_res": detector_res,
        },
        "cap": cap,
        "preflight": preflight,
        "build_seconds": build_seconds,
        "matrix": {
            name: {
                "nnz": matrix.nnz,
                "bytes": _sparse_bytes(matrix),
            }
            for name, matrix in matrices.items()
        },
        "profiles": profile_metrics,
    }

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ideal_buckets = dict(
        (resolution[0], count)
        for resolution, count in preflight[ideal_name]["buckets"]
    )
    capped_buckets = dict(
        (resolution[0], count)
        for resolution, count in preflight[capped_name]["buckets"]
    )
    resolutions = np.arange(
        min(ideal_buckets), max(ideal_buckets) + 1,
    )
    width = 0.38
    axes[0].bar(resolutions - width / 2,
                [ideal_buckets.get(int(value), 0) for value in resolutions],
                width=width, label="ideal")
    axes[0].bar(resolutions + width / 2,
                [capped_buckets.get(int(value), 0) for value in resolutions],
                width=width, label=f"auto cap={cap}")
    axes[0].set(xlabel="source res", ylabel="voxels", title="Resolution buckets")
    axes[0].legend()

    ideal_samples = preflight[ideal_name]["samples"]
    capped_samples = preflight[capped_name]["samples"]
    axes[1].bar(["source samples", "build time"], [
        capped_samples / ideal_samples,
        build_seconds[capped_name] / build_seconds[ideal_name],
    ])
    axes[1].axhline(1.0, color="k", linewidth=1)
    axes[1].set(ylabel="cap / ideal", title="Work reduction")

    names = list(profile_metrics)
    positions = np.arange(len(names))
    for metric in ("l1_relative", "l2_relative", "max_pixel_relative"):
        axes[2].plot(
            positions, [profile_metrics[name][metric] for name in names],
            marker="o", label=metric,
        )
    axes[2].set_yscale("log")
    axes[2].set_xticks(positions, names, rotation=25, ha="right")
    axes[2].set(ylabel="relative error", title="Image error vs ideal")
    axes[2].legend(fontsize=8)

    figure.suptitle(
        f"d=10 mm adaptive cap validation, detector res={detector_res}",
    )
    figure.tight_layout()
    figure_path = output_dir / f"adaptive_cap_detector_res_{detector_res}.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    json_path = output_dir / f"adaptive_cap_detector_res_{detector_res}.json"
    json_path.write_text(json.dumps(result, indent=2))
    return {"result": result, "figure_path": figure_path, "json_path": json_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=5)
    parser.add_argument("--detector-res", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output = run(
        output_dir=arguments.output_dir,
        cap=arguments.cap,
        detector_res=arguments.detector_res,
        parallel=arguments.parallel,
    )
    print(output["figure_path"])
    print(output["json_path"])
