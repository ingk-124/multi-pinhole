"""Evaluate fully-visible adaptive source quadrature on MST geometry.

Partial resolution is held fixed across cases. Its accuracy is deliberately
validated in ``evaluate_partial_resolution.py``, where the discontinuous cell
integral can be isolated from wall geometry and trilinear neighbour coupling.
"""

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

from examples.benchmark_projection import build_mst_world


def _profiles(voxel):
    rho, theta, phi = voxel.normalized_coordinates().T
    return {
        "constant": np.ones(voxel.N),
        "gaussian": np.exp(-(rho / 0.55) ** 2),
        "boundary": (rho <= 0.9).astype(float),
        "asymmetric": np.exp(-(rho / 0.65) ** 2) * (
            1.0 + 0.25 * np.cos(theta) + 0.15 * np.sin(phi)
        ),
    }


def _relative_l2(actual, reference):
    return np.linalg.norm(actual - reference) / np.linalg.norm(reference)


def run(output_dir=None, spacing=75.0, detector_res=1, parallel=4,
        voxel_bounds=None, preflight_only=False):
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(tempfile.gettempdir()) / "multi_pinhole_mst_adaptive")
    output_dir.mkdir(parents=True, exist_ok=True)
    world = build_mst_world(
        (1, 1, 1), voxel_spacing=spacing, detector_res=detector_res,
        voxel_bounds=voxel_bounds,
    )
    world.find_visible_voxels(verbose=0)
    cases = {
        "ideal": dict(
            res=None, partial_res=5, res_mode="ideal",
            point_source_threshold=1.0 / 8.0,
        ),
        "auto cap 5": dict(
            res=5, partial_res=5, res_mode="auto",
            point_source_threshold=1.0 / 8.0,
        ),
        "fixed 5": dict(res=5, partial_res=5),
    }
    matrices, elapsed, reports = {}, {}, {}
    for name, kwargs in cases.items():
        reports[name] = world.preflight_projection(**kwargs)
        if preflight_only:
            continue
        start = time.perf_counter()
        world.set_projection_matrix(
            verbose=0, parallel=parallel, force=True,
            max_working_memory=1_000_000_000, **kwargs,
        )
        elapsed[name] = time.perf_counter() - start
        matrices[name] = {camera: matrix.copy()
                          for camera, matrix in world.P_matrix.items()}

    if preflight_only:
        return {"world": world, "reports": reports}

    source_profiles = _profiles(world.voxel)
    images = {
        case: {
            profile_name: np.concatenate([
                camera_matrix @ profile
                for camera_matrix in camera_matrices.values()
            ])
            for profile_name, profile in source_profiles.items()
        }
        for case, camera_matrices in matrices.items()
    }
    reference_name = "ideal"
    compared = ["auto cap 5", "fixed 5"]
    image_l2 = {
        case: [_relative_l2(images[case][profile],
                            images[reference_name][profile])
               for profile in source_profiles]
        for case in compared
    }
    flux_error = {
        case: [(images[case][profile].sum() -
                images[reference_name][profile].sum()) /
               images[reference_name][profile].sum()
               for profile in source_profiles]
        for case in compared
    }

    estimates = {}
    visible_counts = {}
    for camera in world.cameras:
        state = world.visible_voxels[camera][0]
        full = np.flatnonzero(state == 2)
        partial = np.flatnonzero(state == 1)
        estimates[camera] = world.estimate_source_resolution(
            camera, 0, full, max_resolution=5,
            point_source_threshold=1.0 / 8.0, detector_grid="psf",
        )
        visible_counts[camera] = (full.size, partial.size)

    sample_counts = {
        name: report.total_samples_upper_bound
        for name, report in reports.items()
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    camera_key = "right"
    state = world.visible_voxels[camera_key][0]
    full = np.flatnonzero(state == 2)
    centers = world.voxel.get_gravity_center(full)
    radius = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
    scatter = axes[0, 0].scatter(
        radius, centers[:, 2],
        c=np.prod(estimates[camera_key].resolution, axis=1),
        s=16, cmap="viridis",
    )
    axes[0, 0].set_title("Right camera: adaptive full samples/voxel")
    axes[0, 0].set_xlabel("major radius R [mm]")
    axes[0, 0].set_ylabel("Z [mm]")
    fig.colorbar(scatter, ax=axes[0, 0])

    profile_names = list(source_profiles)
    positions = np.arange(len(profile_names))
    width = 0.2
    for index, case in enumerate(compared):
        offset = (index - 0.5 * (len(compared) - 1)) * width
        axes[0, 1].bar(positions + offset, image_l2[case], width, label=case)
        axes[0, 2].bar(positions + offset, np.abs(flux_error[case]), width,
                       label=case)
    for axis, title in zip(
            axes[0, 1:],
            ("Image relative L2 vs ideal full",
             "Absolute flux error vs ideal full")):
        axis.set_yscale("log")
        axis.set_xticks(positions, profile_names, rotation=20)
        axis.set_title(title)
        axis.legend(fontsize=8)

    case_names = list(cases)
    axes[1, 0].bar(case_names, [elapsed[name] for name in case_names])
    axes[1, 0].set_title("Projection construction time")
    axes[1, 0].set_ylabel("seconds")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 1].bar(case_names, [sample_counts[name] for name in case_names])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Expanded source samples before partial masking")
    axes[1, 1].tick_params(axis="x", rotation=20)

    pixel_shape = tuple(world.cameras[camera_key].screen.pixel_shape)
    middle = pixel_shape[0] // 2
    for case in case_names:
        image = (matrices[case][camera_key] @ source_profiles["gaussian"]).reshape(
            pixel_shape,
        )
        axes[1, 2].plot(image[middle], label=case)
    axes[1, 2].set_title("Right Gaussian image center row")
    axes[1, 2].set_xlabel("pixel index")
    axes[1, 2].set_ylabel("signal")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(
        f"MST full-voxel quadrature: d={spacing:g} mm, "
        f"detector res={detector_res}, partial res=5 fixed",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path = output_dir / f"mst_adaptive_resolution_d{spacing:g}.png"
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)

    summary = {
        "spacing_mm": spacing,
        "voxel_bounds": voxel_bounds,
        "voxel_shape": world.voxel.shape,
        "voxel_count": world.voxel.N,
        "visible_counts": {
            camera: {"full": counts[0], "partial": counts[1]}
            for camera, counts in visible_counts.items()
        },
        "elapsed_seconds": elapsed,
        "sample_counts_upper_bound": sample_counts,
        "image_relative_l2": image_l2,
        "flux_relative": flux_error,
        "profiles": list(source_profiles),
    }
    json_path = output_dir / f"mst_adaptive_resolution_d{spacing:g}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    return {
        "world": world,
        "elapsed": elapsed,
        "sample_counts": sample_counts,
        "image_l2": image_l2,
        "flux_error": flux_error,
        "reports": reports,
        "estimates": estimates,
        "output_path": output_path,
        "json_path": json_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=75.0)
    parser.add_argument("--detector-res", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limited-roi", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    limited_bounds = (
        ((-2000.0, 1500.0), (-2000.0, -500.0), (-500.0, 200.0))
        if arguments.limited_roi else None
    )
    result = run(
        output_dir=arguments.output_dir, spacing=arguments.spacing,
        detector_res=arguments.detector_res, parallel=arguments.parallel,
        voxel_bounds=limited_bounds, preflight_only=arguments.preflight_only,
    )
    if arguments.preflight_only:
        for name, report in result["reports"].items():
            print(name)
            print(report.summary())
    else:
        print(result["output_path"])
        print(result["json_path"])
