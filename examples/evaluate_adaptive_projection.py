"""Compare geometry-adaptive and fixed source quadrature on a wall-free case."""

from pathlib import Path
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from multi_pinhole import Camera, Eye, Screen, Voxel, World


def build_case():
    camera = Camera(
        eyes=[Eye(position=(0.0, 0.0), focal_length=20.0,
                  eye_type="pinhole", eye_shape="circle", eye_size=1.0)],
        apertures=[],
        screen=Screen(
            screen_shape="rectangle", screen_size=(24.0, 32.0),
            pixel_shape=(24, 32), subpixel_resolution=2,
        ),
        camera_position=(0.0, 0.0, 0.0),
    )
    voxel = Voxel.uniform_voxel(
        ranges=((-12.0, 12.0), (-1.0, 1.0), (60.0, 240.0)),
        shape=(24, 1, 30),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def profiles(voxel):
    x, _, z = voxel.gravity_center.T
    return {
        "constant": np.ones(voxel.N),
        "linear": 1.0 + 0.45 * x / 12.0 + 0.3 * (z - 150.0) / 90.0,
        "square": ((np.abs(x) < 5.0) & (z > 100.0) & (z < 180.0)).astype(float),
        "gaussian": np.exp(-(x / 5.0) ** 2 - ((z - 145.0) / 42.0) ** 2),
    }


def _relative_l2(actual, reference):
    denominator = np.linalg.norm(reference)
    return np.linalg.norm(actual - reference) / denominator if denominator else 0.0


def run(output_dir=None, reference_res=4, point_source_threshold=1.0 / 8.0):
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(tempfile.gettempdir()) / "multi_pinhole_adaptive_projection")
    output_dir.mkdir(parents=True, exist_ok=True)
    world = build_case()
    cases = {
        "fixed 1": dict(res=1, res_mode="fixed"),
        "fixed 2": dict(res=2, res_mode="fixed"),
        "adaptive": dict(res=reference_res, res_mode="auto",
                         point_source_threshold=point_source_threshold),
        f"fixed {reference_res}": dict(res=reference_res, res_mode="fixed"),
    }
    matrices, elapsed = {}, {}
    # Warm visibility, rasterization kernels, and sparse assembly once before
    # recording construction times. Each measured case is still recomputed.
    world.find_visible_voxels(verbose=0)
    world.set_projection_matrix(
        res=1, verbose=0, parallel=1, force=True,
        max_working_memory=256 * 2 ** 20,
    )
    for name, kwargs in cases.items():
        start = time.perf_counter()
        world.set_projection_matrix(
            verbose=0, parallel=1, force=True,
            max_working_memory=256 * 2 ** 20, **kwargs,
        )
        elapsed[name] = time.perf_counter() - start
        matrices[name] = world.P_matrix[0].copy()

    reference_name = f"fixed {reference_res}"
    reference = matrices[reference_name]
    source_profiles = profiles(world.voxel)
    images = {
        case: {name: matrix @ profile for name, profile in source_profiles.items()}
        for case, matrix in matrices.items()
    }
    compared_cases = [name for name in cases if name != reference_name]
    image_l2 = {
        case: [_relative_l2(images[case][name], images[reference_name][name])
               for name in source_profiles]
        for case in compared_cases
    }
    flux_error = {
        case: [(images[case][name].sum() - images[reference_name][name].sum()) /
               images[reference_name][name].sum()
               for name in source_profiles]
        for case in compared_cases
    }
    matrix_l2 = {
        case: sparse.linalg.norm(matrices[case] - reference) /
        sparse.linalg.norm(reference)
        for case in compared_cases
    }

    full_voxels = np.flatnonzero(world.visible_voxels[0][0] == 2)
    estimate = world.estimate_source_resolution(
        0, 0, full_voxels, max_resolution=reference_res,
        point_source_threshold=point_source_threshold,
    )
    sample_counts = {
        "fixed 1": world.voxel.N,
        "fixed 2": world.voxel.N * 2 ** 3,
        "adaptive": int(np.prod(estimate.resolution, axis=1).sum()),
        reference_name: world.voxel.N * reference_res ** 3,
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    centers = world.voxel.get_gravity_center(full_voxels)
    scatter = axes[0, 0].scatter(
        centers[:, 0], centers[:, 2],
        c=np.prod(estimate.resolution, axis=1), s=18, cmap="viridis",
    )
    axes[0, 0].set_title("Adaptive samples per voxel")
    axes[0, 0].set_xlabel("world x [mm]")
    axes[0, 0].set_ylabel("world z [mm]")
    fig.colorbar(scatter, ax=axes[0, 0])

    profile_names = list(source_profiles)
    positions = np.arange(len(profile_names))
    width = 0.24
    for case_index, case in enumerate(compared_cases):
        axes[0, 1].bar(positions + (case_index - 1) * width,
                       image_l2[case], width, label=case)
        axes[0, 2].bar(positions + (case_index - 1) * width,
                       np.abs(flux_error[case]), width, label=case)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title(f"Image relative L2 vs {reference_name}")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title(f"Absolute flux error vs {reference_name}")
    for axis in axes[0, 1:]:
        axis.set_xticks(positions, profile_names, rotation=20)
        axis.legend()

    case_names = list(cases)
    bars = axes[1, 0].bar(case_names, [elapsed[name] for name in case_names])
    axes[1, 0].set_title("Projection construction time")
    axes[1, 0].set_ylabel("seconds")
    axes[1, 0].tick_params(axis="x", rotation=20)
    for bar, name in zip(bars, case_names):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{sample_counts[name]:,} samples", ha="center", va="bottom",
                        rotation=90, fontsize=8)

    axes[1, 1].bar(compared_cases, [matrix_l2[name] for name in compared_cases])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title(f"Sparse P relative Frobenius error vs {reference_name}")
    axes[1, 1].tick_params(axis="x", rotation=20)

    gaussian_reference = images[reference_name]["gaussian"].reshape(
        tuple(world.cameras[0].screen.pixel_shape),
    )
    middle = gaussian_reference.shape[0] // 2
    for case in case_names:
        image = images[case]["gaussian"].reshape(gaussian_reference.shape)
        axes[1, 2].plot(image[middle], label=case)
    axes[1, 2].set_title("Gaussian image center row")
    axes[1, 2].set_xlabel("pixel index")
    axes[1, 2].set_ylabel("signal")
    axes[1, 2].legend()

    fig.suptitle(
        f"Wall-free adaptive source quadrature; threshold={point_source_threshold:g}",
    )
    fig.tight_layout()
    output_path = output_dir / "adaptive_projection_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "world": world,
        "matrices": matrices,
        "elapsed": elapsed,
        "sample_counts": sample_counts,
        "image_l2": image_l2,
        "flux_error": flux_error,
        "matrix_l2": matrix_l2,
        "estimate": estimate,
        "output_path": output_path,
    }


if __name__ == "__main__":
    result = run()
    print(result["output_path"])
