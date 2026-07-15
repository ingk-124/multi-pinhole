"""Evaluate adaptive source quadrature on the d=75 mm MST geometry."""

from pathlib import Path
import tempfile
import time

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


def run(output_dir=None, spacing=75.0, detector_res=1, parallel=4):
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(tempfile.gettempdir()) / "multi_pinhole_mst_adaptive")
    output_dir.mkdir(parents=True, exist_ok=True)
    world = build_mst_world((1, 1, 1), voxel_spacing=spacing,
                            detector_res=detector_res)
    world.find_visible_voxels(verbose=0)
    world.set_projection_matrix(res=1, partial_res=1, verbose=0,
                                parallel=parallel, force=True)
    cases = {
        "fixed 1": dict(res=1, partial_res=1),
        "fixed 2": dict(res=2, partial_res=2),
        "fixed 3": dict(res=3, partial_res=3),
        "fixed 4": dict(res=4, partial_res=4),
        "adaptive p5": dict(
            res=5, partial_res=5, res_mode="auto",
            point_source_threshold=1.0 / 8.0,
        ),
        "adaptive p3": dict(
            res=5, partial_res=3, res_mode="auto",
            point_source_threshold=1.0 / 8.0,
        ),
        "fixed 5": dict(res=5, partial_res=5),
    }
    matrices, elapsed = {}, {}
    for name, kwargs in cases.items():
        start = time.perf_counter()
        world.set_projection_matrix(
            verbose=0, parallel=parallel, force=True,
            max_working_memory=1_000_000_000, **kwargs,
        )
        elapsed[name] = time.perf_counter() - start
        matrices[name] = {camera: matrix.copy()
                          for camera, matrix in world.P_matrix.items()}

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
    reference_name = "fixed 5"
    compared = ["fixed 1", "fixed 3", "adaptive p5", "adaptive p3"]
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

    def estimated_samples(name):
        if name.startswith("fixed"):
            resolution = int(name.split()[-1])
            return sum((full + partial) * resolution ** 3
                       for full, partial in visible_counts.values())
        partial_resolution = 5 if name.endswith("p5") else 3
        full_samples = sum(
            int(np.prod(estimate.resolution, axis=1).sum())
            for estimate in estimates.values()
        )
        partial_samples = sum(partial * partial_resolution ** 3
                              for _, partial in visible_counts.values())
        return full_samples + partial_samples

    sample_counts = {name: estimated_samples(name) for name in cases}

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
        offset = (index - 1.5) * width
        axes[0, 1].bar(positions + offset, image_l2[case], width, label=case)
        axes[0, 2].bar(positions + offset, np.abs(flux_error[case]), width,
                       label=case)
    for axis, title in zip(
            axes[0, 1:],
            ("Image relative L2 vs fixed 5", "Absolute flux error vs fixed 5")):
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
        f"MST adaptive source quadrature: d={spacing:g} mm, detector res={detector_res}",
    )
    fig.tight_layout()
    output_path = output_dir / "mst_adaptive_resolution.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    partial_errors = {resolution: [] for resolution in range(1, 5)}
    for camera in world.cameras:
        partial = np.flatnonzero(world.visible_voxels[camera][0] == 1)
        reference_columns = matrices["fixed 5"][camera][:, partial].tocsc()
        denominator = np.asarray(abs(reference_columns).sum(axis=0)).ravel()
        for resolution in range(1, 5):
            candidate = matrices[f"fixed {resolution}"][camera][:, partial].tocsc()
            numerator = np.asarray(abs(candidate - reference_columns).sum(axis=0)).ravel()
            relative = np.full(denominator.shape, np.inf)
            np.divide(numerator, denominator, out=relative, where=denominator > 0.0)
            relative[(denominator == 0.0) & (numerator == 0.0)] = 0.0
            partial_errors[resolution].append(relative)
    partial_errors = {
        resolution: np.concatenate(parts)
        for resolution, parts in partial_errors.items()
    }
    tolerances = (0.01, 0.03, 0.05)
    oracle_resolution = {}
    for tolerance in tolerances:
        selected = np.full(partial_errors[1].shape, 5, dtype=int)
        for resolution in range(4, 0, -1):
            selected[partial_errors[resolution] <= tolerance] = resolution
        oracle_resolution[tolerance] = selected

    oracle_fig, oracle_axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for resolution, errors in partial_errors.items():
        finite_errors = np.sort(errors[np.isfinite(errors)])
        fraction = np.arange(1, finite_errors.size + 1) / finite_errors.size
        oracle_axes[0].plot(finite_errors, fraction, label=f"res {resolution}")
    oracle_axes[0].set_xscale("log")
    oracle_axes[0].set_xlabel("partial-column relative L1 vs res 5")
    oracle_axes[0].set_ylabel("cumulative fraction")
    oracle_axes[0].set_title("Per-voxel impulse-response convergence")
    oracle_axes[0].legend()

    width = 0.25
    positions = np.arange(1, 6)
    for index, tolerance in enumerate(tolerances):
        selected = oracle_resolution[tolerance]
        counts = np.array([(selected == resolution).sum()
                           for resolution in positions])
        oracle_axes[1].bar(positions + (index - 1) * width, counts, width,
                           label=f"tol {tolerance:g}")
    oracle_axes[1].set_xticks(positions)
    oracle_axes[1].set_xlabel("oracle-selected partial res")
    oracle_axes[1].set_ylabel("camera–voxel columns")
    oracle_axes[1].set_title("Best possible selection using known res-5 P")
    oracle_axes[1].legend()

    sample_ratios = [
        np.sum(selected ** 3) / (selected.size * 5 ** 3)
        for selected in oracle_resolution.values()
    ]
    oracle_axes[2].bar([str(tolerance) for tolerance in tolerances], sample_ratios)
    oracle_axes[2].set_xlabel("allowed per-column relative L1")
    oracle_axes[2].set_ylabel("partial samples / fixed res 5")
    oracle_axes[2].set_title("Oracle partial-sample reduction")
    oracle_fig.suptitle("MST d=75 mm partial-visibility adaptation potential")
    oracle_fig.tight_layout()
    oracle_path = output_dir / "mst_partial_resolution_oracle.png"
    oracle_fig.savefig(oracle_path, dpi=180)
    plt.close(oracle_fig)
    return {
        "world": world,
        "elapsed": elapsed,
        "sample_counts": sample_counts,
        "image_l2": image_l2,
        "flux_error": flux_error,
        "estimates": estimates,
        "partial_errors": partial_errors,
        "oracle_resolution": oracle_resolution,
        "output_path": output_path,
        "oracle_path": oracle_path,
    }


if __name__ == "__main__":
    result = run()
    print(result["output_path"])
