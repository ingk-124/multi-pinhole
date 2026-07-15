"""Sweep the geometry threshold for adaptive source quadrature."""

from pathlib import Path
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np

from examples.evaluate_adaptive_projection import build_case, profiles, _relative_l2


def run(output_dir=None, thresholds=(0.5, 0.25, 0.125, 0.0625), reference_res=4):
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(tempfile.gettempdir()) / "multi_pinhole_adaptive_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    world = build_case()
    world.find_visible_voxels(verbose=0)
    world.set_projection_matrix(res=1, verbose=0, parallel=1, force=True)
    source_profiles = profiles(world.voxel)

    world.set_projection_matrix(
        res=reference_res, verbose=0, parallel=1, force=True,
        max_working_memory=256 * 2 ** 20,
    )
    reference = world.P_matrix[0].copy()
    reference_images = {name: reference @ profile
                        for name, profile in source_profiles.items()}
    full_voxels = np.flatnonzero(world.visible_voxels[0][0] == 2)

    rows = []
    for threshold in thresholds:
        estimate = world.estimate_source_resolution(
            0, 0, full_voxels, max_resolution=reference_res,
            point_source_threshold=threshold,
        )
        start = time.perf_counter()
        world.set_projection_matrix(
            res=reference_res, verbose=0, parallel=1, force=True,
            res_mode="auto",
            point_source_threshold=threshold,
            max_working_memory=256 * 2 ** 20,
        )
        elapsed = time.perf_counter() - start
        matrix = world.P_matrix[0]
        image_errors = {
            name: _relative_l2(matrix @ profile, reference_images[name])
            for name, profile in source_profiles.items()
        }
        rows.append({
            "threshold": float(threshold),
            "samples": int(np.prod(estimate.resolution, axis=1).sum()),
            "elapsed": elapsed,
            "capped_fraction": float(estimate.capped.mean()),
            "image_errors": image_errors,
        })

    reference_samples = world.voxel.N * reference_res ** 3
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    sample_fraction = np.array([row["samples"] / reference_samples for row in rows])
    for profile_name in source_profiles:
        errors = [row["image_errors"][profile_name] for row in rows]
        axes[0].plot(sample_fraction, errors, marker="o", label=profile_name)
    for x, row in zip(sample_fraction, rows):
        axes[0].annotate(f"{row['threshold']:g}", (x, row["image_errors"]["constant"]),
                         xytext=(3, 3), textcoords="offset points", fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(f"source samples / fixed {reference_res}")
    axes[0].set_ylabel("image relative L2")
    axes[0].set_title("Accuracy–sample tradeoff\nlabels: point-source threshold")
    axes[0].legend()

    axes[1].plot([row["elapsed"] for row in rows],
                 [max(row["image_errors"].values()) for row in rows], marker="o")
    for row in rows:
        axes[1].annotate(f"{row['threshold']:g}",
                         (row["elapsed"], max(row["image_errors"].values())),
                         xytext=(3, 3), textcoords="offset points", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("projection time [s]")
    axes[1].set_ylabel("worst profile relative L2")
    axes[1].set_title("Accuracy–time tradeoff")

    axes[2].bar([str(row["threshold"]) for row in rows],
                [row["capped_fraction"] for row in rows])
    axes[2].set_xlabel("point-source threshold [PSF scale]")
    axes[2].set_ylabel("fraction of capped source axes")
    axes[2].set_title(f"Axes requesting more than res={reference_res}")

    fig.suptitle("Wall-free adaptive source-resolution threshold sweep")
    fig.tight_layout()
    output_path = output_dir / "adaptive_threshold_sweep.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"world": world, "rows": rows, "output_path": output_path}


if __name__ == "__main__":
    result = run()
    print(result["output_path"])
