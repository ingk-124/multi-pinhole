"""Sweep small voxel size and source depth in a fully-visible Toy model."""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy import sparse

from multi_pinhole import Camera, Eye, Screen, Voxel, World


def build_world(voxel_size, depth_ratio, focal_length=20.0):
    """Build a 5x1x5 wall/aperture-free voxel block at a given Eye depth."""
    eye = Eye(position=(0.0, 0.0), focal_length=focal_length,
              eye_type="pinhole", eye_shape="circle", eye_size=1.0)
    camera = Camera(
        eyes=[eye], apertures=[],
        screen=Screen(
            screen_shape="rectangle", screen_size=(40.0, 40.0),
            pixel_shape=(40, 40), subpixel_resolution=2,
        ),
        camera_position=(0.0, 0.0, 0.0),
    )
    source_depth = depth_ratio * focal_length
    source_world_z = eye.position[2] + source_depth
    half_widths = np.array([2.5, 0.5, 2.5]) * voxel_size
    center = np.array([0.0, 0.0, source_world_z])
    ranges = np.column_stack((center - half_widths, center + half_widths))
    voxel = Voxel.uniform_voxel(ranges=ranges, shape=(5, 1, 5))
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def source_profiles(voxel):
    indices = voxel.get_voxel_position(np.arange(voxel.N)).astype(float)
    x_index = indices[:, 0] - 2.0
    z_index = indices[:, 2] - 2.0
    return {
        "constant": np.ones(voxel.N),
        "linear": 1.0 + 0.12 * x_index + 0.08 * z_index,
        "square": ((indices[:, 0] >= 2) & (indices[:, 2] <= 2)).astype(float),
        "gaussian": np.exp(-0.5 * (x_index ** 2 + z_index ** 2)),
    }


def _relative_l2(actual, reference):
    denominator = np.linalg.norm(reference)
    return np.linalg.norm(actual - reference) / denominator if denominator else 0.0


def run(output_dir=None, voxel_sizes=(2.0, 5.0, 10.0, 25.0),
        depth_ratios=(5.0, 10.0, 20.0, 50.0, 100.0),
        adaptive_max_res=8, reference_res=12,
        max_projected_step=0.25):
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(tempfile.gettempdir()) / "multi_pinhole_small_voxel_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for voxel_size in voxel_sizes:
        for depth_ratio in depth_ratios:
            world = build_world(voxel_size, depth_ratio)
            world.find_visible_voxels(verbose=0)
            visibility = world.visible_voxels[0][0]
            if np.any(visibility != 2):
                raise RuntimeError(
                    "Toy sweep requires every voxel to be fully visible; "
                    f"got states {np.unique(visibility)}",
                )
            estimate = world.estimate_source_resolution(
                0, 0, max_resolution=adaptive_max_res,
                max_projected_step=max_projected_step,
                detector_grid="psf",
            )
            world.set_projection_matrix(
                res=adaptive_max_res, verbose=0, parallel=1, force=True,
                adaptive_source_resolution=True,
                max_projected_step=max_projected_step,
            )
            adaptive = world.P_matrix[0].copy()
            world.set_projection_matrix(
                res=adaptive_max_res, verbose=0, parallel=1, force=True,
            )
            fixed_max = world.P_matrix[0].copy()
            world.set_projection_matrix(
                res=reference_res, verbose=0, parallel=1, force=True,
            )
            reference = world.P_matrix[0].copy()

            profiles = source_profiles(world.voxel)
            profile_errors = {
                name: _relative_l2(adaptive @ profile, reference @ profile)
                for name, profile in profiles.items()
            }
            rows.append({
                "voxel_size": float(voxel_size),
                "depth_ratio": float(depth_ratio),
                "resolution": estimate.resolution.copy(),
                "median_resolution": np.median(estimate.resolution, axis=0),
                "max_resolution": np.max(estimate.resolution, axis=0),
                "sample_ratio": float(
                    np.prod(estimate.resolution, axis=1).sum() /
                    (world.voxel.N * reference_res ** 3)
                ),
                "capped_fraction": float(estimate.capped.mean()),
                "matrix_l2": float(
                    sparse.linalg.norm(adaptive - reference) /
                    sparse.linalg.norm(reference)
                ),
                "fixed_max_matrix_l2": float(
                    sparse.linalg.norm(fixed_max - reference) /
                    sparse.linalg.norm(reference)
                ),
                "profile_errors": profile_errors,
                "worst_profile_l2": float(max(profile_errors.values())),
            })

    n_d, n_z = len(voxel_sizes), len(depth_ratios)

    def grid(key, component=None):
        values = []
        for row in rows:
            value = row[key]
            values.append(value if component is None else value[component])
        return np.asarray(values).reshape(n_d, n_z)

    panels = [
        (grid("median_resolution", 0), "median $r_x$", None),
        (grid("median_resolution", 1), "median $r_y$", None),
        (grid("median_resolution", 2), "median $r_z$", None),
        (grid("sample_ratio"), f"adaptive samples / fixed res {reference_res}", None),
        (grid("matrix_l2"), f"adaptive P relative L2 vs fixed {reference_res}", "log"),
        (grid("worst_profile_l2"), "worst profile image relative L2", "log"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for axis, (values, title, scale) in zip(axes.ravel(), panels):
        if scale == "log":
            plotted = np.maximum(values, 1e-16)
            vmin, vmax = plotted.min(), plotted.max()
            if vmax <= vmin:
                vmin, vmax = 0.9 * vmin, 1.1 * vmax
            image = axis.imshow(
                plotted, origin="lower", aspect="auto", cmap="viridis",
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
        else:
            plotted = values
            image = axis.imshow(plotted, origin="lower", aspect="auto", cmap="viridis")
        axis.set_xticks(np.arange(n_z), [f"{value:g}" for value in depth_ratios])
        axis.set_yticks(np.arange(n_d), [f"{value:g}" for value in voxel_sizes])
        axis.set_xlabel("source depth $Z/f$")
        axis.set_ylabel("voxel size d [mm]")
        axis.set_title(title)
        for i in range(n_d):
            for j in range(n_z):
                label = (f"{values[i, j]:.1e}" if scale == "log" else
                         f"{values[i, j]:.2g}")
                normalized = image.norm(plotted[i, j])
                text_color = "black" if normalized > 0.68 else "white"
                axis.text(j, i, label, ha="center", va="center", color=text_color,
                          fontsize=8)
        fig.colorbar(image, ax=axis)
    fig.suptitle(
        "Fully-visible small-voxel sweep; "
        f"adaptive max={adaptive_max_res}, step={max_projected_step:g} PSF scale",
    )
    fig.tight_layout()
    heatmap_path = output_dir / "small_voxel_depth_sweep.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    profile_names = list(rows[0]["profile_errors"])
    profile_fig, profile_axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, profile_name in zip(profile_axes.ravel(), profile_names):
        for voxel_size in voxel_sizes:
            selected = [row for row in rows if row["voxel_size"] == voxel_size]
            axis.plot(depth_ratios,
                      [row["profile_errors"][profile_name] for row in selected],
                      marker="o", label=f"d={voxel_size:g} mm")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(profile_name)
        axis.set_xlabel("source depth $Z/f$")
        axis.set_ylabel(f"image relative L2 vs fixed {reference_res}")
        axis.legend(fontsize=8)
    profile_fig.suptitle("Adaptive image error by profile")
    profile_fig.tight_layout()
    profile_path = output_dir / "small_voxel_profile_errors.png"
    profile_fig.savefig(profile_path, dpi=180)
    plt.close(profile_fig)
    return {
        "rows": rows,
        "heatmap_path": heatmap_path,
        "profile_path": profile_path,
    }


if __name__ == "__main__":
    result = run()
    print(result["heatmap_path"])
    print(result["profile_path"])
