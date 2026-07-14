"""Visualize geometry-only adaptive source-resolution diagnostics."""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from multi_pinhole import Camera, Eye, Screen, Voxel
from multi_pinhole.projection import projected_axis_spans, select_source_resolution


def build_case():
    """Return a wall-free x-z source slice viewed by an unrotated camera."""
    camera = Camera(
        eyes=[Eye(position=(0.0, 0.0), focal_length=20.0,
                  eye_type="pinhole", eye_shape="circle", eye_size=1.0)],
        apertures=[],
        screen=Screen(
            screen_shape="rectangle", screen_size=(30.0, 40.0),
            pixel_shape=(30, 40), subpixel_resolution=4,
        ),
        camera_position=(0.0, 0.0, 0.0),
    )
    voxel = Voxel.uniform_voxel(
        ranges=((-15.0, 15.0), (-2.0, 2.0), (60.0, 400.0)),
        shape=(30, 1, 50),
    )
    return camera, voxel


def voxel_edge_lengths(voxel, indices):
    """Return compact per-cell world-axis lengths for selected voxels."""
    positions = voxel.get_voxel_position(indices).astype(int)
    return np.column_stack((
        voxel.dx_axis[positions[:, 0]],
        voxel.dy_axis[positions[:, 1]],
        voxel.dz_axis[positions[:, 2]],
    ))


def run(output_dir=None, max_resolution=8, max_projected_step=1.0):
    """Calculate and plot projected spans and selected axis resolutions."""
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(tempfile.gettempdir()) / "multi_pinhole_adaptive_res")
    output_dir.mkdir(parents=True, exist_ok=True)
    camera, voxel = build_case()
    indices = np.arange(voxel.N_voxel)
    centers = voxel.get_gravity_center(indices)
    edge_lengths = voxel_edge_lengths(voxel, indices)
    spans = projected_axis_spans(camera, 0, centers, edge_lengths)
    estimate = select_source_resolution(
        spans,
        detector_pitch=camera.screen.subpixel_size,
        max_resolution=max_resolution,
        max_projected_step=max_projected_step,
    )

    x, z = centers[:, 0], centers[:, 2]
    values = [
        np.max(estimate.projected_span_cells, axis=1),
        estimate.resolution[:, 0],
        estimate.resolution[:, 1],
        estimate.resolution[:, 2],
    ]
    titles = [
        "Largest unrefined span [subpixels]",
        "Selected source $r_x$",
        "Selected source $r_y$",
        "Selected source $r_z$",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, value, title in zip(axes.ravel(), values, titles):
        scatter = axis.scatter(x, z, c=value, s=18, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("world x [mm]")
        axis.set_ylabel("world z [mm]")
        fig.colorbar(scatter, ax=axis)
    fig.suptitle(
        "Geometry-only adaptive source resolution\n"
        f"reference pitch={camera.screen.subpixel_size[0]:g} mm, "
        f"max step={max_projected_step:g} subpixel, max res={max_resolution}",
    )
    fig.tight_layout()
    output_path = output_dir / "adaptive_source_resolution_geometry.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "camera": camera,
        "voxel": voxel,
        "spans": spans,
        "estimate": estimate,
        "output_path": output_path,
    }


if __name__ == "__main__":
    result = run()
    print(result["output_path"])
