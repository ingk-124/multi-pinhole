"""Compare numerical pinhole etendue against an analytic X-axis scan."""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multi_pinhole_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "multi_pinhole_cache"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from multi_pinhole import Eye, Screen
from multi_pinhole.core import _spot_cell_overlap


def analytic_point_source_etendue(eye, points):
    """Point-source pinhole solid angle divided by 4 pi.

    The detector plane is perpendicular to the camera z axis.  ``Z`` is the
    axial source-to-pinhole distance and ``cos_theta`` accounts for off-axis
    viewing of the pinhole aperture.
    """
    axial_distance = points[:, 2] - eye.position[2]
    radial_offset = np.linalg.norm(points[:, :2] - eye.position[:2], axis=1)
    cos_theta = axial_distance / np.sqrt(axial_distance ** 2 + radial_offset ** 2)
    pinhole_area = np.pi * (eye.eye_size[0] / 2) ** 2
    return pinhole_area * cos_theta ** 3 / (4 * np.pi * axial_distance ** 2)


def analytic_spot_image(screen, eye, point):
    """Analytic top-hat spot integrated over each screen subpixel."""
    rays = eye.calc_rays(point[None, :])
    uv_center = screen.xy2uv(rays.XY)[0]
    spot_size = eye.eye_size * rays.zoom_rate[0]
    half = 0.5 * spot_size
    u_axis = np.unique(screen.subpixel_position[:, 0])
    v_axis = np.unique(screen.subpixel_position[:, 1])
    du, dv = screen.subpixel_size
    i0 = max(0, int(np.floor((uv_center[0] - half[0]) / du)))
    i1 = min(len(u_axis) - 1, int(np.ceil((uv_center[0] + half[0]) / du) - 1))
    j0 = max(0, int(np.floor((uv_center[1] - half[1]) / dv)))
    j1 = min(len(v_axis) - 1, int(np.ceil((uv_center[1] + half[1]) / dv) - 1))
    overlap = np.zeros(screen.N_subpixel)
    use_ellipse = eye.eye_shape in ("circle", "ellipse")
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            flat_index = i * len(v_axis) + j
            overlap[flat_index] = _spot_cell_overlap(
                u_axis[i] - 0.5 * du, u_axis[i] + 0.5 * du,
                v_axis[j] - 0.5 * dv, v_axis[j] + 0.5 * dv,
                uv_center[0], uv_center[1], half[0], half[1], use_ellipse,
            )

    z = point[2] - eye.position[2]
    ray_offset = rays.XY[0] - eye.position[:2]
    ray_tangent = np.linalg.norm(ray_offset) / eye.focal_length
    ray_cosine = 1.0 / np.sqrt(1.0 + ray_tangent ** 2)
    weight_density = (screen.etendue_per_subpixel(eye) / screen.A_subpixel
                      / ((rays.zoom_rate[0] ** 2) * (z ** 2) * ray_cosine))
    return overlap * weight_density


def analytic_screen_image(screen, eye, points, source_strength):
    """Sum analytic finite-pinhole spots for a line of point sources."""
    image = np.zeros(screen.N_subpixel)
    for point, strength in zip(points, source_strength):
        image += strength * analytic_spot_image(screen, eye, point)
    return image


def run(output_dir=None, n_points=81, pixel_shape=(220, 220), subpixel_resolution=8,
        axial_distance=100.0, x_extent=30.0):
    """Run the X-axis etendue scan and save a comparison plot."""
    output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "multi_pinhole_etendue"
    output_dir.mkdir(parents=True, exist_ok=True)

    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=1.0)
    screen = Screen(screen_shape="square", screen_size=24.0, pixel_shape=pixel_shape,
                    subpixel_resolution=subpixel_resolution)

    x = np.linspace(-x_extent, x_extent, n_points)
    points = np.column_stack([x, np.zeros_like(x), np.full_like(x, eye.position[2] + axial_distance)])
    rays = eye.calc_rays(points)
    P_sources = screen.ray2image_grid(eye, rays)
    source_strength = np.ones(n_points)
    numerical_screen = P_sources @ source_strength
    analytic_screen = analytic_screen_image(screen, eye, points, source_strength)
    screen_relative_error = ((numerical_screen.sum() - analytic_screen.sum()) / analytic_screen.sum())

    numerical = np.asarray(P_sources.sum(axis=0)).ravel()
    analytic = analytic_point_source_etendue(eye, points)
    relative_error = (numerical - analytic) / analytic

    spot_index = n_points // 2
    numerical_spot = P_sources[:, spot_index].toarray().ravel()
    analytic_spot = analytic_spot_image(screen, eye, points[spot_index])
    spot_relative_error = ((numerical_spot.sum() - analytic_spot.sum()) / analytic_spot.sum())

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].plot(x, analytic, label="analytic", color="black", linewidth=1.5)
    axes[0].scatter(x, numerical, label="ray2image_grid", color="tab:blue", s=18)
    axes[0].set_xlabel("source x")
    axes[0].set_ylabel("total etendue")
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].plot(x, relative_error, marker="o", color="tab:orange", linewidth=1.0, markersize=3)
    axes[1].set_xlabel("source x")
    axes[1].set_ylabel("relative error")

    fig.tight_layout()
    output_path = output_dir / "etendue_x_scan.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    spot_rays = eye.calc_rays(points[spot_index][None, :])
    spot_uv_center = screen.xy2uv(spot_rays.XY)[0]
    spot_half_width = 0.5 * eye.eye_size * spot_rays.zoom_rate[0]
    margin = 3.0 * float(np.max(spot_half_width))

    image_shape = tuple(screen.subpixel_shape)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    extent = [0.0, screen.screen_size[1], screen.screen_size[0], 0.0]
    vmax = max(numerical_spot.max(), analytic_spot.max())
    im0 = axes[0].imshow(numerical_spot.reshape(image_shape), origin="upper", extent=extent, vmax=vmax)
    axes[0].set_title("numerical")
    axes[0].set_xlabel("v")
    axes[0].set_ylabel("u")
    axes[0].set_xlim(spot_uv_center[1] - margin, spot_uv_center[1] + margin)
    axes[0].set_ylim(spot_uv_center[0] + margin, spot_uv_center[0] - margin)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(analytic_spot.reshape(image_shape), origin="upper", extent=extent, vmax=vmax)
    axes[1].set_title("analytic top-hat")
    axes[1].set_xlabel("v")
    axes[1].set_xlim(spot_uv_center[1] - margin, spot_uv_center[1] + margin)
    axes[1].set_ylim(spot_uv_center[0] + margin, spot_uv_center[0] - margin)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    diff = numerical_spot - analytic_spot
    im2 = axes[2].imshow(diff.reshape(image_shape), origin="upper", extent=extent, cmap="coolwarm")
    axes[2].set_title("difference")
    axes[2].set_xlabel("v")
    axes[2].set_xlim(spot_uv_center[1] - margin, spot_uv_center[1] + margin)
    axes[2].set_ylim(spot_uv_center[0] + margin, spot_uv_center[0] - margin)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    spot_output_path = output_dir / "etendue_spot_distribution.png"
    fig.savefig(spot_output_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    vmax = max(numerical_screen.max(), analytic_screen.max())
    im0 = axes[0].imshow(numerical_screen.reshape(image_shape), origin="upper", extent=extent, vmax=vmax)
    axes[0].set_title("P @ f")
    axes[0].set_xlabel("v")
    axes[0].set_ylabel("u")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(analytic_screen.reshape(image_shape), origin="upper", extent=extent, vmax=vmax)
    axes[1].set_title("analytic convolution")
    axes[1].set_xlabel("v")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    screen_diff = numerical_screen - analytic_screen
    im2 = axes[2].imshow(screen_diff.reshape(image_shape), origin="upper", extent=extent, cmap="coolwarm")
    axes[2].set_title("difference")
    axes[2].set_xlabel("v")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    screen_output_path = output_dir / "etendue_screen_convolution.png"
    fig.savefig(screen_output_path, dpi=180)
    plt.close(fig)

    center_row = image_shape[0] // 2
    numerical_profile = numerical_screen.reshape(image_shape)[center_row]
    analytic_profile = analytic_screen.reshape(image_shape)[center_row]
    v_axis = np.linspace(screen.subpixel_size[1] * 0.5,
                         screen.screen_size[1] - screen.subpixel_size[1] * 0.5,
                         image_shape[1])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(v_axis, analytic_profile, label="analytic convolution", color="black", linewidth=1.5)
    ax.scatter(v_axis, numerical_profile, label="P @ f", color="tab:blue", s=4)
    ax.set_xlabel("v")
    ax.set_ylabel("center-row intensity")
    ax.legend()
    fig.tight_layout()
    profile_output_path = output_dir / "etendue_screen_profile.png"
    fig.savefig(profile_output_path, dpi=180)
    plt.close(fig)

    return {
        "x": x,
        "points": points,
        "P_sources": P_sources,
        "source_strength": source_strength,
        "numerical": numerical,
        "analytic": analytic,
        "relative_error": relative_error,
        "numerical_spot": numerical_spot,
        "analytic_spot": analytic_spot,
        "spot_relative_error": spot_relative_error,
        "numerical_screen": numerical_screen,
        "analytic_screen": analytic_screen,
        "screen_relative_error": screen_relative_error,
        "output_path": output_path,
        "spot_output_path": spot_output_path,
        "screen_output_path": screen_output_path,
        "profile_output_path": profile_output_path,
    }


if __name__ == "__main__":
    result = run()
    max_error = np.max(np.abs(result["relative_error"]))
    print(f"plot: {result['output_path']}")
    print(f"spot plot: {result['spot_output_path']}")
    print(f"screen plot: {result['screen_output_path']}")
    print(f"profile plot: {result['profile_output_path']}")
    print(f"max relative error: {max_error:.3%}")
    print(f"spot relative error: {result['spot_relative_error']:.3%}")
    print(f"screen relative error: {result['screen_relative_error']:.3%}")
