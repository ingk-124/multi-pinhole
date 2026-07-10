"""Verify that any intersected aperture blocks a ray.

Run from the repository root with:

    python examples/verify_aperture_blocking.py
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multi_pinhole_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/multi_pinhole_cache")

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from multi_pinhole import Aperture, Camera, Eye, Screen
from multi_pinhole.utils import stl_utils


def _point_on_ray_at_z(start, end, z):
    t = (z - start[2]) / (end[2] - start[2])
    return start + t * (end - start)


def plot_geometry(eye, apertures, sources, combined_visible, output_path="/tmp/aperture_blocking_rays.png"):
    eye_position = eye.position

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    x_lim = (-3, 5)
    y_lim = (-3, 3)
    z_lim = (0, 32)

    # Screen plane at z=0.
    screen_x = np.array([-10, 10, 10, -10, -10])
    screen_y = np.array([-10, -10, 10, 10, -10])
    screen_z = np.zeros_like(screen_x)
    ax.plot(screen_x, screen_y, screen_z, color="0.45", lw=1.2, label="screen")

    aperture_styles = [
        ("large aperture", "tab:blue", 0.22),
        ("small aperture", "tab:orange", 0.35),
    ]
    for aperture, (label, color, alpha) in zip(apertures, aperture_styles):
        stl_utils.show_stl(
            aperture.stl_model,
            ax=ax,
            facecolors=color,
            edgecolors=color,
            alpha=alpha,
            lw=0.2,
            x_lim=x_lim,
            y_lim=y_lim,
            z_lim=z_lim,
            full_model=False,
        )
        ax.plot([], [], color=color, lw=4, alpha=alpha, label=f"{label} (z={aperture.position[2]:.1f})")

    ax.scatter(*eye_position, color="black", marker="x", s=80, label="eye")

    source_labels = ["source A: visible", "source B: blocked"]
    colors = ["tab:green" if v else "tab:red" for v in combined_visible]
    for source, label, color in zip(sources, source_labels, colors):
        ax.scatter(*source, color=color, s=55, label=label)
        ax.plot(
            [eye_position[0], source[0]],
            [eye_position[1], source[1]],
            [eye_position[2], source[2]],
            color=color,
            lw=2.0,
        )
        for aperture in apertures:
            hit = _point_on_ray_at_z(eye_position, source, aperture.position[2])
            ax.scatter(*hit, color=color, s=22)

    blocked_hit = _point_on_ray_at_z(eye_position, sources[1], apertures[1].position[2])
    ax.text(*blocked_hit, "  blocked by small aperture", color="tab:red")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.view_init(elev=18, azim=-62)
    ax.set_title("Two rays through two aperture masks")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return output_path


def main():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(4, 4), subpixel_resolution=20)

    # The apertures are placed at different z positions. Source A passes through both holes.
    # Source B passes through the larger hole but hits the smaller aperture mask.
    large_aperture = Aperture(shape="circle", size=1.5, position=(0.0, 0.0, 14.0)).set_model(
        resolution=40, max_size=10
    )
    small_aperture = Aperture(shape="circle", size=0.5, position=(0.0, 0.0, 17.0)).set_model(
        resolution=40, max_size=10
    )
    apertures = [large_aperture, small_aperture]

    sources = np.array(
        [
            [0.0, 0.0, 30.0],  # A: passes both apertures
            [4.0, 0.0, 30.0],  # B: blocked by the small aperture
        ]
    )

    per_aperture_visible = np.array(
        [
            stl_utils.check_visible(
                mesh_obj=aperture.stl_model,
                start=eye.position,
                grid_points=sources,
                behind_start_included=True,
            )
            for aperture in apertures
        ]
    )
    combined_visible = np.all(per_aperture_visible, axis=0)

    np.testing.assert_array_equal(per_aperture_visible, np.array([[True, True], [True, False]]))
    np.testing.assert_array_equal(combined_visible, np.array([True, False]))

    camera = Camera(
        eyes=[eye],
        apertures=apertures,
        screen=screen,
        camera_position=(0.0, 0.0, 0.0),
    )
    image = camera.calc_image_vec(0, sources, check_visibility=True).tocsc()

    assert image[:, 0].nnz > 0, "source A should reach the screen"
    assert image[:, 1].nnz == 0, "source B should be blocked by one aperture"

    print("per aperture visible:")
    print(per_aperture_visible)
    print("combined visible:", combined_visible)
    output_path = plot_geometry(eye, apertures, sources, combined_visible)
    print("plot:", output_path)
    print("OK: source A is visible, source B is blocked.")


if __name__ == "__main__":
    main()
