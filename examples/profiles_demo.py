"""Visual demonstration of the generic poloidal profile helpers.

Run from the repository root with::

    python examples/profiles_demo.py

The coordinates and amplitudes are dimensionless. Applications can assign
physical units to the profile amplitude as needed.
"""

import matplotlib.pyplot as plt
import numpy as np

from multi_pinhole import profiles


def parameter_title(name, parameters, keys):
    """Build a plot title from the parameters actually used."""
    values = ", ".join(rf"${key}={parameters[key]:g}$" for key in keys)
    return f"{name}: {values}"


def plot_cross_sections(x, cases, title, *, show_radius=False):
    """Compare profile, and optionally radius, along the horizontal chord."""
    nrows = 2 if show_radius else 1
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(8, 6 if show_radius else 4.5),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    profile_ax = axes[-1, 0]
    for case in cases:
        profile_ax.plot(
            x,
            case["profile"](x, 0, **case["parameters"]),
            label=case["label"],
        )
        if show_radius:
            radius, _ = case["radius"](x, 0, **case["radius_parameters"])
            axes[0, 0].plot(x, radius, label=case["label"])

    if show_radius:
        axes[0, 0].set_ylabel(r"Normalized radius $\rho$")
        axes[0, 0].legend()
    profile_ax.set_xlabel("Normalized poloidal coordinate x")
    profile_ax.set_ylabel("Profile amplitude")
    profile_ax.legend()
    fig.suptitle(title)
    return fig


def plot_maps(x, y, cases, title):
    """Plot representative two-dimensional profiles in one comparison figure."""
    xx, yy = np.meshgrid(x, y, indexing="xy")
    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=(3.3 * len(cases) + 0.8, 3.3),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    levels = np.linspace(0, 1, 16)
    contour = None
    for ax, case in zip(axes[0], cases, strict=True):
        values = case["profile"](xx, yy, **case["parameters"])
        contour = ax.contourf(xx, yy, values, levels=levels, cmap="viridis")
        ax.set_title(case["label"])
        ax.set_aspect("equal")
        ax.set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.suptitle(title)
    fig.colorbar(contour, ax=axes, label="Profile amplitude", shrink=0.85)
    return fig


def main():
    x = np.linspace(-1, 1, 201)
    y = np.linspace(-1, 1, 201)
    base = dict(A=1.0, delta=0.2, alpha=2.0, beta=3.0)

    flat_kink = base | dict(
        xi_0=0.2,
        rho_s=0.3,
        d=2.0,
        center_angle_xy=0.0,
        normalize_kink=False,
    )
    flat_cases = [
        {
            "label": "Kinked",
            "profile": profiles.kinked_profile,
            "parameters": flat_kink,
        },
        *[
            {
                "label": rf"Full angle, $\lambda_0={lam_0:g}$",
                "profile": profiles.flattening_profile,
                "parameters": flat_kink | dict(lam_0=lam_0),
            }
            for lam_0 in (1.0, 0.5)
        ],
    ]
    localized_case = {
        "label": r"Localized option, $\lambda_0=1$",
        "profile": profiles.flattening_profile,
        "parameters": flat_kink | dict(localized=True, lam_0=1.0, rho_flat=0.3, w=0.4),
    }
    flat_title = parameter_title(
        "Density-island flattening", flat_kink, ("xi_0", "rho_s", "d")
    )
    plot_cross_sections(x, flat_cases, flat_title)
    plot_maps(x, y, flat_cases + [localized_case], flat_title)

    crescent_kink = base | dict(
        xi_0=0.7,
        rho_s=0.3,
        d=4.0,
        center_angle_xy=0.0,
        normalize_kink=False,
    )
    crescent_parameters = {
        key: value for key, value in crescent_kink.items() if key != "normalize_kink"
    }
    radius_keys = ("delta", "xi_0", "rho_s", "d", "center_angle_xy")
    crescent_cases = [
        {
            "label": "Kinked",
            "profile": profiles.kinked_profile,
            "parameters": crescent_kink,
            "radius": profiles.kinked_rho,
            "radius_parameters": {key: crescent_kink[key] for key in radius_keys},
        },
        {
            "label": "Crescent",
            "profile": profiles.crescent_profile,
            "parameters": crescent_parameters,
            "radius": profiles.crescent_rho,
            "radius_parameters": {key: crescent_parameters[key] for key in radius_keys},
        },
    ]
    crescent_title = parameter_title("Crescent vs. kink", crescent_kink, ("xi_0", "d"))
    plot_cross_sections(x, crescent_cases, crescent_title, show_radius=True)
    plot_maps(x, y, crescent_cases, crescent_title)
    plt.show()


if __name__ == "__main__":
    main()
