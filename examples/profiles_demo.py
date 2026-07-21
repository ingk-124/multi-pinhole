"""Visual demonstration of the generic poloidal profile helpers.

Run from the repository root with::

    python examples/profiles_demo.py

The coordinates and amplitudes in this example are dimensionless.  Applications
can assign physical units to the profile amplitude as needed.
"""

import matplotlib.pyplot as plt
import numpy as np

from multi_pinhole import profiles


def plot_radial_cross_sections(x, parameters):
    """Compare transformed radii and profiles along the horizontal midplane."""
    axisymmetric = parameters["axisymmetric"]
    kinked = parameters["kinked"]
    flattened = parameters["flattened"]
    full_flattening = parameters["full_flattening"]

    rho_shifted, _ = profiles.shifted_polar(
        x, 0, cx=axisymmetric["delta"], cy=0,
    )
    rho_kinked, _ = profiles.kinked_rho(
        x, 0, **{key: kinked[key] for key in ("delta", "xi_0", "rho_s", "d")},
        phi=0,
        psi_0=kinked["psi_0"],
    )
    rho_flattened, _ = profiles.flattening_rho(
        x,
        0,
        **{
            key: flattened[key]
            for key in ("delta", "xi_0", "rho_s", "d", "w", "gamma", "lam_0")
        },
        phi=0,
        psi_0=flattened["psi_0"],
        psi_1=flattened["psi_1"],
    )

    values = [
        profiles.axisymmetric_profile(x, 0, **axisymmetric),
        profiles.kinked_profile(x, 0, **kinked, phi=0),
        profiles.flattening_profile(x, 0, **flattened, phi=0),
        profiles.flattening_profile(x, 0, **full_flattening, phi=0),
    ]
    rho_full_flattening, _ = profiles.flattening_rho(
        x,
        0,
        **{
            key: full_flattening[key]
            for key in ("delta", "xi_0", "rho_s", "d", "w", "gamma", "lam_0")
        },
        phi=0,
        psi_0=full_flattening["psi_0"],
        psi_1=full_flattening["psi_1"],
    )
    radii = [rho_shifted, rho_kinked, rho_flattened, rho_full_flattening]
    labels = [
        "Shifted axisymmetric",
        "Kinked",
        r"Flattened ($\lambda_0=0.5$)",
        r"Full flattening ($\lambda_0=1$)",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for radius, value, label in zip(radii, values, labels):
        axes[0].plot(x, radius, label=label)
        axes[1].plot(x, value, label=label)

    axes[0].set_ylabel(r"Normalized radius $\rho$")
    axes[1].set_ylabel("Profile amplitude")
    axes[1].set_xlabel("Normalized poloidal coordinate x")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(r"Poloidal profiles along $y=0$ at $\phi=0$")
    return fig


def plot_phase_slices(x, y, phi, parameters, profile_name):
    """Plot one non-axisymmetric profile at several toroidal phases."""
    xx, yy, pp = np.meshgrid(x, y, phi, indexing="ij")
    axisymmetric = profiles.axisymmetric_profile(
        xx, yy, **parameters["axisymmetric"],
    )

    if profile_name == "kinked":
        values = profiles.kinked_profile(xx, yy, **parameters["kinked"], phi=pp)
    elif profile_name == "flattened":
        values = profiles.flattening_profile(
            xx, yy, **parameters["flattened"], phi=pp,
        )
    elif profile_name == "full flattening":
        values = profiles.flattening_profile(
            xx, yy, **parameters["full_flattening"], phi=pp,
        )
    else:
        raise ValueError(f"Unknown profile name: {profile_name}")

    ncols = 4
    nrows = int(np.ceil(phi.size / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols + 1, 2.8 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    levels = np.linspace(0, parameters["axisymmetric"]["A"], 16)
    contour = None
    for index, ax in enumerate(axes.flat):
        if index >= phi.size:
            ax.set_visible(False)
            continue
        contour = ax.contourf(
            x, y, values[:, :, index].T, levels=levels, cmap="viridis",
        )
        ax.contour(
            x,
            y,
            axisymmetric[:, :, index].T,
            levels=8,
            colors="white",
            linewidths=0.5,
            linestyles="--",
        )
        ax.set_title(rf"$\phi={phi[index] / np.pi:.2g}\pi$")
        ax.set_aspect("equal")

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.suptitle(f"{profile_name.capitalize()} profile over toroidal phase")
    fig.colorbar(
        contour,
        ax=axes,
        label="Profile amplitude",
        shrink=0.9,
        pad=0.02,
    )
    return fig


def main():
    x = np.linspace(-1, 1, 101)
    y = np.linspace(-1, 1, 101)
    phi = np.linspace(-np.pi, np.pi, 8, endpoint=False)

    axisymmetric = dict(A=1.0, delta=0.2, alpha=2.0, beta=3.0)
    kinked = axisymmetric | dict(xi_0=0.4, rho_s=0.3, d=2.0, psi_0=0.0)
    flattened = kinked | dict(w=0.4, gamma=0.1, lam_0=0.5, psi_1=np.pi)
    full_flattening = flattened | dict(lam_0=1.0)
    parameters = {
        "axisymmetric": axisymmetric,
        "kinked": kinked,
        "flattened": flattened,
        "full_flattening": full_flattening,
    }

    plot_radial_cross_sections(x, parameters)
    plot_phase_slices(x, y, phi, parameters, "kinked")
    plot_phase_slices(x, y, phi, parameters, "flattened")
    plot_phase_slices(x, y, phi, parameters, "full flattening")
    plt.show()


if __name__ == "__main__":
    main()
