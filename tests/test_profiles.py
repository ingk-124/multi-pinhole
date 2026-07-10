import numpy as np
import pytest

from multi_pinhole import Voxel
from multi_pinhole import profiles


def test_torus_to_poloidal_cartesian_passes_phi_through():
    r = np.array([0.0, 1.0, 2.0])
    theta = np.array([0.0, np.pi / 2, np.pi])
    phi = np.array([0.1, 0.2, 0.3])

    x, y, phi_out = profiles.torus_to_poloidal_cartesian(r, theta, phi)

    np.testing.assert_allclose(x, np.array([0.0, 0.0, -2.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y, np.array([0.0, 1.0, 0.0]), rtol=1e-12, atol=1e-12)
    assert phi_out is phi


def test_helical_poloidal_coordinates_use_displacement_phase_convention():
    r = np.array([1.0, 2.0])
    theta = np.array([np.pi / 3, np.pi / 2])
    phi = np.array([np.pi / 5, np.pi / 7])

    x, y, psi = profiles.helical_poloidal_coordinates(r, theta, phi, m_=2, n_=-3, phi_0=0.4)

    np.testing.assert_allclose(x, r * np.cos(theta), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y, r * np.sin(theta), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(psi, 2 * theta - (-3) * (phi - 0.4), rtol=1e-12, atol=1e-12)


def make_torus_voxel(coordinate_type="torus"):
    return Voxel(
        x_axis=np.linspace(0.8, 1.2, 3),
        y_axis=np.linspace(-0.2, 0.2, 3),
        z_axis=np.linspace(-0.2, 0.2, 3),
        coordinate_type=coordinate_type,
        coordinate_parameters={"major_radius": 1.0, "minor_radius": 0.2},
    )


def test_torus_coordinates_from_voxel_accepts_torus_and_torus_inverse():
    for coordinate_type in ("torus", "torus_inverse"):
        voxel = make_torus_voxel(coordinate_type=coordinate_type)

        r, theta, phi = profiles.torus_coordinates_from_voxel(voxel)

        assert r.shape == (voxel.N_voxel,)
        assert theta.shape == (voxel.N_voxel,)
        assert phi.shape == (voxel.N_voxel,)
        np.testing.assert_allclose(np.stack([r, theta, phi], axis=1), voxel.normalized_coordinates())


def test_torus_coordinates_from_voxel_rejects_non_torus_voxel():
    voxel = Voxel(
        x_axis=np.linspace(-1.0, 1.0, 3),
        y_axis=np.linspace(-1.0, 1.0, 3),
        z_axis=np.linspace(-1.0, 1.0, 3),
    )

    with pytest.raises(ValueError, match="torus"):
        profiles.torus_coordinates_from_voxel(voxel)


def test_evaluate_torus_profile_matches_direct_profile_call():
    voxel = make_torus_voxel()
    r, theta, phi = voxel.normalized_coordinates().T

    def torus_profile(r, theta, phi, A):
        return A * r * np.cos(theta - phi)

    wrapped = profiles.evaluate_torus_profile(voxel, torus_profile, A=2.0)
    direct = torus_profile(r, theta, phi, A=2.0)

    np.testing.assert_allclose(wrapped, direct, rtol=1e-12, atol=1e-12)


def test_evaluate_poloidal_profile_matches_direct_profile_call():
    voxel = make_torus_voxel(coordinate_type="torus_inverse")
    r, theta, phi = voxel.normalized_coordinates().T
    x, y, phi = profiles.torus_to_poloidal_cartesian(r, theta, phi)
    params = dict(A=2.0, delta=0.1, alpha=2, beta=3, xi_0=0.1, rho_s=0.5, d=2)

    wrapped = profiles.evaluate_poloidal_profile(voxel, profiles.kinked_profile, **params)
    direct = profiles.kinked_profile(x, y, phi=phi, **params)

    np.testing.assert_allclose(wrapped, direct, rtol=1e-12, atol=1e-12)


def test_shifted_polar_keeps_original_boundary_at_unit_radius():
    rho, theta = profiles.shifted_polar(np.array([1.0, 0.0]), np.array([0.0, 1.0]), cx=0.2, cy=-0.1)

    np.testing.assert_allclose(rho, np.ones(2), rtol=1e-12, atol=1e-12)
    assert theta.shape == rho.shape


def test_shifted_polar_can_skip_boundary_normalization():
    rho, theta = profiles.shifted_polar(
        np.array([1.0, -1.0]),
        np.array([0.0, 0.0]),
        cx=-0.2,
        cy=0,
        normalize_boundary=False,
    )

    np.testing.assert_allclose(rho, np.array([1.2, 0.8]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(theta, np.array([0.0, np.pi]), rtol=1e-12, atol=1e-12)


def test_gaussian_odd_exponent_decays_on_both_sides_and_supports_scalar_input():
    left = profiles.gaussian(0.25, rho_s=0.5, w=0.5, d=3, edge=None)
    center = profiles.gaussian(0.5, rho_s=0.5, w=0.5, d=3, edge=None)
    outside = profiles.gaussian(1.2, rho_s=0.5, w=0.5, d=3, edge=None)

    assert np.isscalar(left) or left.shape == ()
    assert 0 < left < center
    assert center == 1
    assert outside == 0


def test_two_power_derivative_is_zero_not_nan_outside_boundary():
    rho = np.array([0.0, 0.5, 1.0, 1.2])

    derivative = profiles.two_power_derivative(rho, alpha=2, beta=3)

    assert np.isfinite(derivative).all()
    assert derivative[-1] == 0


def test_two_power_and_derivative_support_scalar_inputs():
    profile = profiles.two_power(1.2, alpha=2, beta=3)
    derivative = profiles.two_power_derivative(1.2, alpha=2, beta=3)

    assert profile == 0
    assert derivative == 0


def test_profiles_accept_arrays_and_scalars_without_global_constants():
    x = np.array([-0.5, 0.0, 0.5])
    y = np.zeros_like(x)
    phi = np.linspace(0, np.pi, x.size)

    axisymmetric = profiles.axisymmetric_profile(x, y, A=2.0, delta=0.1, alpha=2, beta=3)
    kinked = profiles.kinked_profile(
        x, y, A=2.0, delta=0.1, alpha=2, beta=3, xi_0=0.1, rho_s=0.5, d=2, phi=phi
    )
    flattened = profiles.flattening_profile(
        x, y, A=2.0, delta=0.1, alpha=2, beta=3,
        xi_0=0.1, rho_s=0.5, d=2, w=0.2, gamma=0.1, phi=phi
    )
    scalar = profiles.axisymmetric_profile(0.0, 0.0, A=2.0, delta=0.1, alpha=2, beta=3)

    assert axisymmetric.shape == x.shape
    assert kinked.shape == x.shape
    assert flattened.shape == x.shape
    assert np.isfinite(axisymmetric).all()
    assert np.isfinite(kinked).all()
    assert np.isfinite(flattened).all()
    assert np.isfinite(scalar)
