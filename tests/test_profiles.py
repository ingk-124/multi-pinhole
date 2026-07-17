import numpy as np
from multi_pinhole import profiles
from multi_pinhole.coordinates import spherical_coordinates


def test_spherical_coordinates_angles_do_not_depend_on_reference_radius():
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 2.0], [0.0, -3.0, 0.0]])
    unit = spherical_coordinates(1.0)(points)
    scaled = spherical_coordinates(4.0)(points)

    np.testing.assert_allclose(scaled[:, 0], unit[:, 0] / 4.0)
    np.testing.assert_allclose(scaled[:, 1:], unit[:, 1:])


def test_spherical_coordinates_axes_and_general_point():
    points = np.array([
        [0.0, 0.0, 2.0],
        [0.0, 0.0, -2.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, -2.0, 3.0],
    ])
    result = spherical_coordinates(7.0)(points)

    np.testing.assert_allclose(result[:4, 1], [0.0, np.pi, np.pi / 2, np.pi / 2])
    np.testing.assert_allclose(result[2:4, 2], [0.0, np.pi / 2])
    expected_theta = np.arccos(points[-1, 2] / np.linalg.norm(points[-1]))
    np.testing.assert_allclose(result[-1, 1], expected_theta)


def test_spherical_coordinates_origin_and_roundoff_handling():
    points = np.array([[0.0, 0.0, 0.0], [1e-300, 0.0, 1.0]])
    result = spherical_coordinates(2.0)(points)

    assert np.isnan(result[0, 1])
    assert np.isfinite(result[1, 1])
    np.testing.assert_allclose(result[1, 1], 0.0)


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


def test_kinked_profile_axisymmetric_limit_and_phase_periodicity():
    x = np.linspace(-0.7, 0.7, 15)
    y = np.linspace(0.3, -0.3, 15)
    parameters = dict(A=2.0, delta=0.1, alpha=2.5, beta=3.0)

    axisymmetric = profiles.axisymmetric_profile(x, y, **parameters)
    kink_limit = profiles.kinked_profile(
        x, y, **parameters, xi_0=0.0, rho_s=0.4, d=2.0, phi=0.7,
    )
    kink = profiles.kinked_profile(
        x, y, **parameters, xi_0=0.15, rho_s=0.4, d=2.0, phi=0.7,
        psi_0=-0.2,
    )
    periodic = profiles.kinked_profile(
        x, y, **parameters, xi_0=0.15, rho_s=0.4, d=2.0,
        phi=0.7 + 2 * np.pi, psi_0=-0.2,
    )

    np.testing.assert_allclose(kink_limit, axisymmetric, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(periodic, kink, rtol=1e-12, atol=1e-12)


def test_minimum_flattening_matches_independent_reference_formula():
    x = np.linspace(-0.6, 0.8, 17)
    y = np.linspace(0.2, -0.4, 17)
    parameters = dict(delta=0.08, xi_0=0.12, rho_s=0.45, d=2.3,
                      phi=0.6, psi_0=-0.1)

    rho_shifted, _ = profiles.shifted_polar(x, y, parameters["delta"], 0)
    rho_kinked, theta_kinked = profiles.kinked_rho(x, y, **parameters)
    expected_rho = np.minimum(
        rho_kinked,
        np.logaddexp(parameters["rho_s"] / 0.03, rho_shifted / 0.03) * 0.03,
    )
    rho, theta = profiles.flattening_rho_min(x, y, **parameters)

    np.testing.assert_allclose(rho, expected_rho, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(theta, theta_kinked, rtol=1e-12, atol=1e-12)
    expected_profile = 1.7 * profiles.two_power(expected_rho, 2.0, 3.0)
    actual_profile = profiles.flattening_profile_min(
        x, y, A=1.7, alpha=2.0, beta=3.0, **parameters,
    )
    np.testing.assert_allclose(actual_profile, expected_profile, rtol=1e-12, atol=1e-12)
