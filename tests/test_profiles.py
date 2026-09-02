import inspect

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from multi_pinhole import profiles
from multi_pinhole.coordinates import spherical_coordinates


def test_spherical_coordinates_angles_do_not_depend_on_reference_radius():
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 2.0], [0.0, -3.0, 0.0]])
    unit = spherical_coordinates(1.0)(points)
    scaled = spherical_coordinates(4.0)(points)

    np.testing.assert_allclose(scaled[:, 0], unit[:, 0] / 4.0)
    np.testing.assert_allclose(scaled[:, 1:], unit[:, 1:])


def test_spherical_coordinates_axes_and_general_point():
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, -2.0, 3.0],
        ]
    )
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


def test_helical_center_angle_uses_signed_mode_numbers_and_reference():
    phi = np.array([-0.4, 0.2, 0.8])
    center = profiles.helical_center_angle(
        phi,
        center_angle_xy_ref=0.3,
        m=2,
        n=-1,
        phi_ref=0.2,
    )

    np.testing.assert_allclose(center, 0.3 - 0.5 * (phi - 0.2))
    assert center[1] == 0.3


def test_helical_center_angle_broadcasts_voxels_and_phase_bins():
    phi = np.linspace(-0.5, 0.5, 4)[:, None]
    center_ref = np.linspace(-np.pi, np.pi, 12, endpoint=False)[None, :]

    center = profiles.helical_center_angle(
        phi,
        center_ref,
        m=1,
        n=-1,
        phi_ref=0.1,
    )

    assert center.shape == (4, 12)
    np.testing.assert_allclose(
        center,
        center_ref - (phi - 0.1),
    )


def test_helical_center_angle_matches_phase_residual_identity():
    theta_xy = np.linspace(-1.0, 1.0, 7)
    phi = np.linspace(-0.6, 0.9, 7)
    center_ref = 0.25
    m, n, phi_ref = 2, -3, 0.15
    center = profiles.helical_center_angle(
        phi,
        center_ref,
        m=m,
        n=n,
        phi_ref=phi_ref,
    )

    expected = m * (theta_xy - center_ref) - n * (phi - phi_ref)
    np.testing.assert_allclose(m * (theta_xy - center), expected)


def test_helical_center_angle_rejects_zero_m():
    with np.testing.assert_raises_regex(ValueError, "m must be nonzero"):
        profiles.helical_center_angle(
            0.0,
            center_angle_xy_ref=0.0,
            m=0,
            n=-1,
            phi_ref=0.0,
        )


def test_shifted_polar_keeps_original_boundary_at_unit_radius():
    rho, theta = profiles.shifted_polar(
        np.array([1.0, 0.0]), np.array([0.0, 1.0]), cx=0.2, cy=-0.1
    )

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


def test_kink_displaced_polar_uses_explicit_poloidal_center_angle():
    center_angle = np.array([0.0, np.pi / 2, np.pi])
    xi = 0.2
    x = 0.1 + xi * np.cos(center_angle)
    y = xi * np.sin(center_angle)

    rho, _ = profiles._kink_displaced_polar(
        x,
        y,
        delta=0.1,
        xi=xi,
        center_angle_xy=center_angle,
    )

    np.testing.assert_allclose(rho, 0.0, atol=1e-12)


def test_kink_normalization_is_optional_after_static_boundary_normalization():
    x = np.array([1.0, -1.0])
    y = np.zeros(2)
    parameters = dict(
        delta=0.1,
        xi_0=0.2,
        rho_s=10.0,
        d=2.0,
        center_angle_xy=0.0,
    )

    rho_default, _ = profiles.kinked_rho(x, y, **parameters)
    rho_normalized, _ = profiles.kinked_rho(x, y, **parameters, normalize_kink=True)

    assert not np.allclose(rho_default, 1.0)
    np.testing.assert_allclose(rho_normalized, 1.0, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("call", "parameter"),
    [
        (lambda: profiles.shifted_polar(0, 0, cx=[0.1], cy=0), "cx"),
        (lambda: profiles.gaussian(0.5, rho_s=[0.4], w=0.2), "rho_s"),
        (lambda: profiles.two_power(0.5, alpha=[2], beta=3), "alpha"),
        (
            lambda: profiles.kinked_rho(0, 0, delta=0.1, xi_0=[0.2], rho_s=0.4, d=2),
            "xi_0",
        ),
        (
            lambda: profiles.axisymmetric_profile(
                0, 0, A=[1], delta=0.1, alpha=2, beta=3
            ),
            "A",
        ),
    ],
)
def test_scalar_profile_parameters_reject_arrays(call, parameter):
    with pytest.raises(ValueError, match=parameter):
        call()


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: profiles.shifted_polar(0, 0, cx=1, cy=0),
            "shifted origin",
        ),
        (lambda: profiles.gaussian(0.5, rho_s=0.4, w=0), "w"),
        (lambda: profiles.two_power(0.5, alpha=0, beta=3), "alpha"),
        (lambda: profiles.smooth_maximum(0, 1, eps=-0.1), "eps"),
        (
            lambda: profiles.kinked_rho(0, 0, delta=0.1, xi_0=0.2, rho_s=0, d=2),
            "rho_s",
        ),
        (
            lambda: profiles.crescent_rho(0, 0, delta=0.1, xi_0=0.2, rho_s=0.4, d=1),
            "d must be greater than one",
        ),
        (
            lambda: profiles.crescent_rho(0, 0, delta=0.1, xi_0=0, rho_s=0.4, d=2),
            "xi_0",
        ),
        (
            lambda: profiles.shifted_polar(0, 0, cx=0.1, cy=0, normalize_boundary=1),
            "normalize_boundary",
        ),
        (
            lambda: profiles.kinked_rho(
                0,
                0,
                delta=0.1,
                xi_0=0.2,
                rho_s=0.4,
                d=2,
                normalize_kink=0,
            ),
            "normalize_kink",
        ),
        (
            lambda: profiles.flattening_profile(
                0,
                0,
                A=1,
                delta=0.1,
                alpha=2,
                beta=3,
                xi_0=0.2,
                rho_s=0.4,
                d=2,
                localized=True,
                w=0.2,
                blend_edge=0,
            ),
            "blend_edge",
        ),
        (
            lambda: profiles.kinked_rho(
                0,
                0,
                delta=0.1,
                xi_0=0.2,
                rho_s=0.4,
                d=2,
                center_angle_xy=1j,
            ),
            "center_angle_xy",
        ),
        (
            lambda: profiles.kinked_rho(
                0,
                0,
                delta=0.1,
                xi_0=0.2,
                rho_s=0.4,
                d=2,
                center_angle_xy=np.inf,
            ),
            "center_angle_xy",
        ),
        (
            lambda: profiles.axisymmetric_profile(
                0, 0, A=1, delta=0.1, alpha=2, beta=3, edge_value=np.nan
            ),
            "edge_value",
        ),
    ],
)
def test_numerically_invalid_profile_parameters_fail_explicitly(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_zero_smoothing_uses_exact_minimum_and_maximum():
    a = np.array([-1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, 3.0])

    np.testing.assert_array_equal(profiles.smooth_maximum(a, b), np.maximum(a, b))
    np.testing.assert_array_equal(profiles.smooth_minimum(a, b), np.minimum(a, b))
    assert profiles.smooth_maximum(1.0, 1.0, eps=0.1) > 1.0
    assert profiles.smooth_minimum(1.0, 1.0, eps=0.1) < 1.0


def test_optional_profile_controllers_are_keyword_only():
    functions = (
        profiles.shifted_polar,
        profiles.gaussian,
        profiles.smooth_maximum,
        profiles.smooth_minimum,
        profiles.kinked_rho,
        profiles.crescent_rho,
        profiles.axisymmetric_profile,
        profiles.kinked_profile,
        profiles.flattening_profile,
        profiles.crescent_profile,
    )

    for function in functions:
        optional = [
            parameter
            for parameter in inspect.signature(function).parameters.values()
            if parameter.default is not inspect.Parameter.empty
        ]
        assert optional
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in optional
        )


@pytest.mark.parametrize(
    ("function", "defaults"),
    [
        (profiles.shifted_polar, {"normalize_boundary": True}),
        (profiles.gaussian, {"d": 2, "edge": None}),
        (profiles.smooth_maximum, {"eps": 0}),
        (profiles.smooth_minimum, {"eps": 0}),
        (
            profiles.kinked_rho,
            {"center_angle_xy": 0, "normalize_kink": False},
        ),
        (
            profiles.crescent_rho,
            {"center_angle_xy": 0, "smooth_eps": 0},
        ),
        (profiles.axisymmetric_profile, {"edge_value": 0}),
        (
            profiles.kinked_profile,
            {
                "center_angle_xy": 0,
                "normalize_kink": False,
                "edge_value": 0,
            },
        ),
        (
            profiles.flattening_profile,
            {
                "localized": False,
                "rho_flat": None,
                "lam_0": 1,
                "w": None,
                "gamma": 0,
                "blend_edge": None,
                "center_angle_xy": 0,
                "flattening_angle_offset": np.pi,
                "smooth_eps": 0,
                "normalize_kink": False,
                "edge_value": 0,
            },
        ),
        (
            profiles.crescent_profile,
            {"center_angle_xy": 0, "smooth_eps": 0, "edge_value": 0},
        ),
    ],
)
def test_profile_controller_defaults_are_part_of_the_api(function, defaults):
    signature = inspect.signature(function)

    for name, expected in defaults.items():
        actual = signature.parameters[name].default
        assert actual == expected


def test_flattening_localized_defaults_to_full_angle_and_is_keyword_only():
    parameter = inspect.signature(profiles.flattening_profile).parameters["localized"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


def test_kink_coordinate_helper_is_private_and_old_public_name_is_removed():
    assert callable(profiles._kink_displaced_polar)
    assert not hasattr(profiles, "rigid_shifted_polar")


def test_crescent_fold_matches_the_radial_stationary_condition():
    xi_0 = 1.0
    rho_s = 0.4
    d = 2.0

    fold = profiles._crescent_fold(xi_0, rho_s, d)

    assert fold is not None
    base_radius, effective_radius = fold
    normalized_power = (base_radius / rho_s) ** d
    derivative = 1 - xi_0 * d / rho_s * (base_radius / rho_s) ** (d - 1) * np.exp(
        -normalized_power
    )
    np.testing.assert_allclose(derivative, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        effective_radius,
        base_radius + xi_0 * np.exp(-normalized_power),
        rtol=1e-12,
        atol=1e-12,
    )


def test_crescent_fold_returns_none_when_the_radial_map_is_monotonic():
    assert profiles._crescent_fold(kink_amplitude=0.2, rho_s=0.4, d=2.0) is None


def test_crescent_rho_preserves_a_monotonic_kink():
    x = np.linspace(-0.8, 0.8, 21)
    y = np.linspace(0.2, -0.2, 21)
    parameters = dict(
        delta=0.1,
        xi_0=0.2,
        rho_s=0.4,
        d=2.0,
        center_angle_xy=0.3,
    )

    expected = profiles.kinked_rho(x, y, **parameters)
    actual = profiles.crescent_rho(x, y, **parameters)

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)


def test_fully_flattened_rho_defaults_to_exact_hard_clipping():
    x = np.linspace(-0.8, 0.8, 31)
    y = np.linspace(0.3, -0.3, 31)
    parameters = dict(
        delta=0.1,
        xi_0=0.5,
        rho_s=0.4,
        d=2.0,
        center_angle_xy=0.3,
    )

    rho_0, _ = profiles.shifted_polar(x, y, parameters["delta"], 0)
    rho_kinked, _ = profiles.kinked_rho(x, y, **parameters)
    expected = np.minimum(
        rho_kinked,
        np.maximum(parameters["rho_s"], rho_0),
    )

    actual = profiles._fully_flattened_rho(
        rho_0, rho_kinked, parameters["rho_s"], smooth_eps=0
    )

    np.testing.assert_array_equal(actual, expected)


def test_fully_flattened_rho_accepts_a_separate_flattening_radius():
    x = np.linspace(-0.8, 0.8, 31)
    y = np.zeros_like(x)
    parameters = dict(delta=0.1, xi_0=0.2, rho_s=0.4, d=2.0)

    rho_0, _ = profiles.shifted_polar(x, y, parameters["delta"], 0)
    rho_kinked, _ = profiles.kinked_rho(
        x,
        y,
        **parameters,
        normalize_kink=True,
    )
    rho_flat = 0.25
    expected = np.minimum(
        rho_kinked,
        np.maximum(rho_flat, rho_0),
    )

    actual = profiles._fully_flattened_rho(rho_0, rho_kinked, rho_flat, smooth_eps=0)

    np.testing.assert_array_equal(actual, expected)


def test_fully_flattened_rho_applies_one_epsilon_to_both_clip_operations():
    x = np.linspace(-0.8, 0.8, 31)
    y = np.linspace(0.3, -0.3, 31)
    eps = 0.02
    parameters = dict(delta=0.1, xi_0=0.5, rho_s=0.4, d=2.0)

    rho_0, _ = profiles.shifted_polar(x, y, parameters["delta"], 0)
    rho_kinked, _ = profiles.kinked_rho(x, y, **parameters)
    smoothed_floor = eps * np.logaddexp(
        parameters["rho_s"] / eps,
        rho_0 / eps,
    )
    expected = -eps * np.logaddexp(-rho_kinked / eps, -smoothed_floor / eps)

    actual = profiles._fully_flattened_rho(
        rho_0, rho_kinked, parameters["rho_s"], smooth_eps=eps
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_crescent_rho_defaults_to_exact_hard_clipping_at_the_fold():
    x = np.linspace(-0.9, 0.9, 61)
    y = np.zeros_like(x)
    parameters = dict(delta=0.0, xi_0=1.0, rho_s=0.4, d=2.0)
    fold = profiles._crescent_fold(
        parameters["xi_0"], parameters["rho_s"], parameters["d"]
    )
    assert fold is not None
    _, effective_radius_at_fold = fold

    rho_0, _ = profiles.shifted_polar(x, y, parameters["delta"], 0)
    rho_kinked, theta_kinked = profiles.kinked_rho(x, y, **parameters)
    expected = np.minimum(
        rho_kinked,
        np.maximum(effective_radius_at_fold, rho_0),
    )

    actual, theta = profiles.crescent_rho(x, y, **parameters)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(theta, theta_kinked)


def test_crescent_rho_applies_one_positive_epsilon_to_both_clip_operations():
    x = np.linspace(-0.9, 0.9, 61)
    y = np.zeros_like(x)
    eps = 0.02
    parameters = dict(delta=0.0, xi_0=1.0, rho_s=0.4, d=2.0)
    fold = profiles._crescent_fold(
        parameters["xi_0"], parameters["rho_s"], parameters["d"]
    )
    assert fold is not None
    _, effective_radius_at_fold = fold

    rho_0, _ = profiles.shifted_polar(x, y, parameters["delta"], 0)
    rho_kinked, _ = profiles.kinked_rho(x, y, **parameters)
    smoothed_floor = eps * np.logaddexp(
        effective_radius_at_fold / eps,
        rho_0 / eps,
    )
    expected = -eps * np.logaddexp(-rho_kinked / eps, -smoothed_floor / eps)

    actual, _ = profiles.crescent_rho(x, y, **parameters, smooth_eps=eps)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("delta", "center_angle_xy"),
    [(0.0, 0.0), (0.1, 0.0), (0.1, 0.7), (-0.2, 1.1)],
)
def test_crescent_fold_matches_numerical_kink_minimum(delta, center_angle_xy):
    xi_0 = 1.0
    rho_s = 0.4
    d = 2.0
    boundary_radius = delta * np.cos(center_angle_xy) + np.sqrt(
        1 - delta**2 * np.sin(center_angle_xy) ** 2
    )
    predicted = profiles._crescent_fold(xi_0 / boundary_radius, rho_s, d)
    assert predicted is not None

    def kinked_radius(rho_0):
        x = delta - rho_0 * boundary_radius * np.cos(center_angle_xy)
        y = -rho_0 * boundary_radius * np.sin(center_angle_xy)
        rho_kinked, _ = profiles.kinked_rho(
            x,
            y,
            delta=delta,
            xi_0=xi_0,
            rho_s=rho_s,
            d=d,
            center_angle_xy=center_angle_xy,
            normalize_kink=False,
        )
        return float(rho_kinked)

    numerical = minimize_scalar(
        kinked_radius,
        bounds=(0.3, 1.0),
        method="bounded",
        options={"xatol": 1e-13},
    )

    np.testing.assert_allclose(numerical.x, predicted[0], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(numerical.fun, predicted[1], rtol=1e-9, atol=1e-10)


def test_gaussian_odd_exponent_decays_on_both_sides_and_supports_scalar_input():
    left = profiles.gaussian(0.25, rho_s=0.5, w=0.5, d=3, edge=None)
    center = profiles.gaussian(0.5, rho_s=0.5, w=0.5, d=3, edge=None)
    outside = profiles.gaussian(1.2, rho_s=0.5, w=0.5, d=3, edge=None)

    assert np.isscalar(left) or left.shape == ()
    assert 0 < left < center
    assert center == 1
    assert outside == 0


def test_gaussian_edge_taper_is_disabled_by_default_and_requires_positive_width():
    rho = np.array([0.0, 0.5, 1.0])

    untapered = profiles.gaussian(rho, rho_s=0.5, w=1.0)
    explicit_untapered = profiles.gaussian(rho, rho_s=0.5, w=1.0, edge=None)
    tapered = profiles.gaussian(rho, rho_s=0.5, w=1.0, edge=0.02)

    np.testing.assert_array_equal(untapered, explicit_untapered)
    assert untapered[0] > 0
    assert untapered[-1] > 0
    assert tapered[0] == 0
    assert tapered[-1] == 0
    with pytest.raises(ValueError, match="edge"):
        profiles.gaussian(rho, rho_s=0.5, w=1.0, edge=0)


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
    center_angle_xy = np.linspace(0, np.pi, x.size)

    axisymmetric = profiles.axisymmetric_profile(
        x, y, A=2.0, delta=0.1, alpha=2, beta=3
    )
    kinked = profiles.kinked_profile(
        x,
        y,
        A=2.0,
        delta=0.1,
        alpha=2,
        beta=3,
        xi_0=0.1,
        rho_s=0.5,
        d=2,
        center_angle_xy=center_angle_xy,
    )
    flattened = profiles.flattening_profile(
        x,
        y,
        A=2.0,
        delta=0.1,
        alpha=2,
        beta=3,
        xi_0=0.1,
        rho_s=0.5,
        d=2,
        localized=True,
        w=0.2,
        gamma=0.1,
        center_angle_xy=center_angle_xy,
    )
    scalar = profiles.axisymmetric_profile(0.0, 0.0, A=2.0, delta=0.1, alpha=2, beta=3)

    assert axisymmetric.shape == x.shape
    assert kinked.shape == x.shape
    assert flattened.shape == x.shape
    assert np.isfinite(axisymmetric).all()
    assert np.isfinite(kinked).all()
    assert np.isfinite(flattened).all()
    assert np.isfinite(scalar)


def test_kinked_profile_axisymmetric_limit_and_center_angle_periodicity():
    x = np.linspace(-0.7, 0.7, 15)
    y = np.linspace(0.3, -0.3, 15)
    parameters = dict(A=2.0, delta=0.1, alpha=2.5, beta=3.0)

    axisymmetric = profiles.axisymmetric_profile(x, y, **parameters)
    kink_limit = profiles.kinked_profile(
        x,
        y,
        **parameters,
        xi_0=0.0,
        rho_s=0.4,
        d=2.0,
        center_angle_xy=0.7,
    )
    kink = profiles.kinked_profile(
        x,
        y,
        **parameters,
        xi_0=0.15,
        rho_s=0.4,
        d=2.0,
        center_angle_xy=0.5,
    )
    periodic = profiles.kinked_profile(
        x,
        y,
        **parameters,
        xi_0=0.15,
        rho_s=0.4,
        d=2.0,
        center_angle_xy=0.5 + 2 * np.pi,
    )

    np.testing.assert_allclose(kink_limit, axisymmetric, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(periodic, kink, rtol=1e-12, atol=1e-12)


def test_localized_flattening_matches_independent_density_blend_formula():
    x = np.linspace(-0.7, 0.8, 17)
    y = np.linspace(0.25, -0.35, 17)
    parameters = dict(
        delta=0.08,
        xi_0=0.12,
        rho_s=0.45,
        rho_flat=0.35,
        d=2.3,
        w=0.25,
        gamma=0.15,
        lam_0=0.7,
        center_angle_xy=0.6,
        flattening_angle_offset=-0.2,
    )

    rho_kinked, theta_kinked = profiles.kinked_rho(
        x,
        y,
        **{
            key: parameters[key]
            for key in ("delta", "xi_0", "rho_s", "d", "center_angle_xy")
        },
    )
    relative_theta = (
        theta_kinked
        - parameters["center_angle_xy"]
        - parameters["flattening_angle_offset"]
    )
    distorted_theta = relative_theta + parameters["gamma"] * np.sin(relative_theta)
    angular_weight = 0.5 * (1 + np.cos(distorted_theta))
    lam = (
        profiles.gaussian(
            rho_kinked,
            parameters["rho_flat"],
            parameters["w"],
        )
        * angular_weight
        * parameters["lam_0"]
    )
    rho_0, _ = profiles.shifted_polar(x, y, parameters["delta"], 0)
    rho_island = profiles._fully_flattened_rho(
        rho_0,
        rho_kinked,
        parameters["rho_flat"],
        smooth_eps=0,
    )
    inside = x**2 + y**2 <= 1
    kinked_density = np.where(
        inside,
        2.0 * profiles.two_power(rho_kinked, alpha=2.5, beta=3.0),
        0,
    )
    island_density = np.where(
        inside,
        2.0 * profiles.two_power(rho_island, alpha=2.5, beta=3.0),
        0,
    )
    expected = (1 - lam) * kinked_density + lam * island_density

    actual = profiles.flattening_profile(
        x,
        y,
        A=2.0,
        alpha=2.5,
        beta=3.0,
        localized=True,
        **parameters,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_full_angle_flattening_blends_density_and_has_exact_lam_endpoints():
    x = np.linspace(-0.8, 0.8, 31)
    y = np.linspace(0.25, -0.25, 31)
    parameters = dict(
        A=3.0,
        delta=0.1,
        alpha=2.5,
        beta=3.0,
        xi_0=0.5,
        rho_s=0.4,
        rho_flat=0.3,
        d=2.0,
        center_angle_xy=0.3,
        edge_value=0.6,
    )
    kinked = profiles.kinked_profile(
        x,
        y,
        **{key: value for key, value in parameters.items() if key != "rho_flat"},
    )
    rho_0, _ = profiles.shifted_polar(x, y, parameters["delta"], 0)
    rho_kinked, _ = profiles.kinked_rho(
        x,
        y,
        parameters["delta"],
        parameters["xi_0"],
        parameters["rho_s"],
        parameters["d"],
        center_angle_xy=parameters["center_angle_xy"],
    )
    rho_island = profiles._fully_flattened_rho(
        rho_0,
        rho_kinked,
        parameters["rho_flat"],
        smooth_eps=0,
    )
    island = np.where(
        x**2 + y**2 <= 1,
        parameters["edge_value"]
        + (parameters["A"] - parameters["edge_value"])
        * profiles.two_power(
            rho_island,
            parameters["alpha"],
            parameters["beta"],
        ),
        0,
    )

    at_zero = profiles.flattening_profile(x, y, **parameters, localized=False, lam_0=0)
    at_one = profiles.flattening_profile(x, y, **parameters, localized=False, lam_0=1)
    blended = profiles.flattening_profile(
        x, y, **parameters, localized=False, lam_0=0.35
    )

    np.testing.assert_array_equal(at_zero, kinked)
    np.testing.assert_array_equal(at_one, island)
    np.testing.assert_allclose(blended, 0.65 * kinked + 0.35 * island)


def test_flattening_profile_localized_flag_is_keyword_only():
    arguments = (0.1, 0.2, 1.0, 0.1, 2.0, 3.0, 0.2, 0.4, 2.0)

    implicit_full_angle = profiles.flattening_profile(*arguments)
    explicit_full_angle = profiles.flattening_profile(*arguments, localized=False)

    np.testing.assert_array_equal(implicit_full_angle, explicit_full_angle)
    with pytest.raises(TypeError):
        profiles.flattening_profile(*arguments, False)


@pytest.mark.parametrize("lam_0", [-0.1, 1.1])
def test_flattening_profile_rejects_density_blends_outside_unit_interval(lam_0):
    with pytest.raises(ValueError, match="lam_0"):
        profiles.flattening_profile(
            0,
            0,
            1,
            0.1,
            2,
            3,
            0.2,
            0.4,
            2,
            localized=False,
            lam_0=lam_0,
        )


def test_localized_flattening_requires_a_width():
    with pytest.raises(ValueError, match="w"):
        profiles.flattening_profile(
            0,
            0,
            1,
            0.1,
            2,
            3,
            0.2,
            0.4,
            2,
            localized=True,
        )


@pytest.mark.parametrize(
    "local_controls",
    [
        {"w": 0.2},
        {"gamma": 0.1},
        {"blend_edge": 0.02},
        {"flattening_angle_offset": 0.0},
    ],
)
def test_full_angle_flattening_ignores_local_only_controls(local_controls):
    expected = profiles.flattening_profile(0, 0, 1, 0.1, 2, 3, 0.2, 0.4, 2)
    actual = profiles.flattening_profile(
        0,
        0,
        1,
        0.1,
        2,
        3,
        0.2,
        0.4,
        2,
        localized=False,
        **local_controls,
    )

    np.testing.assert_array_equal(actual, expected)


def test_flattening_profile_rejects_non_boolean_localized_flag():
    with pytest.raises(ValueError, match="localized"):
        profiles.flattening_profile(
            0,
            0,
            1,
            0.1,
            2,
            3,
            0.2,
            0.4,
            2,
            localized=1,
        )


def test_profile_edge_value_preserves_center_boundary_and_vacuum_values():
    x = np.array([0.0, 0.5, 1.0, 1.2])
    actual = profiles.axisymmetric_profile(
        x,
        0.0,
        A=5.0,
        delta=0.0,
        alpha=2.0,
        beta=1.0,
        edge_value=2.0,
    )
    expected = np.array([5.0, 4.25, 2.0, 0.0])

    np.testing.assert_allclose(actual, expected)


def test_all_generic_profiles_apply_edge_value_to_their_effective_radius():
    x = np.linspace(-0.8, 1.2, 19)
    y = np.linspace(0.3, -0.2, 19)
    common = dict(A=4.0, alpha=2.0, beta=3.0)
    edge_value = 0.7
    cases = [
        (
            profiles.axisymmetric_profile,
            profiles.shifted_polar(x, y, 0.1, 0)[0],
            dict(delta=0.1),
        ),
        (
            profiles.kinked_profile,
            profiles.kinked_rho(
                x,
                y,
                0.1,
                0.12,
                0.45,
                2.0,
                center_angle_xy=0.3,
            )[0],
            dict(
                delta=0.1,
                xi_0=0.12,
                rho_s=0.45,
                d=2.0,
                center_angle_xy=0.3,
            ),
        ),
        (
            profiles.crescent_profile,
            profiles.crescent_rho(
                x,
                y,
                0.1,
                1.0,
                0.45,
                2.0,
                center_angle_xy=0.3,
            )[0],
            dict(
                delta=0.1,
                xi_0=1.0,
                rho_s=0.45,
                d=2.0,
                center_angle_xy=0.3,
            ),
        ),
    ]

    for profile, rho, parameters in cases:
        old = profile(x, y, **common, **parameters)
        with_zero_edge = profile(x, y, **common, **parameters, edge_value=0)
        expected = np.where(
            x**2 + y**2 <= 1,
            edge_value
            + (common["A"] - edge_value)
            * profiles.two_power(rho, common["alpha"], common["beta"]),
            0,
        )
        actual = profile(x, y, **common, **parameters, edge_value=edge_value)

        np.testing.assert_array_equal(with_zero_edge, old)
        np.testing.assert_allclose(actual, expected)
