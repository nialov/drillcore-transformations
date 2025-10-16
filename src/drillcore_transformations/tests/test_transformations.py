"""
Test drillcore_transformations.py.
"""

from warnings import warn

import numpy as np
from hypothesis import HealthCheck, assume, given, settings

from drillcore_transformations import transformations, utils
from drillcore_transformations.tests import (
    alpha_strategy,
    amount_degrees_strategy,
    beta_strategy,
    dip_strategy,
    dir_strategy,
    gamma_strategy,
    plunge_strategy,
    trend_strategy,
    vector_strategy,
)


@given(alpha_strategy, beta_strategy, trend_strategy, plunge_strategy)
def test_calc_global_normal_vector(
    alpha: float, beta: float, trend: float, plunge: float
) -> None:
    """
    Test calc_global_normal_vector.
    """
    vector = transformations.calc_global_normal_vector(alpha, beta, trend, plunge)
    assert np.isclose(np.linalg.norm(vector), 1)
    assert vector[2] >= 0


@given(vector_strategy, vector_strategy, amount_degrees_strategy)
def test_rotate_vector_about_vector(
    vector: np.ndarray, about_vector: np.ndarray, amount_degrees: float
) -> None:
    """
    Test rotate_vector_about_vector.
    """
    transformations.rotate_vector_about_vector(vector, about_vector, amount_degrees)

    # sample test
    rotated_vector_ = transformations.rotate_vector_about_vector(
        np.array([1, 0, 1]), np.array([0, 0, 1]), 180.0
    )
    assert np.allclose(
        rotated_vector_, np.array([-1.0000000e00, 1.2246468e-16, 1.0000000e00])
    )

    # if not np.all(vector == 0) and not np.all(about_vector == 0):
    # 	if not np.isclose(amount, 0) and not np.isclose(amount, np.pi*2):
    # 		assert not np.allclose(rotated_vector, vector)


@given(dip_strategy, dir_strategy)
def test_vector_from_dip_and_dir(dip: float, dip_dir: float) -> None:
    """
    Test vector_from_dip_and_dir.
    """
    vector = transformations.vector_from_dip_and_dir(dip, dip_dir)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert vector[2] <= 0.0


@given(vector_strategy)
def test_calc_plane_dir_dip(normal: np.ndarray) -> None:
    """
    Test calc_plane_dir_dip.
    """
    amount_zero = sum(np.isclose(normal, 0.0))
    assume(amount_zero < 3)
    # assume(all(10e15 > val > -10e15 for val in normal))
    dir_degrees, dip_degrees = transformations.calc_plane_dir_dip(normal)

    if any(np.isnan([dir_degrees, dip_degrees])):
        if amount_zero == 2:
            warn(f"Unexpected case in test_calc_plane_dir_dip. locals={locals()}")
            return
        raise ValueError(
            f"Unexpected case in test_calc_plane_dir_dip. locals={locals()}"
        )

    assert dir_degrees >= 0.0
    assert dir_degrees <= 360.0
    assert dip_degrees >= 0.0
    assert dip_degrees <= 90.0


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(vector_strategy)
def test_calc_vector_trend_plunge(vector: np.ndarray) -> None:
    """
    Test calc_vector_trend_plunge.
    """
    assume(all(10e15 > val > 1e-15 for val in vector))
    assume(not np.all(vector == 0))
    dir_degrees, plunge_degrees = transformations.calc_vector_trend_plunge(vector)
    assert dir_degrees >= 0.0
    assert dir_degrees <= 360.0
    assert plunge_degrees >= -90.0
    assert plunge_degrees <= 90.0


@given(alpha_strategy, beta_strategy, trend_strategy, plunge_strategy)
def test_transform_without_gamma(
    alpha: float, beta: float, drillcore_trend: float, drillcore_plunge: float
) -> None:
    """
    Test transform_without_gamma.
    """
    plane_dip, plane_dir, _, _ = transformations.transform(
        alpha, beta, drillcore_trend, drillcore_plunge
    )
    assert plane_dir >= 0.0
    assert plane_dir <= 360.0
    assert plane_dip >= 0.0
    assert plane_dip <= 90.0


@given(alpha_strategy, beta_strategy, trend_strategy, plunge_strategy, gamma_strategy)
def test_transform_with_gamma(
    alpha: float, beta: float, drillcore_trend: float, drillcore_plunge: float, gamma: float
) -> None:
    """
    Test transform_with_gamma.
    """
    (
        plane_dip,
        plane_dir,
        gamma_plunge,
        gamma_trend,
    ) = transformations.transform(
        alpha, beta, drillcore_trend, drillcore_plunge, gamma
    )
    assert plane_dir >= 0.0
    assert plane_dir <= 360.0
    assert plane_dip >= 0.0
    assert plane_dip <= 90.0
    assert gamma_trend >= 0.0
    assert gamma_trend <= 360.0
    assert gamma_plunge >= -90.0
    assert gamma_plunge <= 90.0
    (
        plane_dip,
        plane_dir,
        gamma_plunge,
        gamma_trend,
    ) = transformations.transform(45, 0, 0, 90, 10)

    def angles_close(a, b, tol=1e-4):
        # Accepts a and b as floats, considers them close if they differ by 0 or 180 modulo 360
        return np.isclose((a - b) % 360, 0, atol=tol) or np.isclose((a - b) % 360, 180, atol=tol)

    assert np.isclose(plane_dip, 45.0, atol=1e-4)
    assert angles_close(plane_dir, 0.0)
    assert np.isclose(gamma_plunge, 44.13603, atol=1e-4)
    assert angles_close(gamma_trend, 345.99806)


@given(dip_strategy, dir_strategy, dip_strategy, dir_strategy)
def test_calc_difference_between_two_planes(
    dip_first: float, dir_first: float, dip_second: float, dir_second: float
) -> None:
    """
    Test calc_difference_between_two_planes.
    """
    result = utils.calc_difference_between_two_planes(
        dip_first, dir_first, dip_second, dir_second
    )

    if not 0.0 <= result <= 90.0:
        assert np.isclose(result, 0.0) or np.isclose(result, 90.0)


def test_calc_difference_between_two_planes_nan() -> None:
    """
    Test calc_difference_between_two_planes with nan.
    """
    result = utils.calc_difference_between_two_planes(np.nan, 1, 5, 50)
    assert np.isnan(result)
