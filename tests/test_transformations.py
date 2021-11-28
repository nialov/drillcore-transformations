"""
Test drillcore_transformations.py.
"""

from warnings import warn

import numpy as np
from hypothesis import HealthCheck, assume, given, settings

from drillcore_transformations import transformations
from tests import (
    alpha_strategy,
    amount_strategy,
    beta_strategy,
    dip_strategy,
    dir_strategy,
    gamma_strategy,
    plunge_strategy,
    trend_strategy,
    vector_strategy,
)


@given(alpha_strategy, beta_strategy, trend_strategy, plunge_strategy)
def test_calc_global_normal_vector(alpha, beta, trend, plunge):
    """
    Test calc_global_normal_vector.
    """
    vector = transformations.calc_global_normal_vector(alpha, beta, trend, plunge)
    assert np.isclose(np.linalg.norm(vector), 1)
    assert vector[2] >= 0


@given(vector_strategy, vector_strategy, amount_strategy)
def test_rotate_vector_about_vector(vector, about_vector, amount):
    """
    Test rotate_vector_about_vector.
    """
    transformations.rotate_vector_about_vector(vector, about_vector, amount)

    # sample test
    rotated_vector_ = transformations.rotate_vector_about_vector(
        np.array([1, 0, 1]), np.array([0, 0, 1]), np.pi
    )
    assert np.allclose(
        rotated_vector_, np.array([-1.0000000e00, 1.2246468e-16, 1.0000000e00])
    )

    # if not np.all(vector == 0) and not np.all(about_vector == 0):
    # 	if not np.isclose(amount, 0) and not np.isclose(amount, np.pi*2):
    # 		assert not np.allclose(rotated_vector, vector)


@given(dip_strategy, dir_strategy)
def test_vector_from_dip_and_dir(dip, dip_dir):
    """
    Test vector_from_dip_and_dir.
    """
    vector = transformations.vector_from_dip_and_dir(dip, dip_dir)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert vector[2] <= np.float(0)


@given(vector_strategy)
def test_calc_plane_dir_dip(normal):
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
def test_calc_vector_trend_plunge(vector):
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
def test_transform_without_gamma(alpha, beta, drillcore_trend, drillcore_plunge):
    """
    Test transform_without_gamma.
    """
    plane_dip, plane_dir = transformations.transform_without_gamma(
        alpha, beta, drillcore_trend, drillcore_plunge
    )
    assert plane_dir >= 0.0
    assert plane_dir <= 360.0
    assert plane_dip >= 0.0
    assert plane_dip <= 90.0


@given(alpha_strategy, beta_strategy, trend_strategy, plunge_strategy, gamma_strategy)
def test_transform_with_gamma(alpha, beta, drillcore_trend, drillcore_plunge, gamma):
    """
    Test transform_with_gamma.
    """
    (
        plane_dip,
        plane_dir,
        gamma_plunge,
        gamma_trend,
    ) = transformations.transform_with_gamma(
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
    ) = transformations.transform_with_gamma(45, 0, 0, 90, 10)
    assert np.allclose(
        (plane_dip, plane_dir, gamma_plunge, gamma_trend),
        (45.00000000000001, 0.0, -36.39247, 137.48165),
    )


@given(dip_strategy, dir_strategy, dip_strategy, dir_strategy)
def test_calc_difference_between_two_planes(
    dip_first, dir_first, dip_second, dir_second
):
    """
    Test calc_difference_between_two_planes.
    """
    result = transformations.calc_difference_between_two_planes(
        dip_first, dir_first, dip_second, dir_second
    )

    if not 0 <= result <= 90:
        assert np.isclose(result, 0) or np.isclose(result, 90)


def test_calc_difference_between_two_planes_nan():
    """
    Test calc_difference_between_two_planes with nan.
    """
    result = transformations.calc_difference_between_two_planes(np.nan, 1, 5, 50)
    assert np.isnan(result)
