"""
Test drillcore_transformations.py.
"""

import matplotlib.pyplot as plt
import numpy as np
from hypothesis import assume, given, settings

from drillcore_transformations import visualizations
from tests import (
    alpha_strategy,
    beta_strategy,
    gamma_strategy,
    plunge_strategy,
    trend_strategy,
    vector_strategy,
)

plt.rcParams["figure.max_open_warning"] = 100


@given(vector_strategy)
@settings(max_examples=5, deadline=None)
def test_visualize_plane(plane_normal):
    """
    Test visualize_plane.
    """
    assume(not np.all(plane_normal == 0))
    fig = plt.figure(figsize=(9, 9))
    ax = fig.gca(projection="3d")
    visualizations.visualize_plane(plane_normal, ax)
    plt.close()


@given(vector_strategy)
@settings(max_examples=5, deadline=None)
def test_visualize_vector(vector):
    """
    Test visualize_vector.
    """
    assume(not np.all(vector == 0))
    fig = plt.figure(figsize=(9, 9))
    ax = fig.gca(projection="3d")
    visualizations.visualize_vector(vector, ax)
    plt.close()


@given(
    vector_strategy,
    vector_strategy,
    vector_strategy,
    trend_strategy,
    plunge_strategy,
    alpha_strategy,
    beta_strategy,
    vector_strategy,
    gamma_strategy,
)
@settings(max_examples=5, deadline=None)
def test_visualize_results(
    plane_normal,
    plane_vector,
    drillcore_vector,
    drillcore_trend,
    drillcore_plunge,
    alpha,
    beta,
    gamma_vector,
    gamma,
):
    """
    Test visualize_results.
    """
    visualizations.visualize_results(
        plane_normal,
        plane_vector,
        drillcore_vector,
        drillcore_trend,
        drillcore_plunge,
        alpha,
        beta,
        gamma_vector,
        gamma,
    )
    plt.close()
    visualizations.visualize_results(
        plane_normal,
        plane_vector,
        drillcore_vector,
        drillcore_trend,
        drillcore_plunge,
        alpha,
        beta,
    )
    plt.close()
