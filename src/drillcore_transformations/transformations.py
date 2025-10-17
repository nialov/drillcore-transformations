"""
Module with all calculations.
"""

from typing import NamedTuple, Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
from functools import partial

array_float64 = partial(np.array, dtype=np.float64)

DEFAULT_CONVENTION_MAP = {"beta": lambda beta: beta - 180}


class Measurement(NamedTuple):
    alpha: float
    beta: float
    drillcore_trend: float
    drillcore_plunge: float
    gamma: Optional[float] = None


def calc_global_normal_vector(
    alpha: float, beta: float, trend: float, plunge: float
) -> npt.NDArray[np.float64]:
    """
    Calculate the normal vector of a measured plane.

    Based on alpha and beta measurements and the trend and plunge
    of the drillcore.

    Help and code snippets from:
    https://tinyurl.com/tqr84ww

    :param alpha: Alpha of the measured plane in degrees.
    :param beta: Beta of the measured plane in degrees.
    :param trend: Trend of the drillcore
    :param plunge: Plunge of the drillcore
    :return: Normalized normal vector of a plane. Always points upwards (z >= 0)
    """
    # Degrees to radians
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    trend = np.deg2rad(trend)
    plunge = np.deg2rad(plunge)
    # Calculate normal vector of the plane
    ng_1 = (
        np.cos(np.pi / 2 - trend)
        * np.cos(np.pi / 2 - plunge)
        * np.cos(beta)
        * np.cos(alpha)
        - np.sin(np.pi / 2 - trend) * np.sin(beta) * np.cos(alpha)
        + np.cos(np.pi / 2 - trend) * np.sin(np.pi / 2 - plunge) * np.sin(alpha)
    )
    ng_2 = (
        np.sin(np.pi / 2 - trend)
        * np.cos(np.pi / 2 - plunge)
        * np.cos(beta)
        * np.cos(alpha)
        + np.cos(np.pi / 2 - trend) * np.sin(beta) * np.cos(alpha)
        + np.sin(np.pi / 2 - trend) * np.sin(np.pi / 2 - plunge) * np.sin(alpha)
    )
    ng_3 = -np.sin(np.pi / 2 - plunge) * np.cos(beta) * np.cos(alpha) + np.cos(
        np.pi / 2 - plunge
    ) * np.sin(alpha)

    # Always return a normalized vector pointing upwards.
    if ng_3 < 0:
        return array_float64([-ng_1, -ng_2, -ng_3]) / np.linalg.norm(
            array_float64([-ng_1, -ng_2, -ng_3])
        )
    return array_float64([ng_1, ng_2, ng_3]) / np.linalg.norm(
        array_float64([ng_1, ng_2, ng_3])
    )


def rotate_vector_about_vector(
    vector: npt.NDArray[np.float64],
    about_vector: npt.NDArray[np.float64],
    amount_degrees: float,
) -> npt.NDArray[np.float64]:
    """
    Rotate a given vector about another vector.

    Implements Rodrigues' rotation formula:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    E.g.

    >>> rotate_vector_about_vector(np.array([1, 0, 1]), np.array([0, 0, 1]), 180.0)
    array([-1.0000000e+00,  1.2246468e-16,  1.0000000e+00])

    TODO: Is gamma axial or vector data? Right now treated as vector. =>
    Negative plunges possible.

    :param vector: Vector to rotate.
    :param about_vector: Vector to rotate about.
    :param amount_degrees: How many radians to rotate.
    :return: Rotated vector.
    """
    if (
        np.all(vector == 0)
        or np.all(about_vector == 0)
        or np.isclose(np.linalg.norm(vector), 0.0)
        or np.isclose(np.linalg.norm(about_vector), 0.0)
    ):
        return array_float64([0.0, 0.0, 0.0])
    if np.allclose(
        vector / np.linalg.norm(vector), about_vector / np.linalg.norm(about_vector)
    ):
        return vector
    if np.all(np.cross(vector, about_vector) == 0):
        return vector
    if np.isclose(amount_degrees, 0.0):
        return vector
    about_vector = about_vector / np.linalg.norm(about_vector)
    amount_rad = np.deg2rad(amount_degrees)
    try:
        v_rot = (
            vector * np.cos(amount_rad)
            + np.cross(about_vector, vector) * np.sin(amount_rad)
            + about_vector * np.dot(about_vector, vector) * (1 - np.cos(amount_rad))
        )
    except ValueError:
        return array_float64([np.nan, np.nan, np.nan])
    return np.asarray(v_rot, dtype=np.float64)


def vector_from_dip_and_dir(dip: float, dip_dir: float) -> npt.NDArray[np.float64]:
    """
    Assemble a normalized vector that always points downwards from dip data.

    Assumes dip is positive. Uses dip and dip direction.  Credits to PhD Jussi
    Mattila for this snippet.

    E.g.

    >>> vector_from_dip_and_dir(45, 0)
    array([ 0.        ,  0.70710678, -0.70710678])

    :param dip: Dip of a feature. Between [0, 90]
    :param dip_dir: Dip direction of feature.
    :return: Normalized vector pointing in the direction and the dip.
    """
    # Raise if dip is negative.
    if dip < 0:
        raise ValueError(f"Dip is negative. Dip: {dip} (In {__name__})")

    nx = np.sin(np.deg2rad(dip_dir)) * np.cos(np.deg2rad(dip))
    ny = np.cos(np.deg2rad(dip_dir)) * np.cos(np.deg2rad(dip))
    nz = -np.sin(np.deg2rad(dip))
    n = array_float64([nx, ny, nz])
    # Normalize output vector
    n = n / np.linalg.norm(n)
    return n


def calc_plane_dir_dip(normal: npt.NDArray[np.float64]) -> Tuple[float, float]:
    """
    Calculate direction of dip and dip of a plane.

    Based on normal vector of plane. Normal vector should point upwards but it
    will be reversed if not.

    :param normal: Normal vector of a plane.
    :return: Direction of dip and dip in degrees
    """
    if np.all(normal == 0):
        return np.nan, np.nan
    if normal[2] < 0:
        normal = -normal

    # plane dip
    dip_radians = np.pi / 2 - np.arcsin(normal[2])
    dip_degrees = np.rad2deg(dip_radians)
    # Get plane vector trend from plane normal vector

    normal_xy = normal[:2]
    xy_norm = np.linalg.norm(normal_xy)
    if np.isclose(xy_norm, 0.0):
        return 0.0, 0.0

    normal_xy = normal_xy / np.linalg.norm(normal_xy)
    dir_0 = np.array([0, 1.0])
    # y is negative
    if normal_xy[1] < 0:
        # x is negative
        if normal_xy[0] < 0:
            dir_radians = np.pi * 2 - np.arccos(np.dot(normal_xy, dir_0))
        # x is positive
        else:
            dir_radians = np.arccos(np.dot(normal_xy, dir_0))
    # y is positive
    else:
        # x is negative
        if normal_xy[0] < 0:
            dir_radians = np.pi * 2 - np.arccos(np.dot(normal_xy, dir_0))
        # x is positive
        else:
            dir_radians = np.arccos(np.dot(normal_xy, dir_0))

    dir_degrees = np.rad2deg(dir_radians)
    if 90.1 > dip_degrees > 90.0:
        dip_degrees = 90.0
    elif dip_degrees > 90.1:
        raise ValueError(f"dip_degrees too high: {dip_degrees}")
    return dir_degrees, dip_degrees


def calc_vector_trend_plunge(vector: npt.NDArray[np.float64]) -> Tuple[float, float]:
    """
    Calculate trend and plunge of a vector.

    TODO: No longer valid, no negative plunges currently possible.
    Does not assume that the data is axial and a negative plunge result implies
    that the gamma feature is pointed upwards.

    :param vector: vector vector of a plane.
    :return: Direction of dip and dip in degrees
    """
    if np.all(vector == 0):
        return np.nan, np.nan

    plunge_radians = np.arcsin(vector[2])
    plunge_degrees = -np.rad2deg(plunge_radians)
    if plunge_degrees < 0.0:
        # plunge_degrees = -plunge_degrees
        plunge_degrees = 90 + plunge_degrees

    assert 0.0 <= plunge_degrees <= 90.0
    # if vector[2] > 0:
    #     plunge_radians = np.arcsin(vector[2])
    #     plunge_degrees = -np.rad2deg(plunge_radians)
    # else:
    #     plunge_radians = np.arcsin(vector[2])
    #     plunge_degrees = -np.rad2deg(plunge_radians)

    # Get vector trend
    vector_xy = vector[:2]
    vector_xy = vector_xy / np.linalg.norm(vector_xy)
    dir_0 = np.array([0, 1.0])
    # y is negative
    if vector_xy[1] < 0:
        # x is negative
        if vector_xy[0] < 0:
            trend_radians = np.pi * 2 - np.arccos(np.dot(vector_xy, dir_0))
        # x is positive
        else:
            trend_radians = np.arccos(np.dot(vector_xy, dir_0))
    # y is positive
    else:
        # x is negative
        if vector_xy[0] < 0:
            trend_radians = np.pi * 2 - np.arccos(np.dot(vector_xy, dir_0))
        # x is positive
        else:
            trend_radians = np.arccos(np.dot(vector_xy, dir_0))

    trend_degrees = np.rad2deg(trend_radians)
    return round(trend_degrees, 5), round(plunge_degrees, 5)


def calc_normal_vector_of_plane(dip: float, dip_dir: float) -> npt.NDArray[np.float64]:
    """
    Calculate normalized normal vector of plane based on dip and dip dir.

    :param dip: Dip of the plane
    :param dir: Dip direction of the plane
    :return: Normalized normal vector of the plane
    """
    plane_vector_1 = vector_from_dip_and_dir(dip, dip_dir)
    plane_vector_2 = vector_from_dip_and_dir(dip=0, dip_dir=dip_dir + 90)
    plane_normal = np.cross(plane_vector_1, plane_vector_2)
    plane_normal = plane_normal if plane_normal[2] > 0 else -plane_normal
    return plane_normal / np.linalg.norm(plane_normal)


def apply_convention_map(
    convention_map: dict[str, Callable[[Optional[float]], Optional[float]]],
    **kwargs: dict[str, Optional[float]],
) -> dict[str, Optional[float]]:
    return {k: convention_map.get(k, lambda x: x)(v) for k, v in kwargs.items()}


def transform(
    alpha: float,
    beta: float,
    drillcore_trend: float,
    drillcore_plunge: float,
    gamma: Optional[float] = None,
    convention_map: dict[str, Callable[[Optional[float]], Optional[float]]] = DEFAULT_CONVENTION_MAP,
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """
    Transform alpha, beta and, optionally, gamma measurements from core.

    E.g.

    >>> transform(45, 0, 0, 90)
    (45.00000000000001, 180.0, None, None)

    >>> transform(45, 0, 0, 90, 0.0)
    (45.0, 180.0, 45.0, 180.0)

    :param alpha: Angle in degrees between drillcore axis and plane.
    :param drillcore_trend: Trend of the drillcore.
    :param drillcore_plunge: Plunge of the drillcore.
    :param gamma: Linear feature on a plane. Measured in clockwise direction from ellipse long axis at DOWN hole end.
    """

    if any(np.isnan(x) for x in (alpha, beta, drillcore_trend, drillcore_plunge)):
        return np.nan, np.nan, np.nan, np.nan
    try:
        # apply convention map
        alpha, beta, drillcore_trend, drillcore_plunge, gamma = apply_convention_map(
            alpha=alpha,
            beta=beta,
            drillcore_trend=drillcore_trend,
            drillcore_plunge=drillcore_plunge,
            gamma=gamma,
            convention_map=convention_map,
        ).values()

        # plane normal vector
        plane_normal = calc_global_normal_vector(
            alpha, beta, drillcore_trend, drillcore_plunge
        )

        # plane direction of dip and dip
        plane_dir, plane_dip = calc_plane_dir_dip(plane_normal)

        # Vector in the direction of plane dir and dip
        plane_vector = vector_from_dip_and_dir(plane_dip, plane_dir)

        if gamma is not None:
            # Gamma vector
            gamma_vector = rotate_vector_about_vector(plane_vector, plane_normal, gamma)

            # Gamma trend and plunge
            gamma_trend, gamma_plunge = calc_vector_trend_plunge(gamma_vector)
            gamma_plunge = float(gamma_plunge)
            gamma_trend = float(gamma_trend)
        else:
            gamma_plunge, gamma_trend = None, None

        return float(plane_dip), float(plane_dir), gamma_plunge, gamma_trend
    except ValueError as e:
        print(str(e))
        if gamma is not None:
            return float("nan"), float("nan"), float("nan"), float("nan")
        else:
            return float("nan"), float("nan"), None, None
