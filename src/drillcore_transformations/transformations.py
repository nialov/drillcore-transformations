"""
Module with all calculations.
"""

import logging

import numpy as np
import numpy.typing as npt


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
    # Due to differences in nomenclature, some dumb transformations are made.
    beta = beta - 180
    trend = trend - 180

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
        return np.array([-ng_1, -ng_2, -ng_3], dtype=np.float64) / np.linalg.norm(
            np.array([-ng_1, -ng_2, -ng_3], dtype=np.float64)
        )
    return np.array([ng_1, ng_2, ng_3], dtype=np.float64) / np.linalg.norm(np.array([ng_1, ng_2, ng_3], dtype=np.float64))


def rotate_vector_about_vector(
    vector: npt.NDArray[np.float64], about_vector: npt.NDArray[np.float64], amount: float
) -> npt.NDArray[np.float64]:
    """
    Rotate a given vector about another vector.

    Implements Rodrigues' rotation formula:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    E.g.

    >>> rotate_vector_about_vector(np.array([1, 0, 1]), np.array([0, 0, 1]), np.pi)
    array([-1.0000000e+00,  1.2246468e-16,  1.0000000e+00])

    TODO: Is gamma axial or vector data? Right now treated as vector. =>
    Negative plunges possible.

    :param vector: Vector to rotate.
    :param about_vector: Vector to rotate about.
    :param amount: How many radians to rotate.
    :return: Rotated vector.
    """
    if np.all(vector == 0) or np.all(about_vector == 0):
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if np.allclose(
        vector / np.linalg.norm(vector), about_vector / np.linalg.norm(about_vector)
    ):
        return vector
    if np.all(np.cross(vector, about_vector) == 0):
        return vector
    if np.isclose(amount, 0):
        return vector
    about_vector = about_vector / np.linalg.norm(about_vector)
    amount_rad = amount
    try:
        v_rot = (
            vector * np.cos(amount_rad)
            + np.cross(about_vector, vector) * np.sin(amount_rad)
            + about_vector * np.dot(about_vector, vector) * (1 - np.cos(amount_rad))
        )
    except ValueError:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
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
    # Print warning if dip is negative.
    if dip < 0:
        logging.error(f"Dip is negative. Dip: {dip} (In {__name__})")

    nx = np.sin(np.deg2rad(dip_dir)) * np.cos(np.deg2rad(dip))
    ny = np.cos(np.deg2rad(dip_dir)) * np.cos(np.deg2rad(dip))
    nz = -np.sin(np.deg2rad(dip))
    n = np.array([nx, ny, nz], dtype=np.float64)
    # Normalize output vector
    n = n / np.linalg.norm(n)
    return n


def calc_plane_dir_dip(normal: npt.NDArray[np.float64]) -> tuple[float, float]:
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


def calc_vector_trend_plunge(vector: npt.NDArray[np.float64]) -> tuple[float, float]:
    """
    Calculate trend and plunge of a vector.

    Does not assume that the data is axial and a negative plunge result implies
    that the gamma feature is pointed upwards.

    :param vector: vector vector of a plane.
    :return: Direction of dip and dip in degrees
    """
    if np.all(vector == 0):
        return np.nan, np.nan

    if vector[2] > 0:
        plunge_radians = np.arcsin(vector[2])
        plunge_degrees = -np.rad2deg(plunge_radians)
    else:
        plunge_radians = np.arcsin(vector[2])
        plunge_degrees = -np.rad2deg(plunge_radians)

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


def transform_without_gamma(
    alpha: float, beta: float, drillcore_trend: float, drillcore_plunge: float
) -> tuple[float, float]:
    """
    Transform alpha and beta measurements from core.

    E.g.

    >>> transform_without_gamma(45, 0, 0, 90)
    (45.00000000000001, 0.0)

    :param alpha: Angle in degrees between drillcore axis and plane.
    :param beta: Angle in degrees between TOP mark of core and ellipse long axis at
        DOWN hole end.
    :param drillcore_trend: Trend of the drillcore.
    :param drillcore_plunge: Plunge of the drillcore.
    :return: Plane dip and direction
    """
    if any(np.isnan(x) for x in (alpha, beta, drillcore_trend, drillcore_plunge)):
        return np.nan, np.nan
    try:
        plane_normal = calc_global_normal_vector(
            alpha, beta, drillcore_trend, drillcore_plunge
        )

        plane_dir, plane_dip = calc_plane_dir_dip(plane_normal)
        return plane_dip, plane_dir
    except ValueError as e:
        print(str(e))
        return np.nan, np.nan


def transform_with_gamma(
    alpha: float,
    beta: float,
    drillcore_trend: float,
    drillcore_plunge: float,
    gamma: float,
) -> tuple[float, float, float, float]:
    """
    Transform alpha, beta and gamma measurements from core.

    E.g.

    >>> transform_with_gamma(45, 0, 0, 90, 10)
    (45.00000000000001, 0.0, -36.39247, 137.48165)

    :param alpha: Angle in degrees between drillcore axis and plane.
    :param beta: Angle in degrees between TOP mark of core and ellipse
        long axis at DOWN hole end in counterclockwise direction.
    :param drillcore_trend: Trend of the drillcore.
    :param drillcore_plunge: Plunge of the drillcore.
    :param gamma: Linear feature on a plane. Measured in clockwise direction
        from ellipse long axis at DOWN hole end.
    :return: Plane dip and direction + Linear feature plunge and trend.
    """
    if any(np.isnan(x) for x in (alpha, beta, drillcore_trend, drillcore_plunge)):
        return np.nan, np.nan, np.nan, np.nan
    try:
        # plane normal vector
        plane_normal = calc_global_normal_vector(
            alpha, beta, drillcore_trend, drillcore_plunge
        )

        # plane direction of dip and dip
        plane_dir, plane_dip = calc_plane_dir_dip(plane_normal)

        # Vector in the direction of plane dir and dip
        plane_vector = vector_from_dip_and_dir(plane_dip, plane_dir)

        # Gamma vector
        gamma_vector = rotate_vector_about_vector(plane_vector, plane_normal, gamma)

        # Gamma trend and plunge
        gamma_trend, gamma_plunge = calc_vector_trend_plunge(gamma_vector)

        return plane_dip, plane_dir, gamma_plunge, gamma_trend
    except ValueError as e:
        print(str(e))
        return np.nan, np.nan, np.nan, np.nan


def fix_to_numerical(values: list) -> list:
    """
    Fix values to numerical.
    """
    new_values = []
    for val in values:
        if isinstance(val, str):
            if len(val) == 0:
                val = np.nan
            else:
                val = val.strip(" ").replace("\xa0", "")
        try:
            val = float(val)
        except ValueError:
            val = np.nan
        new_values.append(val)
    return new_values

def calc_difference_between_two_planes(
    dip_first: float, dir_first: float, dip_second: float, dir_second: float
) -> float:
    """
    Calculate difference between two measured planes.

    Result is in range [0, 180].
    """
    if any(np.isnan(np.array([dip_first, dir_first, dip_second, dir_second]))):
        return np.nan
    if np.isclose(dip_first, dip_second) and np.isclose(dir_first, dir_second):
        return 0.0
    vec_first = calc_normal_vector_of_plane(dip_first, dir_first)
    vec_second = calc_normal_vector_of_plane(dip_second, dir_second)
    diff = np.rad2deg(np.arccos(np.dot(vec_first, vec_second)))  # type: ignore
    diff = diff if diff <= 90 else 180 - diff
    return diff
