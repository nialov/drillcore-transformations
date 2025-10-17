import numpy as np

from drillcore_transformations.transformations import calc_normal_vector_of_plane



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
    dot = np.dot(vec_first, vec_second)
    # Clamp the dot product to [-1, 1] to avoid invalid values due to floating-point errors.
    dot = np.clip(dot, -1.0, 1.0)
    diff = np.rad2deg(np.arccos(dot))
    diff = diff if diff <= 90 else 180 - diff
    return float(diff)


