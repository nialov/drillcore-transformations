"""
Main module.
"""
import numpy as np
from drillcore_transformations_py.visualizations import visualize_results


def calc_global_normal_vector(alpha, beta, trend, plunge):
	"""
	Calculates the normal vector of a measured plane based on alpha and beta measurements and the trend and plunge
	of the drillcore.
	Help and code snippets from:
	https://tinyurl.com/tqr84ww

	:param alpha: Alpha of the measured plane in degrees.
	:type alpha: float
	:param beta: Beta of the measured plane in degrees.
	:type beta: float
	:param trend: Trend of the drillcore
	:type trend: float
	:param plunge: Plunge of the drillcore
	:type plunge: float
	:return: Normalized normal vector of a plane. Always points upwards (z >= 0)
	:rtype: numpy.ndarray
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
	ng_1 = np.cos(np.pi / 2 - trend) * np.cos(np.pi / 2 - plunge) * np.cos(beta) * np.cos(alpha) - np.sin(
		np.pi / 2 - trend) * np.sin(beta) * np.cos(alpha) + np.cos(np.pi / 2 - trend) * np.sin(
		np.pi / 2 - plunge) * np.sin(alpha)
	ng_2 = np.sin(np.pi / 2 - trend) * np.cos(np.pi / 2 - plunge) * np.cos(beta) * np.cos(alpha) + np.cos(
		np.pi / 2 - trend) * np.sin(beta) * np.cos(alpha) + np.sin(np.pi / 2 - trend) * np.sin(
		np.pi / 2 - plunge) * np.sin(alpha)
	ng_3 = -np.sin(np.pi / 2 - plunge) * np.cos(beta) * np.cos(alpha) + np.cos(np.pi / 2 - plunge) * np.sin(
		alpha)

	# Always return a normalized vector pointing downwards.
	if ng_3 < 0:
		return np.array([-ng_1, -ng_2, -ng_3]) / np.linalg.norm(np.array([-ng_1, -ng_2, -ng_3]))
	else:
		return np.array([ng_1, ng_2, ng_3]) / np.linalg.norm(np.array([ng_1, ng_2, ng_3]))


def rotate_vector_about_vector(vector, about_vector, amount):
	"""
	Rotates a given vector about another vector.

	Implements Rodrigues' rotation formula:
	https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

	Example:

	>>> rotate_vector_about_vector(np.array([1, 0, 1]), np.array([0, 0, 1]), np.pi)
	array([-1.0000000e+00,  1.2246468e-16,  1.0000000e+00])

	TODO: Is gamma axial or vector data? Right now treated as vector. => Negative plunges possible.

	:param vector: Vector to rotate.
	:type vector: numpy.ndarray
	:param about_vector: Vector to rotate about.
	:type about_vector: numpy.ndarray
	:param amount: How many radians to rotate.
	:type amount: float
	:return: Rotated vector.
	:rtype: np.ndarray
	"""
	about_vector = about_vector / np.linalg.norm(about_vector)
	amount_rad = amount
	v_rot = vector * np.cos(amount_rad) \
			+ np.cross(about_vector, vector) \
			* np.sin(amount_rad) + about_vector \
			* np.dot(about_vector, vector) \
			* (1 - np.cos(amount_rad))

	return v_rot


def vector_from_dip_and_dir(dip, dir):
	"""
	Assembles a normalized vector that always points downwards, if dip is positive, from dip and dip direction.
	Credits to PhD Jussi Mattila for this snippet.

	Example:

	>>> vector_from_dip_and_dir(45, 0)
	array([ 0.        ,  0.70710678, -0.70710678])

	:param dip: Dip of a feature. Between [0, 90]
	:type dip: float
	:param dir: Dip direction of feature.
	:type dir: float
	:return: Normalized vector pointing in the direction and the dip.
	:rtype: numpy.ndarray
	"""
	# Print warning if dip is negative.
	if dip < 0:
		print(f"Warning!\n"
			  f"Dip is negative. Dip: {dip}\n"
			  f"In {__name__}")

	nx = np.sin(np.deg2rad(dir)) * np.cos(np.deg2rad(dip))
	ny = np.cos(np.deg2rad(dir)) * np.cos(np.deg2rad(dip))
	nz = -np.sin(np.deg2rad(dip))
	n = np.array([nx, ny, nz])
	# Normalize output vector
	n = n / np.linalg.norm(n)
	return n


def calc_plane_dir_dip(normal):
	"""
	Calculate direction of dip and dip of a plane based on normal vector of plane.

	:param normal: Normal vector of a plane.
	:type normal: numpy.ndarray
	:return: Direction of dip and dip in degrees
	:rtype: tuple[float, float]
	"""
	# plane dip
	dip_radians = np.pi / 2 - np.arcsin(normal[2])
	dip_degrees = np.rad2deg(dip_radians)
	# Get plane vector trend from plane normal vector

	normal_xy = normal[:2]
	normal_xy = normal_xy / np.linalg.norm(normal_xy)
	dir_0 = np.array([0, 1.])
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
	return dir_degrees, dip_degrees


def calc_vector_trend_plunge(vector):
	"""
	Calculate trend and plunge of a vector. Does not assume that the data is axial and a negative plunge result implies
	that the gamma feature is pointed upwards.

	:param vector: vector vector of a plane.
	:type vector: numpy.ndarray
	:return: Direction of dip and dip in degrees
	:rtype: tuple[float, float]
	"""

	if vector[2] > 0:
		plunge_radians = np.arcsin(vector[2])
		plunge_degrees = -np.rad2deg(plunge_radians)
	else:
		plunge_radians = np.arcsin(vector[2])
		plunge_degrees = -np.rad2deg(plunge_radians)

	# Get vector trend
	vector_xy = vector[:2]
	vector_xy = vector_xy / np.linalg.norm(vector_xy)
	dir_0 = np.array([0, 1.])
	# y is negative
	if vector_xy[1] < 0:
		# x is negative
		if vector_xy[0] < 0:
			dir_radians = np.pi * 2 - np.arccos(np.dot(vector_xy, dir_0))
		# x is positive
		else:
			dir_radians = np.arccos(np.dot(vector_xy, dir_0))
	# y is positive
	else:
		# x is negative
		if vector_xy[0] < 0:
			dir_radians = np.pi * 2 - np.arccos(np.dot(vector_xy, dir_0))
		# x is positive
		else:
			dir_radians = np.arccos(np.dot(vector_xy, dir_0))

	dir_degrees = np.rad2deg(dir_radians)
	return dir_degrees, plunge_degrees


def transform_without_gamma(alpha, beta, drillcore_trend, drillcore_plunge):
	"""
	Transforms alpha and beta measurements from core.

	:param alpha: Angle in degrees between drillcore axis and plane.
	:type alpha: float
	:param beta: Angle in degrees between TOP mark of core and ellipse long axis at DOWN hole end.
	:type beta: float
	:param drillcore_trend: Trend of the drillcore.
	:type drillcore_trend: float
	:param drillcore_plunge: Plunge of the drillcore.
	:type drillcore_plunge: float
	:return: Plane dip and direction
	:rtype: Tuple
	"""

	if np.NaN in (alpha, beta, drillcore_trend, drillcore_plunge):
		return np.NaN, np.NaN
	try:
		# plane normal vector
		# >>> timeit.timeit(lambda: calc_global_normal_vector(41, 195, 44, 85), number=1000)
		# 0.04800009999996746
		plane_normal = calc_global_normal_vector(alpha, beta, drillcore_trend, drillcore_plunge)

		plane_dir, plane_dip = calc_plane_dir_dip(plane_normal)
		return plane_dip, plane_dir
	except ValueError as e:
		print(str(e))
		return np.NaN, np.NaN


def transform_with_gamma(alpha, beta, drillcore_trend, drillcore_plunge, gamma):
	"""
	Transforms alpha, beta and gamma measurements from core.

	:param alpha: Angle in degrees between drillcore axis and plane.
	:type alpha: float
	:param beta: Angle in degrees between TOP mark of core and ellipse long axis at DOWN hole end in counterclockwise
		direction.
	:type beta: float
	:param drillcore_trend: Trend of the drillcore.
	:type drillcore_trend: float
	:param drillcore_plunge: Plunge of the drillcore.
	:type drillcore_plunge: float
	:param gamma: Linear feature on a plane. Measured in clockwise direction from ellipse long axis at DOWN hole end.
	:type gamma: float
	:return: Plane dip and direction + Linear feature plunge and trend.
	:rtype: tuple[float, float, float, float]
	"""
	if np.NaN in (alpha, beta, drillcore_trend, drillcore_plunge):
		return np.NaN, np.NaN, np.NaN, np.NaN
	try:
		# plane normal vector
		plane_normal = calc_global_normal_vector(alpha, beta, drillcore_trend, drillcore_plunge)

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
		return np.NaN, np.NaN, np.NaN, np.NaN


def transform_with_visualization(alpha, beta, drillcore_trend, drillcore_plunge, with_gamma=False, gamma=None):
	"""
	Transforms alpha, beta and gamma measurements from core and visualizes in Matplotlib 3D-plot.

	:param alpha: Angle in degrees between drillcore axis and plane.
	:type alpha: float
	:param beta: Angle in degrees between TOP mark of core and ellipse long axis at DOWN hole end.
	:type beta: float
	:param drillcore_trend: Trend of the drillcore.
	:type drillcore_trend: float
	:param drillcore_plunge: Plunge of the drillcore.
	:type drillcore_plunge: float
	:param with_gamma: Visualize gamma measurement or not.
	:type with_gamma: bool
	:param gamma: Linear feature on a plane. Measured in clockwise direction from ellipse long axis at DOWN hole end.
	:type gamma: float
	:return: Plane dip and direction + Linear feature plunge and trend.
	:rtype: tuple[float, float, float, float] | tuple[float, float]
	"""
	if with_gamma:
		plane_dip, plane_dir, gamma_plunge, gamma_trend = transform_with_gamma(alpha, beta, drillcore_trend, drillcore_plunge, gamma)
	else:
		plane_dip, plane_dir = transform_without_gamma(alpha, beta, drillcore_trend, drillcore_plunge)

	drillcore_vector = vector_from_dip_and_dir(drillcore_plunge, drillcore_trend)
	# plane normal vector
	plane_normal = calc_global_normal_vector(alpha, beta, drillcore_trend, drillcore_plunge)
	# Vector in the direction of plane dir and dip
	plane_vector = vector_from_dip_and_dir(plane_dip, plane_dir)

	if with_gamma:

		gamma_vector = rotate_vector_about_vector(plane_vector, plane_normal, gamma)

		# Gamma trend and plunge
		gamma_trend, gamma_plunge = calc_vector_trend_plunge(gamma_vector)

		visualize_results(plane_normal, plane_vector, drillcore_vector, drillcore_trend, drillcore_plunge, alpha, -beta,
						  gamma_vector, gamma)

		return plane_dip, plane_dir, gamma_plunge, gamma_trend

	visualize_results(plane_normal, plane_vector, drillcore_vector, drillcore_trend, drillcore_plunge, alpha, -beta)
	return plane_dip, plane_dir

