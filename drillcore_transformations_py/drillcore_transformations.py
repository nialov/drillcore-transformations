"""
Main module.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from mpl_toolkits.mplot3d import Axes3D
from sympy import Plane, Point3D, Line3D

from drillcore_transformations_py.gamma_rotation import rotateAbout


def calc_global_normal_vector(alpha, beta, trend, plunge):
	"""
	Calculates the normal vector of a measured plane based on alpha and beta measurements and the trend and plunge
	of the drillcore.
	Help and code snippets from:
	https://www.researchgate.net/publication/256939047_Orientation_uncertainty_goes_bananas_An_algorithm_to_visualise_the_uncertainty_sample_space_on_stereonets_for_oriented_objects_measured_in_boreholes

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


def visualize_vector(vector: np.array, ax: Axes3D, **kwargs):
	vector = vector.round(decimals=2)
	ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], **kwargs)


def visualize_plane(xyz: list, ax: Axes3D, **kwargs):
	surf = ax.plot_surface(xyz[0], xyz[1], xyz[2], alpha=0.2, **kwargs)
	surf._facecolors2d = surf._facecolors3d
	surf._edgecolors2d = surf._edgecolors3d


def vector_from_dip_and_dir(dip, dir):
	"""
	Assembles a normalized vector that always points downwards from dip and dip direction.
	Credits to PhD Jussi Mattila (Rock Mechanics Consulting Finland) for this snippet.

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
	# Fracture vector plane plunge
	measured_plane = Plane(Point3D(0, 0, 0), normal_vector=tuple(normal))
	surface_plane = Plane(Point3D(0, 0, 0), Point3D(1, 1, 0), Point3D(1, -1, 0))
	dip_radians = measured_plane.angle_between(surface_plane)
	dip_degrees = \
		np.rad2deg(float(dip_radians)) \
			if np.rad2deg(float(dip_radians)) <= 90 else \
			180 - np.rad2deg(float(dip_radians))

	# Get fracture vector trend from fracture normal vector
	normal_projection = surface_plane.projection_line(Line3D(Point3D(0, 0, 0), direction_ratio=tuple(normal)))
	north_line = Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))
	# y is negative
	if float(normal_projection.p2[1]) < 0:
		# x is negative
		if float(normal_projection.p2[0]) < 0:
			dir_radians = sympy.pi + normal_projection.smallest_angle_between(north_line)
		# x is positive
		else:
			dir_radians = sympy.pi - normal_projection.smallest_angle_between(north_line)
	# y is positive
	else:
		# x is negative
		if float(normal_projection.p2[0]) < 0:
			dir_radians = 2 * sympy.pi - normal_projection.smallest_angle_between(north_line)
		# x is positive
		else:
			dir_radians = normal_projection.smallest_angle_between(north_line)

	dir_degrees = np.rad2deg(float(dir_radians))
	return dir_degrees, dip_degrees

def trend_plunge_of_vector(vector):
	"""
	Calculate trend and plunge of a vector.

	:param vector: Vector
	:type vector: numpy.ndarray
	:return: Trend and plunge
	:rtype: tuple[float, float]
	"""
	# Get vector trend from plane normal vector
	surface_plane = Plane(Point3D(0, 0, 0), Point3D(1, 1, 0), Point3D(1, -1, 0))
	vector_projection = surface_plane.projection_line(Line3D(Point3D(0, 0, 0), direction_ratio=tuple(vector)))
	north_line = Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))
	# y is negative
	if float(vector_projection.p2[1]) < 0:
		# x is negative
		if float(vector_projection.p2[0]) < 0:
			trend_radians = sympy.pi + vector_projection.smallest_angle_between(north_line)
		# x is positive
		else:
			trend_radians = sympy.pi - vector_projection.smallest_angle_between(north_line)
	# y is positive
	else:
		# x is negative
		if float(vector_projection.p2[0]) < 0:
			trend_radians = 2 * sympy.pi - vector_projection.smallest_angle_between(north_line)
		# x is positive
		else:
			trend_radians = vector_projection.smallest_angle_between(north_line)

	trend_degrees = np.rad2deg(float(trend_radians))
	plunge_degrees = np.rad2deg(-np.arcsin(vector[2]))

	return trend_degrees, plunge_degrees

def visualize_results(plane_normal, plane_vector, borehole_vector, alpha, beta, gamma_vector=None):
	"""
	Visualizes alpha-beta measured plane and gamma measured linear feature if given.

	:param plane_normal: Normal of measured plane.
	:type plane_normal: numpy.ndarray
	:param plane_vector: Vector in the dip direction and dip of plane.
	:type plane_vector: numpy.ndarray
	:param borehole_vector: Vector in the direction of the borehole.
	:type borehole_vector: numpy.ndarray
	:param gamma_vector:
	:type gamma_vector: numpy.ndarray
	:return:
	:rtype:
	"""

	# 3D Figure
	fig = plt.figure(figsize=(9, 9))
	ax = fig.gca(projection='3d')
	ax: Axes3D

	# Create plane plane for visualization
	xx, yy = np.meshgrid(np.linspace(-1, 1), np.linspace(-1, 1))
	d = -np.array([0, 0, 0]).dot(plane_normal)
	zz = (-plane_normal[0] * xx - plane_normal[1] * yy - d) * 1. / plane_normal[2]

	# Plot in 3D
	visualize_vector(plane_normal, ax, label='normal of plane', color='red')
	visualize_plane([xx, yy, zz], ax, label='plane plane', color='red')
	visualize_vector(plane_vector, ax, label=r'plunge vector of plane', color='brown', linewidth=2)
	visualize_vector(borehole_vector, ax, label='borehole', color='black', linewidth=2)
	if gamma_vector is not None:
		visualize_vector(gamma_vector, ax, label='gamma', color='darkgreen', linewidth=2)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)
	ax.set_zlim(-1, 1)
	if gamma_vector is not None:
		ax.text(0, 0, 1.4, s=f'bh dir {borehole_dir}\n'
							 f'bh dip {borehole_dip}\n'
							 f'alpha {alpha}\n'
							 f'beta {beta}\n'
							 f'gamma {gamma}\n')
	else:
		ax.text(0, 0, 1.4, s=f'bh dir {borehole_dir}\n'
							 f'bh dip {borehole_dip}\n'
							 f'alpha {alpha}\n'
							 f'beta {beta}\n')

	plt.legend()


def transform_without_gamma(alpha, beta, borehole_trend, borehole_plunge):
	"""
	Transforms alpha and beta measurements from core.

	:param alpha: Angle in degrees between drillcore axis and plane.
	:type alpha: float
	:param beta: Angle in degrees between TOP mark of core and ellipse long axis at DOWN hole end.
	:type beta: float
	:param borehole_trend: Trend of the borehole.
	:type borehole_trend: float
	:param borehole_plunge: Plunge of the borehole.
	:type borehole_plunge: float
	:return: Plane dip and direction
	:rtype: Tuple
	"""
	# Achieve clockwise measurement rotation of beta.
	beta = -beta
	# plane normal vector
	plane_normal = calc_global_normal_vector(alpha, beta, borehole_trend, borehole_plunge)

	# plane direction of dip and dip
	plane_dir, plane_dip = calc_plane_dir_dip(plane_normal)

	return plane_dip, plane_dir


def transform_with_gamma(alpha, beta, borehole_trend, borehole_plunge, gamma):
	"""
	Transforms alpha, beta and gamma measurements from core.
	Rotation of vector about another vector	code is from:
	https://gist.github.com/fasiha/6c331b158d4c40509bd180c5e64f7924
	Big thanks to github user fasiha!

	:param alpha: Angle in degrees between drillcore axis and plane.
	:type alpha: float
	:param beta: Angle in degrees between TOP mark of core and ellipse long axis at DOWN hole end in counterclockwise
		direction.
	:type beta: float
	:param borehole_trend: Trend of the borehole.
	:type borehole_trend: float
	:param borehole_plunge: Plunge of the borehole.
	:type borehole_plunge: float
	:param gamma: Linear feature on a plane. Measured in clockwise direction from ellipse long axis at DOWN hole end.
	:type gamma: float
	:return: Plane dip and direction + Linear feature plunge and trend.
	:rtype: Tuple
	"""
	# Achieve clockwise measurement rotation of beta.
	beta = -beta

	# plane normal vector
	plane_normal = calc_global_normal_vector(alpha, beta, borehole_trend, borehole_plunge)

	# plane direction of dip and dip
	plane_dir, plane_dip = calc_plane_dir_dip(plane_normal)

	# Vector in the direction of plane dir and dip
	plane_vector = vector_from_dip_and_dir(plane_dip, plane_dir)

	# Gamma vector
	# Rotates with right-hand curl. Needs to be reversed to achieve clockwise(!) gamma measurement
	rotation = -np.deg2rad(gamma)
	gamma_vector = rotateAbout(plane_vector, plane_normal, rotation)

	# Gamma trend and plunge
	gamma_trend, gamma_plunge = trend_plunge_of_vector(gamma_vector)

	return plane_dip, plane_dir, gamma_plunge, gamma_trend


def transform_with_visualization(alpha, beta, borehole_trend, borehole_plunge, with_gamma=False, gamma=None):
	"""
	Transforms alpha, beta and gamma measurements from core and visualizes in Matplotlib 3D-plot.

	:param alpha: Angle in degrees between drillcore axis and plane.
	:type alpha: float
	:param beta: Angle in degrees between TOP mark of core and ellipse long axis at DOWN hole end.
	:type beta: float
	:param borehole_trend: Trend of the borehole.
	:type borehole_trend: float
	:param borehole_plunge: Plunge of the borehole.
	:type borehole_plunge: float
	:param with_gamma: Visualize gamma measurement or not.
	:type with_gamma: bool
	:param gamma: Linear feature on a plane. Measured in clockwise direction from ellipse long axis at DOWN hole end.
	:type gamma: float
	:return: Plane dip and direction + Linear feature plunge and trend.
	:rtype: Tuple
	"""
	# Achieve clockwise measurement rotation of beta.
	beta = -beta
	# Borehole vector
	borehole_vector = vector_from_dip_and_dir(borehole_plunge, borehole_trend)

	# plane normal vector
	plane_normal = calc_global_normal_vector(alpha, beta, borehole_trend, borehole_plunge)

	# plane direction of dip and dip
	plane_dir, plane_dip = calc_plane_dir_dip(plane_normal)
	print(plane_dir, plane_dip)

	# Vector in the direction of plane dir and dip
	plane_vector = vector_from_dip_and_dir(plane_dip, plane_dir)
	if with_gamma:
		# Gamma vector
		# Rotates with right-hand curl. Needs to be reversed to achieve clockwise(!) gamma measurement
		rotation = -np.deg2rad(gamma)
		gamma_vector = rotateAbout(plane_vector, plane_normal, rotation)

		# Gamma trend and plunge
		gamma_trend, gamma_plunge = trend_plunge_of_vector(gamma_vector)

		visualize_results(plane_normal, plane_vector, borehole_vector, alpha, -beta, gamma_vector)

		return plane_dip, plane_dir, gamma_plunge, gamma_trend

	visualize_results(plane_normal, plane_vector, borehole_vector, alpha, -beta)
	return plane_dip, plane_dir


def main(alpha, beta, borehole_trend, borehole_plunge, with_gamma, gamma):
	pass
	# transform_with_visualization(alpha, beta, borehole_trend, borehole_plunge, with_gamma, gamma)


if __name__ == "__main__":
	alpha, beta, borehole_dir, borehole_dip, gamma = 55, 1, 0, 70, 45
	with_gamma = True
	main(alpha, beta, borehole_dir, borehole_dip, with_gamma, gamma)
