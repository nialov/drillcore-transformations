import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_vector(vector: np.array, ax: Axes3D, **kwargs):
	vector = vector.round(decimals=2)
	ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], **kwargs)


def visualize_plane(xyz: list, ax: Axes3D, **kwargs):
	surf = ax.plot_surface(xyz[0], xyz[1], xyz[2], alpha=0.2, **kwargs)
	surf._facecolors2d = surf._facecolors3d
	surf._edgecolors2d = surf._edgecolors3d

def visualize_results(plane_normal, plane_vector, drillcore_vector, drillcore_trend, drillcore_plunge, alpha, beta, gamma_vector=None, gamma=None):
	"""
	Visualizes alpha-beta measured plane and gamma measured linear feature if given.

	:param plane_normal: Normal of measured plane.
	:type plane_normal: numpy.ndarray
	:param plane_vector: Vector in the dip direction and dip of plane.
	:type plane_vector: numpy.ndarray
	:param drillcore_vector: Vector in the direction of the drillcore.
	:type drillcore_vector: numpy.ndarray
	:param gamma_vector:
	:type gamma_vector: numpy.ndarray
	:return:
	:rtype:
	"""

	# 3D Figure
	fig = plt.figure(figsize=(9, 9))
	ax = fig.gca(projection='3d')
	ax: Axes3D

	# Create plane for visualization
	xx, yy = np.meshgrid(np.linspace(-1, 1), np.linspace(-1, 1))
	d = -np.array([0, 0, 0]).dot(plane_normal)
	# Only for visualization purposes
	if plane_normal[2] == 0:
		plane_normal[2] = 0.00000000001
	zz = (-plane_normal[0] * xx - plane_normal[1] * yy - d) * 1. / plane_normal[2]

	# Plot in 3D
	visualize_vector(plane_normal, ax, label='normal of plane', color='red')
	visualize_plane([xx, yy, zz], ax, label='plane plane', color='red')
	visualize_vector(plane_vector, ax, label=r'plunge vector of plane', color='brown', linewidth=2)
	visualize_vector(drillcore_vector, ax, label='drillcore', color='black', linewidth=2)
	if gamma_vector is not None:
		visualize_vector(gamma_vector, ax, label='gamma', color='darkgreen', linewidth=2)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)
	ax.set_zlim(-1, 1)
	if gamma_vector is not None:
		ax.text(0, 0, 1.4, s=f'bh dir {drillcore_trend}\n'
							 f'bh dip {drillcore_plunge}\n'
							 f'alpha {alpha}\n'
							 f'beta {beta}\n'
							 f'gamma {gamma}\n')
	else:
		ax.text(0, 0, 1.4, s=f'bh dir {drillcore_trend}\n'
							 f'bh dip {drillcore_plunge}\n'
							 f'alpha {alpha}\n'
							 f'beta {beta}\n')

	plt.legend()
