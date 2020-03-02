from hypothesis import given, assume
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
import numpy as np

import drillcore_transformations_py.transformations as transformations


class TestTransformations:
	alpha_strategy = floats(min_value=0, max_value=90)
	beta_strategy = floats(min_value=0, max_value=360)
	trend_strategy = floats(min_value=0, max_value=90)
	plunge_strategy = floats(min_value=0, max_value=90)
	gamma_strategy = floats(min_value=0, max_value=360)
	vector_strategy = arrays(np.float, shape=3, elements=floats(-1, 1))
	amount_strategy = floats(0, np.pi*2)
	dip_strategy = floats(min_value=0, max_value=90)
	dir_strategy = floats(min_value=0, max_value=360)

	@given(alpha_strategy, beta_strategy, trend_strategy, plunge_strategy)
	def test_calc_global_normal_vector(self, alpha, beta, trend, plunge):
		vector = transformations.calc_global_normal_vector(alpha, beta, trend, plunge)
		assert np.isclose(np.linalg.norm(vector), 1)
		assert vector[2] >= 0

	@given(vector_strategy, vector_strategy, amount_strategy)
	def test_rotate_vector_about_vector(self, vector, about_vector, amount):
		transformations.rotate_vector_about_vector(vector, about_vector, amount)

		# sample test
		rotated_vector_ = transformations.rotate_vector_about_vector(np.array([1, 0, 1]), np.array([0, 0, 1]), np.pi)
		assert np.allclose(rotated_vector_, np.array([-1.0000000e+00,  1.2246468e-16,  1.0000000e+00]))

		# if not np.all(vector == 0) and not np.all(about_vector == 0):
		# 	if not np.isclose(amount, 0) and not np.isclose(amount, np.pi*2):
		# 		assert not np.allclose(rotated_vector, vector)

	@given(dip_strategy, dir_strategy)
	def test_vector_from_dip_and_dir(self, dip, dir):
		vector = transformations.vector_from_dip_and_dir(dip, dir)
		assert np.isclose(np.linalg.norm(vector), 1.)
		assert vector[2] <= np.float(0)

	@given(vector_strategy)
	def test_calc_plane_dir_dip(self, normal):
		dir_degrees, dip_degrees = transformations.calc_plane_dir_dip(normal)
		assume(not np.all(normal == 0))
		assert dir_degrees >= 0.
		assert dir_degrees <= 360.
		assert dip_degrees >= 0.
		assert dip_degrees <= 90.


	@given(vector_strategy)
	def test_calc_vector_trend_plunge(self, vector):
		dir_degrees, plunge_degrees = transformations.calc_vector_trend_plunge(vector)
		assume(not np.all(vector == 0))
		assert dir_degrees >= 0.
		assert dir_degrees <= 360.
		assert plunge_degrees >= -90.
		assert plunge_degrees <= 90.

	@given(alpha_strategy, beta_strategy, trend_strategy, plunge_strategy)
	def test_transform_without_gamma(self, alpha, beta, drillcore_trend, drillcore_plunge):
		plane_dip, plane_dir = transformations.transform_without_gamma(alpha, beta, drillcore_trend, drillcore_plunge)
		assert plane_dir >= 0.
		assert plane_dir <= 360.
		assert plane_dip >= 0.
		assert plane_dip <= 90.

	@given(alpha_strategy, beta_strategy, trend_strategy, plunge_strategy, gamma_strategy)
	def test_transform_with_gamma(self, alpha, beta, drillcore_trend, drillcore_plunge, gamma):
		plane_dip, plane_dir, gamma_plunge, gamma_trend =\
			transformations.transform_with_gamma(alpha, beta, drillcore_trend, drillcore_plunge, gamma)
		assert plane_dir >= 0.
		assert plane_dir <= 360.
		assert plane_dip >= 0.
		assert plane_dip <= 90.
		assert gamma_trend >= 0.
		assert gamma_trend <= 360.
		assert gamma_plunge >= -90.
		assert gamma_plunge <= 90.
		plane_dip, plane_dir, gamma_plunge, gamma_trend = transformations.transform_with_gamma(45, 0, 0, 90, 10)
		assert np.allclose((plane_dip, plane_dir, gamma_plunge, gamma_trend)
						   , (45.00000000000001, 0.0, -36.39247, 137.48165))
