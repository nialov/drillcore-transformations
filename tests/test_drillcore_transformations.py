"""
Test drillcore_transformations.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from click.testing import CliRunner
from drillcore_transformations import cli, transformations, usage, visualizations
from hypothesis import assume, given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import dictionaries, floats, functions, lists, text

alpha_strategy = floats(min_value=-90, max_value=90)
beta_strategy = floats(min_value=-360, max_value=360)
trend_strategy = floats(min_value=0, max_value=90)
plunge_strategy = floats(min_value=-90, max_value=90)
gamma_strategy = floats(min_value=-360, max_value=360)
vector_strategy = arrays(np.float, shape=3, elements=floats(-1, 1))
amount_strategy = floats(0, np.pi * 2)
dip_strategy = floats(min_value=0, max_value=90)
dir_strategy = floats(min_value=0, max_value=360)
function_strategy = functions()
text_strategy = text()
dict_strategy = dictionaries(text_strategy, text_strategy)

sample_csv = (
    Path(__file__).parent.parent / Path("sample_data/Logging_sheet.csv")
).absolute()
sample_csv_result = (
    Path(__file__).parent.parent / Path("sample_data/Logging_sheet_transformed.csv")
).absolute()


class TestTransformations:

    """
    Test Transformations.
    """

    @given(alpha_strategy, beta_strategy, trend_strategy, plunge_strategy)
    def test_calc_global_normal_vector(self, alpha, beta, trend, plunge):
        """
        Test calc_global_normal_vector.
        """
        vector = transformations.calc_global_normal_vector(alpha, beta, trend, plunge)
        assert np.isclose(np.linalg.norm(vector), 1)
        assert vector[2] >= 0

    @given(vector_strategy, vector_strategy, amount_strategy)
    def test_rotate_vector_about_vector(self, vector, about_vector, amount):
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
    def test_vector_from_dip_and_dir(self, dip, dip_dir):
        """
        Test vector_from_dip_and_dir.
        """
        vector = transformations.vector_from_dip_and_dir(dip, dip_dir)
        assert np.isclose(np.linalg.norm(vector), 1.0)
        assert vector[2] <= np.float(0)

    @given(vector_strategy)
    def test_calc_plane_dir_dip(self, normal):
        """
        Test calc_plane_dir_dip.
        """
        assume(not np.all(normal == 0))
        assume(all(10e15 > val > 1e-15 for val in normal))
        dir_degrees, dip_degrees = transformations.calc_plane_dir_dip(normal)
        assert dir_degrees >= 0.0
        assert dir_degrees <= 360.0
        assert dip_degrees >= 0.0
        assert dip_degrees <= 90.0

    @given(vector_strategy)
    def test_calc_vector_trend_plunge(self, vector):
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
        self, alpha, beta, drillcore_trend, drillcore_plunge
    ):
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

    @given(
        alpha_strategy, beta_strategy, trend_strategy, plunge_strategy, gamma_strategy
    )
    def test_transform_with_gamma(
        self, alpha, beta, drillcore_trend, drillcore_plunge, gamma
    ):
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
        self, dip_first, dir_first, dip_second, dir_second
    ):
        """
        Test calc_difference_between_two_planes.
        """
        result = transformations.calc_difference_between_two_planes(
            dip_first, dir_first, dip_second, dir_second
        )

        if not 0 <= result <= 90:
            assert np.isclose(result, 0) or np.isclose(result, 90)

    def test_calc_difference_between_two_planes_nan(self):
        """
        Test calc_difference_between_two_planes with nan.
        """
        result = transformations.calc_difference_between_two_planes(np.nan, 1, 5, 50)
        assert np.isnan(result)


class TestUsage:

    """
    Test Usage.
    """

    @given(function_strategy)
    def test_check_config(self, method):
        """
        Test check_config.
        """
        usage.check_config(method)

    def test_get_config_identifiers(self):
        """
        Test get_config_identifiers.
        """
        base_measurements, headers, conf = usage.get_config_identifiers()
        for s in base_measurements + headers:
            assert isinstance(s, str)
        assert isinstance(*conf, Path)
        return base_measurements, headers, conf

    def test_initialize_config(self):
        """
        Test initialize_config.
        """
        _, _, _ = self.test_get_config_identifiers()
        usage.initialize_config()
        self.test_check_config()

    @given(text_strategy)
    def test_add_and_remove_column_name(self, name):
        """
        Test add_and_remove_column_name.
        """
        try:
            base_measurements, headers, _ = self.test_get_config_identifiers()
            usage.add_column_name(headers[0], base_measurements[0], name)
            assert not usage.add_column_name(
                headers[0], base_measurements[0], base_measurements[0]
            )
            # testing removal
            usage.remove_column_name(headers[0], base_measurements[0], name)
            assert not usage.remove_column_name(
                headers[0], base_measurements[0], "prettysurethisisnotinthelist"
            )
        except Exception:
            self.test_initialize_config()
            raise

    @given(lists(elements=text_strategy))
    def test_parse_columns_two_files(self, list_with_texts):
        """
        Test parse_columns_two_files.
        """
        bm, _, _ = self.test_get_config_identifiers()
        with_gamma = True
        d = usage.parse_columns_two_files(bm, with_gamma)
        for k, v in d.items():
            assert k in bm
            assert v in bm
            assert k == v
        with_gamma = False
        d = usage.parse_columns_two_files(bm, with_gamma)
        for k, v in d.items():
            assert k in bm
            assert v in bm
            assert k == v
        assume(len(list_with_texts) > 3)
        try:
            usage.parse_columns_two_files(list_with_texts, with_gamma)
            usage.parse_columns_two_files(list_with_texts, True)
        except usage.ColumnException:
            # This is fine and expected.
            pass

    def test_transform_csv_two_files(self, tmp_path):
        """
        Test transform_csv_two_files.
        """
        ms_file = Path("sample_data/measurement_sample.csv")
        d_file = Path("sample_data/depth_sample.csv")
        assert ms_file.exists() and d_file.exists()
        temp_file = tmp_path / "csv_ms_transformed.csv"
        usage.transform_csv_two_files(ms_file, d_file, False, temp_file)
        assert temp_file.exists()

    def test_transform_excel_two_files_xlsx(self, tmp_path):
        """
        Test transform_excel_two_files_xlsx.
        """
        ms_file = Path("sample_data/measurement_sample.xlsx")
        d_file = Path("sample_data/depth_sample.xlsx")
        assert ms_file.exists() and d_file.exists()
        temp_file = tmp_path / "xlsx_ms_transformed.csv"
        usage.transform_excel_two_files(ms_file, d_file, False, temp_file)
        assert temp_file.exists()

    def test_transform_excel_two_files_xls(self, tmp_path):
        """
        Test transform_excel_two_files_xls.
        """
        ms_file = Path("sample_data/measurement_sample.xls")
        d_file = Path("sample_data/depth_sample.xls")
        assert ms_file.exists() and d_file.exists()
        temp_file = tmp_path / "xls_ms_transformed.csv"
        usage.transform_excel_two_files(ms_file, d_file, False, temp_file)
        assert temp_file.exists()

    @given(dict_strategy)
    def test_change_conventions(self, convention_dict):
        """
        Test change_conventions.
        """
        self.test_initialize_config()
        result = usage.change_conventions(convention_dict)
        self.test_initialize_config()
        assert result is False
        none_result = usage.change_conventions({"alpha": "negative"})
        self.test_initialize_config()
        assert none_result is None


class TestVisualizations:

    """
    Test visualizations.py.
    """

    plt.rcParams["figure.max_open_warning"] = 100

    @given(vector_strategy)
    @settings(max_examples=5, deadline=None)
    def test_visualize_plane(self, plane_normal):
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
    def test_visualize_vector(self, vector):
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
        self,
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


class TestCli:

    """
    Test cli.py.
    """

    def test_transform(self):
        """
        Test transform.
        """
        runner = CliRunner()
        result = runner.invoke(cli.transform, [str(sample_csv), "--gamma"])
        assert result.exit_code == 0
        assert sample_csv_result.exists()

    def test_conventions(self):
        """
        Test conventions.
        """
        runner = CliRunner()
        result = runner.invoke(cli.conventions, [])
        assert result.exit_code == 0
        assert "Changing conventions" in result.output

    def test_config(self):
        """
        Test config.
        """
        runner = CliRunner()
        result = runner.invoke(cli.config, ["--initialize"])
        assert result.exit_code == 0
        assert "Initializing new config.ini" in result.output

        result = runner.invoke(cli.config, [])
        assert result.exit_code == 0
        assert "Config File Path" in result.output
