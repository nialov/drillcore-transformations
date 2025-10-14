"""
Test drillcore_transformations.py.
"""
import os
from contextlib import contextmanager
from pathlib import Path
from shutil import copy
from tempfile import TemporaryDirectory

from hypothesis import assume, given, settings
from hypothesis.strategies import lists

from drillcore_transformations import usage
from tests import dict_strategy, function_strategy, text_strategy


@contextmanager
def change_dir(path: Path):
    """
    Change dir to path.
    """
    current_path = Path(".").resolve()
    try:
        yield os.chdir(path)
    finally:
        os.chdir(current_path)


def test_get_config_identifiers():
    """
    Test get_config_identifiers.
    """
    base_measurements, headers, conf = usage.get_config_identifiers()
    for s in base_measurements + headers:
        assert isinstance(s, str)
    assert all(isinstance(path, Path) for path in conf)
    return base_measurements, headers, conf


def test_initialize_config(tmp_path):
    """
    Test initialize_config.
    """
    _, _, _ = test_get_config_identifiers()
    config_path = tmp_path / "config.ini"
    usage.initialize_config(config_path=config_path)
    assert config_path.exists()


@settings(deadline=None)
@given(name=text_strategy)
def test_add_and_remove_column_name(name):
    """
    Test add_and_remove_column_name.
    """
    with TemporaryDirectory() as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        config_path = tmp_path / usage._CONFIG.name
        copy(usage._CONFIG, config_path)
        with change_dir(tmp_path):
            base_measurements, headers, _ = test_get_config_identifiers()
            usage.add_column_name(
                headers[0], base_measurements[0], name, config_path=config_path
            )
            assert not usage.add_column_name(
                headers[0],
                base_measurements[0],
                base_measurements[0],
                config_path=config_path,
            )
            # testing removal
            usage.remove_column_name(
                headers[0], base_measurements[0], name, config_path=config_path
            )
            assert not usage.remove_column_name(
                headers[0],
                base_measurements[0],
                "prettysurethisisnotinthelist",
                config_path=config_path,
            )


@given(lists(elements=text_strategy))
def test_parse_columns_two_files(list_with_texts):
    """
    Test parse_columns_two_files.
    """
    bm, _, _ = test_get_config_identifiers()
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


def test_transform_csv_two_files(tmp_path):
    """
    Test transform_csv_two_files.
    """
    ms_file = Path("sample_data/measurement_sample.csv")
    d_file = Path("sample_data/depth_sample.csv")
    assert ms_file.exists() and d_file.exists()
    temp_file = tmp_path / "csv_ms_transformed.csv"
    usage.transform_csv_two_files(ms_file, d_file, False, temp_file)
    assert temp_file.exists()


def test_transform_excel_two_files_xlsx(tmp_path):
    """
    Test transform_excel_two_files_xlsx.
    """
    ms_file = Path("sample_data/measurement_sample.xlsx")
    d_file = Path("sample_data/depth_sample.xlsx")
    assert ms_file.exists() and d_file.exists()
    temp_file = tmp_path / "xlsx_ms_transformed.csv"
    usage.transform_excel_two_files(ms_file, d_file, False, temp_file)
    assert temp_file.exists()


def test_transform_excel_two_files_xls(tmp_path):
    """
    Test transform_excel_two_files_xls.
    """
    ms_file = Path("sample_data/measurement_sample.xls")
    d_file = Path("sample_data/depth_sample.xls")
    assert ms_file.exists() and d_file.exists()
    temp_file = tmp_path / "xls_ms_transformed.csv"
    usage.transform_excel_two_files(ms_file, d_file, False, temp_file)
    assert temp_file.exists()


@settings(deadline=None)
@given(convention_dict=dict_strategy)
def test_change_conventions(convention_dict):
    """
    Test change_conventions.
    """
    with TemporaryDirectory() as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        test_initialize_config(tmp_path=tmp_path)
        result = usage.change_conventions(convention_dict)
        assert isinstance(result, bool) or result is None
