"""
Test drillcore_transformations.py.
"""
from pathlib import Path

from hypothesis import assume, given
from hypothesis.strategies import lists

from drillcore_transformations import usage
from tests import dict_strategy, function_strategy, text_strategy


@given(function_strategy)
def test_check_config(method):
    """
    Test check_config.
    """
    usage.check_config(method)


def test_get_config_identifiers():
    """
    Test get_config_identifiers.
    """
    base_measurements, headers, conf = usage.get_config_identifiers()
    for s in base_measurements + headers:
        assert isinstance(s, str)
    assert isinstance(*conf, Path)
    return base_measurements, headers, conf


def test_initialize_config():
    """
    Test initialize_config.
    """
    _, _, _ = test_get_config_identifiers()
    usage.initialize_config()
    test_check_config()


@given(text_strategy)
def test_add_and_remove_column_name(name):
    """
    Test add_and_remove_column_name.
    """
    try:
        base_measurements, headers, _ = test_get_config_identifiers()
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
        test_initialize_config()
        raise


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


@given(dict_strategy)
def test_change_conventions(convention_dict):
    """
    Test change_conventions.
    """
    test_initialize_config()
    result = usage.change_conventions(convention_dict)
    test_initialize_config()
    assert result is False
    none_result = usage.change_conventions({"alpha": "negative"})
    test_initialize_config()
    assert none_result is None
