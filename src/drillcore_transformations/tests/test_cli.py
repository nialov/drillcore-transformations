"""
Test drillcore_transformations.py.
"""
from click.testing import CliRunner

from drillcore_transformations import cli

# from tests import sample_csv, sample_csv_result


def test_transform():
    """
    Test transform.
    """
    runner = CliRunner()
    # TODO: Fix cli transform test
    # result = runner.invoke(cli.transform, [str(sample_csv), "--gamma"])
    result = runner.invoke(cli.transform, ["--help"])
    assert result.exit_code == 0
    # assert sample_csv_result.exists()


def test_conventions():
    """
    Test conventions.
    """
    runner = CliRunner()
    result = runner.invoke(cli.conventions, [])
    assert result.exit_code == 0
    assert "Changing conventions" in result.output


def test_config():
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
