"""
Module with all usage methods i.e. for converting data files and manipulating the config.
"""
import bisect
import configparser
import json
from pathlib import Path

import pandas as pd

from drillcore_transformations.transformations import (
    transform_with_gamma,
    transform_without_gamma,
)

# Identifiers within the module. DO NOT CHANGE TO MATCH YOUR DATA FILE COLUMNS.
# Matching your data file to module identifiers is done in the config file (config.ini).
_ALPHA, _BETA, _GAMMA, _MEASUREMENT_DEPTH, _DEPTH, _BOREHOLE_TREND, _BOREHOLE_PLUNGE = (
    "alpha",
    "beta",
    "gamma",
    "measurement_depth",
    "depth",
    "borehole_trend",
    "borehole_plunge",
)

# Headers within config.
_MEASUREMENTS, _DEPTHS, _BOREHOLE, _CONVENTIONS = (
    "MEASUREMENTS",
    "DEPTHS",
    "BOREHOLE",
    "CONVENTIONS",
)

# Config absolute path
_CONFIG = (Path(__file__).parent / Path("config.ini")).absolute()

# Conventions in config
_MEASUREMENT_CONVENTIONS = ["negative", "none"]


class ColumnException(Exception):

    """
    Raise when there are errors with the columns of your data.

    These can be related to not recognizing the column or if multiple columns
    match identifiers in config.ini.

    Most issues can be fixed by checking the config.ini config file and adding
    your data file column names as identifiers or by removing identical
    identifiers.
    """


def check_config(method):
    """
    Check config.
    """

    def inner(*args, **kwargs):
        assert _CONFIG.exists()
        if not _CONFIG.exists():
            raise FileNotFoundError(
                "config.ini file not found. Run usage.initialize_config()."
            )
        result = method(*args, **kwargs)
        return result

    return inner


def get_config_identifiers():
    """
    Get config identifiers.
    """
    base_measurements = [
        _ALPHA,
        _BETA,
        _GAMMA,
        _MEASUREMENT_DEPTH,
        _DEPTH,
        _BOREHOLE_TREND,
        _BOREHOLE_PLUNGE,
    ]
    headers = [_MEASUREMENTS, _DEPTHS, _BOREHOLE, _CONVENTIONS]
    conf = [_CONFIG]
    return base_measurements, headers, conf


def find_config():
    """
    Return config.ini file path.

    :return: config.ini file path
    :rtype: Path
    """
    return _CONFIG


def initialize_config():
    """
    Create a configfile with default names for alpha, beta, etc.

    Filename will be config.ini. Manual editing of this file is allowed but
    editing methods are also present for adding column names.

    Will overwrite if needed.
    """
    config = configparser.ConfigParser()

    # Measurement file identifiers
    config[_MEASUREMENTS] = {}
    config[_MEASUREMENTS][_ALPHA] = json.dumps([_ALPHA, "ALPHA", "ALPHA_CORE"])
    config[_MEASUREMENTS][_BETA] = json.dumps([_BETA, "BETA", "BETA_CORE"])
    config[_MEASUREMENTS][_GAMMA] = json.dumps([_GAMMA, "GAMMA", "GAMMA_CORE"])
    config[_MEASUREMENTS][_MEASUREMENT_DEPTH] = json.dumps(
        [_MEASUREMENT_DEPTH, "MEASUREMENT_DEPTH", "LENGTH_FROM"]
    )

    # Depth file identifiers
    config[_DEPTHS] = {}
    config[_DEPTHS][_DEPTH] = json.dumps([_DEPTH, "DEPTH"])

    # Borehole trend and plunge identifiers
    config[_BOREHOLE] = {}
    config[_BOREHOLE][_BOREHOLE_TREND] = json.dumps(
        [_BOREHOLE_TREND, "BOREHOLE_TREND", "AZIMUTH", "azimuth", "BEARING", "bearing"]
    )
    config[_BOREHOLE][_BOREHOLE_PLUNGE] = json.dumps(
        [
            _BOREHOLE_PLUNGE,
            "BOREHOLE_PLUNGE",
            "INCLINATION",
            "inclination",
            "PLUNGE",
            "plunge",
            "DIP",
        ]
    )

    # Convention related settings
    config[_CONVENTIONS] = {}
    config[_CONVENTIONS][_ALPHA] = "negative"
    config[_CONVENTIONS][_BETA] = "negative"
    config[_CONVENTIONS][_GAMMA] = "negative"
    config[_CONVENTIONS][_BOREHOLE_TREND] = "none"
    config[_CONVENTIONS][_BOREHOLE_PLUNGE] = "none"

    # Write to .ini file. Will overwrite old one or make a new one.
    save_config(config)


def add_column_name(header, base_column, name):
    """
    Add a column name to recognize measurement type.

    E.g. if your alpha measurements are in a column that is named
    "alpha_measurements" you can add it to the config.ini file with:

    >>> add_column_name(_MEASUREMENTS, _ALPHA, "alpha_measurements")
    True

    If the inputted column name is already in the config file, this will be
    printed out and config will not be changed.

    :param header: You may add new column names to the measurements file
        and to the file containing measurement depth
        information.
    :type header: str
    :param base_column: Which type of measurement is the column name.
        Base types for measurements are:
        "alpha" "beta" "gamma" "measurement_depth"
        Base types for depths are:
        "depth"
        Base types for borehole are:
        "borehole_trend" "borehole_plunge"
    :type base_column: str
    :param name: Name of the new column you want to add.
    :type name: str
    """
    if "%" in name:
        name = name.replace("%", "")
    assert header in [_MEASUREMENTS, _DEPTHS, _BOREHOLE]
    config = configparser.ConfigParser()
    configname = _CONFIG
    if not Path(configname).exists():
        print("config.ini configfile not found. Making a new one with default values.")
        initialize_config()
    assert Path(configname).exists()
    config.read(configname)
    column_list = json.loads(config.get(header, base_column))
    assert isinstance(column_list, list)
    if name in column_list:
        print("Given column name is already in the config. No changes required.")
        return False
    column_list.append(name)
    config[header][base_column] = json.dumps(column_list)
    save_config(config)
    return True


def remove_column_name(header, base_column, name):
    """
    Remove a column name from config.ini.

    >>> remove_column_name(_MEASUREMENTS, _ALPHA, "alpha_measurements")
    True

    :param header: Input the the header in which under the name is.
    :type header: str
    :param base_column: Which type of measurement is the column name.
            Base types for measurements are:
            "alpha" "beta" "gamma" "measurement_depth"
            Base types for depths are:
            "depth"
            Base types for borehole are:
            "borehole_trend" "borehole_plunge"
    :type base_column: str
    :param name: Name of the new column you want to remove.
    :type name: str
    """
    if "%" in name:
        name = name.replace("%", "")
    if header not in [_MEASUREMENTS, _DEPTHS, _BOREHOLE]:
        raise ColumnException(
            "Given header was not a base header.\n" f"header: {header}\n"
        )
    config = configparser.ConfigParser()
    configname = _CONFIG
    if not Path(configname).exists():
        print("config.ini configfile not found. Making a new one with default values.")
        initialize_config()
    assert Path(configname).exists()
    config.read(configname)
    column_list = json.loads(config.get(header, base_column))
    assert isinstance(column_list, list)
    if name in column_list:
        column_list.remove(name)
    else:
        print(
            f"Could not remove name: {name}.\n"
            f"It was not found in the config.ini under header: {header}"
            f" and base_column: {base_column}"
        )
        return False
    config[header][base_column] = json.dumps(column_list)
    save_config(config)
    return True


def save_config(config):
    """
    Save config.ini.
    """
    # Write to .ini file. Will overwrite or make a new one.
    with open(_CONFIG, "w+") as configfile:
        config.write(configfile)


@check_config
def parse_column(header, base_column, columns):
    """
    Find a given base_column in given columns.

    Tries to match it to identifiers in config.ini.

    E.g.

    >>> parse_column(
    ...     "BOREHOLE",
    ...     _BOREHOLE_TREND,
    ...     ["borehole_trend", "alpha", "beta", "borehole_plunge"],
    ... )
    'borehole_trend'

    :param header: "MEASUREMENTS", "DEPTHS" or "BOREHOLE"
    :type header: str
    :param base_column: The base measurement type to identify. (E.g. "alpha", "beta")
    :type base_column: str
    :param columns: Columns from given data file.
    :type columns: list
    :return: Column name in your data file that matches the given base_column
    :rtype: str
    :raises ColumnException: When there are problems with identifying columns.
    """
    config = configparser.ConfigParser()
    assert _CONFIG.exists()
    config.read(_CONFIG)
    column_identifiers = json.loads(config.get(header, base_column))
    assert isinstance(column_identifiers, list)
    matching_columns = list(set(column_identifiers) & set(columns))

    # Check for errors
    if len(matching_columns) == 0:
        raise ColumnException(
            "{base_column} of {header} was not recognized in columns of given file. \n"
            f"Columns:{columns}\n"
            f"Column identifiers in config.ini: {column_identifiers}\n"
            "You must add it to config.ini as an identifier for recognition. \n"
            f"{Path('config.ini').absolute()}\n"
        )
    if len(matching_columns) > 1:
        raise ColumnException(
            f"Multiple {base_column} type column names were found in identifiers. \n"
            "Check config.ini file for identical identifiers. \n"
            f"{Path('config.ini').absolute()}\n"
            "(E.g. alpha_measurement is in both ALPHA and BETA)\n"
        )

    # Column in config.ini & given columns that matches given base_column
    return matching_columns[0]


@check_config
def parse_columns_two_files(columns, with_gamma):
    """
    Match columns to column bases in config.ini.

    Used when there's a separate file with depth data.

    E.g.

    >>> from pprint import pprint
    >>> result = parse_columns_two_files(
    ...     [
    ...         "alpha",
    ...         "beta",
    ...         "gamma",
    ...         "borehole_trend",
    ...         "borehole_plunge",
    ...         "depth",
    ...         "measurement_depth",
    ...     ],
    ...     True,
    ... )
    >>> pprint(result)
    {'alpha': 'alpha',
     'beta': 'beta',
     'borehole_plunge': 'borehole_plunge',
     'borehole_trend': 'borehole_trend',
     'depth': 'depth',
     'gamma': 'gamma',
     'measurement_depth': 'measurement_depth'}

    :param columns: Given columns
    :type columns: list
    :param with_gamma: Whether there are gamma measurements in file or not.
    :type with_gamma: bool
    :return: Matched columns as a dictionary.
    :rtype: dict
    """
    if len(columns) < 4:
        raise ColumnException("Invalid, too short, columns list:\n" f"{columns}")
    # depths file
    find_columns_d = [_DEPTH]
    find_columns_bh = [_BOREHOLE_TREND, _BOREHOLE_PLUNGE]
    # measurements file
    if with_gamma:
        find_columns_m = [_ALPHA, _BETA, _GAMMA, _MEASUREMENT_DEPTH]
    else:
        find_columns_m = [_ALPHA, _BETA, _MEASUREMENT_DEPTH]

    matched_dict = dict()
    for f in find_columns_d:
        col = parse_column(_DEPTHS, f, columns)
        matched_dict[f] = col
    for f in find_columns_bh:
        col = parse_column(_BOREHOLE, f, columns)
        matched_dict[f] = col
    for f in find_columns_m:
        col = parse_column(_MEASUREMENTS, f, columns)
        matched_dict[f] = col
    if with_gamma:
        if len(matched_dict) != 7:
            raise ColumnException(
                f"Invalid column dictionary length.\n"
                f"with_gamma == {with_gamma}\n"
                f"dict:\n"
                f"{matched_dict}"
            )
    else:
        if len(matched_dict) != 6:
            raise ColumnException(
                f"Invalid column dictionary length.\n"
                f"with_gamma == {with_gamma}\n"
                f"dict:\n"
                f"{matched_dict}"
            )
    return matched_dict


@check_config
def parse_columns_one_file(columns, with_gamma):
    """
    Match columns to column bases in config.ini.

    Used when there is only one data file with all required data. I.e. at
    minimum: alpha, beta, borehole trend, borehole plunge

    If gamma data exists => with_gamma should be given as True

    E.g.

    >>> from pprint import pprint
    >>> result = parse_columns_one_file(
    ...     ["alpha", "beta", "borehole_trend", "borehole_plunge"], False
    ... )
    >>> pprint(result)
    {'alpha': 'alpha',
     'beta': 'beta',
     'borehole_plunge': 'borehole_plunge',
     'borehole_trend': 'borehole_trend'}

    :param columns: Given columns
    :type columns: list
    :param with_gamma: Whether there are gamma measurements in file or not.
    :type with_gamma: bool
    :return: Matched columns as a dictionary.
    :rtype: dict
    """
    find_columns_bh = [_BOREHOLE_TREND, _BOREHOLE_PLUNGE]
    # measurements file
    if with_gamma:
        find_columns_m = [_ALPHA, _BETA, _GAMMA]
    else:
        find_columns_m = [_ALPHA, _BETA]

    matched_dict = dict()
    for f in find_columns_bh:
        col = parse_column(_BOREHOLE, f, columns)
        matched_dict[f] = col
    for f in find_columns_m:
        col = parse_column(_MEASUREMENTS, f, columns)
        matched_dict[f] = col

    # Check that all are found.
    # Exceptions should be raised in parse_column if not successfully found
    # i.e. this in only a backup/debug.
    if with_gamma:
        assert len(matched_dict) == 5
    else:
        assert len(matched_dict) == 4

    return matched_dict


def round_outputs(number):
    """
    Round number.
    """
    return round(number, 2)


def apply_conventions(df, col_dict):
    """
    Apply conventions from the config.ini file to given DataFrame.

    col_dict is used to identify the correct columns.

    Recognized conventions are: "negative" "none"

    :param df: DataFrame
    :type df: pandas.DataFrame
    :param col_dict: Dictionary with identifiers for measurement data.
    :type col_dict: dict
    :return: DataFrame with applied conventions in new columns.
            Modified col_dict.
    :rtype: tuple[pandas.DataFrame, dict]
    """
    config = configparser.ConfigParser()
    config.read(_CONFIG)
    alpha_convention = _ALPHA, config[_CONVENTIONS][_ALPHA]
    beta_convention = _BETA, config[_CONVENTIONS][_BETA]
    trend_convention = _BOREHOLE_TREND, config[_CONVENTIONS][_BOREHOLE_TREND]
    plunge_convention = _BOREHOLE_PLUNGE, config[_CONVENTIONS][_BOREHOLE_PLUNGE]
    gamma_convention = _GAMMA, "none"
    if _GAMMA in col_dict:
        gamma_convention = _GAMMA, config[_CONVENTIONS][_GAMMA]

    for conv in (
        alpha_convention,
        beta_convention,
        trend_convention,
        plunge_convention,
        gamma_convention,
    ):
        if conv[1] == "negative":
            new_column = f"{col_dict[conv[0]]}_negative"
            df[new_column] = -df[col_dict[conv[0]]]
            col_dict[conv[0]] = new_column

    return df, col_dict


def apply_conventions_manual(df, col_dict, convention_dict):
    """
    Apply conventions from manual input to given DataFrame.

    Creates new columns with conventioned data.  col_dict is used to identify
    the correct columns.

    Recognized conventions are: "negative" "none"

    :param df: DataFrame
    :type df: pandas.DataFrame
    :param col_dict: Dictionary with identifiers for measurement data.
    :type col_dict: dict
    :param convention_dict: Dictionary with conventions
    :type convention_dict: dict
    :return: DataFrame with applied conventions in new columns.
    :rtype: pandas.DataFrame
    """
    if _GAMMA in col_dict:
        pass

    for conv in convention_dict.items():
        if conv[1] == "negative":
            # Check if column with negative data already exists.
            if "negative" in f"{col_dict[conv[0]]}":
                # Convention has already been applied to column.
                return df, col_dict
            # Conventioned column doesn't exist.
            new_column = f"{col_dict[conv[0]]}_negative"
            df[new_column] = -df[col_dict[conv[0]]]
            col_dict[conv[0]] = new_column
        elif conv[1] == "none":

            # TODO: Do it better.
            if "negative" in f"{col_dict[conv[0]]}":
                new_column = f"{col_dict[conv[0]]}".replace("_negative", "")
            else:
                new_column = f"{col_dict[conv[0]]}"
            col_dict[conv[0]] = new_column

        else:
            print(f"weird conv: {conv}")
    return df, col_dict


def transform_csv(filename, with_gamma=False, output=None):
    """
    Transform data from a given .csv file.

    File must have columns with alpha and beta measurements and borehole trend
    and plunge.

    Saves new .csv file in the given output path.

    :param filename: Path to file for reading.
    :type filename: str
    :param output: Path for output file. Will default filename+_transformed.csv.
    :type output: str
    :param with_gamma: Do gamma calculations or not
    :type with_gamma: bool
    """
    measurements = pd.read_csv(filename, sep=";")
    col_dict = parse_columns_one_file(measurements.columns.tolist(), with_gamma)
    # Check and apply conventions
    measurements, col_dict = apply_conventions(measurements, col_dict)

    # Creates and calculates new columns
    if with_gamma:
        measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ] = measurements.apply(
            lambda row: pd.Series(
                transform_with_gamma(
                    row[col_dict[_ALPHA]],
                    row[col_dict[_BETA]],
                    row[col_dict[_BOREHOLE_TREND]],
                    row[col_dict[_BOREHOLE_PLUNGE]],
                    row[col_dict[_GAMMA]],
                )
            ),
            axis=1,
        )
        measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ] = measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ].applymap(
            round_outputs
        )
    else:
        measurements[["plane_dip", "plane_dir"]] = measurements.apply(
            lambda row: pd.Series(
                transform_without_gamma(
                    row[col_dict[_ALPHA]],
                    row[col_dict[_BETA]],
                    row[col_dict[_BOREHOLE_TREND]],
                    row[col_dict[_BOREHOLE_PLUNGE]],
                )
            ),
            axis=1,
        )
        measurements[["plane_dip", "plane_dir"]] = measurements[
            ["plane_dip", "plane_dir"]
        ].applymap(round_outputs)

    # Savename
    if output is not None:
        savepath = Path(output)
    else:
        savename = Path(filename).stem.split(".")[0] + "_transformed.csv"
        savedir = str(Path(filename).parent)
        savepath = Path(savedir + "/" + savename)
    # Save new .csv. Overwrites old and creates new if needed.
    measurements.to_csv(savepath, sep=";", mode="w+")


def transform_excel(measurement_filename, with_gamma, output=None):
    """
    Transform excel file.
    """
    measurements = pd.read_excel(measurement_filename)
    col_dict = parse_columns_two_files(measurements.columns.tolist(), with_gamma)
    # Check and apply conventions
    measurements, col_dict = apply_conventions(measurements, col_dict)

    if with_gamma:
        measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ] = measurements.apply(
            lambda row: pd.Series(
                transform_with_gamma(
                    row[col_dict[_ALPHA]],
                    row[col_dict[_BETA]],
                    row[col_dict[_BOREHOLE_TREND]],
                    row[col_dict[_BOREHOLE_PLUNGE]],
                    row[col_dict[_GAMMA]],
                )
            ),
            axis=1,
        )
        measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ] = measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ].applymap(
            round_outputs
        )
    else:
        # ALPHA must be reversed to achieve correct result.
        measurements[["plane_dip", "plane_dir"]] = measurements.apply(
            lambda row: pd.Series(
                transform_without_gamma(
                    row[col_dict[_ALPHA]],
                    row[col_dict[_BETA]],
                    row[col_dict[_BOREHOLE_TREND]],
                    row[col_dict[_BOREHOLE_PLUNGE]],
                )
            ),
            axis=1,
        )
        measurements[["plane_dip", "plane_dir"]] = measurements[
            ["plane_dip", "plane_dir"]
        ].applymap(round_outputs)

    # Savename
    if output is not None:
        savepath = Path(output)
    else:
        savename = Path(measurement_filename).stem.split(".")[0] + "_transformed.csv"
        savedir = str(Path(measurement_filename).parent)
        savepath = Path(savedir + "/" + savename)
    # Save new .csv. Overwrites old and creates new if needed.
    measurements.to_csv(savepath, sep=";", mode="w+")


def transform_csv_two_files(
    measurement_filename, depth_filename, with_gamma, output=None
):
    """
    Transform with two csv files.
    """
    measurements = pd.read_csv(measurement_filename, sep=";")
    depth = pd.read_csv(depth_filename, sep=";")
    assert isinstance(depth, pd.DataFrame)
    col_dict = parse_columns_two_files(
        measurements.columns.tolist() + depth.columns.tolist(), with_gamma
    )
    # Check and apply conventions
    measurements, col_dict = apply_conventions(measurements, col_dict)

    trend_plunge = []
    for _, row in measurements.iterrows():
        val = row[col_dict[_MEASUREMENT_DEPTH]]
        right = bisect.bisect(depth[col_dict[_DEPTH]].values, val)
        if right == len(depth):
            right = right - 1
        # Check if index is -1 in which case right and left both work.
        # Depth must be ordered!
        left = right - 1 if right - 1 != -1 else right

        # Check which is closer, left or right to value.
        take = (
            right
            if depth[col_dict[_DEPTH]].iloc[right] - val
            <= val - depth[col_dict[_DEPTH]].iloc[left]
            else left
        )
        trend, plunge = depth[
            [col_dict[_BOREHOLE_TREND], col_dict[_BOREHOLE_PLUNGE]]
        ].iloc[take]
        plunge = -plunge
        trend_plunge.append((trend, plunge))

    measurements["borehole_trend"], measurements["borehole_plunge"] = [
        tr[0] for tr in trend_plunge
    ], [tr[1] for tr in trend_plunge]
    # dict must be updated with new fields in measurements file.
    col_dict[_BOREHOLE_TREND], col_dict[_BOREHOLE_PLUNGE] = (
        "borehole_trend",
        "borehole_plunge",
    )

    # ALPHA must be reversed to achieve correct result.
    measurements[["plane_dip", "plane_dir"]] = measurements.apply(
        lambda row: pd.Series(
            transform_without_gamma(
                row[col_dict[_ALPHA]],
                row[col_dict[_BETA]],
                row[col_dict[_BOREHOLE_TREND]],
                row[col_dict[_BOREHOLE_PLUNGE]],
            )
        ),
        axis=1,
    )
    measurements[["plane_dip", "plane_dir"]] = measurements[
        ["plane_dip", "plane_dir"]
    ].applymap(round_outputs)

    # Savename
    if output is not None:
        savepath = Path(output)
    else:
        savename = Path(measurement_filename).stem.split(".")[0] + "_transformed.csv"
        savedir = str(Path(measurement_filename).parent)
        savepath = Path(savedir + "/" + savename)
    # Save new .csv. Overwrites old and creates new if needed.
    measurements.to_csv(savepath, sep=";", mode="w+")


def transform_excel_two_files(measurement_filename, depth_filename, with_gamma, output):
    """
    Transform with two excel files.
    """
    measurements = pd.read_excel(measurement_filename)
    depth = pd.read_excel(depth_filename)
    col_dict = parse_columns_two_files(
        measurements.columns.tolist() + depth.columns.tolist(), with_gamma
    )

    # Check and apply conventions
    measurements, col_dict = apply_conventions(measurements, col_dict)

    trend_plunge = []
    for _, row in measurements.iterrows():
        val = row[col_dict[_MEASUREMENT_DEPTH]]
        right = bisect.bisect(depth[col_dict[_DEPTH]].values, val)
        if right == len(depth):
            right = right - 1
        # Check if index is -1 in which case right and left both work.
        # Depth must be ordered!
        left = right - 1 if right - 1 != -1 else right

        # Check which is closer, left or right to value.
        take = (
            right
            if depth[col_dict[_DEPTH]].iloc[right] - val
            <= val - depth[col_dict[_DEPTH]].iloc[left]
            else left
        )
        trend, plunge = depth[
            [col_dict[_BOREHOLE_TREND], col_dict[_BOREHOLE_PLUNGE]]
        ].iloc[take]
        plunge = -plunge
        trend_plunge.append((trend, plunge))

    measurements["borehole_trend"], measurements["borehole_plunge"] = [
        tr[0] for tr in trend_plunge
    ], [tr[1] for tr in trend_plunge]
    # dict must be updated with new fields in measurements file.
    col_dict[_BOREHOLE_TREND], col_dict[_BOREHOLE_PLUNGE] = (
        "borehole_trend",
        "borehole_plunge",
    )

    if with_gamma:
        measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ] = measurements.apply(
            lambda row: pd.Series(
                transform_with_gamma(
                    row[col_dict[_ALPHA]],
                    row[col_dict[_BETA]],
                    row[col_dict[_BOREHOLE_TREND]],
                    row[col_dict[_BOREHOLE_PLUNGE]],
                    row[col_dict[_GAMMA]],
                )
            ),
            axis=1,
        )
        measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ] = measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ].applymap(
            round_outputs
        )
    else:
        # ALPHA must be reversed to achieve correct result.
        measurements[["plane_dip", "plane_dir"]] = measurements.apply(
            lambda row: pd.Series(
                transform_without_gamma(
                    row[col_dict[_ALPHA]],
                    row[col_dict[_BETA]],
                    row[col_dict[_BOREHOLE_TREND]],
                    row[col_dict[_BOREHOLE_PLUNGE]],
                )
            ),
            axis=1,
        )
        measurements[["plane_dip", "plane_dir"]] = measurements[
            ["plane_dip", "plane_dir"]
        ].applymap(round_outputs)

    # Savename
    savedir = Path(measurement_filename).parent
    if output is not None:
        savepath = Path(output)
    else:
        savename = Path(measurement_filename).stem.split(".")[0] + "_transformed.csv"
        savepath = savedir / savename
    # Save new .csv. Overwrites old and creates new if needed.
    measurements.to_csv(savepath, sep=";", mode="w+")


def transform_with_two_files(
    measurements: pd.DataFrame, depth: pd.DataFrame, with_gamma, output=None
):
    """
    Transform with two loaded DataFrames.
    """
    col_dict = parse_columns_two_files(
        measurements.columns.tolist() + depth.columns.tolist(), with_gamma
    )

    # Check and apply conventions
    measurements, col_dict = apply_conventions(measurements, col_dict)

    trend_plunge = []
    for _, row in measurements.iterrows():
        val = row[col_dict[_MEASUREMENT_DEPTH]]
        right = bisect.bisect(depth[col_dict[_DEPTH]].values, val)
        if right == len(depth):
            right = right - 1
        # Check if index is -1 in which case right and left both work.
        # Depth must be ordered!
        left = right - 1 if right - 1 != -1 else right

        # Check which is closer, left or right to value.
        take = (
            right
            if depth[col_dict[_DEPTH]].iloc[right] - val
            <= val - depth[col_dict[_DEPTH]].iloc[left]
            else left
        )
        trend, plunge = depth[
            [col_dict[_BOREHOLE_TREND], col_dict[_BOREHOLE_PLUNGE]]
        ].iloc[take]
        plunge = -plunge
        trend_plunge.append((trend, plunge))

    measurements["borehole_trend"], measurements["borehole_plunge"] = [
        tr[0] for tr in trend_plunge
    ], [tr[1] for tr in trend_plunge]
    # dict must be updated with new fields in measurements file.
    col_dict[_BOREHOLE_TREND], col_dict[_BOREHOLE_PLUNGE] = (
        "borehole_trend",
        "borehole_plunge",
    )

    if with_gamma:
        measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ] = measurements.apply(
            lambda row: pd.Series(
                transform_with_gamma(
                    row[col_dict[_ALPHA]],
                    row[col_dict[_BETA]],
                    row[col_dict[_BOREHOLE_TREND]],
                    row[col_dict[_BOREHOLE_PLUNGE]],
                    row[col_dict[_GAMMA]],
                )
            ),
            axis=1,
        )
        measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ] = measurements[
            ["plane_dip", "plane_dir", "gamma_plunge", "gamma_trend"]
        ].applymap(
            round_outputs
        )
    else:
        # ALPHA must be reversed to achieve correct result.
        measurements[["plane_dip", "plane_dir"]] = measurements.apply(
            lambda row: pd.Series(
                transform_without_gamma(
                    row[col_dict[_ALPHA]],
                    row[col_dict[_BETA]],
                    row[col_dict[_BOREHOLE_TREND]],
                    row[col_dict[_BOREHOLE_PLUNGE]],
                )
            ),
            axis=1,
        )
        measurements[["plane_dip", "plane_dir"]] = measurements[
            ["plane_dip", "plane_dir"]
        ].applymap(round_outputs)

    # Savename
    savepath = Path(output)
    # Save new .csv. Overwrites old and creates new if needed.
    measurements.to_csv(savepath, sep=";", mode="w+")


def convention_testing_csv(
    filename, with_gamma=False, output=None, visualize=False, img_dir=None
):
    """
    Test multitudes of conventions on data in a .csv file.

    TODO: Currently a crude, alpha, method, desperately needs improving and specifying.

    :param filename: .csv file with data
    :type filename: str
    :param with_gamma: If data contains gamma measurements
    :type with_gamma: bool
    :param output: output file with tested conventions
    :type output: str
    :param visualize: Whether to visualize each calculation
        (not worth it unless only 1 row of data)
    :type visualize: bool
    :param img_dir: Directory to save visualization images
    :type img_dir: str
    """
    measurements = pd.read_csv(filename, sep=";")
    col_dict = parse_columns_one_file(
        measurements.columns.tolist(), with_gamma=with_gamma
    )
    borehole_trend_convention = "none"
    for convention in _MEASUREMENT_CONVENTIONS:
        alpha_convention = convention
        for convention in _MEASUREMENT_CONVENTIONS:
            beta_convention = convention
            for convention in _MEASUREMENT_CONVENTIONS:
                borehole_plunge_convention = convention
                for idx, convention in enumerate(_MEASUREMENT_CONVENTIONS):
                    if with_gamma:
                        gamma_convention = convention
                    else:
                        gamma_convention = "none"
                        if idx != 0:
                            break
                    convention_dict = dict()
                    convention_dict[_ALPHA] = alpha_convention
                    convention_dict[_BETA] = beta_convention
                    convention_dict[_BOREHOLE_TREND] = borehole_trend_convention
                    convention_dict[_BOREHOLE_PLUNGE] = borehole_plunge_convention
                    if with_gamma:
                        convention_dict[_GAMMA] = gamma_convention

                    measurements, col_dict = apply_conventions_manual(
                        measurements, col_dict, convention_dict
                    )

                    # Current convention setup as a string
                    conventions = [
                        alpha_convention,
                        beta_convention,
                        borehole_trend_convention,
                        borehole_plunge_convention,
                        gamma_convention,
                    ]
                    curr = (
                        "|".join(conventions)
                        .replace("negative", "-")
                        .replace("none", "0")
                    )

                    # Create and calculate new columns

                    # Transformed measured plane dip
                    plane_dip_curr = f"plane_dip_{curr}"
                    # Transformed measured plane dir
                    plane_dir_curr = f"plane_dir_{curr}"

                    if with_gamma:

                        # Transformed measured linear feature plunge
                        gamma_plunge_curr = f"gamma_plunge_{curr}"
                        # Transformed measured linear feature trend
                        gamma_trend_curr = f"gamma_trend_{curr}"

                        measurements[
                            [
                                plane_dip_curr,
                                plane_dir_curr,
                                gamma_plunge_curr,
                                gamma_trend_curr,
                            ]
                        ] = measurements.apply(
                            lambda row: pd.Series(
                                transform_with_gamma(
                                    row[col_dict[_ALPHA]],
                                    row[col_dict[_BETA]],
                                    row[col_dict[_BOREHOLE_TREND]],
                                    row[col_dict[_BOREHOLE_PLUNGE]],
                                    row[col_dict[_GAMMA]],
                                    visualize,
                                    img_dir,
                                    curr,
                                )
                            ),
                            axis=1,
                        )
                        measurements[
                            [
                                plane_dip_curr,
                                plane_dir_curr,
                                gamma_plunge_curr,
                                gamma_trend_curr,
                            ]
                        ] = measurements[
                            [
                                plane_dip_curr,
                                plane_dir_curr,
                                gamma_plunge_curr,
                                gamma_trend_curr,
                            ]
                        ].applymap(
                            round_outputs
                        )

                    else:
                        measurements[
                            [plane_dip_curr, plane_dir_curr]
                        ] = measurements.apply(
                            lambda row: pd.Series(
                                transform_without_gamma(
                                    row[col_dict[_ALPHA]],
                                    row[col_dict[_BETA]],
                                    row[col_dict[_BOREHOLE_TREND]],
                                    row[col_dict[_BOREHOLE_PLUNGE]],
                                )
                            ),
                            axis=1,
                        )
                        measurements[[plane_dip_curr, plane_dir_curr]] = measurements[
                            [plane_dip_curr, plane_dir_curr]
                        ].applymap(round_outputs)
    # Savename
    if output is not None:
        if Path(output).is_absolute():
            savepath = Path(output)
        else:
            savepath = Path(filename).parent / Path(output)
    else:
        savename = Path(filename).stem.split(".")[0] + "_convention_test.csv"
        savedir = str(Path(filename).parent)
        savepath = Path(savedir + "/" + savename)
    # Save new .csv. Overwrites old and creates new if needed.
    measurements.to_csv(savepath, sep=";", mode="w+")


def change_conventions(convention_dict):
    """
    Change config.ini conventions by passing a dictionary.

    :param convention_dict: Dictionary with new conventions.
    :type convention_dict: dict
    :return: None if successful and False if key was not recognized or
        convention was not recognized or dictionary was invalid.
    :rtype: None | bool
    """
    if len(convention_dict) == 0:
        return False
    if len(convention_dict) > 7 or len(convention_dict.keys()) != len(
        set(convention_dict.keys())
    ):
        print(
            "Given dictionary is too long / contains duplicate keys."
            "No changes were made to config.ini"
        )
        return False
    if False in [isinstance(v, str) for v in convention_dict.values()]:
        print("Invalid values is dictionary. No changes were made to config.ini")
        return False
    config = configparser.ConfigParser()
    config.read(_CONFIG)

    any_changes = False
    for key, val in convention_dict.items():
        if len(key) == 0 or len(val) == 0:
            continue

        if key in (_ALPHA, _BETA, _GAMMA):
            if val in _MEASUREMENT_CONVENTIONS:
                config[_MEASUREMENTS][key] = val
                any_changes = True
            else:
                print(f"Given convention: {val} was not recognized as a convention.")
                continue
        elif key in (_BOREHOLE_TREND, _BOREHOLE_PLUNGE):
            if val in _MEASUREMENT_CONVENTIONS:
                config[_BOREHOLE][key] = val
                any_changes = True
            else:
                print(f"Given convention: {val} was not recognized as a convention.")
                continue
        elif key in (_MEASUREMENT_DEPTH, _DEPTH):
            if val in _MEASUREMENT_CONVENTIONS:
                config[_DEPTHS][key] = val
                any_changes = True
            else:
                print(f"Given convention: {val} was not recognized as a convention.")
                continue
        else:
            print(f"Given key: {key} was not recognized to belong under any header.")
            continue

    save_config(config)
    return any_changes
