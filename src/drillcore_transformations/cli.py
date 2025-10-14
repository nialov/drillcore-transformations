"""
Console script for drillcore_transformations_code.
"""

import click

from drillcore_transformations import usage

base_measurements, headers, conf = usage.get_config_identifiers()
(
    _ALPHA,
    _BETA,
    _GAMMA,
    _MEASUREMENT_DEPTH,
    _DEPTH,
    _BOREHOLE_TREND,
    _BOREHOLE_PLUNGE,
) = base_measurements
_MEASUREMENTS, _DEPTHS, _BOREHOLE, _CONVENTIONS = headers
_CONFIG = conf[0]


@click.group()
def cli():
    """
    Transform drillcore structural measurements.
    """


@cli.command()
@click.argument(
    "measurementfile",
    type=click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True),
    nargs=1,
)
@click.option(
    "--depthfile",
    "-d",
    type=click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True),
    help="Separate file that contains depth measurements. Optional.",
)
@click.option(
    "--outputfile",
    "-o",
    type=click.Path(writable=True, dir_okay=False, resolve_path=True),
    help="Output file location and name (Optional).",
)
@click.option("--gamma", is_flag=True, help="Add if measurements contain gamma.")
def transform(measurementfile, depthfile=None, outputfile=None, gamma=False):
    """
    Transform measurement file with or without a depth file.

    """
    assert isinstance(measurementfile, str)
    click.echo(click.style("Starting transform.", fg="yellow"))
    if depthfile is not None:
        # Two files.
        if ".csv" in measurementfile:
            # csv. files
            click.echo(
                click.style(
                    "Transforming with csv measurement and depth files.", fg="cyan"
                )
            )
            usage.transform_csv_two_files(measurementfile, depthfile, gamma, outputfile)
        elif ".xls" in measurementfile:
            # excel files
            click.echo(
                click.style(
                    "Transforming with excel measurement and depth files.", fg="cyan"
                )
            )
            usage.transform_excel_two_files(
                measurementfile, depthfile, gamma, outputfile
            )
    else:
        if ".csv" in measurementfile:
            # csv. file
            click.echo(
                click.style("Transforming with csv measurement file.", fg="cyan")
            )
            usage.transform_csv(measurementfile, gamma, outputfile)
        elif ".xls" in measurementfile:
            # excel file
            click.echo(
                click.style("Transforming with excel measurement file.", fg="cyan")
            )
            usage.transform_excel(measurementfile, outputfile, gamma)
    # click.echo("Transform complete.")
    click.echo(click.style("Transform complete.", fg="green"))
    return 0


@cli.command()
@click.argument("header", type=click.Choice([_MEASUREMENTS, _DEPTHS, _BOREHOLE]))
@click.argument("base_column", type=click.Choice(base_measurements))
@click.argument("name", type=click.STRING)
@click.option(
    "--remove",
    is_flag=True,
    help="If given, removes inputted column name from identifiers.",
)
def columnname(header, base_column, name, remove=False):
    """
    Add/remove column identifiers in config.ini.

    :param header: Header within config.ini
    :type header: str
    :param base_column: Measurement type under which you want to add an identifier.
    :type base_column: str
    :param name: New column to add
    :type name: str
    """
    for i in (header, base_column, name):
        assert isinstance(i, str)
    if remove:
        usage.remove_column_name(header, base_column, name)
    else:
        usage.add_column_name(header, base_column, name)


@cli.command()
@click.option(
    "--initialize",
    "-i",
    is_flag=True,
    help="WARNING: Initializes a new, default, config.ini when run.",
)
def config(initialize=False):
    """
    Config path printer and initializing new config.ini.
    """
    if initialize:
        click.echo(click.style("Initializing new config.ini", fg="yellow"))
        usage.initialize_config()
        click.echo(click.style("New config initialized.", fg="green"))
    c = usage.find_config()
    click.echo(click.style(f"Config File Path: \n{c}", fg="green"))


@cli.command()
def conventions():
    """
    Echoes information about config.ini editing.
    """
    click.echo(
        click.style(
            "Changing conventions through command line not supported.", fg="red"
        )
    )
    click.echo(
        click.style(
            "Instead, you should edit config.ini through the method"
            " change_conventions in usage.py "
            "or manually with a text editor.",
            fg="yellow",
        )
    )
    c = usage.find_config()
    click.echo(click.style(f"Config File Path: \n{c}", fg="green"))
