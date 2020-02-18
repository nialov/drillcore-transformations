"""Console script for borehole_transformations."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for borehole_transformations."""
    click.echo("Replace this message by putting your code into "
               "borehole_transformations.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
