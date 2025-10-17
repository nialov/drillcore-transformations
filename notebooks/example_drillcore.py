import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from drillcore_transformations.transformations import transform
    from pathlib import Path
    import bisect
    from functools import partial
    from itertools import starmap
    return Path, bisect, partial, pd, starmap, transform


@app.cell
def _(Path):
    MEASUREMENTS_PATH = Path("src/drillcore_transformations/tests/sample_data/measurement_sample.csv")
    DEPTHS_PATH = Path("src/drillcore_transformations/tests/sample_data/depth_sample.csv")
    return DEPTHS_PATH, MEASUREMENTS_PATH


@app.cell
def _(DEPTHS_PATH, MEASUREMENTS_PATH, pd):
    measurements = pd.read_csv(MEASUREMENTS_PATH, sep=";")
    depths = pd.read_csv(DEPTHS_PATH, sep=";")
    return depths, measurements


@app.cell
def _(measurements):
    measurements
    return


@app.cell
def _(depths):
    depths
    return


@app.cell
def _(depths):
    depths.plot(x="DEPTH", y="DIP")
    return


@app.cell
def _(bisect, pd):
    def resolve_drillcore_trend_plunge(depth:float, depths:pd.DataFrame, depth_column="DEPTH", trend_column:str = "AZIMUTH", plunge_column:str = "DIP")-> tuple[float, float]:
        # Takes the value to the "left". I.e. if depths are sorted ascending, it takes the orientation value of the lower depth
        # E.g. at depth of 42.6 m, with orientation defined at 40 m and 45 m, orientation at 40 m will be used
        bisect_index = bisect.bisect_left(depths[depth_column], depth) - 1
        trend = depths[trend_column].iloc[bisect_index]
        plunge = depths[plunge_column].iloc[bisect_index]
        return trend, plunge
    return (resolve_drillcore_trend_plunge,)


@app.cell
def _(depths, measurements, partial, resolve_drillcore_trend_plunge):
    drillcore_trend, drillcore_plunge = zip(*measurements["LENGTH_FROM"].apply(partial(resolve_drillcore_trend_plunge, depths=depths)).to_numpy())
    return drillcore_plunge, drillcore_trend


@app.cell
def _(drillcore_plunge, drillcore_trend, measurements):
    measurements["drillcore_trend"] = drillcore_trend
    measurements["drillcore_plunge"] = drillcore_plunge
    return


@app.cell
def _(measurements):
    measurements
    return


@app.cell
def _(measurements, starmap, transform):
    dips, directions, _, _ = zip(*starmap(transform, measurements[["ALPHA_CORE", "BETA_CORE", "drillcore_trend", "drillcore_plunge"]].to_numpy()))
    return dips, directions


@app.cell
def _(dips, directions, measurements):
    measurements["dip"] = dips
    measurements["direction"] = directions
    return


@app.cell
def _(measurements):
    measurements[["ALPHA_CORE", "BETA_CORE", "dip", "direction", "drillcore_trend", "drillcore_plunge"]]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
