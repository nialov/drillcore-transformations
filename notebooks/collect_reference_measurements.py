import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from drillcore_transformations.tests.test_convention import measurement_validations_lope_202510, measurement_validations_synt_2025
    from drillcore_transformations.transformations import transform
    import pandas as pd
    from itertools import starmap
    return (
        measurement_validations_lope_202510,
        measurement_validations_synt_2025,
        pd,
        starmap,
        transform,
    )


@app.cell
def _(
    measurement_validations_lope_202510,
    measurement_validations_synt_2025,
    pd,
):
    df = pd.DataFrame([*measurement_validations_lope_202510, *measurement_validations_synt_2025])
    measurement_df = df['measurement'].apply(lambda m: pd.Series(m._asdict()))
    df = pd.concat([measurement_df, df.drop(columns=['measurement'])], axis=1)
    df
    return (df,)


@app.cell
def _(df, pd, starmap, transform):
    args_iter = (tuple(row) for row in df[['alpha', 'beta', 'drillcore_trend', 'drillcore_plunge', 'gamma']].to_numpy())
    results = list(starmap(transform, args_iter))
    calculated = pd.DataFrame(
        results,
        columns=['calculated_dip', 'calculated_dip_dir', 'calculated_plunge', 'calculated_trend'],
        index=df.index
    )
    df_final = df.join(calculated)

    df_final
    return


if __name__ == "__main__":
    app.run()
