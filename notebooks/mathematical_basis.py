import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sympy as sp
    return mo, sp


@app.cell
def _(mo):
    mo.md('''
    # Mathematical Basis of Drillcore Measurement Transformation

    This notebook introduces the mathematical steps behind the `transform` function,
    which converts drillcore measurements (alpha, beta, trend, plunge, gamma)
    to real-world structural orientations.

    We use symbolic computation (`sympy`) to illustrate each step.
    ''')
    return


@app.cell
def _(mo):
    mo.md("## Define symbols for all input parameters")
    return


@app.cell
def _(sp):
    alpha, beta, trend, plunge, gamma = sp.symbols('alpha beta trend plunge gamma')
    return alpha, beta, gamma, plunge, trend


@app.cell
def _(mo):
    mo.md("## Degrees to radians conversion")
    return


@app.cell
def _(alpha, beta, plunge, sp, trend):

    deg2rad = lambda x: x * sp.pi / 180
    alpha_rad = deg2rad(alpha)
    beta_rad = deg2rad(beta)
    trend_rad = deg2rad(trend)
    plunge_rad = deg2rad(plunge)
    return alpha_rad, beta_rad, plunge_rad, trend_rad


@app.cell
def _(mo):
    mo.md("Symbolic equations for the normal vector components")
    return


@app.cell
def _(alpha_rad, beta_rad, plunge_rad, sp, trend_rad):
    ng_1 = (
        sp.cos(sp.pi / 2 - trend_rad)
        * sp.cos(sp.pi / 2 - plunge_rad)
        * sp.cos(beta_rad)
        * sp.cos(alpha_rad)
        - sp.sin(sp.pi / 2 - trend_rad) * sp.sin(beta_rad) * sp.cos(alpha_rad)
        + sp.cos(sp.pi / 2 - trend_rad) * sp.sin(sp.pi / 2 - plunge_rad) * sp.sin(alpha_rad)
    )
    ng_2 = (
        sp.sin(sp.pi / 2 - trend_rad)
        * sp.cos(sp.pi / 2 - plunge_rad)
        * sp.cos(beta_rad)
        * sp.cos(alpha_rad)
        + sp.cos(sp.pi / 2 - trend_rad) * sp.sin(beta_rad) * sp.cos(alpha_rad)
        + sp.sin(sp.pi / 2 - trend_rad) * sp.sin(sp.pi / 2 - plunge_rad) * sp.sin(alpha_rad)
    )
    ng_3 = -sp.sin(sp.pi / 2 - plunge_rad) * sp.cos(beta_rad) * sp.cos(alpha_rad) + sp.cos(
        sp.pi / 2 - plunge_rad
    ) * sp.sin(alpha_rad)
    normal_vec = sp.Matrix([ng_1, ng_2, ng_3])
    return (normal_vec,)


@app.cell
def _(mo):
    mo.md("Normalization of the normal vector")
    return


@app.cell
def _(normal_vec, sp):
    norm = sp.sqrt(sum([comp**2 for comp in normal_vec]))
    normal_vec_normalized = normal_vec / norm
    normal_vec_normalized_up = sp.Piecewise(
        (normal_vec_normalized, normal_vec[2] >= 0),
        (-normal_vec_normalized, normal_vec[2] < 0)
    )
    normal_vec_normalized_up
    return (normal_vec_normalized_up,)


@app.cell
def _(mo):
    mo.md("Plane dip and direction from the normal vector")
    return


@app.cell
def _(normal_vec_normalized_up, sp):
    n = normal_vec_normalized_up
    # Extract the third component as a Piecewise
    n2 = sp.Piecewise(
        (n.args[0][0][2], n.args[0][1]),
        (n.args[1][0][2], n.args[1][1])
    )
    dip_radians = sp.pi / 2 - sp.asin(n2)
    dip_degrees = sp.deg(dip_radians * 180 / sp.pi)
    # Direction calculation (only for first branch for demonstration)
    normal_xy = sp.Matrix([n.args[0][0][0], n.args[0][0][1]])
    xy_norm = sp.sqrt(n.args[0][0][0]**2 + n.args[0][0][1]**2)
    normal_xy_unit = normal_xy / xy_norm
    _dir_0_plane = sp.Matrix([0, 1])
    dot_prod = normal_xy_unit.dot(_dir_0_plane)
    dir_radians = sp.acos(dot_prod)
    dir_degrees = sp.deg(dir_radians * 180 / sp.pi)
    return


@app.cell
def _():
    r"""
    ## Rodrigues' Rotation Formula

    To rotate a vector about another vector (the normal), we use Rodrigues' rotation formula:

    $$
    \mathbf{v}_{\text{rot}} = \mathbf{v} \cos \theta
        + (\mathbf{k} \times \mathbf{v}) \sin \theta
        + \mathbf{k} (\mathbf{k} \cdot \mathbf{v}) (1 - \cos \theta)
    $$

    where $\mathbf{v}$ is the vector to rotate, $\mathbf{k}$ is the axis (normal), and $\theta$ is the rotation angle (gamma).
    """
    pass
    return


@app.cell
def _(mo):
    mo.md("Apply Rodrigues' rotation formula to rotate the vector about the normal by gamma.")
    return


@app.cell
def _(gamma, normal_vec_normalized_up, sp):
    # Use the first branch of Piecewise for demonstration
    v = normal_vec_normalized_up.args[0][0]
    k = v
    theta = gamma * sp.pi / 180
    v_rot = (
        v * sp.cos(theta)
        + k.cross(v) * sp.sin(theta)
        + k * (k.dot(v)) * (1 - sp.cos(theta))
    )
    return (v_rot,)


@app.cell
def _(mo):
    mo.md("Trend and plunge from the rotated vector")
    return


@app.cell
def _(sp, v_rot):
    plunge_radians = sp.asin(v_rot[2])
    plunge_degrees = sp.deg(plunge_radians * 180 / sp.pi)
    vector_xy = sp.Matrix([v_rot[0], v_rot[1]])
    vector_xy_unit = vector_xy / sp.sqrt(v_rot[0]**2 + v_rot[1]**2)
    _dir_0_vector = sp.Matrix([0, 1])
    trend_radians = sp.acos(vector_xy_unit.dot(_dir_0_vector))
    trend_degrees = sp.deg(trend_radians * 180 / sp.pi)
    return


@app.cell
def _():
    """
    ## Summary

    In this notebook, we have symbolically derived the steps for transforming drillcore measurements
    (alpha, beta, trend, plunge, gamma) into real-world structural feature orientations.

    - Converted input angles to radians
    - Computed the normal vector of the measured plane
    - Normalized the vector and ensured it points upwards
    - Calculated dip and direction from the normal
    - Used Rodrigues' rotation formula to rotate the vector by gamma
    - Computed trend and plunge from the rotated vector

    These steps form the mathematical basis of the `transform` function in the codebase.
    """
    pass
    return


if __name__ == "__main__":
    app.run()
