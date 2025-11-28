import marimo

app = marimo.App(width="medium")

@app.cell
def _():
    """
    # Mathematical Basis of Drillcore Measurement Transformation

    This notebook introduces the mathematical steps behind the `transform` function,
    which converts drillcore measurements (alpha, beta, trend, plunge, gamma)
    to real-world structural orientations.

    We use symbolic computation (`sympy`) to illustrate each step.
    """
    pass

@app.cell
def _():
    import sympy as sp
    pass

@app.cell
def _():
    # Define symbols for all input parameters
    alpha, beta, trend, plunge, gamma = sp.symbols('alpha beta trend plunge gamma')
    return alpha, beta, trend, plunge, gamma

@app.cell
def _(alpha, beta, trend, plunge, gamma):
    # Degrees to radians conversion
    deg2rad = lambda x: x * sp.pi / 180
    alpha_rad = deg2rad(alpha)
    beta_rad = deg2rad(beta)
    trend_rad = deg2rad(trend)
    plunge_rad = deg2rad(plunge)
    return alpha_rad, beta_rad, trend_rad, plunge_rad, gamma

@app.cell
def _(alpha_rad, beta_rad, trend_rad, plunge_rad, gamma):
    # Symbolic equations for the normal vector components
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
    return normal_vec

@app.cell
def _(normal_vec):
    # Normalization of the normal vector
    norm = sp.sqrt(sum([comp**2 for comp in normal_vec]))
    normal_vec_normalized = normal_vec / norm
    normal_vec_normalized_up = sp.Piecewise(
        (normal_vec_normalized, normal_vec[2] >= 0),
        (-normal_vec_normalized, normal_vec[2] < 0)
    )
    normal_vec_normalized_up
    return normal_vec_normalized_up

@app.cell
def _(normal_vec_normalized_up):
    # Plane dip and direction from the normal vector
    n = normal_vec_normalized_up
    dip_radians = sp.pi / 2 - sp.asin(n[2])
    dip_degrees = sp.deg(dip_radians * 180 / sp.pi)
    # Direction calculation
    normal_xy = sp.Matrix([n[0], n[1]])
    xy_norm = sp.sqrt(n[0]**2 + n[1]**2)
    normal_xy_unit = normal_xy / xy_norm
    dir_0 = sp.Matrix([0, 1])
    dot_prod = normal_xy_unit.dot(dir_0)
    dir_radians = sp.acos(dot_prod)
    dir_degrees = sp.deg(dir_radians * 180 / sp.pi)
    return dip_degrees, dir_degrees

@app.cell
def _():
    """
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

@app.cell
def _(normal_vec_normalized_up, gamma):
    # Symbolic Rodrigues' rotation
    v = normal_vec_normalized_up
    k = normal_vec_normalized_up
    theta = gamma * sp.pi / 180
    v_rot = (
        v * sp.cos(theta)
        + k.cross(v) * sp.sin(theta)
        + k * (k.dot(v)) * (1 - sp.cos(theta))
    )
    return v_rot

@app.cell
def _(v_rot):
    # Trend and plunge from the rotated vector
    plunge_radians = sp.asin(v_rot[2])
    plunge_degrees = sp.deg(plunge_radians * 180 / sp.pi)
    vector_xy = sp.Matrix([v_rot[0], v_rot[1]])
    vector_xy_unit = vector_xy / sp.sqrt(v_rot[0]**2 + v_rot[1]**2)
    dir_0 = sp.Matrix([0, 1])
    trend_radians = sp.acos(vector_xy_unit.dot(dir_0))
    trend_degrees = sp.deg(trend_radians * 180 / sp.pi)
    return trend_degrees, plunge_degrees

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
