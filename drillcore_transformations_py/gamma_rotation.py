"""
Rotation of vector about another vector code is from:
https://gist.github.com/fasiha/6c331b158d4c40509bd180c5e64f7924
Big thanks to github user fasiha!
"""
import numpy as np
import numpy.linalg as linalg

def makeUnit(x):
    """Normalize entire input to norm 1. Not what you want for 2D arrays!"""
    return x / linalg.norm(x)


def xParV(x, v):
    """Project x onto v. Result will be parallel to v."""
    # (x' * v / norm(v)) * v / norm(v)
    # = (x' * v) * v / norm(v)^2
    # = (x' * v) * v / (v' * v)
    return np.dot(x, v) / np.dot(v, v) * v


def xPerpV(x, v):
    """Component of x orthogonal to v. Result is perpendicular to v."""
    return x - xParV(x, v)


def xProjectV(x, v):
    """Project x onto v, returning parallel and perpendicular components
    >> d = xProject(x, v)
    >> np.allclose(d['par'] + d['perp'], x)
    True
    """
    par = xParV(x, v)
    perp = x - par
    return {'par': par, 'perp': perp}


def rotateAbout(a, b, theta):
    """Rotate vector a about vector b by theta radians."""
    # Thanks user MNKY at http://math.stackexchange.com/a/1432182/81266
    proj = xProjectV(a, b)
    w = np.cross(b, proj['perp'])
    return (proj['par'] +
            proj['perp'] * np.cos(theta) +
            linalg.norm(proj['perp']) * makeUnit(w) * np.sin(theta))
