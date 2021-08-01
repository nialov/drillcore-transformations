"""
Module for visualizing results.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize_vector(vector: np.array, ax: Axes3D, **kwargs):
    """
    Visualize a vector with 3d plot.
    """
    vector = vector.round(decimals=2)
    ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], **kwargs)


def visualize_plane(plane_normal, ax: Axes3D, **kwargs):
    """
    Visualize a plane with 3d plot.
    """
    # Create plane for visualization
    xx, yy = np.meshgrid(np.linspace(-1, 1), np.linspace(-1, 1))
    d = -np.array([0, 0, 0]).dot(plane_normal)
    # Only for visualization purposes
    if plane_normal[2] == 0:
        plane_normal[2] = 0.00000000001
    zz = (-plane_normal[0] * xx - plane_normal[1] * yy - d) * 1.0 / plane_normal[2]
    ax.plot_surface(xx, yy, zz, alpha=0.2, **kwargs)
    # surf._facecolors = surf._facecolor3d
    # surf._edgecolors = surf._edgecolor3d


def visualize_results(
    plane_normal,
    plane_vector,
    drillcore_vector,
    drillcore_trend,
    drillcore_plunge,
    alpha,
    beta,
    gamma_vector=None,
    gamma=None,
    img_dir=None,
    curr_conv=None,
):
    """
    Visualize alpha-beta measured plane and gamma measured linear feature.

    :param plane_normal: Normal of measured plane.
    :type plane_normal: numpy.ndarray
    :param plane_vector: Vector in the dip direction and dip of plane.
    :type plane_vector: numpy.ndarray
    :param drillcore_vector: Vector in the direction of the drillcore.
    :type drillcore_vector: numpy.ndarray
    :param gamma_vector:
    :type gamma_vector: numpy.ndarray
    """
    # 3D Figure
    fig = plt.figure(figsize=(9, 9))
    ax = fig.gca(projection="3d")
    ax: Axes3D

    # Plot in 3D
    visualize_vector(plane_normal, ax, label="normal of plane", color="red")
    visualize_plane(plane_normal, ax, label="plane", color="red")
    visualize_vector(
        plane_vector, ax, label=r"plunge vector of plane", color="brown", linewidth=2
    )
    visualize_vector(
        drillcore_vector, ax, label="drillcore", color="black", linewidth=2
    )
    if gamma_vector is not None:
        visualize_vector(
            gamma_vector, ax, label="gamma", color="darkgreen", linewidth=2
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    if gamma_vector is not None:
        ax.text(
            0,
            0,
            1.4,
            s=f"bh dir {drillcore_trend}\n"
            f"bh dip {drillcore_plunge}\n"
            f"alpha {alpha}\n"
            f"beta {beta}\n"
            f"gamma {gamma}\n",
        )
    else:
        ax.text(
            0,
            0,
            1.4,
            s=f"bh dir {drillcore_trend}\n"
            f"bh dip {drillcore_plunge}\n"
            f"alpha {alpha}\n"
            f"beta {beta}\n",
        )

    # plt.legend()

    if img_dir is not None:
        curr_conv = curr_conv.replace("|", "_")
        if Path(img_dir).exists():
            pass
        else:
            os.mkdir(img_dir)
        num = 0
        savename = Path(img_dir) / Path(curr_conv + f"_{num}.png")
        while savename.exists():
            num += 1
            savename = Path(img_dir) / Path(curr_conv + f"_{num}")
        plt.savefig(savename)
