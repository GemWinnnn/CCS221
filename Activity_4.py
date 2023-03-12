import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay

import tensorflow as tf
import streamlit as st


def _plt_basic_object_(points):

    tri = Delaunay(points).convex_hull

    # Define a color for each face
    face_colors = np.array([
        [1, 0, 0],  # red
        [0, 1, 0],  # green
        [0, 0, 1],  # blue
        [1, 1, 0],  # yellow
        [1, 0, 1],  # magenta
        [0, 1, 1],  # cyan
    ])

    # Repeat each color for each vertex in the corresponding face
    face_colors = np.repeat(face_colors, 3, axis=0)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    S = ax.plot_trisurf(points[:,0], points[:,1], points[:,2],
                        triangles = tri,
                        shade = True, facecolors = face_colors, lw = 0.5)

    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-15, 15)

    return fig


def _rectangle_(bottom_lower=(0, 0, 0), side_lengths=(1, 1, 1)):
    bottom_lower = np.array(bottom_lower)
    side_lengths = np.array(side_lengths)

    points = np.vstack([
        bottom_lower,
        bottom_lower + [0, side_lengths[1], 0],
        bottom_lower + [side_lengths[0], side_lengths[1], 0],
        bottom_lower + [side_lengths[0], 0, 0],
        bottom_lower + [0, 0, side_lengths[2]],
        bottom_lower + [0, side_lengths[1], side_lengths[2]],
        bottom_lower + [side_lengths[0], side_lengths[1], side_lengths[2]],
        bottom_lower + [side_lengths[0], 0, side_lengths[2]],
    ])

    return points


def _diamond_(bottom_lower=(0, 0, 0,), side_length=5):
    bottom_lower = np.array(bottom_lower)
    u = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    v = np.array([0, np.pi/2, np.pi])
    x = bottom_lower[0] + side_length/2 * np.cos(u)
    y = bottom_lower[1] + side_length/2 * np.sin(u)
    z = bottom_lower[2] + side_length/2 * np.sin(v)
    points = np.vstack([x, y, z]).T
    return points


def _pyramid2_(bottom_center=(0, 0, 0)):
    bottom_center = np.array(bottom_center) 

    points = np.vstack([
        bottom_center + [-3, -3, 0],
        bottom_center + [-3, +3, 0],
        bottom_center + [+3, -3, 0],
        bottom_center + [+3, +3, 0],
        bottom_center + [0, 0, +5]
    ])

    return points
  
def _sphere_(center=(0,0,0), radius=1, num_steps=20):
    center = np.array(center)
    u = np.linspace(0, 2*np.pi, num_steps)
    v = np.linspace(0, np.pi, num_steps)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    
    return points

