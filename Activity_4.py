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
                        triangles=tri,
                        shade=True, facecolors=face_colors, lw=0.5)

    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-15, 15)

    return fig

def _diamond_(bottom_lower=(0, 0, 0), side_length=5):
    bottom_lower = np.array(bottom_lower)

    points = np.vstack([
        bottom_lower + [side_length/2, 0, side_length/2],
        bottom_lower + [0, side_length/2, side_length/2],
        bottom_lower + [side_length/2, side_length/2, 0],
        bottom_lower + [side_length/2, side_length/2, side_length],
        bottom_lower + [side_length, side_length/2, side_length/2],
        bottom_lower + [side_length/2, side_length, side_length/2],
    ])

    return points

def rectangle(bottom_lower=(0, 0, 0), side_lengths=(1, 1, 1)):
    bottom_lower = np.array(bottom_lower)
    side_lengths = np.array(side_lengths)

    points = np.vstack([    bottom_lower,    bottom_lower + [0, side_lengths[1], 0],
                        
        bottom_lower + [side_lengths[0], side_lengths[1], 0],
        bottom_lower + [side_lengths[0], 0, 0],
        bottom_lower + [0, 0, side_lengths[2]],
        bottom_lower + [0, side_lengths[1], side_lengths[2]],
        bottom_lower + [side_lengths[0], side_lengths[1], side_lengths[2]]
    ])

    return points

def sphere(center=(0, 0, 0), radius=1):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    return points

def triangle(p1=(0, 0, 0), p2=(1, 0, 0), p3=(0, 1, 0)):
    points = np.vstack([p1, p2, p3])
    return points
st.sidebar.title("Controls")

shape_type = st.sidebar.selectbox("Select Shape", ["Rectangle", "Triangle", "Sphere"])

if shape_type == "Rectangle":
    bottom_lower = st.sidebar.slider("Bottom Left", -10, 10, 0, 1)
    side_lengths = st.sidebar.slider("Side Lengths", 0, 10, 1, 0.5)
    points = rectangle(bottom_lower=(bottom_lower, bottom_lower, bottom_lower), side_lengths=(side_lengths, side_lengths, side_lengths))

if shape_type == "Triangle":
    p1 = st.sidebar.slider("Point 1", -10, 10, 0, 1)
    p2 = st.sidebar.slider("Point 2", -10, 10, 1, 1)
    p3 = st.sidebar.slider("Point 3", -10, 10, 0, 1)
    points = triangle(p1=(p1, p1, p1), p2=(p2, p2, p2), p3=(p3, p3, p3))

if shape_type == "Sphere":
    center = st.sidebar.slider("Center", -10, 10, 0, 1)
    radius = st.sidebar.slider("Radius", 0, 10, 1, 0.5)
    points = sphere(center=(center, center, center), radius=radius)

fig = _plt_basic_object_(points)
st.pyplot(fig)


 
