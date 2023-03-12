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

def plot_3d_object(shape, *args, **kwargs):
    if shape == "rectangle":
        points = rectangle(*args, **kwargs)

    elif shape == "diamond":
        points = _diamond_(*args, **kwargs)

    elif shape == "triangle":
        points = triangle(*args, **kwargs)

    elif shape == "sphere":
        points = sphere(*args, **kwargs)

    fig = _plt_basic_object_(points)
    return fig

def main():
    st.sidebar.title("Controls")

shapes = st.sidebar.selectbox("Choose a shape", ["rectangle", "diamond", "triangle", "sphere"])

if shapes == "rectangle":
    side_lengths = st.sidebar.slider("Side Lengths", 0.1, 15.0, (1, 1, 1), step=0.1)
    bottom_lower = st.sidebar.slider("Bottom Lower Coordinate", -15.0, 15.0, (0, 0, 0), step=0.1)
    fig = plot_3d_object(shapes, bottom_lower, side_lengths=side_lengths)

elif shapes == "diamond":
    side_length = st.sidebar.slider("Side Length", 0.1, 15.0, 5, step=0.1)
    bottom_lower = st.sidebar.slider("Bottom Lower Coordinate", -15.0, 15.0, (0, 0, 0), step=0.1)
    fig = plot_3d_object(shapes, bottom_lower, side_length=side_length)

elif shapes == "triangle":
    p1 = st.sidebar.slider("Vertex 1", -15.0, 15.0, (0, 0, 0), step=0.1)
    p2 = st.sidebar.slider("Vertex 2", -15.0, 15.0, (1, 0, 0), step=0.1)
    p3 = st.sidebar.slider("Vertex 3", -15.0, 15.0, (0, 1, 0), step=0.1)
    fig = plot_3d_object(shapes, p1, p2, p3)

elif shapes == "sphere":
    center = st.sidebar.slider("Center", -15.0, 15.0, (0, 0, 0), step=0.1)
    radius = st.sidebar.slider("Radius", 0.1, 15.0, 1, step=0.1)
    fig = plot_3d_object(shapes, center=center, radius=radius)

st.pyplot(fig)
if name == "main":
main()

 
