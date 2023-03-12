import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay
import tensorflow as tf

def _plt_basic_object_(points, color=[1, 0, 0]):

    tri = Delaunay(points).convex_hull

    # Define a color for each face
    face_colors = np.array([
        color
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



def _sphere_(center=(0,0,0), radius=1, num_steps=20):
    center = np.array(center)
    u = np.linspace(0, 2*np.pi, num_steps)
    v = np.linspace(0, np.pi, num_steps)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    return points


def _diamond_(bottom_lower=(0, 0, 0,), side_length=5):

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

def _pyramid_(bottom_center=(0,0,0), side_length=1):
    bottom_center = np.array(bottom_center)
    half_side = side_length/2
    points = np.vstack([
        bottom_center + [-half_side, -half_side, 0],
        bottom_center + [-half_side, half_side, 0],
        bottom_center + [half_side, half_side, 0],
        bottom_center + [half_side, -half_side, 0],
        bottom_center + [0, 0, side_length]
    ])
    return points


st.title("3D Shape Translation")

# Create a 3D rectangular prism and display it
init_rectangular_prism = _rectangle_(bottom_lower=(1, 2, 5), side_lengths=(7, 5, 4))
fig = _plt_basic_object_(init_rectangular_prism)
st.pyplot(fig)

# Create a 3D pyramid and display it
init_pyramid = _pyramid_(bottom_center=(0, 0, 0), side_length=10)
fig = _plt_basic_object_(init_pyramid)
st.pyplot(fig)

# Create a 3D sphere and display it
init_sphere = _sphere_(center=(0,0,0), radius=2)
fig = _plt_basic_object_(init_sphere)
st.pyplot(fig)

# Get translation vector from user and display translated shapes
x = st.number_input("Enter the x component of the vector:")
y = st.number_input("Enter the y component of the vector:")
z = st.number_input("Enter the z component of the vector:")
translation_amount = tf.constant([x, y, z], dtype=tf.float32)

# Create a 3D rectangular prism and display it
init_rectangular_prism = _rectangle_(bottom_lower=(1, 2, 5), side_lengths=(7, 5, 4))
color_rectangular_prism = [1, 0, 0] # Red color
fig = _plt_basic_object_(init_rectangular_prism, color_rectangular_prism)
st.pyplot(fig)

# Create a 3D pyramid and display it
init_pyramid = _pyramid_(bottom_center=(0, 0, 0), side_length=10)
color_pyramid = [0, 1, 0] # Green color
fig = _plt_basic_object_(init_pyramid, color_pyramid)
st.pyplot(fig)

# Create a 3D sphere and display it
init_sphere = _sphere_(center=(0,0,0), radius=2)
color_sphere = [0, 0, 1] # Blue color
fig = _plt_basic_object_(init_sphere, color_sphere)
st.pyplot(fig)
