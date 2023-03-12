import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay
import tensorflow as tf

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

st.title("3D Rectangular Prism Translation")
st.write("Enter the translation vector to move the 3D rectangular prism.")

# Create a 3D rectangular prism and display it
init_rectangular_prism = _rectangle_(bottom_lower=(1, 2, 5), side_lengths=(7, 5, 4))
fig = _plt_basic_object_(init_rectangular_prism)
st.pyplot(fig)

# Get translation vector from user and display translated rectangular prism
x = st.number_input("Enter the x component of the vector:")
y = st.number_input("Enter the y component of the vector:")
z = st.number_input("Enter the z component of the vector:")
translation_amount = tf.constant([x, y, z], dtype=tf.float32)
points = tf.constant(init_rectangular_prism, dtype=tf.float32)
translated_object = tf.add(points, translation_amount)
translated_rectangular_prism = translated_object.numpy()
fig = _plt_basic_object_(translated_rectangular_prism)
st.pyplot(fig)
