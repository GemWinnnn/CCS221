import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.spatial import Delaunay
import tensorflow as tf
import streamlit as st

# Add Streamlit components
st.title("3D Object Translator")

x = st.sidebar.slider("Enter the x component of the vector:", -10.0, 10.0, 0.0)
y = st.sidebar.slider("Enter the y component of the vector:", -10.0, 10.0, 0.0)
z = st.sidebar.slider("Enter the z component of the vector:", -10.0, 10.0, 0.0)
translation_amount = tf.constant([x, y, z], dtype=tf.float32)

def _plt_basic_object_(points):
    tri = Delaunay(points).convex_hull

    # Create a colormap to assign colors based on the Z-coordinate of the vertices
    colormap = plt.cm.get_cmap("viridis")
    z_range = points[:, 2].max() - points[:, 2].min()
    face_colors = colormap((points[tri].mean(axis=1)[:, 2] - points[:, 2].min()) / z_range)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create Poly3DCollection and set facecolors
    polyc = art3d.Poly3DCollection(points[tri])
    polyc.set_facecolor(face_colors)
    ax.add_collection(polyc)
    
    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-15, 15)

    st.pyplot(fig)



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


def _sphere_(center=(0, 0, 0), radius=1, num_steps=20):
    center = np.array(center)
    u = np.linspace(0, 2 * np.pi, num_steps)
    v = np.linspace(0, np.pi, num_steps)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    return points


def translate_obj(points, amount):
    return tf.add(points, amount)

# Initialize objects
init_rectangular_prism = _rectangle_(bottom_lower=(1, 2, 5), side_lengths=(7, 5, 4))
init_sphere = _sphere_(center=(0, 0, 0), radius=2)
init_pyramid = _pyramid2_(bottom_center=(0,0,0))

# Define colors for each object
rectangular_prism_color = plt.cm.get_cmap("viridis")(0.5)
sphere_color = plt.cm.get_cmap("viridis")(0.2)
pyramid_color = plt.cm.get_cmap("viridis")(0.8)

# Convert objects to TensorFlow tensors
rectangular_prism_points = tf.constant(init_rectangular_prism, dtype=tf.float32)
sphere_points = tf.constant(init_sphere, dtype=tf.float32)
points_pyramid2 = tf.constant(init_pyramid, dtype=tf.float32)

# Translate the objects
translated_rectangular_prism = translate_obj(rectangular_prism_points, translation_amount)
translated_sphere = translate_obj(sphere_points, translation_amount)
translated_object = translate_obj(points_pyramid2, translation_amount)

# Convert translated objects to NumPy arrays
translated_rectangular_prism = translated_rectangular_prism.numpy()
translated_sphere = translated_sphere.numpy()
translated_object = translated_object.numpy()

# Plot initial objects
_plt_basic_object_(init_rectangular_prism)
_plt_basic_object_(init_sphere)
_plt_basic_object_(init_pyramid)


# Plot translated objects
st.header("Translated Objects")
_plt_basic_object_(translated_rectangular_prism)
_plt_basic_object_(translated_sphere)
_plt_basic_object_(translated_object)
