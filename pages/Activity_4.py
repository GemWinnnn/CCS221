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

separation = st.sidebar.slider("Enter the separation between shapes:", -10.0, 10.0, 0.0)

# Add a select box for choosing the visualization
visualization_options = [
    "Initial",
    "Translated",
    "Separated"
]

# Plotting function for individual shapes
def plot_shape(points, object_type="general", title=None):
    if object_type == "pyramid":
        tri = np.array([[0, 1, 2], [0, 1, 4], [1, 2, 4], [0, 2, 4], [2, 3, 4]])
    else:
        tri = Delaunay(points).convex_hull

    colormap = plt.cm.get_cmap("viridis")
    z_range = points[:, 2].max() - points[:, 2].min()
    face_colors = colormap((points[tri].mean(axis=1)[:, 2] - points[:, 2].min()) / z_range)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    polyc = art3d.Poly3DCollection(points[tri])
    polyc.set_facecolor(face_colors)
    ax.add_collection(polyc)

    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-15, 15)

    if title:
        ax.set_title(title)

    return fig

# Functions to define the initial shapes
def create_pyramid(bottom_center=(0, 0, 0)):
    bottom_center = np.array(bottom_center) 

    points = np.vstack([
        bottom_center + [-3, -3, 0],
        bottom_center + [-3, +3, 0],
        bottom_center + [+3, -3, 0],
        bottom_center + [+3, +3, 0],
        bottom_center + [0, 0, +5]
    ])

    return points

def create_rectangular_prism(bottom_lower=(0, 0, 0), side_lengths=(1, 1, 1)):
    bottom_lower = np.array(bottom_lower)
    side_lengths = np.array(side_lengths)

    points = np.vstack([
        bottom_lower,
        bottom_lower + [0, side_lengths[1], 0],
        bottom_lower + [side_lengths[0], side_lengths[1], 0],
        bottom_lower + [side_lengths[0], 0, 0],
        bottom_lower + [0, 0, side_lengths[2]],
        bottom_lower + [0, side_lengths[1], side_lengths[2]],
        bottom_lower + [



# Functions to define the initial shapes
def create_pyramid(bottom_center=(0, 0, 0)):
    bottom_center = np.array(bottom_center) 

    points = np.vstack([
        bottom_center + [-3, -3, 0],
        bottom_center + [-3, +3, 0],
        bottom_center + [+3, -3, 0],
        bottom_center + [+3, +3, 0],
        bottom_center + [0, 0, +5]
    ])

    return points

def create_rectangular_prism(bottom_lower=(0, 0, 0), side_lengths=(1, 1, 1)):
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
init_rectangular_prism = create_rectangular_prism(bottom_lower=(1, 2, 5), side_lengths=(7, 5, 4))
init_sphere = create_sphere(center=(0, 0, 0), radius=2)
init_pyramid = create_pyramid(bottom_center=(0, 0, 0))

# Convert objects to TensorFlow tensors
rectangular_prism_points = tf.constant(init_rectangular_prism, dtype=tf.float32)
sphere_points = tf.constant(init_sphere, dtype=tf.float32)
points_pyramid = tf.constant(init_pyramid, dtype=tf.float32)

# Translate and seperate the objects 
translated_rectangular_prism = translate_obj(rectangular_prism_points, translation_amount)
translated_sphere = translate_obj(sphere_points, translation_amount)
translated_pyramid = translate_obj(points_pyramid, translation_amount)

separated_rectangular_prism = translate_obj(rectangular_prism_points, tf.constant([0, separation, 0], dtype=tf.float32))
separated_sphere = translate_obj(sphere_points, tf.constant([separation, 0, 0], dtype=tf.float32))
separated_pyramid = translate_obj(points_pyramid, tf.constant([0, -separation, 0], dtype=tf.float32))

# Convert translated and separated objects to NumPy arrays
translated_rectangular_prism = translated_rectangular_prism.numpy()
translated_sphere = translated_sphere.numpy()
translated_pyramid = translated_pyramid.numpy()

separated_rectangular_prism = separated_rectangular_prism.numpy()
separated_sphere = separated_sphere.numpy()
separated_pyramid = separated_pyramid.numpy()

# Separate the objects based on the separation slider value
separation_vector = tf.constant([separation, 0, 0], dtype=tf.float32)
separated_sphere = translate_obj(translated_sphere, separation_vector)
separated_object = translate_obj(translated_object, 2 * separation_vector)

# Convert translated and separated objects to NumPy arrays
translated_rectangular_prism = translated_rectangular_prism.numpy()
separated_sphere = separated_sphere.numpy()
separated_object = separated_object.numpy()

# Display the selected visualization for each shape
shape_options = ["Rectangular Prism", "Sphere", "Pyramid"]
selected_shape = st.sidebar.selectbox("Choose a shape:", shape_options)

if selected_shape == "Rectangular Prism":
    if "Initial" in visualization_options:
        st.header("Initial Rectangular Prism")
        st.pyplot(plot_shape(init_rectangular_prism, title="Initial Rectangular Prism"))
    if "Translated" in visualization_options:
        st.header("Translated Rectangular Prism")
        st.pyplot(plot_shape(translated_rectangular_prism, title="Translated Rectangular Prism"))
    if "Separated" in visualization_options:
        st.header("Separated Rectangular Prism")
        st.pyplot(plot_shape(separated_rectangular_prism, title="Separated Rectangular Prism"))

elif selected_shape == "Sphere":
    if "Initial" in visualization_options:
        st.header("Initial Sphere")
        st.pyplot(plot_shape(init_sphere, title="Initial Sphere"))
    if "Translated" in visualization_options:
        st.header("Translated Sphere")
        st.pyplot(plot_shape(translated_sphere, title="Translated Sphere"))
    if "Separated" in visualization_options:
        st.header("Separated Sphere")
        st.pyplot(plot_shape(separated_sphere, title="Separated Sphere"))

elif selected_shape == "Pyramid":
    if "Initial" in visualization_options:
        st.header("Initial Pyramid")
        st.pyplot(plot_shape(init_pyramid, object_type="pyramid", title="Initial Pyramid"))
    if "Translated" in visualization_options:
        st.header("Translated Pyramid")
        st.pyplot(plot_shape(translated_pyramid, object_type="pyramid", title="Translated Pyramid"))
    if "Separated" in visualization_options:
        st.header("Separated Pyramid")
        st.pyplot(plot_shape(separated_pyramid, object_type="pyramid", title="Separated Pyramid"))
