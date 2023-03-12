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

init_pyramid = _pyramid2_(bottom_center=(0,0,0))
points_pyramid2 = tf.constant(init_pyramid, dtype=tf.float32)

fig = _plt_basic_object_(init_pyramid)
st.pyplot(fig)

x = st.sidebar.slider('x', -15.0, 15.0, 0.0)
y = st.sidebar.slider('y', -15.0, 15.0, 0.0)
z = st.sidebar.slider('z', -15.0, 15.0, 0.0)

@st.cache
def translate_obj(points, amount):
    return tf.add(points, amount)

if st.button('Translate'):
    translation_amount = tf.constant([x, y, z], dtype=tf.float32)
    translated_object = translate_obj(points_pyramid2, translation_amount)
    translated_pyramid = translated_object.numpy()
    fig = _plt_basic_object_(translated_pyramid)
    st.pyplot(fig)
