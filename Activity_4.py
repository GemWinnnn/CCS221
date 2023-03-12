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
    x = bottom_lower[0] + side_length/2 * np.cos(u * np.pi / 180)
    y = bottom_lower[1] + side_length/2 * np.sin(u * np.pi / 180)
    z = bottom_lower[2] + side_length/2 * np.sin(v * np.pi / 180)
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


st.title("3D Shapes Translation")

#Rectangle
st.header("Rectangle")
init_rect = _rectangle_(bottom_lower=(-5, -5, -5), side_lengths=(10, 5, 3))
points_rect = tf.constant(init_rect, dtype=tf.float32)

fig = _plt_basic_object_(init_rect)
st.pyplot(fig)

x1 = st.sidebar.slider('x1', -15.0, 15.0, 0.0)
y1 = st.sidebar.slider('y1', -15.0, 15.0, 0.0)
z1 = st.sidebar.slider('z1', -15.0, 15.0, 0.0)

@st.cache
def translate_obj(points, amount):
    return tf.add(points, amount)

if st.button('Translate Rectangle'):
    translation_amount = tf.constant([x1, y1, z1], dtype=tf.float32)
    translated_object = translate_obj(points_rect, translation_amount)
    translated_rect = translated_object.numpy()
    fig = _plt_basic_object_(translated_rect)
    st.pyplot(fig)

#Diamond
st.header("Diamond")
init_diamond = _diamond_(bottom_lower=(1, 2, 3), side_length=8)
points_diamond = tf.constant(init_diamond, dtype=tf.float32)

fig = _plt_basic_object_(init_diamond)
st.pyplot(fig)

x2 = st.sidebar.slider('x2', -15.0, 15.0, 0.0)
y2 = st.sidebar.slider('y2', -15.0, 15.0, 0.0)
z2 = st.sidebar.slider('z2', -15.0, 15.0, 0.0)

if st.button('Translate Diamond'):
    translation_amount = tf.constant([x2, y2, z2], dtype=tf.float32)
    translated_object = translate_obj(points_diamond, translation_amount)
    translated_diamond = translated_object.numpy()
    fig = _plt_basic_object_(translated_diamond)
    st.pyplot(fig)

#Pyramid
st.header("Pyramid")
init_pyramid = _pyramid2_(bottom_center=(0, 0, 0))
points_pyramid = tf.constant(init_pyramid, dtype=tf.float32)

fig = _plt_basic_object_(init_pyramid)
st.pyplot(fig)

x3 = st.sidebar.slider('x3', -15.0, 15.0, 0.0)
y3 = st.sidebar.slider('y3', -15.0, 15.0, 0.0)
z3 = st.sidebar.slider('z3', -15.0, 15.0, 0.0)

if st.button('Translate Pyramid'):
    translation_amount = tf.constant([x3, y3, z3], dtype=tf.float32)
    translated_object = translate_obj(points_pyramid, translation_amount)
    translated_pyramid = translated_object.numpy()
    fig = _plt_basic_object_(translated_pyramid)
    st.pyplot(fig)
    
_object.numpy()
fig = _plt_basic_object_(translated_pyramid)
st.pyplot(fig)

#Sphere
st.header("Sphere")
init_sphere = _sphere_(center=(0, 0, 0), radius=5, num_steps=20)
points_sphere = tf.constant(init_sphere, dtype=tf.float32)

fig = _plt_basic_object_(init_sphere)
st.pyplot(fig)

x4 = st.sidebar.slider('x4', -15.0, 15.0, 0.0)
y4 = st.sidebar.slider('y4', -15.0, 15.0, 0.0)
z4 = st.sidebar.slider('z4', -15.0, 15.0, 0.0)

if st.button('Translate Sphere'):
    translation_amount = tf.constant([x4, y4, z4], dtype=tf.float32)
    translated_object = translate_obj(points_sphere, translation_amount)
    translated_sphere = translated_object.numpy()
    fig = _plt_basic_object_(translated_sphere)
    st.pyplot(fig)
