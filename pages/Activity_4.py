
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay

def plt_basic_object_(points, counter):
    tri = Delaunay(points).convex_hull

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    S = ax.plot_trisurf(points[:,0], points[:,1], points[:,2],triangles=tri,shade=True, cmap=cm.seismic,lw=0.5)

    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-10, 10)
    if (counter == 1):
        plt.title("Pyramid")
    elif (counter == 2):
        plt.title("Heart")
    elif (counter == 3):
        plt.title("Diamond")
    elif (counter == 4):
        plt.title ("Circle")

    return fig

def _pyramid_(bottom_center=(0, 0, 0)):
    bottom_center = np.array(bottom_center) 

    points = np.vstack([
    bottom_center + [-3, -3, 0],
    bottom_center + [-3, +3, 0],
    bottom_center + [+3, -3, 0],
    bottom_center + [+3, +3, 0],
    bottom_center + [0, 0, +5]
    ])

    return points

init_pyramid = _pyramid_(bottom_center=(0,0,0))
points_pyramid2 = tf.constant(init_pyramid, dtype=tf.float32)
counter = 1
fig1 = plt_basic_object_(init_pyramid, counter)
st.pyplot(fig1)

x = st.slider("Enter for x:", -10, 10, step=1,key='my_slider1')
y = st.slider("Enter for y:", -10, 10, step=1,key='my_slider2')
z = st.slider("Enter for z:", -10, 10, step=1,key='my_slider3')

translation = tf.constant([x, y, z], dtype=tf.float32)

translated_points = points_pyramid2 + translation

fig2 = plt_basic_object_(translated_points.numpy(), counter)
st.pyplot(fig2)

def _heart_(bottom_center = (0, 0, 0)):
    bottom_center = np.array(bottom_center)
    points = np.vstack([
        bottom_center + [+1.5, -1, +3.5],
        bottom_center + [+1.5, +1, +3.5],
        bottom_center + [-1.5, -1, +3.5],
        bottom_center + [-1.5, +1, +3.5],        
        bottom_center + [0, +1, +3],
        bottom_center + [0, -1, +2],
        bottom_center + [+3, 0, +2],
        bottom_center + [-3, 0, +2],
        bottom_center + [0, 1, -2],
        bottom_center + [0, -1, -2]
    ])
    return points

init_heart = _heart_(bottom_center=(0,0,0))
points_heart = tf.constant(init_heart, dtype=tf.float32)
counter = 2
fig3 = plt_basic_object_(init_heart, counter)
st.pyplot(fig3)

x = st.slider("Enter for x:", -10, 10, step=1,key='my_slider4')
y = st.slider("Enter for y:", -10, 10, step=1,key='my_slider5')
z = st.slider("Enter for z:", -10, 10, step=1,key='my_slider6')

translation = tf.constant([x, y, z], dtype=tf.float32)

translated_points = points_heart + translation

fig4 = plt_basic_object_(translated_points.numpy(), counter)
st.pyplot(fig4)



def _diamond_(bottom_center=(0, 0, 0)):             #fucntion for diamond shape
    bottom_center = np.array(bottom_center)

    points = np.vstack([
    bottom_center + [+2.5, +2.5, 0],
    bottom_center + [-2.5, +2.5, 0],
    bottom_center + [+2.5, -2.5, 0],
    bottom_center + [-2.5, -2.5, 0],
    bottom_center + [0, 0, 5],
    bottom_center + [0, 0, -5]
    ])
    return points

init_pyramid = _diamond_(bottom_center=(0,0,0))
points_pyramid = tf.constant(init_pyramid, dtype=tf.float32)
counter = 3
fig5=plt_basic_object_(init_pyramid, counter)
st.pyplot(fig5)

x = st.slider("Enter for x:", -10, 10, step=1,key='my_slider7')
y = st.slider("Enter for y:", -10, 10, step=1,key='my_slider8')
z = st.slider("Enter for z:", -10, 10, step=1,key='my_slider9')

translation = tf.constant([x, y, z], dtype=tf.float32)

translated_points = points_pyramid + translation

fig6 = plt_basic_object_(translated_points.numpy(), counter)
st.pyplot(fig6)

def _sphere_(center=(0,0,0), radius=1, num_steps=20):
    center = np.array(center)
    u = np.linspace(0, 2*np.pi, num_steps)
    v = np.linspace(0, np.pi, num_steps)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    return points

def translate_obj(points, amount):
    return tf.add(points, amount)

init_sphere = _sphere_(center=(0,0,0), radius=2)
counter = 5
points = tf.constant(init_sphere, dtype=tf.float32)

x = st.slider("X Translation", -5.0, 5.0, 0.0, step=0.1)
y = st.slider("Y Translation", -5.0, 5.0, 0.0, step=0.1)
z = st.slider("Z Translation", -5.0, 5.0, 0.0, step=0.1)

translation_amount = tf.constant([x, y, z], dtype=tf.float32)
translated_object = translate_obj(points, translation_amount)

# plot translated sphere
fig5 = plt_basic_object_(translated_object.numpy(), title="Sphere (Translated)")
st.pyplot(fig5)
