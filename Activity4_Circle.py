import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

def _plt_basic_object_(points):

    tri = Delaunay(points).convex_hull.data

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    S = ax.plot_trisurf(points[:,0], points[:,1], points[:,2],
                        triangles = tri,
                        shade = True, cmap = cm.rainbow, lw = 0.5)

    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)

    st.pyplot(fig)

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

def main():
    init_sphere = _sphere_(center=(0,0,0), radius=2)
    points = tf.constant(init_sphere, dtype=tf.float32)

    st.title("3D Sphere Translator")

    x = st.slider("X Translation", -5.0, 5.0, 0.0, step=0.1)
    y = st.slider("Y Translation", -5.0, 5.0, 0.0, step=0.1)
    z = st.slider("Z Translation", -5.0, 5.0, 0.0, step=0.1)

    translation_amount = tf.constant([x, y, z], dtype=tf.float32)
    translated_object = translate_obj(points, translation_amount)

    with tf.Session() as session: 
        translated_sphere = session.run(translated_object)

    _plt_basic_object_(translated_sphere)

if __name__ == "__main__":
    main()
