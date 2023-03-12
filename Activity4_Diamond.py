import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay
import tensorflow as tf
import streamlit as st

def _plt_basic_object_(points):
    tri = Delaunay(points).convex_hull.data
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    S = ax.plot_trisurf(points[:,0], points[:,1], points[:,2],
                        triangles = tri,
                        shade = True, cmap = cm.rainbow, lw = 0.5)
    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-15, 15)
    st.pyplot(fig)

def _diamond_(bottom_lower=(0, 0, 0), side_length=5):
    bottom_lower = np.array(bottom_lower)
    u = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    v = np.array([0, np.pi/2, np.pi])
    x = bottom_lower[0] + side_length/2 * np.cos(u)
    y = bottom_lower[1] + side_length/2 * np.sin(u)
    z = bottom_lower[2] + side_length/2 * np.sin(v)
    points = np.vstack([x, y, z]).T
    return points

init_diamond = _diamond_(bottom_lower=(1, 2, 3), side_length=10)
points = tf.constant(init_diamond, dtype=tf.float32)

st.title("3D Diamond Translator")

x = st.slider("X Translation", -15.0, 15.0, 0.0, step=0.1)
y = st.slider("Y Translation", -15.0, 15.0, 0.0, step=0.1)
z = st.slider("Z Translation", -15.0, 15.0, 0.0, step=0.1)

translation_amount = tf.constant([x, y, z], dtype=tf.float32)
translated_object = tf.add(points, translation_amount)

with tf.Session() as session:
    translated_diamond = session.run(translated_object)

_plt_basic_object_(translated_diamond)
if __name__ == "__main__":
    main()
