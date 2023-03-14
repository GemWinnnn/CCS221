import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay



def _plt_basic_object_(points, counter):
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
        plt.title("Sphere")

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


def main():
    init_sphere = _sphere_(center=(0,0,0), radius=2)
    points_sphere = tf.constant(init_sphere, dtype=tf.float32)
    counter = 4

    st.title("3D Object Translator")
    st.sidebar.title("Object Selection")

    object_choice = st.sidebar.selectbox(
        "Choose an object to translate:",
        ( "Pyramid",
          "Heart",
          "Diamond",
          "Sphere ")
    

  x = st.slider("X Translation", -5.0, 5.0, 0.0, step=0.1)
  y = st.slider("Y Translation", -5.0, 5.0, 0.0, step=0.1)
  z = st.slider("Z Translation", -5.0, 5.0, 0.0, step=0.1)

  if object_choice == "Pyramid":
      init_object = _pyramid_(bottom_center=(0,0,0))
      points = tf.constant(init_object, dtype=tf.float32)
      counter = 1
  elif object_choice == "Heart":
      init_object = _heart_(bottom_center=(0,0,0))
      points = tf.constant(init_object, dtype=tf.float32)
      counter = 2
  elif object_choice == "Diamond":
      init_object = _pyramid_(bottom_center=(0,0,0)) + _pyramid_(bottom_center=(0,0,5))
      points = tf.constant(init_object, dtype=tf.float32)
      counter = 3
  else:
      points = points_sphere

  translation_amount = tf.constant([x, y, z], dtype=tf.float32)
  translated_object = tf.add(points, translation_amount)

  with tf.Session() as session: 
      translated_points = session.run(translated_object)

  fig = _plt_basic_object_(translated_points, counter)
  st.pyplot(fig)

if __name__ == '__main__':
  main()
