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

    plt.show()


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

    plt.show()


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


init_rectangular_prism = _rectangle_(bottom_lower=(1, 2, 5), side_lengths=(7, 5, 4))
points = tf.constant(init_rectangular_prism, dtype=tf.float32)

_plt_basic_object_(init_rectangular_prism)

@tf.function
def translate_obj(points, amount):  
    return tf.add(points, amount)

# Update the values here to move the diamond around x , y , z
x = float(input("Enter the x component of the vector: "))
y = float(input("Enter the y component of the vector: "))
z = float(input("Enter the z component of the vector: "))
translation_amount = tf.constant([x, y, z], dtype=tf.float32)
translated_object = translate_obj(points, translation_amount)

translated_rectangular_prism = translated_object.numpy()
_plt_basic_object_(translated_rectangular_prism)



init_diamond_ = _diamond_(bottom_lower=(1, 2, 3), side_length=10)
points = tf.constant(init_diamond_, dtype=tf.float32)

_plt_basic_object_(init_diamond_)

@tf.function
def translate_obj(points, amount):  
    return tf.add(points, amount)

# Update the values here to move the diamond around x , y , z
x = float(input("Enter the x component of the vector: "))
y = float(input("Enter the y component of the vector: "))
z = float(input("Enter the z component of the vector: "))
translation_amount = tf.constant([x, y, z], dtype=tf.float32)
translated_object = translate_obj(points, translation_amount)

translated_diamond = translated_object.numpy()
_plt_basic_object_(translated_diamond)
