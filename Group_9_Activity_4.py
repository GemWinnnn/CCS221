#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay

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

    plt.show()


# In[3]:



def _sphere_(center=(0,0,0), radius=1, num_steps=20):
    center = np.array(center)
    u = np.linspace(0, 2*np.pi, num_steps)
    v = np.linspace(0, np.pi, num_steps)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    return points

init_sphere = _sphere_(center=(0,0,0), radius=2)
points = tf.constant(init_sphere, dtype=tf.float32)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    _plt_basic_object_(sess.run(points))


# In[ ]:


def translate_obj(points, amount):
    return tf.add(points, amount)

translation_amount = tf.constant([1,2,2], dtype=tf.float32)
translated_object = translate_obj(points,translation_amount)

with tf.Session() as session: 
    translated_cube = session.run(translated_object)

_plt_basic_object_(translated_cube)


# In[ ]:


def scale_obj(points,amount):
    return tf.multiply(points,amount)

scale_amount = tf.constant([4,2,2], dtype=tf.float32)
scaled_object = scale_obj(points,translation_amount)

with tf.compat.v1.Session() as session: 
    scaled_cube = session.run(scaled_object)
    
_plt_basic_object_(scaled_cube)


# In[ ]:


def shear_obj_x(points,xold,xnew,zold,znew):
    sh_x = tf.multiply(xold,xnew)
    sh_z = tf.multiply(zold,znew)
    
    shear_points = tf.stack([
                            [1,sh_x,0],
                            [0,1,0],
                            [0,sh_z,1]
                            ])
    sheare_object = tf.matmul(tf.cast(points,tf.float32),tf.cast(shear_points,tf.float32))
    return sheare_object

with tf.compat.v1.Session() as session: 
    sheared_object_x = session.run(shear_obj_x(points, 1, -1, 1, 1))

_plt_basic_object_(sheared_object_x)


# In[ ]:


def shear_obj_y(points, yold, ynew, zold, znew):
    sh_y = tf.multiply(yold, ynew)
    sh_z = tf.multiply(zold, znew)

    shear_points = tf.stack([
        [1, 0, 0],
        [0, sh_y, 0],
        [0, sh_z, 1]
    ])
    
    sheared_object = tf.matmul(tf.cast(points, tf.float32), tf.cast(shear_points, tf.float32))
    return sheared_object

with tf.compat.v1.Session() as session: 
    sheared_object_y = session.run(shear_obj_y(points, 1, -2, 1, 1))

_plt_basic_object_(sheared_object_y)


# In[ ]:


def rotate_sphere(points, angles):
    # Convert Cartesian coordinates to spherical coordinates
    r = tf.norm(points, axis=-1, keepdims=True)
    theta = tf.atan2(points[:,1], points[:,0])
    phi = tf.acos(tf.divide(points[:,2], r))

    # Apply rotation to the spherical coordinates
    theta += angles[0] * np.pi / 180.0
    phi += angles[1] * np.pi / 180.0

    # Convert spherical coordinates back to Cartesian coordinates
    x = r * tf.sin(phi) * tf.cos(theta)
    y = r * tf.sin(phi) * tf.sin(theta)
    z = r * tf.cos(phi)

    return tf.stack([x, y, z], axis=-1)


init_sphere = _sphere_(center=(0,0,0), radius=2)
points = tf.constant(init_sphere, dtype=tf.float32)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    _plt_basic_object_(sess.run(points))

    rotated_sphere = sess.run(rotate_sphere(points, [45, 0]))
    _plt_basic_object_(rotated_sphere)


# In[ ]:




