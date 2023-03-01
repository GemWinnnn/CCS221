#!/usr/bin/env python
# coding: utf-8
Submitted by Canete, Biaca, Hallares | BSCS 1-B
# In[1]:


#Import Librarys 
import cv2
import numpy as np
#Load the image
image = cv2.imread('gojo.jpg')


# In[2]:


#Translation 
tx = int(input("Enter translation along x-axis (in pixels): "))
ty = int(input("Enter translation along y-axis (in pixels): "))
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))


# In[3]:


# Rotation
# Prompt the user for the rotation angle
angle = float(input('Enter the rotation angle in degrees: '))

# Get the dimensions of the image
(h, w) = image.shape[:2]

# Calculate the rotation matrix
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)

# Calculate the new dimensions of the rotated image
cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])
new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

# Adjust the rotation matrix to take into account the new dimensions
M[0, 2] += (new_w / 2) - center[0]
M[1, 2] += (new_h / 2) - center[1]

# Apply the rotation transformation to the image
rotated_image = cv2.warpAffine(image, M, (new_w, new_h))


# In[4]:


# Scaling
# Prompt the user for the new width and height
new_w = int(input('Enter the new width: '))
new_h = int(input('Enter the new height: '))

# Resize the image using cv2.resize
resized_image = cv2.resize(image, (new_w, new_h))


# In[ ]:


# Reflection
reflection = input("Enter 'x' to reflect along x-axis, 'y' to reflect along y-axis: ")
if reflection == 'x':
    reflection_matrix = np.float32([[-1, 0, image.shape[1]], [0, 1, 0]])
elif reflection == 'y':
    reflection_matrix = np.float32([[1, 0, 0], [0, -1, image.shape[0]]])
else:
    print("Invalid input, defaulting to no reflection.")
    reflection_matrix = np.float32([[1, 0, 0], [0, 1, 0]])
reflected_image = cv2.warpAffine(image, reflection_matrix, (image.shape[1], image.shape[0]))


# In[ ]:


# Shearing
shear = float(input("Enter shear factor: "))
shear_matrix = np.float32([[1, shear, 0], [0, 1, 0]])
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))


# In[ ]:


def shear_(shear_x):
    
    return sheared_img


# In[ ]:


# Display the original and transformed images
cv2.imshow('Original', image)
cv2.imshow('Translated', translated_image)
cv2.imshow('Rotated', rotated_image)
cv2.imshow('Scaled', scaled_image)
cv2.imshow('Reflected', reflected_image)
cv2.imshow('Sheared', sheared_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# #### 
