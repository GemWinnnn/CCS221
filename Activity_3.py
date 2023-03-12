import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def translate_image(img, tx, ty):
    M = np.float32([[1, 0, tx], 
                    [0, 1, ty]])
    translated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return translated_img

def rotate_image(img, angle):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (cols, rows))
    return rotated_img

def scale_image(img, x_scale, y_scale):
    scaled_img = cv2.resize(img, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_CUBIC)
    return scaled_img

def reflect_image(img, axis):
    if axis == 'x':
        reflected_img = cv2.flip(img, 0)
    elif axis == 'y':
        reflected_img = cv2.flip(img, 1)
    else:
        reflected_img = img
    return reflected_img

def shear_image(img, shear_factor):
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return sheared_img

def main():
    st.title("Image Transformations")

    # Read the original images and convert to RGB
    img1 = cv2.cvtColor(cv2.imread("img1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("img2.png"), cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(cv2.imread("img3.png"), cv2.COLOR_BGR2RGB)

    # Define the transformation parameters using sliders
    tx = st.slider("Translation X", -200, 200, 50, 10)
    ty = st.slider("Translation Y", -200, 200, -50, 10)
    angle = st.slider("Rotation Angle", -180, 180, 30, 10)
    x_scale = st.slider("Scaling X", 0.1, 2.0, 1.5, 0.1)
    y_scale = st.slider("Scaling Y", 0.1, 2.0, 0.8, 0.1)
    axis = st.selectbox("Reflection Axis", ['None', 'X', 'Y'])
    shear_factor = st.slider("Shear Factor", -0.5, 0.5, 0.3, 0.1)

    # Loop through the images and apply the transformations
    for img in [img1, img2, img3]:
        plt.figure(figsize=(10,10))
        for i in range(2):
            # Translation
            translated_img = translate_image(img, tx, ty)
            plt.subplot(3, 3, 1 + (i * 6) % 6)
            plt.imshow(translated_img)
            plt.title(f"Translated {tx}, {ty}")
            plt.axis('off')

            # Rotation
            rotated_img = rotate_image(img, angle)
            plt.subplot(3, 3, 2 + (i * 6) % 6)
            plt.imshow(rotated_img)
            plt.title(f"Rotation {tx}, {ty}")
            
             # Scaling
        scaled_img = scale_image(img, x_scale, y_scale)
        plt.subplot(3, 3, 3 + (i * 6) % 6)
        plt.imshow(scaled_img)
        plt.title(f"Scaled {x_scale}, {y_scale}")
        plt.axis('off')

        # Reflection
        if axis != 'None':
            reflected_img = reflect_image(img, axis.lower())
        else:
            reflected_img = img
        plt.subplot(3, 3, 4 + (i * 6) % 6)
        plt.imshow(reflected_img)
        plt.title(f"Reflected {axis}-axis")
        plt.axis('off')

        # Shear
        sheared_img = shear_image(img, shear_factor)
        plt.subplot(3, 3, 5 + (i * 6) % 6)
        plt.imshow(sheared_img)
        plt.title(f"Sheared {shear_factor}")
        plt.axis('off')

    st.pyplot(plt)
    plt.close()
            
            
