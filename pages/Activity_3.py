
import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

# Define the translate_image function
def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(image, M, (cols, rows))
    return translated_img

# Define the shear_image function
def shear_image(image, shear_factor):
    rows, cols = image.shape[:2]
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared_img = cv2.warpAffine(image, M, (cols, rows))
    return sheared_img

# Define the rotate_image function
def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_img = cv2.warpAffine(image, M, (cols, rows))
    return rotated_img

# Define the scale_image function
def scale_image(image, x_scale, y_scale):
    scaled_img = cv2.resize(image, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)
    return scaled_img

# Define the reflect_image function
def reflect_image(image, axis):
    if axis == 'x':
        reflected_img = cv2.flip(image, 0)
    elif axis == 'y':
        reflected_img = cv2.flip(image, 1)
    return reflected_img



st.title("Image Uploader and Transformer")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image file as numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Display original image
    st.subheader("Original Image")
    st.image(opencv_image, caption='Uploaded Image', use_column_width=True)

    # Define the transformation parameters
    tx = st.slider("Translation X", min_value=-100, max_value=100, value=0, step=1)
    ty = st.slider("Translation Y", min_value=-100, max_value=100, value=0, step=1)
    angle = st.slider("Rotation Angle", min_value=-180, max_value=180, value=0, step=1)
    x_scale = st.slider("X Scale", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    y_scale = st.slider("Y Scale", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    axis = st.radio("Reflection Axis", ('None', 'X', 'Y'))
    shear_factor = st.slider("Shear Factor", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)

    # Apply the transformations
    transformed_images = []

    # Translation
    if tx != 0 or ty != 0:
        translated_img = translate_image(opencv_image, tx, ty)
        transformed_images.append(("Translated", translated_img))

    # Rotation
    if angle != 0:
        rotated_img = rotate_image(opencv_image, angle)
        transformed_images.append(("Rotated", rotated_img))

    # Scaling
    if x_scale != 1.0 or y_scale != 1.0:
        scaled_img = scale_image(opencv_image, x_scale, y_scale)
        transformed_images.append(("Scaled", scaled_img))

    # Reflection
    if axis != 'None':
        reflected_img = reflect_image(opencv_image, axis.lower())
        transformed_images.append(("Reflected", reflected_img))

    # Shear
    if shear_factor != 0:
        sheared_img = shear_image(opencv_image, shear_factor)
        transformed_images.append(("Sheared", sheared_img))

    # Display transformed images
    if len(transformed_images) > 0:
        st.subheader("Transformed Images")
        cols = st.columns(len(transformed_images))
        for i, (title, img) in enumerate(transformed_images):
            with cols[i]:
                st.image(img, caption=title, use_column_width=True)