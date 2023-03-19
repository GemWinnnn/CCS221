# Import necessary libraries
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import streamlit as st

# Define a function to perform a translation transformation on an image
def translation(img, tx, ty):
    # Get the number of rows and columns in the image
    rows, cols = img.shape[:2]
    
    # Create a translation matrix
    m_translation = np.float32([[1, 0, tx], [0, 1, ty]])

    # Perform the translation transformation on the image
    translated_img = cv2.warpAffine(img, m_translation, (cols, rows))
    
    # Return the translated image
    return translated_img

# Define the main function that runs the Streamlit app
def main():
    # Set the title of the Streamlit app
    st.title("Image Uploader and Transformer")

     # Ask the user to input the X and Y values for the translation
    tx_1 = st.sidebar.number_input("X Value:", value=0)
    ty_1 = st.sidebar.number_input("Y Value:", value=0)
    
    # Define the old and new coordinates for the transformation
    BXold = 0
    BYold = 0
    BXnew = BXold + tx_1
    BYnew = BYold + ty_1

    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)

    img = cv2.imdecode (img_bytes,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img1 = translation(img,BXold,BYold)
    img2 = translation(img,BXnew,BYnew)

    # Display the plot in the Streamlit app
    st.title("Original Image")
    st.image(img1)

    st.title("Translated Image")
    st.image(img2)

if __name__ == '__main__':
  main()



