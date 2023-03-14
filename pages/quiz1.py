# Import necessary libraries
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import streamlit as st

# If an image is uploaded, display the original image
# if uploaded_file is not None:
#     # Read the uploaded image file as a numpy array
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     opencv_image = cv2.imdecode(file_bytes, 1)

    # Display the original image
    # st.subheader("Original Image")
    # st.image(opencv_image, caption='Uploaded Image', use_column_width=True)

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

    # Create a file uploader widget for the user to select an image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # If an image is uploaded
    if uploaded_file is not None:
        # Set the title for the translation example
        st.title("Translation Example")
        
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


    # # Create a plot with the old and new images side by side
    # fig, axs = plt.subplots(2, 5, figsize=(18, 14))
    # fig = plt.gcf()
    # This line sets the window title to OpenCV Transformations
    # fig.canvas.manager.set_window_title('Quiz')

    # axs[0, 0].imshow(old_translated_img_1)
    # axs[0, 0].set_title("Number #1")
    # axs[0, 0].axis("off")

    # axs[0, 1].imshow(old_translated_img_2)
    # axs[0, 1].set_title("Number #2")
    # axs[0, 1].axis("off")

    # axs[0, 2].imshow(old_translated_img_3)
    # axs[0, 2].set_title("Number #3")
    # axs[0, 2].axis("off")

    # axs[0, 3].imshow(old_translated_img_4)
    # axs[0, 3].set_title("Number #4")
    # axs[0, 3].axis("off")

    # axs[0, 4].imshow(old_translated_img_5)
    # axs[0, 4].set_title("Number #5")
    # axs[1, 4].axis("off")

    # axs[1, 0].imshow(new_translated_img_1)
    # axs[1, 0].set_title("New")
    # axs[1, 0].axis("off")

    # axs[1, 1].imshow(new_translated_img_2)
    # axs[1, 1].set_title("New")
    # axs[1, 1].axis("off")

    # axs[1, 2].imshow(new_translated_img_3)
    # axs[1, 2].set_title("New")
    # axs[1, 2].axis("off")

    # axs[1, 3].imshow(new_translated_img_4)
    # axs[1, 3].set_title("New")
    # axs[1, 3].axis("off")

    # axs[1, 4].imshow(new_translated_img_5)
    # axs[1, 4].set_title("New")
    # axs[1, 4].axis("off")

    # Display the plot in the Streamlit app
    st.title("Original Image")
    st.image(img1)

    st.title("Translated Image")
    st.image(img2)

if __name__ == '__main__':
  main()



