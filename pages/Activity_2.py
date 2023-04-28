import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def change(two_d_arr, x, y, color):
    two_d_arr[x][y] = color
    img = plt.imshow(two_d_arr, interpolation='none', cmap='plasma')
    img.set_clim([0,50])
    plt.colorbar()
    st.pyplot()

def main():
    st.title("2D Array Editor")

    two_d_arr = np.array([[1,0,1], [0,1,0], [1,0,1]])
    x_val = st.number_input("X coordinate (row 0-2):", key="x", min_value=0, max_value=2)
    y_val = st.number_input("Y coordinate (column 0-2):", key="y", min_value=0, max_value=2)
    c_val = st.number_input("Color Value (1-50):", key="c", min_value=1, max_value=50)

    change(two_d_arr, x_val, y_val, c_val)


if __name__ == '__main__':
    main()