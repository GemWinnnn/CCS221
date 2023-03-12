import streamlit as st
import numpy as np
import matplotlib.pyplot as plt



two_d_arr = np.array([[1,0,1], [0,1,0], [1,0,1]])

def change(x, y, color):
    global two_d_arr
    two_d_arr[x][y] = color
    img = plt.imshow(two_d_arr, interpolation='none', cmap='plasma')
    img.set_clim([0,50])
    plt.colorbar()
    st.pyplot()

def main():
    st.title("2D Array Editor")

    for i in range(3):
        x_val = st.number_input("X coordinate (row 0-2):", key=f"x_{i}", min_value=0, max_value=2)
        y_val = st.number_input("Y coordinate (column 0-2):", key=f"y_{i}", min_value=0, max_value=2)
        c_val = st.number_input("Color Value (1-50):", key=f"c_{i}", min_value=1, max_value=50)

        change(x_val, y_val, c_val)
        st.write("Updated Array:")
        st.write(two_d_arr)

if __name__ == '__main__':
    main()
