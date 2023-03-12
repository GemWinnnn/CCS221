import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def main():
    st.title("2D Array Editor")

    two_d_arr = np.array([[1,0,1], [0,1,0], [1,0,1]])
    st.write("Original Array:")
    st.write(two_d_arr)

    for i in range(3):
        x_val = st.number_input("X coordinate (row 0-2):", min_value=0, max_value=2)
        y_val = st.number_input("Y coordinate (column 0-2):", min_value=0, max_value=2)
        c_val = st.number_input("Color Value (1-50):", min_value=1, max_value=50)

        change(two_d_arr, x_val, y_val, c_val)

    st.write("Updated Array:")
    st.write(two_d_arr)
