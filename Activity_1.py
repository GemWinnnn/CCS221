import streamlit as st
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# DDA Line Algorithm
def DDALine(x1, y1, x2, y2): 
    # find absolute differences
    dx = x2 - x1 
    dy = y2 - y1
    # calculate steps required for generating pixels
    steps = abs(dx) if abs(dx) > abs(dy) else abs(dy)
    # calculate the increment in x and y for each step
    Xinc = float(dx / steps)
    Yinc = float(dy / steps)
    # make a list for coordinates
    X = []
    Y = []
    for i in range(0, int(steps + 1)): 
        # increment the values
        x1 += Xinc
        y1 += Yinc
        # append the x,y coordinates in respective list
        X.append(x1)
        Y.append(y1)
    plt.scatter(X, Y, color='red')
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("DDA Algorithm")

def main_DDA():
    x1 = st.number_input("Enter X1: ")
    y1 = st.number_input("Enter Y1: ")
    x2 = st.number_input("Enter X2: ")
    y2 = st.number_input("Enter Y2: ")
    DDALine(x1, y1, x2, y2)
    midX = (x1 + x2) / 2
    midY = (y1 + y2) / 2
    st.write("Midpoint of the line is at ({}, {})".format(midX, midY))
    plt.scatter(midX, midY, color='blue')
    st.pyplot()

# Bresenham Line Algorithm
def bres(x1, y1, x2, y2):
    x, y = x1, y1
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    p = 2 * dy - dx
    # Initialize the plotting points
    xcoordinates = [x]
    ycoordinates = [y]
    for k in range(2, int(dx) + 2):
        if p > 0:
            y = y + 1 if y < y2 else y - 1
            p = p + 2 * (dy - dx)
        else:
            p = p + 2 * dy
        x = x + 1 if x < x2 else x - 1
        xcoordinates.append(x)
        ycoordinates.append(y)
    plt.scatter(xcoordinates, ycoordinates, color='red')
    plt.title("Bresenham Algorithm")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")


def main_bres():
    x1 = st.number_input("Enter the Starting point of x: ")
    y1 = st.number_input("Enter the Starting point of y: ")
    x2 = st.number_input("Enter the end point of x: ")
    y2 = st.number_input("Enter the end point of y: ")
    bres(x1, y1, x2, y2)
    midX = (x1 + x2) // 2
    midY = (y1 + y2) // 2
    st.write("Midpoint of the line is at ({}, {})".format(midX, midY))
    plt.scatter(midX, midY, color='blue')
    st.pyplot()

# Main function for Stream
def main():
    st.title("Line Drawing Algorithms")
    st.sidebar.title("Select Algorithm")
    algorithms = ["DDA", "Bresenham"]
    choice = st.sidebar.selectbox("Choose an Algorithm", algorithms)

    if choice == "DDA":
        st.sidebar.markdown("### Enter the Coordinates")
        main_DDA()
    elif choice == "Bresenham":
        st.sidebar.markdown("### Enter the Coordinates")
        main_bres()

if __name__ == "__main__":
    main()


