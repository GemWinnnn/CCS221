import numpy as np
import matplotlib.pyplot as plt

two_d_arr = np.array([[1,0,1], [0,1,0],[1,0,1]])

def change(x, y, color):
    two_d_arr[x][y] = color
    img = plt.imshow(two_d_arr, interpolation='none', cmap='plasma')
    img.set_clim([0,50])
    plt.colorbar()
    plt.show()

def main():
    two_d_arr = np.array([[1,0,1], [0,1,0],[1,0,1]])
    for i in range(3):
        x_val = int(input("X coordinate(row 0 1 2):"))
        y_val = int(input("Y coordinate(column 0 1 2):"))
        c_val = int(input("Color Value (1-50)"))
        change(x_val, y_val, c_val)

if __name__ == '__main__':
    main()
