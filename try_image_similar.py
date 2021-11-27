from __future__ import annotations
from skimage import io
import pandas as pd
import numpy as np
from typing import *

class Image:
    src: str = None
    original_matrix: np.ndarray = None
    color_matrices: List[List[List[int]]] = None

    def __init__(self, src: str) -> None:
        self.src = src
        self.original_matrix = io.imread(src)
        # print('image resolution:', len(self.original_matrix[0]), len(self.original_matrix))
        self.color_matrices = [[[pixel[c] for pixel in row] for row in self.original_matrix] for c in range(3)]
    
    def get_color_diff(self, other: Image, x_slice_cnt: int, y_slice_cnt: int) -> Tuple[List[float], List[float]]:
        avg_diff_array = []
        stddev_diff_array = []

        for self_c_matrix, other_c_matrix in zip(self.color_matrices, other.color_matrices):
            self_c_avg_matrix = []
            other_c_avg_matrix = []

            self_c_splitted_matrix = []
            other_c_splitted_matrix = []

            self_slices: List[List[List[int]]] = [a.tolist() for a in np.array_split(self_c_matrix, y_slice_cnt)]
            other_slices: List[List[List[int]]] = [a.tolist() for a in np.array_split(other_c_matrix, y_slice_cnt)]
            
            # print('self_slices')
            # print(len(self_slices)) # slices along y axis
            # print(len(self_slices[0])) # rows in each slice
            # print(len(self_slices[0][0])) # columns in each slice

            for self_slice, other_slice in zip(self_slices, other_slices):
                self_splitted_slice: List[List[List[int]]] = [[a.tolist() for a in np.array_split(row, x_slice_cnt)] for row in self_slice]
                other_splitted_slice: List[List[List[int]]] = [[a.tolist() for a in np.array_split(row, x_slice_cnt)] for row in other_slice]

                # print('self_splitted_slice')
                # print(len(self_splitted_slice)) # rows in each slice
                # print(len(self_splitted_slice[0])) # split along x axis in a slice
                # print(len(self_splitted_slice[0][0])) # columns in a split
                
                # transpose
                self_cells = [list(a) for a in zip(*self_splitted_slice)]
                other_cells = [list(a) for a in zip(*other_splitted_slice)]

                self_c_splitted_matrix.append(self_cells)
                other_c_splitted_matrix.append(other_cells)

                # print('self_cells')
                # print(len(self_cells)) # splits along x axis
                # print(len(self_cells[0])) # rows in each slice
                # print(len(self_cells[0][0])) # coluns in a split

                self_c_avg_matrix.append([np.array(cell).mean() for cell in self_cells])
                other_c_avg_matrix.append([np.array(cell).mean() for cell in other_cells])
            
            # print('self_c_avg_matrix')
            # print(len(self_c_avg_matrix)) # length along y axis
            # print(len(self_c_avg_matrix[0])) # length along x axis
            # print('self_c_splitted_matrix')
            # print(len(self_c_splitted_matrix)) # splits along y axis
            # print(len(self_c_splitted_matrix[0])) # length along x axis
            # print(len(self_c_splitted_matrix[0][0])) # rows in each cell
            # print(len(self_c_splitted_matrix[0][0][0])) # columns in each cell

            # print(self_c_avg_matrix)

            c_diff_matrix = np.abs(np.array(self_c_avg_matrix) - np.array(other_c_avg_matrix))
            c_avg_diff = c_diff_matrix.mean()
            c_stddev_diff = c_diff_matrix.std()

            avg_diff_array.append(c_avg_diff)
            stddev_diff_array.append(c_stddev_diff)
        
        return avg_diff_array, stddev_diff_array


img1 = Image('https://cf.shopee.vn/file/b4af57597710a332afd293c40963b52f') # chat spam img 1
img2 = Image('https://cf.shopee.vn/file/263ec692d4a401bdc9bc170a0ab3cce0') # chat spam img 2
img3 = Image('https://cf.shopee.vn/file/be59f4b353e7ef7350005203378734d6') # chat non-spam img
img4 = Image('https://cf.shopee.sg/file/3aec17f4c5e51b54f37096ea62f31abc') # adidas pants image
img5 = Image('https://cf.shopee.vn/file/aafd3e625aaad3265ca0544b976197ab') # chat non-spam img

c_diff_similar, c_stddev_similar = Image.get_color_diff(img1, img2, 20, 20)
print('similar img')
print(c_diff_similar)
print(c_stddev_similar)

c_diff_non_similar, c_stddev_non_similar = Image.get_color_diff(img1, img3, 20, 20)
print('non-similar img')
print(c_diff_non_similar)
print(c_stddev_non_similar)

c_diff_non_similar, c_stddev_non_similar = Image.get_color_diff(img1, img4, 20, 20)
print('non-similar img')
print(c_diff_non_similar)
print(c_stddev_non_similar)

c_diff_non_similar, c_stddev_non_similar = Image.get_color_diff(img1, img5, 20, 20)
print('non-similar img')
print(c_diff_non_similar)
print(c_stddev_non_similar)

# test1 = io.imread('https://cf.shopee.vn/file/b4af57597710a332afd293c40963b52f')

# width = len(test1[0])
# height = len(test1)

# c0_matrix = np.array([[pixel[0] for pixel in row] for row in test1])
# c1_matrix = np.array([[pixel[1] for pixel in row] for row in test1])
# c2_matrix = np.array([[pixel[2] for pixel in row] for row in test1])

# print(type(c0_matrix))

# test2 = io.imread('https://cf.shopee.vn/file/263ec692d4a401bdc9bc170a0ab3cce0')