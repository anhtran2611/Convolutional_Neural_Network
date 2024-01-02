import numpy as np


matrix_1d = np.array([1,2,3]) # Ma trận 1 chiều được biểu diễn như 1 vector

matrix_2d = np.array([[1,2,3],
                      [4,5,6],
                      [7,8,9]]) # Ma trận 2 chiều

matrix_3d = np.array([
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]],
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]]
])  # Ma trận 3 chiều là ma trận 2 chiều xếp chồng (2 layer và 3 row 3 column)

# print(matrix_3d.shape)

# Tạo padding matrix1d
matrix_1d_padding = np.pad(matrix_1d, (2,4), 'constant', constant_values=(3,1)) # thêm bên trái 2 cột, phải 4 cột

print(matrix_1d_padding)
# Tạo padding matrix2d
matrix_2d_padding = np.pad(matrix_2d, ((1,2),(2,3)), 'constant') # (top,botton),(left,right)
print(matrix_2d_padding)



# Shape matrix 3d
print(matrix_3d.shape)





